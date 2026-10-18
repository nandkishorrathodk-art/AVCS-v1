"""Training pipeline for CandidateReranker (Milestone 3.1).

Trains the lightweight neural candidate reranker on top of frozen Coarse Candidate V2 features:
- Discovers Top-K candidate peaks per target.
- Optimizes Cross-Entropy and Margin loss to select the candidate closest to ground truth.
- Validates on validation split.
- Saves candidate checkpoint with metadata to runs/weights/candidates/candidate_m3_reranker_v1.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from avcs.perception.goal_grounder import load_grounding_checkpoint, encode_queries
from avcs.perception.candidate_disambiguator import (
    CandidateReranker,
    GlobalCandidateDisambiguator,
    extract_spatial_peaks_nms,
    save_disambiguator_checkpoint,
)


def train_reranker(
    coarse_checkpoint: str = "runs/weights/candidates/candidate_m3_highres_v2.pt",
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    save_path: str = "runs/weights/candidates/candidate_m3_reranker_v1.pt",
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    k_candidates: int = 4,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dev = torch.device(device)

    print(f"=== Loading Coarse Feature Extractor: {coarse_checkpoint} ===")
    coarse_model = load_grounding_checkpoint(coarse_checkpoint, device=dev)
    coarse_model.eval()
    for param in coarse_model.parameters():
        param.requires_grad = False

    manifest_p = Path(manifest_path)
    data_dir = manifest_p.parent
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_frames = [fr for fr in manifest["frames"] if fr.get("split") == "train"]
    val_frames = [fr for fr in manifest["frames"] if fr.get("split") == "val"]

    print(f"Dataset: {len(train_frames)} train frames, {len(val_frames)} val frames")

    # Pre-cache images in memory for fast training loops
    image_cache: Dict[str, torch.Tensor] = {}
    for fr in train_frames + val_frames:
        fp = str(data_dir / fr["file_path"])
        if os.path.isfile(fp):
            with Image.open(fp) as img:
                arr = np.array(img.resize((256, 256), Image.BILINEAR).convert("RGB"), dtype=np.uint8)
                t = torch.from_numpy(arr.transpose(2, 0, 1)).float().div_(255.0)
                image_cache[fp] = t

    # Build demonstration items
    train_items = []
    for fr in train_frames:
        fp = str(data_dir / fr["file_path"])
        if fp not in image_cache:
            continue
        for pt in fr.get("interactive_points", []):
            train_items.append({
                "file_path": fp,
                "point": pt,
                "query": pt.get("target_query", "click button"),
                "gt_x": float(pt["x"]),
                "gt_y": float(pt["y"]),
                "gt_box": pt.get("bbox", [float(pt["y"]) - 0.04, float(pt["x"]) - 0.04, float(pt["y"]) + 0.04, float(pt["x"]) + 0.04]),
            })

    val_items = []
    for fr in val_frames:
        fp = str(data_dir / fr["file_path"])
        if fp not in image_cache:
            continue
        for pt in fr.get("interactive_points", []):
            val_items.append({
                "file_path": fp,
                "point": pt,
                "query": pt.get("target_query", "click button"),
                "gt_x": float(pt["x"]),
                "gt_y": float(pt["y"]),
                "gt_box": pt.get("bbox", [float(pt["y"]) - 0.04, float(pt["x"]) - 0.04, float(pt["y"]) + 0.04, float(pt["x"]) + 0.04]),
            })

    print(f"Extracted {len(train_items)} train targets, {len(val_items)} val targets")

    # Initialize Reranker network
    reranker = CandidateReranker().to(dev)
    total_params = sum(p.numel() for p in reranker.parameters())
    print(f"CandidateReranker initialized: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(reranker.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_top1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        reranker.train()
        random.shuffle(train_items)
        train_losses = []
        train_correct = 0
        train_total = 0

        # Train in mini-batches
        for b_start in range(0, len(train_items), batch_size):
            batch = train_items[b_start : b_start + batch_size]
            if len(batch) < 2:
                continue

            images = torch.stack([image_cache[item["file_path"]] for item in batch]).to(dev)
            queries = [item["query"] for item in batch]
            q_tokens = encode_queries(queries).to(dev)

            with torch.no_grad():
                decoded = coarse_model._decode(images, q_tokens)
                heatmap_logits = coarse_model.output(decoded)
                global_feats = F.adaptive_avg_pool2d(decoded, output_size=1).flatten(1)
                query_embs = coarse_model.query_encoder(q_tokens)

                batch_peaks = extract_spatial_peaks_nms(
                    heatmap_logits,
                    k=k_candidates,
                    min_separation_cells=4,
                    min_logit_threshold=-6.0,
                )

            # Assemble features for batch
            batch_local_feats = []
            batch_coords = []
            batch_logits = []
            batch_labels = []
            valid_indices = []

            H_feat, W_feat = heatmap_logits.shape[-2:]

            for i, item in enumerate(batch):
                peaks = batch_peaks[i]
                if not peaks:
                    continue

                gt_x = item["gt_x"]
                gt_y = item["gt_y"]

                # Extract coords
                c_coords = []
                c_logits = []
                dists_to_gt = []
                for r, c, l in peaks:
                    cx = (float(c) + 0.5) / float(W_feat)
                    cy = (float(r) + 0.5) / float(H_feat)
                    c_coords.append([cx, cy])
                    c_logits.append([l])
                    dists_to_gt.append(math.hypot((cx - gt_x) * 255, (cy - gt_y) * 255))

                # Best candidate index
                best_cand_idx = int(np.argmin(dists_to_gt))
                # If closest candidate is within reasonable proximity (< 25px)
                if dists_to_gt[best_cand_idx] <= 25.0:
                    valid_indices.append(i)
                    batch_labels.append(best_cand_idx)
                    row_col_pairs = [(r, c) for r, c, _ in peaks]
                    loc_f = reranker.pool_local_context(decoded[i : i + 1], row_col_pairs)  # [1, K, 576]
                    batch_local_feats.append(loc_f[0])
                    batch_coords.append(torch.tensor(c_coords, device=dev, dtype=torch.float32))
                    batch_logits.append(torch.tensor(c_logits, device=dev, dtype=torch.float32))

            if not valid_indices:
                continue

            sub_local = torch.stack(batch_local_feats, dim=0)    # [N, K, 576]
            sub_coords = torch.stack(batch_coords, dim=0)        # [N, K, 2]
            sub_logits = torch.stack(batch_logits, dim=0)        # [N, K, 1]
            sub_global = global_feats[valid_indices]             # [N, 64]
            sub_query = query_embs[valid_indices]                # [N, 192]
            labels = torch.tensor(batch_labels, device=dev, dtype=torch.long)

            optimizer.zero_grad()
            scores = reranker(
                local_features=sub_local,
                global_features=sub_global,
                query_emb=sub_query,
                coords_norm=sub_coords,
                coarse_logits=sub_logits,
            )  # [N, K]

            loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            preds = scores.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        train_acc = (train_correct / max(1, train_total)) * 100.0
        avg_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Validation evaluation
        reranker.eval()
        val_correct = 0
        val_total = 0
        val_coarse_top1_correct = 0

        with torch.no_grad():
            for item in val_items:
                img_t = image_cache[item["file_path"]].unsqueeze(0).to(dev)
                q_tokens = encode_queries([item["query"]]).to(dev)

                decoded = coarse_model._decode(img_t, q_tokens)
                heatmap_logits = coarse_model.output(decoded)
                global_feat = F.adaptive_avg_pool2d(decoded, output_size=1).flatten(1)
                query_emb = coarse_model.query_encoder(q_tokens)

                batch_peaks = extract_spatial_peaks_nms(
                    heatmap_logits, k=k_candidates, min_separation_cells=4, min_logit_threshold=-6.0
                )[0]
                if not batch_peaks:
                    continue

                gt_x = item["gt_x"]
                gt_y = item["gt_y"]
                c_coords = []
                c_logits = []
                dists_to_gt = []
                H_feat, W_feat = heatmap_logits.shape[-2:]
                for r, c, l in batch_peaks:
                    cx = (float(c) + 0.5) / float(W_feat)
                    cy = (float(r) + 0.5) / float(H_feat)
                    c_coords.append([cx, cy])
                    c_logits.append([l])
                    dists_to_gt.append(math.hypot((cx - gt_x) * 255, (cy - gt_y) * 255))

                best_idx = int(np.argmin(dists_to_gt))
                if dists_to_gt[best_idx] > 25.0:
                    continue

                row_col_pairs = [(r, c) for r, c, _ in batch_peaks]
                loc_f = reranker.pool_local_context(decoded, row_col_pairs)
                coords_t = torch.tensor([c_coords], device=dev, dtype=torch.float32)
                logits_t = torch.tensor([c_logits], device=dev, dtype=torch.float32)

                scores = reranker(
                    local_features=loc_f,
                    global_features=global_feat,
                    query_emb=query_emb,
                    coords_norm=coords_t,
                    coarse_logits=logits_t,
                )
                pred_idx = scores[0].argmax().item()

                if pred_idx == best_idx:
                    val_correct += 1
                if 0 == best_idx:
                    val_coarse_top1_correct += 1
                val_total += 1

        val_acc = (val_correct / max(1, val_total)) * 100.0
        val_coarse_acc = (val_coarse_top1_correct / max(1, val_total)) * 100.0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Val Rerank Acc: {val_acc:.1f}% (Coarse Top-1 on Val: {val_coarse_acc:.1f}%)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_rerank_acc": val_acc,
            "val_coarse_acc": val_coarse_acc,
        })

        if val_acc >= best_val_top1:
            best_val_top1 = val_acc
            disambiguator = GlobalCandidateDisambiguator(
                coarse_model=coarse_model,
                reranker=reranker,
                k_candidates=k_candidates,
            )
            save_disambiguator_checkpoint(
                disambiguator,
                save_path=save_path,
                extra_metadata={
                    "best_val_rerank_acc": best_val_top1,
                    "val_coarse_acc": val_coarse_acc,
                    "epochs_trained": epoch,
                    "coarse_checkpoint": coarse_checkpoint,
                },
            )

    print(f"\nTraining Complete. Best Validation Rerank Accuracy: {best_val_top1:.2f}%. Saved to {save_path}")
    return {"best_val_acc": best_val_top1, "history": history, "save_path": save_path}


if __name__ == "__main__":
    train_reranker()

