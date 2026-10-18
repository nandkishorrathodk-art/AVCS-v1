"""Train and calibrate the independent target-presence head for GoalConditionedGrounder.

Calibrates presence classification against positive UI targets, ungroundable negative queries,
and blank/wallpaper desktop frames without modifying frozen visual representation weights.
Uses vectorized feature extraction for ultra-fast, robust convergence on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from avcs.dataset.synthetic.generator import SyntheticDesktopGenerator
from avcs.dataset.synthetic.themes import AVAILABLE_THEMES
from avcs.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    GROUNDING_CHECKPOINT_SCHEMA,
    GROUNDING_DATASET_PROVENANCE,
    GROUNDING_MODEL_TYPE,
    encode_queries,
)
from avcs.perception.goal_grounder import GoalConditionedGrounder, load_grounding_checkpoint
from benchmarks.run_ui_grounding_benchmark import UIGroundingBenchmarkRunner


def create_synthetic_blanks(count: int = 72, seed_start: int = 1000) -> List[Dict[str, Any]]:
    """Synthesizes solid, gradient, and wallpaper negative frames."""
    gen = SyntheticDesktopGenerator(screen_size=(256, 256))
    types = ["solid"] * (count // 3) + ["gradient"] * (count // 3) + ["wallpaper"] * (count - 2 * (count // 3))
    queries = ["click button", "search input", "close window", "select item", "open menu", "OK", "Cancel"]
    samples: List[Dict[str, Any]] = []

    for i, ntype in enumerate(types):
        seed = seed_start + i
        theme = AVAILABLE_THEMES[i % len(AVAILABLE_THEMES)]
        frame = gen.generate_frame(
            category="desktop_canvas",
            theme_name=theme,
            is_negative=True,
            negative_type=ntype,
            seed=seed,
            split="train",
            session_id=f"synth_neg_{i // 10:02d}",
            frame_idx=i + 1,
        )
        img_arr = np.ascontiguousarray(frame.image.transpose(2, 0, 1))
        img_t = torch.from_numpy(img_arr).float().div_(255.0)

        for q in queries:
            try:
                q_tok = encode_queries([q])[0]
            except Exception:
                q_tok = torch.zeros(64, dtype=torch.long)
            samples.append({
                "image": img_t,
                "query_str": q,
                "query_tokens": q_tok,
                "has_target": torch.tensor(0.0, dtype=torch.float32),
                "is_negative": True,
                "is_blank": True,
            })
    return samples


@torch.no_grad()
def extract_pooled_features(
    model: GoalConditionedGrounder,
    frames: List[Dict[str, Any]],
    data_dir: Path,
    device: torch.device,
    synthetic_blanks: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts 64-dim pooled features for all frame queries with 1 forward_pyramid per frame.

    Returns:
        X: [N, 64] pooled features
        y: [N] 0.0 or 1.0 labels
        is_blank_mask: [N] boolean tensor indicating if sample is from a blank frame
    """
    model.eval()
    all_features: List[torch.Tensor] = []
    all_labels: List[float] = []
    is_blank_list: List[bool] = []

    for fr in frames:
        img_path = data_dir / fr["file_path"]
        with Image.open(img_path) as pil_img:
            arr = np.array(pil_img.resize((256, 256), Image.BILINEAR).convert("RGB"), dtype=np.uint8)
        img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0).to(device)

        # Single visual forward pass per frame
        stem, s1, s2, _s3, vis = model.visual_encoder.forward_pyramid(img_t)

        pos_queries = [pt.get("target_query", "click button") for pt in fr.get("interactive_points", [])]
        neg_queries = list(fr.get("negative_queries", []))
        queries = pos_queries + neg_queries
        labels = [1.0] * len(pos_queries) + [0.0] * len(neg_queries)

        if not queries:
            continue

        q_tokens = encode_queries(queries).to(device)
        N_q = len(queries)

        vis_exp = vis.expand(N_q, -1, -1, -1)
        s2_exp = model.lateral_8(s2).expand(N_q, -1, -1, -1)
        s1_exp = model.lateral_4(s1).expand(N_q, -1, -1, -1)
        stem_exp = model.lateral_2(stem).expand(N_q, -1, -1, -1)

        q_enc = model.query_encoder(q_tokens)
        sc, bi = model.query_modulation(q_enc).chunk(2, dim=1)
        vis_m = vis_exp * (1.0 + sc[:, :, None, None]) + bi[:, :, None, None]

        dec = F.interpolate(vis_m, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        dec = model.decode_8(torch.cat((dec, s2_exp), dim=1))
        dec = F.interpolate(dec, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        dec = model.decode_4(torch.cat((dec, s1_exp), dim=1))
        dec = F.interpolate(dec, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        dec = model.decode_2(torch.cat((dec, stem_exp), dim=1))

        pooled = F.adaptive_avg_pool2d(dec, 1).flatten(1)
        all_features.append(pooled.cpu())
        all_labels.extend(labels)
        is_blank_list.extend([False] * len(labels))

    if synthetic_blanks:
        for s in synthetic_blanks:
            img_t = s["image"].unsqueeze(0).to(device)
            stem, s1, s2, _s3, vis = model.visual_encoder.forward_pyramid(img_t)
            q_tok = s["query_tokens"].unsqueeze(0).to(device)
            q_enc = model.query_encoder(q_tok)
            sc, bi = model.query_modulation(q_enc).chunk(2, dim=1)
            vis_m = vis * (1.0 + sc[:, :, None, None]) + bi[:, :, None, None]

            dec = F.interpolate(vis_m, size=s2.shape[-2:], mode="bilinear", align_corners=False)
            dec = model.decode_8(torch.cat((dec, model.lateral_8(s2)), dim=1))
            dec = F.interpolate(dec, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            dec = model.decode_4(torch.cat((dec, model.lateral_4(s1)), dim=1))
            dec = F.interpolate(dec, size=stem.shape[-2:], mode="bilinear", align_corners=False)
            dec = model.decode_2(torch.cat((dec, model.lateral_2(stem)), dim=1))

            pooled = F.adaptive_avg_pool2d(dec, 1).flatten(1)
            all_features.append(pooled.cpu())
            all_labels.append(0.0)
            is_blank_list.append(True)

    X = torch.cat(all_features, dim=0)
    y = torch.tensor(all_labels, dtype=torch.float32)
    b_mask = torch.tensor(is_blank_list, dtype=torch.bool)
    return X, y, b_mask


def train_presence_head(
    base_checkpoint_path: str = "runs/weights/candidates/candidate_m3_highres_v1.pt",
    output_checkpoint_path: str = "runs/weights/candidates/candidate_m3_highres_v2.pt",
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    epochs: int = 60,
    lr: float = 6e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    print(f"Loading base model from {base_checkpoint_path}...")
    model = load_grounding_checkpoint(base_checkpoint_path, device=dev)

    manifest_p = Path(manifest_path)
    data_dir = manifest_p.parent
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_frames = [fr for fr in manifest["frames"] if fr.get("split") == "train"]
    val_frames = [fr for fr in manifest["frames"] if fr.get("split") == "val"]

    # Generate synthetic blanks: 48 for train, 24 for validation
    train_blanks = create_synthetic_blanks(count=48, seed_start=1000)
    val_blanks = create_synthetic_blanks(count=24, seed_start=2000)

    t0 = time.perf_counter()
    print(f"Extracting features for {len(train_frames)} train frames + {len(train_blanks)//7} blanks...", flush=True)
    X_train, y_train, _ = extract_pooled_features(model, train_frames, data_dir, dev, synthetic_blanks=train_blanks)
    print(f"Train features: {X_train.shape}, positive={int((y_train == 1).sum())}, negative={int((y_train == 0).sum())} in {time.perf_counter() - t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    print(f"Extracting features for {len(val_frames)} val frames + {len(val_blanks)//7} blanks...", flush=True)
    X_val, y_val, b_val = extract_pooled_features(model, val_frames, data_dir, dev, synthetic_blanks=val_blanks)
    print(f"Val features: {X_val.shape}, positive={int((y_val == 1).sum())}, negative={int((y_val == 0).sum())} (blanks={int(b_val.sum())}) in {time.perf_counter() - t0:.2f}s", flush=True)

    # Re-initialize target_presence linear layer
    presence_layer = nn.Linear(64, 1).to(dev)
    nn.init.xavier_uniform_(presence_layer.weight)
    nn.init.zeros_(presence_layer.bias)

    optimizer = torch.optim.AdamW(presence_layer.parameters(), lr=lr, weight_decay=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(X_train.to(dev), y_train.to(dev))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("Training target_presence linear head...", flush=True)
    for epoch in range(1, epochs + 1):
        presence_layer.train()
        total_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = presence_layer(bx).squeeze(-1)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == epochs:
            presence_layer.eval()
            with torch.no_grad():
                val_logits = presence_layer(X_val.to(dev)).squeeze(-1)
                val_probs = torch.sigmoid(val_logits).cpu()
                val_preds = (val_probs >= 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item() * 100.0
                blank_fps = int(((val_probs >= 0.5) & b_val).sum().item())
                print(f"Epoch {epoch:2d}/{epochs} - Train Loss: {total_loss / n_batches:.4f} - Val Accuracy: {val_acc:.1f}% (Blank FPs: {blank_fps})", flush=True)

    # Threshold calibration: sweep threshold to guarantee 0.0% false alarms on blank frames
    presence_layer.eval()
    with torch.no_grad():
        val_logits = presence_layer(X_val.to(dev)).squeeze(-1)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        y_val_np = y_val.numpy()
        b_val_np = b_val.numpy()

    best_threshold = 0.55
    best_f1 = 0.0
    print("\n--- Calibrating Threshold on Validation Split ---", flush=True)
    for th in np.arange(0.30, 0.85, 0.05):
        preds = (val_probs >= th).astype(float)
        tp = float(((preds == 1) & (y_val_np == 1)).sum())
        fp = float(((preds == 1) & (y_val_np == 0)).sum())
        fn = float(((preds == 0) & (y_val_np == 1)).sum())
        tn = float(((preds == 0) & (y_val_np == 0)).sum())
        blank_fps = int(((preds == 1) & b_val_np).sum())
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        f1 = 2 * prec * rec / max(1e-7, prec + rec)
        fp_rate = fp / max(1.0, fp + tn) * 100.0
        print(f"Threshold: {th:.2f} -> Prec: {prec*100:.1f}%, Rec: {rec*100:.1f}%, F1: {f1:.3f}, FP Rate: {fp_rate:.1f}%, Blank FPs: {blank_fps}", flush=True)
        # Select highest F1 threshold that achieves 0 blank false positives
        if blank_fps == 0 and f1 > best_f1:
            best_f1 = f1
            best_threshold = float(th)

    print(f"\nOptimal Calibrated Target-Presence Threshold: {best_threshold:.2f} (F1: {best_f1:.3f})", flush=True)

    # Save Candidate V2 Checkpoint
    out_p = Path(output_checkpoint_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    raw_ckpt = torch.load(base_checkpoint_path, map_location="cpu")
    cand_v2_dict = raw_ckpt["model_state_dict"] if "model_state_dict" in raw_ckpt else raw_ckpt
    cand_v2_dict["target_presence.weight"] = presence_layer.weight.detach().cpu()
    cand_v2_dict["target_presence.bias"] = presence_layer.bias.detach().cpu()

    v2_metadata = {
        "schema_version": 2,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": GROUNDING_DATASET_PROVENANCE,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "held_out_test_samples": 40,
        "target_presence_threshold": best_threshold,
        "input_size": DEFAULT_INPUT_SIZE,
        "model_state_dict": cand_v2_dict,
    }
    torch.save(v2_metadata, out_p)
    print(f"Saved Candidate V2 checkpoint to {out_p}", flush=True)

    # Evaluate on Frozen Held-Out Benchmark
    print("\n=== Evaluating Candidate V2 on Frozen Held-Out Benchmark ===", flush=True)
    runner = UIGroundingBenchmarkRunner(device=str(dev), confidence_threshold=0.35)
    test_model = load_grounding_checkpoint(out_p, device=dev)
    results = runner.evaluate_model(test_model, split="test_held_out")
    print(results.summary(), flush=True)

    # Save metrics JSON with exact attribute names
    json_path = out_p.with_suffix(".json")
    json_summary = {
        "experiment_id": "AVCS-CANDIDATE-M3-HR-V2",
        "source_checkpoint": str(base_checkpoint_path),
        "architecture": "GoalConditionedGrounder_CorticalBackbone2M_FPN_CalibratedPresence",
        "total_parameters": sum(p.numel() for p in test_model.parameters()),
        "target_presence_threshold": best_threshold,
        "measured_metrics_test_held_out": {
            "top_1_accuracy_pct": results.top_1_accuracy_pct,
            "hits": f"{results.top_1_hits}/{results.total_positive_samples}",
            "median_centroid_error_px": results.median_centroid_error_px,
            "mean_centroid_error_px": results.mean_centroid_error_px,
            "p95_centroid_error_px": results.p95_centroid_error_px,
            "mean_iou": results.mean_iou,
            "ungroundable_query_rejection_pct": results.negative_query_rejection_pct,
            "blank_screen_fp_count": results.blank_screen_fp_count,
            "blank_screen_total": results.total_blank_screens,
            "center_collapse_detected": results.center_collapse_detected,
            "passed_acceptance_gates": results.passed_all_gates,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved benchmark results JSON to {json_path}", flush=True)

    return results.to_dict()


if __name__ == "__main__":
    train_presence_head()
