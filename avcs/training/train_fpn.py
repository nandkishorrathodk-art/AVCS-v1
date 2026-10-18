"""Training script for Path 1: FPNRegionProposer on UI Grounding Dataset.

Trains multi-scale heads:
- P2 (64x64) for small targets (<24 px)
- P3 (32x32) for medium targets (24-64 px)
- P4 (16x16) for large targets (>64 px) + Category classification

Trains on CPU with train/val isolation from datasets/ui_grounding/manifest.json.
Saves versioned candidate checkpoint: runs/weights/candidates/candidate_m3_fpn_v1.pt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.perception.fpn_proposer import FPNRegionProposer
from avcs.perception.visual_pilot import ELEMENT_KINDS, UI_CATEGORIES


class FPNDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train") -> None:
        self.data_root = Path(data_root)
        manifest_path = self.data_root / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.frames = [f for f in manifest["frames"] if f["split"] == split]
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(UI_CATEGORIES)}
        self.kind_to_idx = {k: idx for idx, k in enumerate(ELEMENT_KINDS)}

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        f_meta = self.frames[idx]
        screen_path = self.data_root / f_meta["file_path"]

        img = Image.open(screen_path).convert("RGB").resize((256, 256))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1) # (3, 256, 256)

        cat_str = f_meta["category"]
        cat_label = torch.tensor(self.cat_to_idx.get(cat_str, 0), dtype=torch.long)

        # Target grids for P2 (64), P3 (32), P4 (16)
        # Class masks: 0=static, 1=interactive
        mask_p2 = torch.zeros(64, 64, dtype=torch.long)
        mask_p3 = torch.zeros(32, 32, dtype=torch.long)
        mask_p4 = torch.zeros(16, 16, dtype=torch.long)

        box_p2 = torch.zeros(4, 64, 64, dtype=torch.float32)
        box_p3 = torch.zeros(4, 32, 32, dtype=torch.float32)
        box_p4 = torch.zeros(4, 16, 16, dtype=torch.float32)

        kind_p2 = torch.zeros(64, 64, dtype=torch.long)
        kind_p3 = torch.zeros(32, 32, dtype=torch.long)
        kind_p4 = torch.zeros(16, 16, dtype=torch.long)

        reg_mask_p2 = torch.zeros(64, 64, dtype=torch.bool)
        reg_mask_p3 = torch.zeros(32, 32, dtype=torch.bool)
        reg_mask_p4 = torch.zeros(16, 16, dtype=torch.bool)

        H, W = f_meta.get("height", 256), f_meta.get("width", 256)

        for pt in f_meta.get("interactive_points", []):
            cx = pt["x"]
            cy = pt["y"]
            bbox = pt.get("bbox", [cy - 0.05, cx - 0.05, cy + 0.05, cx + 0.05])
            max_dim = max((bbox[2] - bbox[0]) * H, (bbox[3] - bbox[1]) * W)
            k_idx = self.kind_to_idx.get(pt.get("element_kind", "button"), 0)

            # Assign to appropriate level
            if max_dim < 24: # Small -> P2 (64x64)
                gx = min(63, max(0, int(cx * 64)))
                gy = min(63, max(0, int(cy * 64)))
                mask_p2[gy, gx] = 1
                box_p2[:, gy, gx] = torch.tensor(bbox, dtype=torch.float32)
                kind_p2[gy, gx] = k_idx
                reg_mask_p2[gy, gx] = True
            elif max_dim <= 64: # Medium -> P3 (32x32)
                gx = min(31, max(0, int(cx * 32)))
                gy = min(31, max(0, int(cy * 32)))
                mask_p3[gy, gx] = 1
                box_p3[:, gy, gx] = torch.tensor(bbox, dtype=torch.float32)
                kind_p3[gy, gx] = k_idx
                reg_mask_p3[gy, gx] = True
            else: # Large -> P4 (16x16)
                gx = min(15, max(0, int(cx * 16)))
                gy = min(15, max(0, int(cy * 16)))
                mask_p4[gy, gx] = 1
                box_p4[:, gy, gx] = torch.tensor(bbox, dtype=torch.float32)
                kind_p4[gy, gx] = k_idx
                reg_mask_p4[gy, gx] = True

        return {
            "image": img_t,
            "category": cat_label,
            "mask_p2": mask_p2, "box_p2": box_p2, "kind_p2": kind_p2, "reg_p2": reg_mask_p2,
            "mask_p3": mask_p3, "box_p3": box_p3, "kind_p3": kind_p3, "reg_p3": reg_mask_p3,
            "mask_p4": mask_p4, "box_p4": box_p4, "kind_p4": kind_p4, "reg_p4": reg_mask_p4,
        }


def train_fpn(
    data_dir: str = "datasets/ui_grounding",
    output_checkpoint: str = "runs/weights/candidates/candidate_m3_fpn_v1.pt",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_ds = FPNDataset(data_dir, split="train")
    val_ds = FPNDataset(data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FPNRegionProposer(pyramid_dim=48)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weighting for static (0) vs interactive (1)
    aff_weights = torch.tensor([0.1, 2.0], dtype=torch.float32)

    print(f"=== TRAINING CANDIDATE M3 FPN REGION PROPOSER ===")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Epochs: {epochs}")

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"]
            cats = batch["category"]

            optimizer.zero_grad()
            out = model(imgs)

            loss_cat = F.cross_entropy(out["cat_logits"], cats)

            # Losses for P2, P3, P4
            loss_levels = []
            for lvl, sz in [("p2", 64), ("p3", 32), ("p4", 16)]:
                head_out = out[lvl]
                aff_logits = head_out[:, :2, :, :]
                box_preds = torch.sigmoid(head_out[:, 2:6, :, :])
                kind_logits = head_out[:, 6:, :, :]

                gt_mask = batch[f"mask_{lvl}"]
                gt_boxes = batch[f"box_{lvl}"]
                gt_kinds = batch[f"kind_{lvl}"]
                reg_mask = batch[f"reg_{lvl}"]

                l_aff = F.cross_entropy(aff_logits, gt_mask, weight=aff_weights)
                l_box = torch.tensor(0.0)
                l_kind = torch.tensor(0.0)

                if reg_mask.any():
                    pred_b = box_preds.permute(0, 2, 3, 1)[reg_mask]
                    target_b = gt_boxes.permute(0, 2, 3, 1)[reg_mask]
                    l_box = F.smooth_l1_loss(pred_b, target_b)

                    pred_k = kind_logits.permute(0, 2, 3, 1)[reg_mask]
                    target_k = gt_kinds[reg_mask]
                    l_kind = F.cross_entropy(pred_k, target_k)

                loss_levels.append(l_aff + 0.5 * l_box + 0.3 * l_kind)

            loss = loss_cat + sum(loss_levels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(imgs)

        scheduler.step()
        train_loss = epoch_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_cat_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"]
                cats = batch["category"]
                out = model(imgs)
                l_cat = F.cross_entropy(out["cat_logits"], cats)
                val_loss += l_cat.item() * len(imgs)
                preds = out["cat_logits"].argmax(dim=-1)
                val_cat_correct += int((preds == cats).sum().item())

        val_loss = val_loss / len(val_ds)
        val_cat_acc = val_cat_correct / len(val_ds) * 100.0

        if epoch % 5 == 0 or epoch == epochs or val_loss < best_val_loss:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.3f} | Val Cat Loss: {val_loss:.3f} | Val Cat Acc: {val_cat_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
            ckpt_data = {
                "schema_version": 2,
                "model_type": "fpn_region_proposer",
                "dataset_provenance": "datasets/ui_grounding/manifest.json",
                "training_samples": len(train_ds),
                "validation_samples": len(val_ds),
                "held_out_test_samples": 40,
                "best_val_loss": val_loss,
                "best_epoch": epoch,
                "trained_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_state_dict": model.state_dict(),
            }
            torch.save(ckpt_data, output_checkpoint)

    ckpt_bytes = Path(output_checkpoint).read_bytes()
    sha256_hash = hashlib.sha256(ckpt_bytes).hexdigest()
    print(f"FPN Training Complete. Checkpoint saved: {output_checkpoint} (SHA-256: {sha256_hash[:16]}...)")
    return {"checkpoint": output_checkpoint, "sha256": sha256_hash}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    train_fpn(epochs=args.epochs)

