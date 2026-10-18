"""Training pipeline for AVCS System 1 Visual Understanding Pilot (Candidate M3-VisualPilot-V1).

Trains:
1. UI Category Classification (8 categories)
2. Spatial Affordance Segmentation (16x16 grid: static background, clickable, typeable)
3. Region Proposal & Element Kind Classification (button, input_field, icon, tab)

Trains on CPU with strict train/val/test-held-out isolation from datasets/ui_grounding/manifest.json.
Saves versioned checkpoint to runs/weights/candidates/candidate_m3_visual_pilot_v1.pt with Schema 2 provenance.
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
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from avcs.perception.visual_pilot import ELEMENT_KINDS, UI_CATEGORIES, VisualUnderstandingPilot


class UIVisualPilotDataset(Dataset):
    """PyTorch Dataset loading UI screens, affordance masks, and region annotations."""

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_meta = self.frames[idx]
        screen_path = self.data_root / f_meta["file_path"]
        mask_path = self.data_root / f_meta["affordance_mask_path"]

        # Load RGB screen
        img = Image.open(screen_path).convert("RGB").resize((256, 256))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1) # (3, 256, 256)

        # Load Affordance mask (16, 16)
        mask_np = np.load(mask_path).astype(np.int64)
        mask_t = torch.from_numpy(mask_np) # (16, 16)

        # Category label
        cat_str = f_meta["category"]
        cat_label = torch.tensor(self.cat_to_idx.get(cat_str, 0), dtype=torch.long)

        # Construct region target tensors (16, 16)
        # box_targets: (4, 16, 16) [ymin, xmin, ymax, xmax]
        # kind_targets: (16, 16) long
        # has_region_mask: (16, 16) bool
        box_targets = torch.zeros(4, 16, 16, dtype=torch.float32)
        kind_targets = torch.zeros(16, 16, dtype=torch.long)
        has_region_mask = torch.zeros(16, 16, dtype=torch.bool)

        for pt in f_meta.get("interactive_points", []):
            cx = pt["x"]
            cy = pt["y"]
            gx = min(15, max(0, int(cx * 16)))
            gy = min(15, max(0, int(cy * 16)))

            bbox = pt.get("bbox", [cy - 0.05, cx - 0.05, cy + 0.05, cx + 0.05])
            # Bounding box span
            b_w = max(0.02, bbox[3] - bbox[1])
            b_h = max(0.02, bbox[2] - bbox[0])

            box_targets[0, gy, gx] = bbox[0]
            box_targets[1, gy, gx] = bbox[1]
            box_targets[2, gy, gx] = bbox[2]
            box_targets[3, gy, gx] = bbox[3]

            k_str = pt.get("element_kind", "button")
            kind_targets[gy, gx] = self.kind_to_idx.get(k_str, 0)
            has_region_mask[gy, gx] = True

        return {
            "image": img_t,
            "category": cat_label,
            "affordance": mask_t,
            "box_targets": box_targets,
            "kind_targets": kind_targets,
            "has_region_mask": has_region_mask,
        }


def train_visual_pilot(
    data_dir: str = "datasets/ui_grounding",
    output_checkpoint: str = "runs/weights/candidates/candidate_m3_visual_pilot_v1.pt",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Trains Candidate M3 Visual Understanding Pilot and saves checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_ds = UIVisualPilotDataset(data_dir, split="train")
    val_ds = UIVisualPilotDataset(data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VisualUnderstandingPilot(feature_dim=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    aff_weights = torch.tensor([0.2, 1.2, 1.5], dtype=torch.float32)

    best_val_score = -1.0
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []

    print(f"=== TRAINING CANDIDATE VISUAL PILOT (M3-HR-V1) ===")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_cat_correct = 0
        train_total = 0

        for batch in train_loader:
            imgs = batch["image"]
            cats = batch["category"]
            affs = batch["affordance"]
            box_t = batch["box_targets"]
            kind_t = batch["kind_targets"]
            reg_mask = batch["has_region_mask"]

            optimizer.zero_grad()
            out = model(imgs)

            # Category loss
            loss_cat = F.cross_entropy(out["category_logits"], cats)

            # Affordance segmentation loss
            loss_aff = F.cross_entropy(out["affordance_logits"], affs, weight=aff_weights)

            # Region box and kind loss on interactive cells
            loss_box = torch.tensor(0.0)
            loss_kind = torch.tensor(0.0)
            if reg_mask.any():
                pred_boxes = out["box_pred"].permute(0, 2, 3, 1)[reg_mask] # (N, 4)
                target_boxes = box_t.permute(0, 2, 3, 1)[reg_mask]         # (N, 4)
                loss_box = F.smooth_l1_loss(pred_boxes, target_boxes)

                pred_kinds = out["kind_logits"].permute(0, 2, 3, 1)[reg_mask] # (N, 4)
                target_kinds = kind_t[reg_mask]                                 # (N,)
                loss_kind = F.cross_entropy(pred_kinds, target_kinds)

            loss = loss_cat + 0.8 * loss_aff + 0.5 * loss_box + 0.5 * loss_kind
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(imgs)
            preds = out["category_logits"].argmax(dim=-1)
            train_cat_correct += int((preds == cats).sum().item())
            train_total += len(imgs)

        scheduler.step()
        train_loss = total_loss / train_total
        train_acc = train_cat_correct / train_total * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_cat_correct = 0
        val_aff_correct = 0
        val_aff_total = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"]
                cats = batch["category"]
                affs = batch["affordance"]

                out = model(imgs)
                l_cat = F.cross_entropy(out["category_logits"], cats)
                l_aff = F.cross_entropy(out["affordance_logits"], affs, weight=aff_weights)
                val_loss += (l_cat + 0.8 * l_aff).item() * len(imgs)

                preds = out["category_logits"].argmax(dim=-1)
                val_cat_correct += int((preds == cats).sum().item())

                aff_preds = out["affordance_logits"].argmax(dim=1)
                val_aff_correct += int((aff_preds == affs).sum().item())
                val_aff_total += affs.numel()
                val_total += len(imgs)

        val_loss = val_loss / val_total
        val_cat_acc = val_cat_correct / val_total * 100.0
        val_aff_acc = val_aff_correct / val_aff_total * 100.0

        # Balanced score prioritizing category accuracy and affordance fidelity
        composite_score = val_cat_acc * 0.7 + val_aff_acc * 0.3

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_cat_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_cat_acc": round(val_cat_acc, 2),
            "val_aff_acc": round(val_aff_acc, 2),
        })

        if epoch % 5 == 0 or epoch == epochs or composite_score > best_val_score:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_cat_acc:.1f}% | Aff Acc: {val_aff_acc:.1f}%")

        if composite_score > best_val_score:
            best_val_score = composite_score
            best_metrics = {
                "val_cat_acc": val_cat_acc,
                "val_aff_acc": val_aff_acc,
                "val_loss": val_loss,
                "epoch": epoch,
            }
            # Save checkpoint
            os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
            ckpt_data = {
                "schema_version": 2,
                "model_type": "visual_understanding_pilot",
                "dataset_provenance": "datasets/ui_grounding/manifest.json",
                "training_samples": len(train_ds),
                "validation_samples": len(val_ds),
                "held_out_test_samples": 40,
                "best_val_cat_acc": val_cat_acc,
                "best_val_aff_acc": val_aff_acc,
                "best_epoch": epoch,
                "trained_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_state_dict": model.state_dict(),
                "history": history,
            }
            torch.save(ckpt_data, output_checkpoint)

    # Compute SHA-256 of saved checkpoint
    ckpt_bytes = Path(output_checkpoint).read_bytes()
    sha256_hash = hashlib.sha256(ckpt_bytes).hexdigest()

    summary = {
        "checkpoint": output_checkpoint,
        "sha256": sha256_hash,
        "best_epoch": best_metrics.get("epoch", epochs),
        "best_val_cat_acc": best_metrics.get("val_cat_acc", 0.0),
        "best_val_aff_acc": best_metrics.get("val_aff_acc", 0.0),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    print(f"Training Complete. Saved checkpoint: {output_checkpoint} (SHA-256: {sha256_hash[:16]}...)")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_visual_pilot(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
