"""AVCS Milestone 3.5 — Pixel-Native Word Recognizer Training Pipeline.

Trains PixelWordRecognizer on labelled UI element crops extracted from procedural desktop screens.
Strict pure-pixel regime: Zero OCR libraries, zero UIA, zero DOM.
Splits are strictly session-disjoint and frame-disjoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from avcs.perception.word_recognizer import (
    CANONICAL_CROP_HEIGHT,
    CANONICAL_CROP_WIDTH,
    PixelWordRecognizer,
    UI_WORD_VOCABULARY,
    WORD_TO_ID,
    extract_word_label_from_point,
)


class UIWordCropDataset(Dataset):
    """Dataset of cropped UI patches and corresponding word vocabulary IDs."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        data_root: str,
        is_train: bool = True,
        negative_ratio: float = 0.15,
        target_size: Tuple[int, int] = (CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH),
    ) -> None:
        self.samples = samples
        self.data_root = data_root
        self.is_train = is_train
        self.negative_ratio = negative_ratio
        self.target_size = target_size
        self._image_cache: Dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_image(self, file_path: str) -> Image.Image:
        if file_path not in self._image_cache:
            full_path = os.path.join(self.data_root, file_path)
            self._image_cache[file_path] = Image.open(full_path).convert("RGB")
        return self._image_cache[file_path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        file_path = item["file_path"]
        img = self._get_image(file_path)
        W, H = img.size

        if item.get("is_background", False):
            # Extract background / negative crop
            bg_box = item["bbox_px"]
            x1, y1, x2, y2 = bg_box
            crop = img.crop((x1, y1, x2, y2))
            label_id = 0  # "<none>"
        else:
            bbox_norm = item["bbox_norm"]
            ymin, xmin, ymax, xmax = bbox_norm
            x1 = int(xmin * W)
            y1 = int(ymin * H)
            x2 = int(xmax * W)
            y2 = int(ymax * H)

            # Ensure minimum viable crop size
            if x2 <= x1 + 4:
                x2 = min(W, x1 + 8)
                x1 = max(0, x1 - 4)
            if y2 <= y1 + 4:
                y2 = min(H, y1 + 8)
                y1 = max(0, y1 - 4)

            # Training spatial jitter (scale-aware perturbation up to 12% width/height)
            if self.is_train:
                w_box = max(4, x2 - x1)
                h_box = max(4, y2 - y1)
                max_jx = max(1, int(0.12 * w_box))
                max_jy = max(1, int(0.12 * h_box))
                jx1 = random.randint(-max_jx, max_jx)
                jy1 = random.randint(-max_jy, max_jy)
                jx2 = random.randint(-max_jx, max_jx)
                jy2 = random.randint(-max_jy, max_jy)
                x1 = max(0, min(W - 4, x1 + jx1))
                y1 = max(0, min(H - 4, y1 + jy1))
                x2 = max(x1 + 4, min(W, x2 + jx2))
                y2 = max(y1 + 4, min(H, y2 + jy2))

            crop = img.crop((x1, y1, x2, y2))

            # Training photometric augmentation
            if self.is_train:
                if random.random() < 0.4:
                    factor = random.uniform(0.85, 1.15)
                    crop = ImageEnhance.Brightness(crop).enhance(factor)
                if random.random() < 0.4:
                    factor = random.uniform(0.85, 1.15)
                    crop = ImageEnhance.Contrast(crop).enhance(factor)

            label_str = item["label_str"]
            label_id = WORD_TO_ID.get(label_str, 0)

        # Resize to canonical (CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH)
        pil_resized = crop.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)
        arr = np.array(pil_resized, dtype=np.float32) / 255.0
        # (H, W, 3) -> (3, H, W)
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return t, label_id


def extract_crops_from_manifest(
    manifest_path: str,
    data_root: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parses manifest.json and partitions element crops into train, val, test_held_out, and test_negative."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []
    negative_bg_samples: List[Dict[str, Any]] = []

    for frame in manifest["frames"]:
        split = frame["split"]
        file_path = frame["file_path"]
        is_neg = frame.get("is_negative", False)
        W = frame.get("width", 256)
        H = frame.get("height", 256)

        if is_neg:
            # Generate negative background crops from wallpaper screens
            for i in range(4):
                bx1 = random.randint(10, W - 80)
                by1 = random.randint(10, H - 40)
                bx2 = bx1 + random.randint(40, 70)
                by2 = by1 + random.randint(16, 30)
                negative_bg_samples.append({
                    "file_path": file_path,
                    "is_background": True,
                    "bbox_px": (bx1, by1, bx2, by2),
                    "label_str": "<none>",
                    "split": split,
                })
            continue

        pts = frame.get("interactive_points", [])
        for pt in pts:
            lbl = extract_word_label_from_point(pt)
            sample = {
                "file_path": file_path,
                "is_background": False,
                "bbox_norm": pt.get("bbox", [pt["y"] - 0.04, pt["x"] - 0.08, pt["y"] + 0.04, pt["x"] + 0.08]),
                "label_str": lbl,
                "element_id": pt.get("element_id", ""),
                "element_kind": pt.get("element_kind", ""),
                "split": split,
            }
            if split == "train":
                train_samples.append(sample)
            elif split == "val":
                val_samples.append(sample)
            elif split == "test_held_out":
                test_samples.append(sample)

        # In train and val, extract random non-element background patches
        if split == "train":
            for _ in range(2):
                bx1 = random.randint(5, W - 70)
                by1 = random.randint(5, H - 30)
                train_samples.append({
                    "file_path": file_path,
                    "is_background": True,
                    "bbox_px": (bx1, by1, bx1 + 60, by1 + 20),
                    "label_str": "<none>",
                    "split": "train",
                })
        elif split == "val":
            bx1 = random.randint(5, W - 70)
            by1 = random.randint(5, H - 30)
            val_samples.append({
                "file_path": file_path,
                "is_background": True,
                "bbox_px": (bx1, by1, bx1 + 60, by1 + 20),
                "label_str": "<none>",
                "split": "val",
            })

    return train_samples, val_samples, test_samples, negative_bg_samples


def train_word_recognizer(
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    data_root: str = "datasets/ui_grounding",
    output_dir: str = "runs/weights/candidates",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes the training loop for PixelWordRecognizer on CPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    out_checkpoint_path = os.path.join(output_dir, "candidate_m3_word_recognizer_v1.pt")
    out_json_path = os.path.join(output_dir, "candidate_m3_word_recognizer_v1.json")

    print(f"[WordRecognizer] Loading samples from {manifest_path}...")
    train_samples, val_samples, test_samples, negative_samples = extract_crops_from_manifest(
        manifest_path, data_root
    )
    print(f"[WordRecognizer] Train crops: {len(train_samples)}, Val crops: {len(val_samples)}, Test crops: {len(test_samples)}, Negative: {len(negative_samples)}")

    train_ds = UIWordCropDataset(train_samples, data_root, is_train=True)
    val_ds = UIWordCropDataset(val_samples, data_root, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PixelWordRecognizer(num_classes=len(UI_WORD_VOCABULARY), dropout_p=0.15).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[WordRecognizer] Initialized PixelWordRecognizer: {param_count:,} parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_state_dict: Optional[Dict[str, Any]] = None
    history: List[Dict[str, float]] = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=-1)
            train_correct += int((preds == y_b).sum().item())
            train_total += len(y_b)

        scheduler.step()
        avg_train_loss = total_loss / max(1, train_total)
        train_acc = (train_correct / max(1, train_total)) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_top3_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)

                logits = model(x_b)
                v_loss = criterion(logits, y_b)
                val_loss += v_loss.item() * len(y_b)

                preds = torch.argmax(logits, dim=-1)
                val_correct += int((preds == y_b).sum().item())

                _, top3 = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1)
                for i in range(len(y_b)):
                    if y_b[i] in top3[i]:
                        val_top3_correct += 1
                val_total += len(y_b)

        avg_val_loss = val_loss / max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0
        val_top3_acc = (val_top3_correct / max(1, val_total)) * 100.0

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc_pct": round(train_acc, 2),
            "val_loss": round(avg_val_loss, 4),
            "val_acc_pct": round(val_acc, 2),
            "val_top3_acc_pct": round(val_top3_acc, 2),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}% (Top-3: {val_top3_acc:.1f}%)")

    total_time_s = time.time() - t_start

    # Save best checkpoint
    assert best_state_dict is not None
    checkpoint_payload = {
        "schema_version": 2,
        "checkpoint_name": "candidate_m3_word_recognizer_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "PixelWordRecognizer",
        "parameter_count": param_count,
        "num_classes": len(UI_WORD_VOCABULARY),
        "vocabulary": UI_WORD_VOCABULARY,
        "canonical_crop_shape": [3, CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH],
        "best_val_accuracy_pct": round(best_val_acc, 2),
        "total_training_time_s": round(total_time_s, 2),
        "model_state_dict": best_state_dict,
    }
    torch.save(checkpoint_payload, out_checkpoint_path)

    # Compute SHA-256
    with open(out_checkpoint_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    metadata = {
        "checkpoint_name": "candidate_m3_word_recognizer_v1",
        "checkpoint_file": out_checkpoint_path,
        "sha256": ckpt_sha256,
        "parameter_count": param_count,
        "num_classes": len(UI_WORD_VOCABULARY),
        "canonical_crop_shape": [3, CANONICAL_CROP_HEIGHT, CANONICAL_CROP_WIDTH],
        "best_val_accuracy_pct": round(best_val_acc, 2),
        "total_epochs": epochs,
        "total_training_time_s": round(total_time_s, 2),
        "history": history,
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[WordRecognizer] Training complete in {total_time_s:.1f}s.")
    print(f"[WordRecognizer] Checkpoint saved: {out_checkpoint_path}")
    print(f"[WordRecognizer] SHA-256: {ckpt_sha256}")
    print(f"[WordRecognizer] Best Val Accuracy: {best_val_acc:.2f}%\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    args = parser.parse_args()

    train_word_recognizer(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
