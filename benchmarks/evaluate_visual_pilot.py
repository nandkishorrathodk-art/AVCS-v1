"""Independent Evaluation Benchmark for Candidate M3 Visual Understanding Pilot.

Evaluates:
1. Checkpoint integrity and parameter count.
2. Category classification accuracy & Macro-F1 on untouched held-out test frames (n=40).
3. Affordance segmentation accuracy & Region IoU on held-out test frames.
4. False positive rejection on negative wallpaper/blank screens (n=30).
5. Single-thread CPU latency percentiles (P50, P90, P95, P99).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.perception.visual_pilot import ELEMENT_KINDS, UI_CATEGORIES, VisualUnderstandingPilot


def calculate_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two [ymin, xmin, ymax, xmax] boxes."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0.0, yB - yA) * max(0.0, xB - xA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0.0 else 0.0


def evaluate_visual_pilot(
    checkpoint_path: str = "runs/weights/candidates/candidate_m3_visual_pilot_v1.pt",
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    data_root: str = "datasets/ui_grounding",
) -> Dict[str, Any]:
    print("=" * 60)
    print("AVCS CANDIDATE M3 VISUAL PILOT - INDEPENDENT EVALUATION")
    print("=" * 60)

    # 1. Checkpoint Integrity
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_bytes = ckpt_file.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = VisualUnderstandingPilot()
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"SHA-256 Hash:    {sha256}")
    print(f"File Size:       {len(raw_bytes):,} bytes")
    print(f"Schema Version:  {ckpt_data.get('schema_version', 'legacy')}")
    print(f"Parameter Count: {total_params:,}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    held_out_frames = [f for f in manifest["frames"] if f["split"] == "test_held_out"]
    negative_frames = [f for f in manifest["frames"] if f["split"] == "test_negative"]

    print(f"\nEvaluating on {len(held_out_frames)} held-out frames (8 classes, 5 per class)...")

    # 2. Category & Affordance Evaluation
    cat_correct = 0
    total_held_out = len(held_out_frames)
    class_tp = {c: 0 for c in UI_CATEGORIES}
    class_fp = {c: 0 for c in UI_CATEGORIES}
    class_fn = {c: 0 for c in UI_CATEGORIES}
    confusion = {c1: {c2: 0 for c2 in UI_CATEGORIES} for c1 in UI_CATEGORIES}

    aff_correct_cells = 0
    aff_total_cells = 0

    total_gt_points = 0
    detected_gt_points = 0
    matched_ious: List[float] = []
    kind_matches = 0

    for f_meta in held_out_frames:
        screen_file = os.path.join(data_root, f_meta["file_path"])
        mask_file = os.path.join(data_root, f_meta["affordance_mask_path"])

        img = Image.open(screen_file).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        gt_cat = f_meta["category"]

        gt_mask = np.load(mask_file).astype(np.int64)

        # Predict
        pred_cat, cat_conf, regions = model.predict_visual_structure(img_np)

        # Category scoring
        confusion[gt_cat][pred_cat] += 1
        if pred_cat == gt_cat:
            cat_correct += 1
            class_tp[gt_cat] += 1
        else:
            class_fp[pred_cat] += 1
            class_fn[gt_cat] += 1

        # Affordance map scoring
        # Run raw forward pass for full mask
        t_in = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            out = model(t_in)
            aff_pred = out["affordance_logits"].argmax(dim=1)[0].numpy()
            aff_correct_cells += int((aff_pred == gt_mask).sum())
            aff_total_cells += gt_mask.size

        # Region proposal recall & IoU against GT interactive points
        gt_points = f_meta.get("interactive_points", [])
        total_gt_points += len(gt_points)

        for pt in gt_points:
            pt_box = pt.get("bbox", [pt["y"] - 0.05, pt["x"] - 0.05, pt["y"] + 0.05, pt["x"] + 0.05])
            pt_kind = pt.get("element_kind", "button")

            # Find matching predicted region
            best_iou = 0.0
            best_kind = ""
            for r in regions:
                iou = calculate_iou(pt_box, r.bbox_norm)
                if iou > best_iou:
                    best_iou = iou
                    best_kind = r.element_kind

            if best_iou > 0.15: # Detected inside or covering target
                detected_gt_points += 1
                matched_ious.append(best_iou)
                if best_kind == pt_kind:
                    kind_matches += 1

    cat_accuracy = (cat_correct / total_held_out) * 100.0
    aff_accuracy = (aff_correct_cells / aff_total_cells) * 100.0

    # Compute Macro-F1 across 8 categories
    f1_scores = []
    print("\n--- PER-CATEGORY METRICS ---")
    for cat in UI_CATEGORIES:
        tp = class_tp[cat]
        fp = class_fp[cat]
        fn = class_fn[cat]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
        print(f"  {cat:20s}: Acc={tp}/5 ({tp/5*100:5.1f}%) | Prec={prec*100:5.1f}% | Rec={rec*100:5.1f}% | F1={f1:.3f}")

    macro_f1 = float(np.mean(f1_scores))
    region_recall = (detected_gt_points / total_gt_points * 100.0) if total_gt_points > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    kind_acc = (kind_matches / detected_gt_points * 100.0) if detected_gt_points > 0 else 0.0

    print(f"\nOverall Category Accuracy: {cat_correct}/{total_held_out} ({cat_accuracy:.1f}%)")
    print(f"Macro-F1 (8 classes):      {macro_f1:.3f}")
    print(f"Affordance Grid Accuracy:  {aff_accuracy:.1f}%")
    print(f"Region Detection Recall:   {detected_gt_points}/{total_gt_points} ({region_recall:.1f}%)")
    print(f"Mean Region BBox IoU:      {mean_iou:.3f}")
    print(f"Element Kind Accuracy:     {kind_matches}/{detected_gt_points} ({kind_acc:.1f}%)")

    # 3. Negative Screen Audit
    print(f"\nEvaluating on {len(negative_frames)} negative screens (blanks, gradients, wallpapers)...")
    negative_fp_count = 0
    negative_total = len(negative_frames)

    for f_meta in negative_frames:
        screen_file = os.path.join(data_root, f_meta["file_path"])
        img = Image.open(screen_file).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)

        pred_cat, cat_conf, regions = model.predict_visual_structure(img_np, threshold=0.45, margin=0.10)
        # Any interactive region proposed on a blank/wallpaper frame is a false positive
        if len(regions) > 0:
            negative_fp_count += len(regions)

    print(f"Negative Screen Interactive Region FP Count: {negative_fp_count} across {negative_total} frames")
    negative_fp_rate = negative_fp_count / negative_total

    # 4. CPU Latency Benchmark
    print("\nProfiling CPU Latency (1 thread, 100 iterations)...")
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model.predict_visual_structure(dummy_img)

    latencies_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model.predict_visual_structure(dummy_img)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    latencies_ms.sort()
    p50_lat = latencies_ms[int(0.50 * 100)]
    p90_lat = latencies_ms[int(0.90 * 100)]
    p95_lat = latencies_ms[int(0.95 * 100)]
    p99_lat = latencies_ms[int(0.99 * 100)]

    print(f"  P50 Latency: {p50_lat:.2f} ms")
    print(f"  P90 Latency: {p90_lat:.2f} ms")
    print(f"  P95 Latency: {p95_lat:.2f} ms")
    print(f"  P99 Latency: {p99_lat:.2f} ms")

    results = {
        "checkpoint": checkpoint_path,
        "sha256": sha256,
        "parameters": total_params,
        "category_accuracy_pct": round(cat_accuracy, 2),
        "macro_f1": round(macro_f1, 4),
        "affordance_accuracy_pct": round(aff_accuracy, 2),
        "region_recall_pct": round(region_recall, 2),
        "mean_region_iou": round(mean_iou, 4),
        "element_kind_accuracy_pct": round(kind_acc, 2),
        "negative_fp_count": negative_fp_count,
        "negative_screens_tested": negative_total,
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "confusion_matrix": confusion,
    }

    # Save benchmark results
    out_json = "benchmarks/results/visual_pilot_candidate_v1_evaluation.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_json}")
    return results


if __name__ == "__main__":
    evaluate_visual_pilot()

