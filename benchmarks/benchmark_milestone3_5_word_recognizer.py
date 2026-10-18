"""AVCS Milestone 3.5 — Pixel-Native Word Recognizer Independent Benchmark.

Evaluates the trained candidate_m3_word_recognizer_v1.pt on held-out test crops (n=40 frames, 316 targets)
and negative wallpaper screens (n=30 frames, 50 background crops).

Evaluates:
1. Standalone Word Recognition Quality:
   - Top-1 Exact Match Accuracy
   - Top-3 Accuracy
   - Character Error Rate (CER / normalized edit distance)
   - Background False Positive Rate (rejection of non-text/wallpaper patches)
   - CPU Inference Latency (P50 and P95 in ms for single crop and 20-crop batch)
2. Integrated End-to-End Pipeline:
   - FPNRegionProposer + PixelWordRecognizer joint spatial-localization & word recognition
   - CombinedVisualProposer + PixelWordRecognizer joint spatial-localization & word recognition

Zero external OCR libraries, zero UIA, zero DOM inspection.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.perception.combined_proposer import CombinedVisualProposer
from avcs.perception.fpn_proposer import FPNRegionProposer
from avcs.perception.tiled_proposer import TiledNativeProposer, _calc_iou
from avcs.perception.word_recognizer import (
    CANONICAL_CROP_HEIGHT,
    CANONICAL_CROP_WIDTH,
    PixelWordRecognizer,
    UI_WORD_VOCABULARY,
    WORD_TO_ID,
    extract_word_label_from_point,
    load_word_recognizer,
)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer(gt: str, pred: str) -> float:
    """Computes normalized Character Error Rate [0.0, 1.0]."""
    if not gt and not pred:
        return 0.0
    dist = levenshtein_distance(gt.lower(), pred.lower())
    max_len = max(len(gt), len(pred), 1)
    return min(1.0, dist / float(max_len))


def evaluate_standalone_recognizer(
    recognizer: PixelWordRecognizer,
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    data_root: str = "datasets/ui_grounding",
) -> Dict[str, Any]:
    """Evaluates PixelWordRecognizer directly on held-out test element crops."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_frames = [f for f in manifest["frames"] if f["split"] == "test_held_out"]
    negative_frames = [f for f in manifest["frames"] if f.get("is_negative", False)]

    test_crops: List[Tuple[np.ndarray, str]] = []
    bg_crops: List[np.ndarray] = []

    # Extract test crops
    for f in test_frames:
        img_path = os.path.join(data_root, f["file_path"])
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        for pt in f.get("interactive_points", []):
            gt_label = extract_word_label_from_point(pt)
            b = pt.get("bbox", [pt["y"] - 0.04, pt["x"] - 0.08, pt["y"] + 0.04, pt["x"] + 0.08])
            x1 = max(0, int(b[1] * W))
            y1 = max(0, int(b[0] * H))
            x2 = min(W, int(b[3] * W))
            y2 = min(H, int(b[2] * H))
            if x2 <= x1 + 2:
                x2 = min(W, x1 + 8)
            if y2 <= y1 + 2:
                y2 = min(H, y1 + 8)
            crop_np = np.array(img.crop((x1, y1, x2, y2)), dtype=np.uint8)
            test_crops.append((crop_np, gt_label))

    # Extract negative background crops
    for f in negative_frames:
        img_path = os.path.join(data_root, f["file_path"])
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        for _ in range(2):
            bx1 = random.randint(10, W - 80)
            by1 = random.randint(10, H - 40)
            crop_np = np.array(img.crop((bx1, by1, bx1 + 60, by1 + 24)), dtype=np.uint8)
            bg_crops.append(crop_np)
        if len(bg_crops) >= 50:
            break

    print(f"[Benchmark] Evaluating on {len(test_crops)} held-out test crops and {len(bg_crops)} negative crops...")

    top1_correct = 0
    top3_correct = 0
    cer_scores: List[float] = []
    single_latencies_ms: List[float] = []

    for crop_np, gt_label in test_crops:
        t0 = time.perf_counter()
        pred_word, conf = recognizer.predict_crop(crop_np, threshold=0.20)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        single_latencies_ms.append(dt_ms)

        # Check top-1
        if pred_word.lower() == gt_label.lower():
            top1_correct += 1

        # Check top-3 via raw logits
        t = recognizer.preprocess_crop(crop_np)
        with torch.no_grad():
            logits = recognizer(t)
            _, top3_idx = torch.topk(logits[0], k=3)
            top3_words = [UI_WORD_VOCABULARY[int(idx.item())].lower() for idx in top3_idx]
            if gt_label.lower() in top3_words:
                top3_correct += 1

        cer = compute_cer(gt_label, pred_word)
        cer_scores.append(cer)

    # Negative background rejection
    bg_false_positives = 0
    for bg_crop in bg_crops:
        pred_word, conf = recognizer.predict_crop(bg_crop, threshold=0.35)
        if pred_word != "<none>":
            bg_false_positives += 1

    bg_rejection_rate = ((len(bg_crops) - bg_false_positives) / max(1, len(bg_crops))) * 100.0

    # Batched inference latency (20 crops)
    batch_latencies_ms: List[float] = []
    sample_20_crops = [test_crops[i % len(test_crops)][0] for i in range(20)]
    for _ in range(30):
        t0 = time.perf_counter()
        _ = recognizer.predict_crops_batch(sample_20_crops, threshold=0.30)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        batch_latencies_ms.append(dt_ms)

    top1_acc = (top1_correct / max(1, len(test_crops))) * 100.0
    top3_acc = (top3_correct / max(1, len(test_crops))) * 100.0
    mean_cer = float(np.mean(cer_scores)) * 100.0

    return {
        "test_crops_count": len(test_crops),
        "negative_crops_count": len(bg_crops),
        "top1_accuracy_pct": round(top1_acc, 2),
        "top3_accuracy_pct": round(top3_acc, 2),
        "mean_cer_pct": round(mean_cer, 2),
        "bg_rejection_rate_pct": round(bg_rejection_rate, 2),
        "bg_false_positives": bg_false_positives,
        "single_crop_latency_p50_ms": round(float(np.median(single_latencies_ms)), 2),
        "single_crop_latency_p95_ms": round(float(np.percentile(single_latencies_ms, 95)), 2),
        "batch_20_latency_p50_ms": round(float(np.median(batch_latencies_ms)), 2),
        "batch_20_latency_p95_ms": round(float(np.percentile(batch_latencies_ms, 95)), 2),
    }


def evaluate_end_to_end_proposer_with_text(
    proposer: Any,
    recognizer: PixelWordRecognizer,
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    data_root: str = "datasets/ui_grounding",
) -> Dict[str, Any]:
    """Evaluates end-to-end spatial proposal + text recognition on held-out frames."""
    proposer.attach_word_recognizer(recognizer)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_frames = [f for f in manifest["frames"] if f["split"] == "test_held_out"]

    total_gt = 0
    spatially_detected = 0
    joint_spatial_and_text_correct = 0
    total_proposals = 0
    proposals_with_text = 0

    for f in test_frames:
        img_path = os.path.join(data_root, f["file_path"])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)

        if hasattr(proposer, "propose_regions"):
            _, _, regions = proposer.propose_regions(img_np, threshold=0.35, margin=0.03, max_proposals=36)
        else:
            _, _, regions, _ = proposer.propose_all(img_np, fpn_threshold=0.35, fpn_margin=0.03)

        total_proposals += len(regions)
        for r in regions:
            if r.detected_text is not None and r.detected_text != "<none>":
                proposals_with_text += 1

        gt_pts = f.get("interactive_points", [])
        total_gt += len(gt_pts)

        for pt in gt_pts:
            gt_label = extract_word_label_from_point(pt)
            bbox = pt.get("bbox", [pt["y"] - 0.05, pt["x"] - 0.05, pt["y"] + 0.05, pt["x"] + 0.05])

            best_iou = 0.0
            matched_region = None
            for r in regions:
                iou = _calc_iou(bbox, r.bbox_norm)
                if iou > best_iou:
                    best_iou = iou
                    matched_region = r

            if best_iou >= 0.15 and matched_region is not None:
                spatially_detected += 1
                if matched_region.detected_text and matched_region.detected_text.lower() == gt_label.lower():
                    joint_spatial_and_text_correct += 1

    spatial_recall = (spatially_detected / max(1, total_gt)) * 100.0
    joint_accuracy = (joint_spatial_and_text_correct / max(1, total_gt)) * 100.0
    text_accuracy_on_detected = (joint_spatial_and_text_correct / max(1, spatially_detected)) * 100.0

    return {
        "total_ground_truth": total_gt,
        "total_proposals": total_proposals,
        "proposals_with_text": proposals_with_text,
        "spatial_recall_pct": round(spatial_recall, 2),
        "joint_spatial_and_text_accuracy_pct": round(joint_accuracy, 2),
        "text_accuracy_on_localized_pct": round(text_accuracy_on_detected, 2),
    }


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs/weights/candidates/candidate_m3_word_recognizer_v1.pt")
    parser.add_argument("--manifest", type=str, default="datasets/ui_grounding/manifest.json")
    parser.add_argument("--output", type=str, default="benchmarks/results/milestone3_5_word_recognition.json")
    args = parser.parse_args()

    print(f"=== AVCS Milestone 3.5 — Word Recognizer Benchmark ===")
    recognizer = load_word_recognizer(args.checkpoint, device="cpu")

    # Compute SHA-256 of checkpoint
    with open(args.checkpoint, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    param_count = sum(p.numel() for p in recognizer.parameters())
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"SHA-256: {ckpt_sha256}")
    print(f"Parameters: {param_count:,}")

    # 1. Standalone Evaluation
    standalone_results = evaluate_standalone_recognizer(recognizer, args.manifest)
    print("\n--- Standalone Crop Recognition Results ---")
    for k, v in standalone_results.items():
        print(f"  {k}: {v}")

    # 2. End-to-End Evaluation with FPN Proposer
    print("\n--- End-to-End FPN Proposer + Word Recognizer ---")
    fpn_ckpt = "runs/weights/candidates/candidate_m3_fpn_v1.pt"
    fpn = FPNRegionProposer()
    if os.path.isfile(fpn_ckpt):
        ckpt_data = torch.load(fpn_ckpt, map_location="cpu", weights_only=True)
        fpn.load_state_dict(ckpt_data["model_state_dict"])
    fpn_results = evaluate_end_to_end_proposer_with_text(fpn, recognizer, args.manifest)
    for k, v in fpn_results.items():
        print(f"  {k}: {v}")

    # 3. End-to-End Evaluation with Combined Proposer
    print("\n--- End-to-End Combined Proposer + Word Recognizer ---")
    combined = CombinedVisualProposer(fpn_proposer=fpn)
    combined_results = evaluate_end_to_end_proposer_with_text(combined, recognizer, args.manifest)
    for k, v in combined_results.items():
        print(f"  {k}: {v}")

    # Compile benchmark report
    benchmark_report = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_evaluated": args.checkpoint,
        "checkpoint_sha256": ckpt_sha256,
        "parameter_count": param_count,
        "vocabulary_size": len(UI_WORD_VOCABULARY),
        "standalone_crop_evaluation": standalone_results,
        "fpn_proposer_integration": fpn_results,
        "combined_proposer_integration": combined_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"\n[Benchmark] Results written to: {args.output}\n")


if __name__ == "__main__":
    run_benchmark()
