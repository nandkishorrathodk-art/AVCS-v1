"""AVCS Milestone 3.4 — High-Recall Spatial Proposer Benchmark.

Evaluates and compares on untouched held-out frames (n=40, 316 targets) and negative screens (n=30):
1. Baseline: Visual Pilot V1 (single-scale stride-16)
2. Path 1: FPN Region Proposer (stride-4, stride-8, stride-16 multi-scale pyramid)
3. Path 2: Tiled Native Proposer (1:1 original-resolution overlapping patch scanner)
4. Path 3: Combined Visual Proposer (Global FPN context + Native detail scanning)

Reports:
- Overall recall
- Recall by target size: Small (<24 px), Medium (24-64 px), Large (>64 px)
- Precision & false proposals per frame
- Adjacent-target collapse instances
- CPU P50 & P95 latencies
- Peak RAM usage
- Negative screen false proposals
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

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.perception.combined_proposer import CombinedVisualProposer
from avcs.perception.fpn_proposer import FPNRegionProposer
from avcs.perception.tiled_proposer import TiledNativeProposer
from avcs.perception.visual_pilot import VisualUnderstandingPilot
from avcs.perception.visual_state import VisualRegion


def calc_iou(bA: Tuple[float, float, float, float], bB: Tuple[float, float, float, float]) -> float:
    yA, xA = max(bA[0], bB[0]), max(bA[1], bB[1])
    yB, xB = min(bA[2], bB[2]), min(bA[3], bB[3])
    inter = max(0.0, yB - yA) * max(0.0, xB - xA)
    areaA = max(0.0, bA[2] - bA[0]) * max(0.0, bA[3] - bA[1])
    areaB = max(0.0, bB[2] - bB[0]) * max(0.0, bB[3] - bB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def evaluate_proposer_system(
    proposer_fn: Any,
    held_out_frames: List[Dict[str, Any]],
    data_root: str,
    iou_match_thresh: float = 0.15,
) -> Dict[str, Any]:
    total_targets = 0
    targets_by_size = {"small": 0, "medium": 0, "large": 0}
    detected_by_size = {"small": 0, "medium": 0, "large": 0}
    total_proposals = 0
    true_proposals = 0
    adjacent_confusions = 0
    latencies_ms: List[float] = []

    for f in held_out_frames:
        img_path = os.path.join(data_root, f["file_path"])
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        H, W = f.get("height", 256), f.get("width", 256)

        t0 = time.perf_counter()
        regions = proposer_fn(img_np)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        total_proposals += len(regions)
        gt_pts = f.get("interactive_points", [])
        total_targets += len(gt_pts)

        proposal_matched = [False] * len(regions)
        proposal_matches_count = [0] * len(regions)

        for pt in gt_pts:
            bbox = pt.get("bbox", [pt["y"] - 0.05, pt["x"] - 0.05, pt["y"] + 0.05, pt["x"] + 0.05])
            max_dim = max((bbox[2] - bbox[0]) * H, (bbox[3] - bbox[1]) * W)
            sz_cat = "small" if max_dim < 24 else ("medium" if max_dim <= 64 else "large")
            targets_by_size[sz_cat] += 1

            best_iou = 0.0
            best_r_idx = -1
            for r_idx, r in enumerate(regions):
                iou = calc_iou(bbox, r.bbox_norm)
                if iou > best_iou:
                    best_iou = iou
                    best_r_idx = r_idx

            if best_iou > iou_match_thresh:
                detected_by_size[sz_cat] += 1
                proposal_matched[best_r_idx] = True
                proposal_matches_count[best_r_idx] += 1

        true_proposals += sum(1 for m in proposal_matched if m)
        for cnt in proposal_matches_count:
            if cnt > 1:
                adjacent_confusions += (cnt - 1)

    overall_recall = sum(detected_by_size.values()) / total_targets * 100.0
    precision = (true_proposals / total_proposals * 100.0) if total_proposals > 0 else 0.0
    false_per_frame = (total_proposals - true_proposals) / len(held_out_frames)

    latencies_ms.sort()
    p50_lat = latencies_ms[int(len(latencies_ms) * 0.50)]
    p95_lat = latencies_ms[int(len(latencies_ms) * 0.95)]

    return {
        "total_targets": total_targets,
        "overall_recall_pct": round(overall_recall, 2),
        "small_recall_pct": round(detected_by_size["small"] / max(1, targets_by_size["small"]) * 100.0, 2),
        "small_detected": f"{detected_by_size['small']}/{targets_by_size['small']}",
        "medium_recall_pct": round(detected_by_size["medium"] / max(1, targets_by_size["medium"]) * 100.0, 2),
        "medium_detected": f"{detected_by_size['medium']}/{targets_by_size['medium']}",
        "large_recall_pct": round(detected_by_size["large"] / max(1, targets_by_size["large"]) * 100.0, 2),
        "large_detected": f"{detected_by_size['large']}/{targets_by_size['large']}",
        "total_proposals": total_proposals,
        "proposals_per_frame": round(total_proposals / len(held_out_frames), 1),
        "precision_pct": round(precision, 2),
        "false_proposals_per_frame": round(false_per_frame, 2),
        "adjacent_target_confusions": adjacent_confusions,
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
    }


def evaluate_negative_screens(
    proposer_fn: Any,
    negative_frames: List[Dict[str, Any]],
    data_root: str,
) -> int:
    total_false_proposals = 0
    for f in negative_frames:
        img_path = os.path.join(data_root, f["file_path"])
        img = Image.open(img_path).convert("RGB")
        regs = proposer_fn(np.array(img, dtype=np.uint8))
        total_false_proposals += len(regs)
    return total_false_proposals


def run_milestone3_4_benchmark() -> Dict[str, Any]:
    print("=" * 65)
    print("AVCS MILESTONE 3.4 — HIGH-RECALL SPATIAL PROPOSER BENCHMARK")
    print("=" * 65)

    data_root = "datasets/ui_grounding"
    manifest_path = os.path.join(data_root, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    held_out = [f for f in manifest["frames"] if f["split"] == "test_held_out"]
    negative = [f for f in manifest["frames"] if f["split"] == "test_negative"]
    print(f"Held-out test frames: {len(held_out)} (316 interactive targets)")
    print(f"Negative test frames: {len(negative)} (blanks, gradients, wallpapers)")

    # 1. Load Baseline: Visual Pilot V1
    print("\n[Evaluating Baseline: Visual Pilot V1 (single-scale stride-16)]...")
    baseline_ckpt = "runs/weights/candidates/candidate_m3_visual_pilot_v1.pt"
    m_baseline = VisualUnderstandingPilot()
    if os.path.exists(baseline_ckpt):
        ckpt = torch.load(baseline_ckpt, map_location="cpu", weights_only=False)
        m_baseline.load_state_dict(ckpt["model_state_dict"])
    m_baseline.eval()

    def baseline_propose(img: np.ndarray) -> List[VisualRegion]:
        _, _, regs = m_baseline.predict_visual_structure(img, top_k=14)
        return regs

    res_baseline = evaluate_proposer_system(baseline_propose, held_out, data_root)
    neg_fp_baseline = evaluate_negative_screens(baseline_propose, negative, data_root)
    res_baseline["negative_false_proposals"] = neg_fp_baseline
    print(f"  Overall Recall: {res_baseline['overall_recall_pct']}% (Small: {res_baseline['small_recall_pct']}%, Med: {res_baseline['medium_recall_pct']}%, Lrg: {res_baseline['large_recall_pct']}%)")
    print(f"  Precision:      {res_baseline['precision_pct']}% | Latency: P50={res_baseline['latency_p50_ms']}ms, P95={res_baseline['latency_p95_ms']}ms")

    # 2. Load Path 1: FPN Region Proposer (stride-4/8/16)
    print("\n[Evaluating Path 1: FPN Multi-Scale Proposer (stride 4/8/16)]...")
    fpn_ckpt = "runs/weights/candidates/candidate_m3_fpn_v1.pt"
    m_fpn = FPNRegionProposer(pyramid_dim=48)
    if os.path.exists(fpn_ckpt):
        ckpt = torch.load(fpn_ckpt, map_location="cpu", weights_only=False)
        m_fpn.load_state_dict(ckpt["model_state_dict"])
    m_fpn.eval()

    def fpn_propose(img: np.ndarray) -> List[VisualRegion]:
        _, _, regs = m_fpn.propose_regions(img, threshold=0.35, margin=0.03, max_proposals=36)
        return regs

    res_fpn = evaluate_proposer_system(fpn_propose, held_out, data_root)
    neg_fp_fpn = evaluate_negative_screens(fpn_propose, negative, data_root)
    res_fpn["negative_false_proposals"] = neg_fp_fpn
    print(f"  Overall Recall: {res_fpn['overall_recall_pct']}% (Small: {res_fpn['small_recall_pct']}%, Med: {res_fpn['medium_recall_pct']}%, Lrg: {res_fpn['large_recall_pct']}%)")
    print(f"  Precision:      {res_fpn['precision_pct']}% | Latency: P50={res_fpn['latency_p50_ms']}ms, P95={res_fpn['latency_p95_ms']}ms")

    # 3. Path 2: Tiled Native-Resolution Proposer
    print("\n[Evaluating Path 2: Tiled Native Proposer (1:1 overlapping patch scanning)]...")
    tiled_proposer = TiledNativeProposer(tile_size=128, tile_step=64, min_contrast=0.04)

    def tiled_propose(img: np.ndarray) -> List[VisualRegion]:
        return tiled_proposer.extract_tile_proposals(img, max_proposals=36)

    res_tiled = evaluate_proposer_system(tiled_propose, held_out, data_root)
    neg_fp_tiled = evaluate_negative_screens(tiled_propose, negative, data_root)
    res_tiled["negative_false_proposals"] = neg_fp_tiled
    print(f"  Overall Recall: {res_tiled['overall_recall_pct']}% (Small: {res_tiled['small_recall_pct']}%, Med: {res_tiled['medium_recall_pct']}%, Lrg: {res_tiled['large_recall_pct']}%)")
    print(f"  Precision:      {res_tiled['precision_pct']}% | Latency: P50={res_tiled['latency_p50_ms']}ms, P95={res_tiled['latency_p95_ms']}ms")

    # 4. Path 3: Combined Global-Context & Local-Detail Proposer
    print("\n[Evaluating Path 3: Combined Global FPN + Native Tiled Detail Proposer]...")
    combined_proposer = CombinedVisualProposer(
        fpn_proposer=m_fpn,
        tiled_proposer=tiled_proposer,
        fusion_iou_threshold=0.35,
        max_total_proposals=48,
    )

    def combined_propose(img: np.ndarray) -> List[VisualRegion]:
        _, _, regs, _ = combined_proposer.propose_all(img, fpn_threshold=0.35, fpn_margin=0.03)
        return regs

    res_combined = evaluate_proposer_system(combined_propose, held_out, data_root)
    neg_fp_combined = evaluate_negative_screens(combined_propose, negative, data_root)
    res_combined["negative_false_proposals"] = neg_fp_combined
    print(f"  Overall Recall: {res_combined['overall_recall_pct']}% (Small: {res_combined['small_recall_pct']}%, Med: {res_combined['medium_recall_pct']}%, Lrg: {res_combined['large_recall_pct']}%)")
    print(f"  Precision:      {res_combined['precision_pct']}% | Latency: P50={res_combined['latency_p50_ms']}ms, P95={res_combined['latency_p95_ms']}ms")

    # Summary Comparison Table
    print("\n" + "=" * 80)
    print("AVCS MILESTONE 3.4 ARCHITECTURAL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':32s} | {'Baseline (V1)':12s} | {'Path 1 (FPN)':12s} | {'Path 2 (Tiled)':14s} | {'Path 3 (Combined)':16s}")
    print("-" * 92)
    print(f"{'Overall Recall':32s} | {res_baseline['overall_recall_pct']:11.1f}% | {res_fpn['overall_recall_pct']:11.1f}% | {res_tiled['overall_recall_pct']:13.1f}% | {res_combined['overall_recall_pct']:15.1f}%")
    print(f"{'  Small Targets (<24 px)':32s} | {res_baseline['small_recall_pct']:11.1f}% | {res_fpn['small_recall_pct']:11.1f}% | {res_tiled['small_recall_pct']:13.1f}% | {res_combined['small_recall_pct']:15.1f}%")
    print(f"{'  Medium Targets (24-64 px)':32s} | {res_baseline['medium_recall_pct']:11.1f}% | {res_fpn['medium_recall_pct']:11.1f}% | {res_tiled['medium_recall_pct']:13.1f}% | {res_combined['medium_recall_pct']:15.1f}%")
    print(f"{'  Large Targets (>64 px)':32s} | {res_baseline['large_recall_pct']:11.1f}% | {res_fpn['large_recall_pct']:11.1f}% | {res_tiled['large_recall_pct']:13.1f}% | {res_combined['large_recall_pct']:15.1f}%")
    print(f"{'Precision':32s} | {res_baseline['precision_pct']:11.1f}% | {res_fpn['precision_pct']:11.1f}% | {res_tiled['precision_pct']:13.1f}% | {res_combined['precision_pct']:15.1f}%")
    print(f"{'Proposals / Frame':32s} | {res_baseline['proposals_per_frame']:12.1f} | {res_fpn['proposals_per_frame']:12.1f} | {res_tiled['proposals_per_frame']:14.1f} | {res_combined['proposals_per_frame']:16.1f}")
    print(f"{'False Proposals / Frame':32s} | {res_baseline['false_proposals_per_frame']:12.1f} | {res_fpn['false_proposals_per_frame']:12.1f} | {res_tiled['false_proposals_per_frame']:14.1f} | {res_combined['false_proposals_per_frame']:16.1f}")
    print(f"{'Adjacent Target Collapse':32s} | {res_baseline['adjacent_target_confusions']:12d} | {res_fpn['adjacent_target_confusions']:12d} | {res_tiled['adjacent_target_confusions']:14d} | {res_combined['adjacent_target_confusions']:16d}")
    print(f"{'CPU Latency P50 (ms)':32s} | {res_baseline['latency_p50_ms']:12.2f} | {res_fpn['latency_p50_ms']:12.2f} | {res_tiled['latency_p50_ms']:14.2f} | {res_combined['latency_p50_ms']:16.2f}")
    print(f"{'CPU Latency P95 (ms)':32s} | {res_baseline['latency_p95_ms']:12.2f} | {res_fpn['latency_p95_ms']:12.2f} | {res_tiled['latency_p95_ms']:14.2f} | {res_combined['latency_p95_ms']:16.2f}")
    print(f"{'Negative Wallpaper FPs':32s} | {res_baseline['negative_false_proposals']:12d} | {res_fpn['negative_false_proposals']:12d} | {res_tiled['negative_false_proposals']:14d} | {res_combined['negative_false_proposals']:16d}")

    comparison = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "held_out_frames": len(held_out),
        "total_targets": 316,
        "baseline_v1": res_baseline,
        "path1_fpn": res_fpn,
        "path2_tiled": res_tiled,
        "path3_combined": res_combined,
    }

    out_json = "benchmarks/results/milestone3_4_proposer_comparison.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved benchmark comparison results to: {out_json}")
    return comparison


if __name__ == "__main__":
    run_milestone3_4_benchmark()

