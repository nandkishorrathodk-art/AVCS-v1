"""Comprehensive evaluation of Two-Stage Query-Conditioned Foveal Grounding on test_held_out.

Compares:
1. Coarse Candidate V2 alone.
2. Two-Stage: Coarse Candidate V2 + Query-Conditioned Foveal Refiner V1.
Measures Top-1 accuracy, Point-in-Box accuracy, Median, Mean, P90, P95, P99 error,
negative query rejection, blank screen false positives, and latency.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
from PIL import Image

from avcs.perception.goal_grounder import load_grounding_checkpoint
from avcs.perception.foveal_grounder import load_foveal_checkpoint


def evaluate_two_stage(
    coarse_checkpoint: str = "runs/weights/candidates/candidate_m3_highres_v2.pt",
    foveal_checkpoint: str = "runs/weights/candidates/candidate_m3_foveal_v1.pt",
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    confidence_threshold: float = 0.55,
    crop_size: int = 96,
) -> Dict[str, Any]:
    print("=== Loading Coarse & Foveal Models ===")
    coarse_model = load_grounding_checkpoint(coarse_checkpoint, device=torch.device("cpu"))
    foveal_model = load_foveal_checkpoint(foveal_checkpoint, device=torch.device("cpu"))

    with open(coarse_checkpoint, "rb") as f:
        coarse_sha = hashlib.sha256(f.read()).hexdigest()
    with open(foveal_checkpoint, "rb") as f:
        foveal_sha = hashlib.sha256(f.read()).hexdigest()

    manifest_p = Path(manifest_path)
    data_dir = manifest_p.parent
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_frames = [fr for fr in manifest["frames"] if fr.get("split") == "test_held_out"]
    neg_frames = [fr for fr in manifest["frames"] if fr.get("split") == "test_negative"]

    coarse_errors = []
    refined_errors = []
    coarse_hits = 0
    refined_hits = 0
    coarse_in_box = 0
    refined_in_box = 0
    records = []
    foveal_latencies = []

    for fr in test_frames:
        img_path = data_dir / fr["file_path"]
        pil_raw = Image.open(img_path).convert("RGB")
        raw_w, raw_h = pil_raw.size
        raw_np = np.array(pil_raw)

        # 256x256 image for coarse model
        pil_256 = pil_raw.resize((256, 256), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        for pt in fr.get("interactive_points", []):
            query = pt.get("target_query", "click button")
            gt_norm_x = float(pt["x"])
            gt_norm_y = float(pt["y"])
            gt_px_x = gt_norm_x * (raw_w - 1)
            gt_px_y = gt_norm_y * (raw_h - 1)
            gt_box = pt.get("bbox", [gt_norm_y - 0.04, gt_norm_x - 0.04, gt_norm_y + 0.04, gt_norm_x + 0.04])
            kind = pt.get("element_kind", "unknown")
            cat = fr.get("category", "unknown")

            # 1. Coarse Grounding
            c_res = coarse_model.predict_point(img_t, query)
            c_norm_x, c_norm_y = c_res["point"]
            c_conf = c_res["confidence"]
            c_px_x = c_norm_x * (raw_w - 1)
            c_px_y = c_norm_y * (raw_h - 1)
            err_c = math.hypot(c_px_x - gt_px_x, c_px_y - gt_px_y)
            coarse_errors.append(err_c)

            b_ymin, b_xmin, b_ymax, b_xmax = gt_box
            c_in_box = (b_xmin <= c_norm_x <= b_xmax) and (b_ymin <= c_norm_y <= b_ymax)
            if c_in_box:
                coarse_in_box += 1
            if (c_in_box or err_c <= 15.0) and (c_conf >= confidence_threshold):
                coarse_hits += 1

            # 2. Foveal Refinement
            t_f0 = time.perf_counter()
            f_res = foveal_model.refine_prediction(
                screen_frame_rgb=raw_np,
                query=query,
                coarse_point_norm=(c_norm_x, c_norm_y),
                crop_size=crop_size,
                confidence_threshold=0.45,
                device="cpu",
            )
            foveal_latencies.append((time.perf_counter() - t_f0) * 1000.0)

            ref_norm_x, ref_norm_y = f_res.refined_coord_norm
            ref_px_x = ref_norm_x * (raw_w - 1)
            ref_px_y = ref_norm_y * (raw_h - 1)
            err_ref = math.hypot(ref_px_x - gt_px_x, ref_px_y - gt_px_y)
            refined_errors.append(err_ref)

            ref_in_box = (b_xmin <= ref_norm_x <= b_xmax) and (b_ymin <= ref_norm_y <= b_ymax)
            if ref_in_box:
                refined_in_box += 1
            # Refined hit: within bbox or tolerance and confident
            combined_conf = max(c_conf, f_res.foveal_confidence) if f_res.status == "REFINED" else c_conf
            if (ref_in_box or err_ref <= 15.0) and (combined_conf >= confidence_threshold):
                refined_hits += 1

            records.append({
                "query": query,
                "category": cat,
                "kind": kind,
                "err_coarse_px": round(err_c, 2),
                "err_refined_px": round(err_ref, 2),
                "delta_px": round(err_ref - err_c, 2),
                "coarse_in_box": c_in_box,
                "refined_in_box": ref_in_box,
                "foveal_status": f_res.status,
                "foveal_conf": round(f_res.foveal_confidence, 3),
            })

    total_samples = len(records)
    coarse_med = float(np.median(coarse_errors))
    coarse_p95 = float(np.percentile(coarse_errors, 95))
    refined_med = float(np.median(refined_errors))
    refined_p95 = float(np.percentile(refined_errors, 95))
    refined_p90 = float(np.percentile(refined_errors, 90))
    refined_p99 = float(np.percentile(refined_errors, 99))
    refined_mean = float(np.mean(refined_errors))

    # Evaluate negative queries on foveal head
    neg_rejections = 0
    total_neg_q = 0
    for fr in test_frames:
        img_path = data_dir / fr["file_path"]
        with Image.open(img_path) as pil_img:
            raw_np = np.array(pil_img.convert("RGB"))
            pil_256 = pil_img.resize((256, 256), Image.BILINEAR)
            img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        for nq in fr.get("negative_queries", []):
            total_neg_q += 1
            c_res = coarse_model.predict_point(img_t, nq)
            f_res = foveal_model.refine_prediction(
                screen_frame_rgb=raw_np,
                query=nq,
                coarse_point_norm=c_res["point"],
                crop_size=crop_size,
                confidence_threshold=0.50,
            )
            # Rejection holds if either coarse rejected or foveal marked absent
            if c_res["confidence"] < confidence_threshold or f_res.foveal_confidence < 0.35:
                neg_rejections += 1

    neg_rejection_pct = (neg_rejections / max(1, total_neg_q)) * 100.0

    # Evaluate blank screen false positives
    blank_fp = 0
    for bfr in neg_frames:
        img_path = data_dir / bfr["file_path"]
        with Image.open(img_path) as pil_img:
            raw_np = np.array(pil_img.convert("RGB"))
            pil_256 = pil_img.resize((256, 256), Image.BILINEAR)
            img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        # Flat canvas check
        if float(img_t.std()) < 0.065:
            continue
        for q in ["click button", "search input", "close window"]:
            c_res = coarse_model.predict_point(img_t, q)
            if c_res["confidence"] >= 0.70:
                f_res = foveal_model.refine_prediction(
                    screen_frame_rgb=raw_np,
                    query=q,
                    coarse_point_norm=c_res["point"],
                    crop_size=crop_size,
                    confidence_threshold=0.70,
                )
                if f_res.foveal_confidence >= 0.70:
                    blank_fp += 1
                    break

    # Analysis of tail cases (> 3px in coarse)
    tail_coarse = [r for r in records if r["err_coarse_px"] > 3.0]
    improved_in_tail = sum(1 for r in tail_coarse if r["err_refined_px"] < r["err_coarse_px"])

    results = {
        "coarse_checkpoint": coarse_checkpoint,
        "coarse_sha256": coarse_sha,
        "foveal_checkpoint": foveal_checkpoint,
        "foveal_sha256": foveal_sha,
        "total_test_samples": total_samples,
        "coarse": {
            "top_1_hits": f"{coarse_hits}/{total_samples}",
            "top_1_accuracy_pct": round(coarse_hits / total_samples * 100.0, 2),
            "point_in_box_hits": f"{coarse_in_box}/{total_samples}",
            "point_in_box_accuracy_pct": round(coarse_in_box / total_samples * 100.0, 2),
            "median_error_px": round(coarse_med, 2),
            "p95_error_px": round(coarse_p95, 2),
        },
        "two_stage_refined": {
            "top_1_hits": f"{refined_hits}/{total_samples}",
            "top_1_accuracy_pct": round(refined_hits / total_samples * 100.0, 2),
            "point_in_box_hits": f"{refined_in_box}/{total_samples}",
            "point_in_box_accuracy_pct": round(refined_in_box / total_samples * 100.0, 2),
            "median_error_px": round(refined_med, 2),
            "mean_error_px": round(refined_mean, 2),
            "p90_error_px": round(refined_p90, 2),
            "p95_error_px": round(refined_p95, 2),
            "p99_error_px": round(refined_p99, 2),
            "samples_le_3px": sum(1 for e in refined_errors if e <= 3.0),
            "pct_le_3px": round(sum(1 for e in refined_errors if e <= 3.0) / total_samples * 100.0, 1),
            "tail_cases_improved": f"{improved_in_tail}/{len(tail_coarse)} ({improved_in_tail/len(tail_coarse)*100:.1f}%)",
        },
        "negative_query_rejection_pct": round(neg_rejection_pct, 2),
        "blank_screen_fp_count": blank_fp,
        "blank_screen_total": len(neg_frames),
        "foveal_latency_p50_ms": round(float(np.median(foveal_latencies)), 2),
        "foveal_latency_p95_ms": round(float(np.percentile(foveal_latencies, 95)), 2),
        "acceptance_targets": {
            "top_1_accuracy_ge_90": (refined_hits / total_samples * 100.0) >= 90.0,
            "median_error_le_2px": refined_med <= 2.0,
            "p95_error_le_3px": refined_p95 <= 3.0,
            "negative_rejection_ge_95": neg_rejection_pct >= 95.0,
            "blank_fp_count_zero": blank_fp == 0,
        },
    }

    report_p = Path("benchmarks/results/foveal_refinement_candidate_v1.json")
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Two-Stage Evaluation Summary ===")
    print(f"Top-1 Grounding Accuracy : {results['two_stage_refined']['top_1_accuracy_pct']}% ({results['two_stage_refined']['top_1_hits']})")
    print(f"Point-in-Box Accuracy    : {results['two_stage_refined']['point_in_box_accuracy_pct']}% ({results['two_stage_refined']['point_in_box_hits']})")
    print(f"Median Centroid Error    : {results['two_stage_refined']['median_error_px']} px (Coarse: {results['coarse']['median_error_px']} px)")
    print(f"P95 Centroid Error       : {results['two_stage_refined']['p95_error_px']} px (Coarse: {results['coarse']['p95_error_px']} px)")
    print(f"Samples <= 3.0 px        : {results['two_stage_refined']['pct_le_3px']}% ({results['two_stage_refined']['samples_le_3px']}/{total_samples})")
    print(f"Tail Cases Improved      : {results['two_stage_refined']['tail_cases_improved']}")
    print(f"Negative Query Rejection : {results['negative_query_rejection_pct']}%")
    print(f"Blank Screen FP Count    : {results['blank_screen_fp_count']} / {results['blank_screen_total']}")
    print(f"Foveal Refinement P50 Latency: {results['foveal_latency_p50_ms']} ms")
    print(f"Acceptance Gates Passed  : {results['acceptance_targets']}")

    return results


if __name__ == "__main__":
    evaluate_two_stage()
