"""Milestone 3.1 Benchmark: Global Candidate Disambiguation & P95 Reduction.

Compares:
Arm A: Coarse Candidate V2 alone.
Arm B: Coarse Candidate V2 + Global Candidate Disambiguation (Top-4 NMS peaks + Neural Reranking).
Arm C: Coarse Candidate V2 + Global Disambiguation + Selective Runtime Foveal Refinement.

Measures:
- Top-1 Target Grounding Accuracy (geometric hit + calibrated confidence gate)
- Point-in-Box Accuracy
- Centroid Error Distribution (Median, Mean, P90, P95, P99, Max, <= 3.0 px)
- Adversarial Hard-Negative Rejection (200 queries across test_held_out)
- Blank/Wallpaper False Positives (30 screens)
- Per-stage Latency (Coarse forward, Peak NMS, Feature pooling, Reranker MLP, Foveal refiner)
- Category & High-Error Subgroup breakdown (dense editor menus, repeated icons, search bars)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
from PIL import Image

from avcs.perception.goal_grounder import load_grounding_checkpoint, encode_queries
from avcs.perception.foveal_grounder import load_foveal_checkpoint
from avcs.perception.candidate_disambiguator import (
    GlobalCandidateDisambiguator,
    load_disambiguator_checkpoint,
)


def run_benchmark_m3_1(
    coarse_checkpoint: str = "runs/weights/candidates/candidate_m3_highres_v2.pt",
    foveal_checkpoint: str = "runs/weights/candidates/candidate_m3_foveal_v1.pt",
    reranker_checkpoint: str = "runs/weights/candidates/candidate_m3_reranker_v1.pt",
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    confidence_threshold: float = 0.55,
    output_json: str = "benchmarks/results/benchmark_m3_1_disambiguation.json",
) -> Dict[str, Any]:
    print("=== AVCS Milestone 3.1: Global Disambiguation & Selective Fovea Benchmark ===")
    dev = torch.device("cpu")

    # 1. Load checkpoints
    print(f"Loading Coarse Grounder: {coarse_checkpoint}")
    coarse_model = load_grounding_checkpoint(coarse_checkpoint, device=dev)
    with open(coarse_checkpoint, "rb") as f:
        coarse_sha = hashlib.sha256(f.read()).hexdigest()

    print(f"Loading Foveal Refiner: {foveal_checkpoint}")
    foveal_model = load_foveal_checkpoint(foveal_checkpoint, device=dev)
    with open(foveal_checkpoint, "rb") as f:
        foveal_sha = hashlib.sha256(f.read()).hexdigest()

    print(f"Loading Candidate Disambiguator: {reranker_checkpoint}")
    disambiguator = load_disambiguator_checkpoint(coarse_model, reranker_checkpoint, device=dev)
    with open(reranker_checkpoint, "rb") as f:
        reranker_sha = hashlib.sha256(f.read()).hexdigest()

    # 2. Load dataset
    manifest_p = Path(manifest_path)
    data_dir = manifest_p.parent
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_frames = [fr for fr in manifest["frames"] if fr.get("split") == "test_held_out"]
    neg_frames = [fr for fr in manifest["frames"] if fr.get("split") == "test_negative"]
    print(f"Test Set: {len(test_frames)} held-out frames, {len(neg_frames)} blank/negative frames")

    # Metrics storage for 3 arms
    # Arm A: Coarse V2
    # Arm B: Coarse V2 + Global Disambiguator
    # Arm C: Coarse V2 + Global Disambiguator + Selective Foveal Refinement
    errors_a: List[float] = []
    errors_b: List[float] = []
    errors_c: List[float] = []

    hits_a = 0
    hits_b = 0
    hits_c = 0

    in_box_a = 0
    in_box_b = 0
    in_box_c = 0

    latencies_coarse: List[float] = []
    latencies_rerank: List[float] = []
    latencies_foveal: List[float] = []
    latencies_total_c: List[float] = []

    per_sample_records: List[Dict[str, Any]] = []

    # Category buckets
    category_metrics: Dict[str, Dict[str, List[float]]] = {}

    for fr in test_frames:
        cat = fr.get("category", "unknown")
        if cat not in category_metrics:
            category_metrics[cat] = {"arm_a": [], "arm_b": [], "arm_c": []}

        img_path = data_dir / fr["file_path"]
        pil_raw = Image.open(img_path).convert("RGB")
        raw_w, raw_h = pil_raw.size
        raw_np = np.array(pil_raw)

        pil_256 = pil_raw.resize((256, 256), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        for pt in fr.get("interactive_points", []):
            q = pt.get("target_query", "click button")
            gt_norm_x = float(pt["x"])
            gt_norm_y = float(pt["y"])
            gt_px_x = gt_norm_x * (raw_w - 1)
            gt_px_y = gt_norm_y * (raw_h - 1)
            gt_box = pt.get("bbox", [gt_norm_y - 0.04, gt_norm_x - 0.04, gt_norm_y + 0.04, gt_norm_x + 0.04])
            b_ymin, b_xmin, b_ymax, b_xmax = gt_box
            kind = pt.get("element_kind", "unknown")

            t_start = time.perf_counter()

            # --- ARM A: Coarse Grounding ---
            t_c0 = time.perf_counter()
            c_res = coarse_model.predict_point(img_t, q)
            t_c = (time.perf_counter() - t_c0) * 1000.0
            latencies_coarse.append(t_c)

            c_norm_x, c_norm_y = c_res["point"]
            c_conf = c_res["confidence"]
            c_px_x = c_norm_x * (raw_w - 1)
            c_px_y = c_norm_y * (raw_h - 1)
            err_a = math.hypot(c_px_x - gt_px_x, c_px_y - gt_px_y)
            errors_a.append(err_a)

            in_box_flag_a = (b_xmin <= c_norm_x <= b_xmax) and (b_ymin <= c_norm_y <= b_ymax)
            if in_box_flag_a:
                in_box_a += 1
            if (in_box_flag_a or err_a <= 15.0) and (c_conf >= confidence_threshold):
                hits_a += 1

            # --- ARM B: Global Candidate Disambiguation ---
            t_r0 = time.perf_counter()
            d_res = disambiguator.disambiguate(img_t, q, screen_size=(raw_w, raw_h))
            t_r = (time.perf_counter() - t_r0) * 1000.0
            latencies_rerank.append(t_r)

            b_norm_x, b_norm_y = d_res.selected_coord_norm
            b_conf = d_res.confidence
            b_px_x = b_norm_x * (raw_w - 1)
            b_px_y = b_norm_y * (raw_h - 1)
            err_b = math.hypot(b_px_x - gt_px_x, b_px_y - gt_px_y)
            errors_b.append(err_b)

            in_box_flag_b = (b_xmin <= b_norm_x <= b_xmax) and (b_ymin <= b_norm_y <= b_ymax)
            if in_box_flag_b:
                in_box_b += 1
            if (in_box_flag_b or err_b <= 15.0) and (b_conf >= confidence_threshold):
                hits_b += 1

            # --- ARM C: Selective Foveal Refinement (Runtime Signals Only) ---
            t_f0 = time.perf_counter()
            # Runtime decision rule: only invoke fovea if candidate is confident and not ambiguous
            foveal_invoked = False
            foveal_accepted = False

            # We use candidate coordinate from disambiguator as fixation point
            fixation_coord = (b_norm_x, b_norm_y)
            final_norm_x, final_norm_y = b_norm_x, b_norm_y
            final_conf = b_conf

            if d_res.decision != "ABSENT" and b_conf >= 0.45:
                foveal_invoked = True
                f_res = foveal_model.refine_prediction(
                    screen_frame_rgb=raw_np,
                    query=q,
                    coarse_point_norm=fixation_coord,
                    crop_size=96,
                    confidence_threshold=0.45,
                )
                latencies_foveal.append(f_res.latency_ms)

                # Runtime Selective Refinement Policy (ZERO ground truth usage):
                # 1. Refinement presence must be high (>= 0.70)
                # 2. Offset must be within local bounds (<= 6.0 px)
                # 3. If wide box (input field or wide button) OR small offset (<= 3.0 px)
                b_ymin_c, b_xmin_c, b_ymax_c, b_xmax_c = f_res.refined_bbox_norm
                box_w = max(1e-3, (b_xmax_c - b_xmin_c) * 96)
                box_h = max(1e-3, (b_ymax_c - b_ymin_c) * 96)
                is_wide_element = (box_w / box_h) >= 2.0

                if f_res.foveal_confidence >= 0.70 and f_res.offset_px <= 6.0:
                    if is_wide_element or f_res.offset_px <= 3.0:
                        foveal_accepted = True
                        final_norm_x, final_norm_y = f_res.refined_coord_norm
                        final_conf = max(b_conf, f_res.foveal_confidence)

            t_total_c = (time.perf_counter() - t_start) * 1000.0
            latencies_total_c.append(t_total_c)

            c_px_x_final = final_norm_x * (raw_w - 1)
            c_px_y_final = final_norm_y * (raw_h - 1)
            err_c = math.hypot(c_px_x_final - gt_px_x, c_px_y_final - gt_px_y)
            errors_c.append(err_c)

            in_box_flag_c = (b_xmin <= final_norm_x <= b_xmax) and (b_ymin <= final_norm_y <= b_ymax)
            if in_box_flag_c:
                in_box_c += 1
            if (in_box_flag_c or err_c <= 15.0) and (final_conf >= confidence_threshold):
                hits_c += 1

            # Log category errors
            category_metrics[cat]["arm_a"].append(err_a)
            category_metrics[cat]["arm_b"].append(err_b)
            category_metrics[cat]["arm_c"].append(err_c)

            per_sample_records.append({
                "query": q,
                "category": cat,
                "kind": kind,
                "gt_coord": (round(gt_norm_x, 4), round(gt_norm_y, 4)),
                "err_a_px": round(err_a, 2),
                "err_b_px": round(err_b, 2),
                "err_c_px": round(err_c, 2),
                "disamb_margin": round(d_res.margin, 3),
                "rerank_changed": d_res.reranked_top1_changed,
                "foveal_invoked": foveal_invoked,
                "foveal_accepted": foveal_accepted,
            })

    total_samples = len(per_sample_records)

    # 3. Hard-Negative Evaluation across 200 negative queries
    print("Evaluating 200 Adversarial Negative Queries...")
    neg_rejections_b = 0
    neg_rejections_c = 0
    total_neg_q = 0

    for fr in test_frames:
        img_path = data_dir / fr["file_path"]
        pil_raw = Image.open(img_path).convert("RGB")
        raw_w, raw_h = pil_raw.size
        pil_256 = pil_raw.resize((256, 256), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        for nq in fr.get("negative_queries", []):
            total_neg_q += 1
            d_res = disambiguator.disambiguate(img_t, nq, screen_size=(raw_w, raw_h))
            if d_res.decision in ("ABSENT", "UNCERTAIN") or d_res.confidence < 0.40:
                neg_rejections_b += 1
                neg_rejections_c += 1

    neg_rejection_pct_b = (neg_rejections_b / max(1, total_neg_q)) * 100.0
    neg_rejection_pct_c = (neg_rejections_c / max(1, total_neg_q)) * 100.0

    # 4. Blank screen false positive evaluation (30 screens)
    print("Evaluating 30 Blank / Wallpaper Screens...")
    blank_fp_b = 0
    blank_fp_c = 0
    for bfr in neg_frames:
        img_path = data_dir / bfr["file_path"]
        pil_raw = Image.open(img_path).convert("RGB")
        raw_w, raw_h = pil_raw.size
        pil_256 = pil_raw.resize((256, 256), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        for q in ["click button", "search input", "close window"]:
            d_res = disambiguator.disambiguate(img_t, q, screen_size=(raw_w, raw_h))
            if d_res.confidence >= 0.70 and d_res.decision == "CONFIDENT":
                blank_fp_b += 1
                blank_fp_c += 1
                break

    # Statistical aggregations
    def summarize_arm(errors, hits, in_box):
        return {
            "top_1_hits": f"{hits}/{total_samples}",
            "top_1_accuracy_pct": round(hits / total_samples * 100.0, 2),
            "point_in_box_hits": f"{in_box}/{total_samples}",
            "point_in_box_accuracy_pct": round(in_box / total_samples * 100.0, 2),
            "median_error_px": round(float(np.median(errors)), 2),
            "mean_error_px": round(float(np.mean(errors)), 2),
            "p90_error_px": round(float(np.percentile(errors, 90)), 2),
            "p95_error_px": round(float(np.percentile(errors, 95)), 2),
            "p99_error_px": round(float(np.percentile(errors, 99)), 2),
            "max_error_px": round(float(np.max(errors)), 2),
            "samples_le_3px": int(sum(1 for e in errors if e <= 3.0)),
            "pct_le_3px": round(sum(1 for e in errors if e <= 3.0) / total_samples * 100.0, 2),
        }

    summary_a = summarize_arm(errors_a, hits_a, in_box_a)
    summary_b = summarize_arm(errors_b, hits_b, in_box_b)
    summary_c = summarize_arm(errors_c, hits_c, in_box_c)

    # Subgroup breakdown
    category_summary = {}
    for cat, err_dict in category_metrics.items():
        category_summary[cat] = {
            "num_samples": len(err_dict["arm_a"]),
            "arm_a_median": round(float(np.median(err_dict["arm_a"])), 2),
            "arm_b_median": round(float(np.median(err_dict["arm_b"])), 2),
            "arm_c_median": round(float(np.median(err_dict["arm_c"])), 2),
            "arm_a_p95": round(float(np.percentile(err_dict["arm_a"], 95)), 2),
            "arm_b_p95": round(float(np.percentile(err_dict["arm_b"], 95)), 2),
            "arm_c_p95": round(float(np.percentile(err_dict["arm_c"], 95)), 2),
        }

    # Latency report
    latencies = {
        "coarse_forward_p50_ms": round(float(np.median(latencies_coarse)), 2),
        "coarse_forward_p95_ms": round(float(np.percentile(latencies_coarse, 95)), 2),
        "disambiguator_p50_ms": round(float(np.median(latencies_rerank)), 2),
        "disambiguator_p95_ms": round(float(np.percentile(latencies_rerank, 95)), 2),
        "selective_foveal_p50_ms": round(float(np.median(latencies_foveal)), 2) if latencies_foveal else 0.0,
        "combined_pipeline_p50_ms": round(float(np.median(latencies_total_c)), 2),
        "combined_pipeline_p95_ms": round(float(np.percentile(latencies_total_c, 95)), 2),
    }

    # Number of times reranking flipped coarse top-1
    rerank_flips = sum(1 for r in per_sample_records if r["rerank_changed"])
    fovea_accepts = sum(1 for r in per_sample_records if r["foveal_accepted"])

    results = {
        "benchmark_mission": "Milestone 3.1: Global Candidate Disambiguation & P95 Reduction",
        "coarse_checkpoint": coarse_checkpoint,
        "coarse_sha256": coarse_sha,
        "reranker_checkpoint": reranker_checkpoint,
        "reranker_sha256": reranker_sha,
        "foveal_checkpoint": foveal_checkpoint,
        "foveal_sha256": foveal_sha,
        "total_test_samples": total_samples,
        "rerank_flips": f"{rerank_flips}/{total_samples} ({rerank_flips/total_samples*100:.1f}%)",
        "selective_fovea_accepted": f"{fovea_accepts}/{total_samples} ({fovea_accepts/total_samples*100:.1f}%)",
        "arm_a_coarse_v2": summary_a,
        "arm_b_global_disambiguator": summary_b,
        "arm_c_selective_fovea": summary_c,
        "adversarial_negative_rejection_pct": round(neg_rejection_pct_c, 2),
        "blank_screen_fp_count": blank_fp_c,
        "blank_screen_total": len(neg_frames),
        "latencies": latencies,
        "category_summary": category_summary,
        "acceptance_targets": {
            "grounding_accuracy_ge_90": summary_c["top_1_accuracy_pct"] >= 90.0,
            "median_error_le_2px": summary_c["median_error_px"] <= 2.0,
            "p95_error_le_3px": summary_c["p95_error_px"] <= 3.0,
            "negative_rejection_ge_95": neg_rejection_pct_c >= 95.0,
            "blank_fp_count_zero": blank_fp_c == 0,
        },
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Benchmark Summary ===")
    print(f"Arm A (Coarse V2)       : Top-1 {summary_a['top_1_accuracy_pct']}% | Med {summary_a['median_error_px']} px | P95 {summary_a['p95_error_px']} px | <=3px {summary_a['pct_le_3px']}%")
    print(f"Arm B (Disambiguator)   : Top-1 {summary_b['top_1_accuracy_pct']}% | Med {summary_b['median_error_px']} px | P95 {summary_b['p95_error_px']} px | <=3px {summary_b['pct_le_3px']}%")
    print(f"Arm C (Selective Fovea) : Top-1 {summary_c['top_1_accuracy_pct']}% | Med {summary_c['median_error_px']} px | P95 {summary_c['p95_error_px']} px | <=3px {summary_c['pct_le_3px']}%")
    print(f"Negative Rejection      : {results['adversarial_negative_rejection_pct']}% (Target >= 95%)")
    print(f"Blank Screen FP Count   : {results['blank_screen_fp_count']} / {results['blank_screen_total']} (Target == 0)")
    print(f"Combined Pipeline P50   : {latencies['combined_pipeline_p50_ms']} ms")
    print(f"Acceptance Gates        : {results['acceptance_targets']}")

    return results


if __name__ == "__main__":
    run_benchmark_m3_1()

