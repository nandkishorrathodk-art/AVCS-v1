"""AVCS Milestone 3.2: Comprehensive Error Forensics Pipeline.

Performs exhaustive per-sample diagnostics on all held-out targets:
- Records exact coordinates, bounding boxes, dimensions, transforms, and errors for Arm A, B, and C.
- Classifies every failure into the 8 required taxonomy categories.
- Generates machine-readable JSON and CSV forensic reports.
- Produces an annotated visual outlier gallery for top error cases.
- Audits metric definitions, percentile conventions, and valid-click vs centroid divergences.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from avcs.perception.goal_grounder import load_grounding_checkpoint, encode_queries
from avcs.perception.foveal_grounder import load_foveal_checkpoint
from avcs.perception.candidate_disambiguator import (
    GlobalCandidateDisambiguator,
    load_disambiguator_checkpoint,
    extract_spatial_peaks_nms,
)


def classify_failure(
    err_px: float,
    in_box: bool,
    gt_box_px: Tuple[float, float, float, float],
    pred_px: Tuple[float, float],
    gt_px: Tuple[float, float],
    candidate_peaks: List[Tuple[float, float, float]],
    raw_size: Tuple[int, int],
    other_points: List[Dict[str, Any]],
    target_query: str,
    foveal_invoked: bool,
    foveal_accepted: bool,
    err_b_px: float,
    err_c_px: float,
) -> Tuple[str, str]:
    """Classifies a sample into one of the 8 canonical forensic categories."""
    if err_px <= 3.0 and in_box:
        return "SUCCESS_WITHIN_TARGET", "Prediction is within 3.0 px and inside bounding box."

    ymin, xmin, ymax, xmax = gt_box_px
    px, py = pred_px
    box_w = xmax - xmin
    box_h = ymax - ymin

    # Category 3: Correct target region but inaccurate center
    if in_box:
        if box_w >= 30.0 or box_h >= 20.0 or (box_w / max(1.0, box_h)) >= 2.0:
            return (
                "CORRECT_TARGET_REGION_INACCURATE_CENTER",
                f"Prediction is inside the {box_w:.1f}x{box_h:.1f}px bounding box (valid click), but offset {err_px:.1f}px from annotated geometric center."
            )
        return (
            "CORRECT_TARGET_REGION_INACCURATE_CENTER",
            f"Prediction is inside bounding box, offset {err_px:.1f}px from annotated center."
        )

    # Category 5: Crop-origin or foveal offset error
    if foveal_accepted and err_c_px > err_b_px and err_b_px <= 6.0:
        return (
            "CROP_ORIGIN_OR_FOVEAL_OFFSET_ERROR",
            f"Arm B was closer ({err_b_px:.1f}px), but local foveal refiner shifted prediction to {err_c_px:.1f}px."
        )

    # Category 6: Ambiguous query or visually indistinguishable candidates
    # Check if a non-top1 candidate peak landed on the true target
    target_captured_in_peaks = False
    for cand in candidate_peaks[1:]:
        pk_px_x, pk_px_y = cand.coord_px
        if (xmin - 2 <= pk_px_x <= xmax + 2) and (ymin - 2 <= pk_px_y <= ymax + 2):
            target_captured_in_peaks = True
            break
        if math.hypot(pk_px_x - gt_px[0], pk_px_y - gt_px[1]) <= 6.0:
            target_captured_in_peaks = True
            break

    if target_captured_in_peaks:
        return (
            "AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES",
            "True target was detected as candidate peak #2-#4, but another candidate won top rank."
        )

    # Check proximity to other interactive elements in the frame
    clicked_other = False
    other_query = ""
    other_kind = ""
    for opt in other_points:
        o_box = opt.get("bbox", [0, 0, 0, 0])
        o_ymin = o_box[0] * (raw_size[1] - 1)
        o_xmin = o_box[1] * (raw_size[0] - 1)
        o_ymax = o_box[2] * (raw_size[1] - 1)
        o_xmax = o_box[3] * (raw_size[0] - 1)
        if (o_xmin - 3 <= px <= o_xmax + 3) and (o_ymin - 3 <= py <= o_ymax + 3):
            clicked_other = True
            other_query = opt.get("target_query", "")
            other_kind = opt.get("element_kind", "")
            break

    # Category 2: Adjacent or repeated target confusion
    if clicked_other:
        # Check if similar type or nearby toolbar/tab
        dist_to_other = math.hypot(px - gt_px[0], py - gt_px[1])
        if dist_to_other <= 35.0 or other_kind in target_query.lower() or "button" in other_kind:
            return (
                "ADJACENT_OR_REPEATED_TARGET_CONFUSION",
                f"Prediction landed on neighboring UI element '{other_query}' ({other_kind}) {dist_to_other:.1f}px away."
            )
        else:
            return (
                "WRONG_SEMANTIC_TARGET",
                f"Prediction clicked distinct element '{other_query}' ({other_kind}) instead of '{target_query}'."
            )

    # Category 4: Coordinate-transform or letterbox inversion error
    # Check if target is near edge where downsampling / padding causes boundary truncation
    if gt_px[0] <= 8 or gt_px[0] >= raw_size[0] - 8 or gt_px[1] <= 8 or gt_px[1] >= raw_size[1] - 8:
        return (
            "COORDINATE_TRANSFORM_OR_LETTERBOX_INVERSION_ERROR",
            f"Target is within 8px of screen edge ({gt_px[0]:.1f}, {gt_px[1]:.1f}); edge convolution padding artifacts."
        )

    # Category 7: Annotation inconsistency
    # If gt reference point is outside its own bounding box
    gt_x, gt_y = gt_px
    if not (xmin - 1e-2 <= gt_x <= xmax + 1e-2 and ymin - 1e-2 <= gt_y <= ymax + 1e-2):
        return (
            "ANNOTATION_INCONSISTENCY",
            f"Ground truth reference point ({gt_x:.1f}, {gt_y:.1f}) is outside bounding box [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]."
        )

    # Category 1: Wrong semantic target (fallback for large distance outliers)
    if err_px > 30.0:
        return (
            "WRONG_SEMANTIC_TARGET",
            f"Prediction landed {err_px:.1f}px away on background or unrelated region."
        )

    return (
        "AMBIGUOUS_QUERY_OR_INDISTINGUISHABLE_CANDIDATES",
        f"Localized near target ({err_px:.1f}px away) but failed bounding box enclosure."
    )


def run_error_forensics_pipeline(
    coarse_checkpoint: str = "runs/weights/candidates/candidate_m3_highres_v2.pt",
    reranker_checkpoint: str = "runs/weights/candidates/candidate_m3_reranker_v1.pt",
    foveal_checkpoint: str = "runs/weights/candidates/candidate_m3_foveal_v1.pt",
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    output_dir: str = "benchmarks/forensics",
) -> Dict[str, Any]:
    print("=== AVCS Milestone 3.2: Error Forensics & Metric Audit Pipeline ===")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    gallery_dir = out_path / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cpu")

    # 1. Load models
    print(f"Loading Coarse Grounder: {coarse_checkpoint}")
    coarse_model = load_grounding_checkpoint(coarse_checkpoint, device=dev)
    with open(coarse_checkpoint, "rb") as f:
        coarse_sha = hashlib.sha256(f.read()).hexdigest()

    print(f"Loading Candidate Disambiguator: {reranker_checkpoint}")
    disambiguator = load_disambiguator_checkpoint(coarse_model, reranker_checkpoint, device=dev)
    with open(reranker_checkpoint, "rb") as f:
        reranker_sha = hashlib.sha256(f.read()).hexdigest()

    print(f"Loading Foveal Refiner: {foveal_checkpoint}")
    foveal_model = load_foveal_checkpoint(foveal_checkpoint, device=dev)
    with open(foveal_checkpoint, "rb") as f:
        foveal_sha = hashlib.sha256(f.read()).hexdigest()

    # 2. Load manifest
    manifest_p = Path(manifest_path)
    data_dir = manifest_p.parent
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_frames = [fr for fr in manifest["frames"] if fr.get("split") == "test_held_out"]
    print(f"Loaded {len(test_frames)} held-out test frames.")

    forensic_records: List[Dict[str, Any]] = []

    # Running errors for metrics
    errors_a: List[float] = []
    errors_b: List[float] = []
    errors_c: List[float] = []

    in_box_a_count = 0
    in_box_b_count = 0
    in_box_c_count = 0

    valid_region_clicks_b = 0

    category_counts: Dict[str, int] = {}

    target_counter = 0

    for fr in test_frames:
        frame_id = fr.get("frame_id", "unknown")
        app_name = fr.get("app_name", "unknown")
        ui_category = fr.get("category", "unknown")
        img_rel = fr["file_path"]
        img_file = data_dir / img_rel
        pil_raw = Image.open(img_file).convert("RGB")
        raw_w, raw_h = pil_raw.size
        raw_np = np.array(pil_raw)

        pil_256 = pil_raw.resize((256, 256), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        interactive_points = fr.get("interactive_points", [])

        for pt_idx, pt in enumerate(interactive_points):
            target_counter += 1
            target_id = pt.get("element_id", f"{frame_id}_t{pt_idx}")
            query = pt.get("target_query", "click button")
            element_kind = pt.get("element_kind", "button")
            gt_norm_x = float(pt["x"])
            gt_norm_y = float(pt["y"])
            gt_px_x = gt_norm_x * (raw_w - 1)
            gt_px_y = gt_norm_y * (raw_h - 1)

            gt_bbox = pt.get("bbox", [gt_norm_y - 0.04, gt_norm_x - 0.04, gt_norm_y + 0.04, gt_norm_x + 0.04])
            b_ymin, b_xmin, b_ymax, b_xmax = gt_bbox
            box_px = (
                b_ymin * (raw_h - 1),
                b_xmin * (raw_w - 1),
                b_ymax * (raw_h - 1),
                b_xmax * (raw_w - 1),
            )
            box_w_px = box_px[3] - box_px[1]
            box_h_px = box_px[2] - box_px[0]
            aspect_ratio = round(box_w_px / max(1e-2, box_h_px), 2)

            # Other points in same frame for confusion diagnosis
            other_points = [p for i, p in enumerate(interactive_points) if i != pt_idx]

            # --- ARM A: Coarse V2 ---
            c_res = coarse_model.predict_point(img_t, query)
            ca_norm_x, ca_norm_y = c_res["point"]
            ca_conf = c_res["confidence"]
            ca_px_x = ca_norm_x * (raw_w - 1)
            ca_px_y = ca_norm_y * (raw_h - 1)
            err_a = math.hypot(ca_px_x - gt_px_x, ca_px_y - gt_px_y)
            errors_a.append(err_a)
            in_box_a = (b_xmin <= ca_norm_x <= b_xmax) and (b_ymin <= ca_norm_y <= b_ymax)
            if in_box_a:
                in_box_a_count += 1

            # --- ARM B: Global Candidate Disambiguator ---
            d_res = disambiguator.disambiguate(img_t, query, screen_size=(raw_w, raw_h))
            cb_norm_x, cb_norm_y = d_res.selected_coord_norm
            cb_conf = d_res.confidence
            cb_px_x = cb_norm_x * (raw_w - 1)
            cb_px_y = cb_norm_y * (raw_h - 1)
            err_b = math.hypot(cb_px_x - gt_px_x, cb_px_y - gt_px_y)
            errors_b.append(err_b)
            in_box_b = (b_xmin <= cb_norm_x <= b_xmax) and (b_ymin <= cb_norm_y <= b_ymax)
            if in_box_b:
                in_box_b_count += 1
                valid_region_clicks_b += 1

            # Candidate peaks info
            cand_peaks = []
            for c in d_res.candidates:
                cand_peaks.append({
                    "id": c.candidate_id,
                    "norm": [round(c.coord_norm[0], 4), round(c.coord_norm[1], 4)],
                    "px": [round(c.coord_px[0], 2), round(c.coord_px[1], 2)],
                    "coarse_logit": round(c.coarse_logit, 2),
                    "rerank_score": round(c.rerank_score, 3),
                })

            # --- ARM C: Selective Foveal Refinement ---
            foveal_invoked = False
            foveal_accepted = False
            cc_norm_x, cc_norm_y = cb_norm_x, cb_norm_y
            cc_conf = cb_conf
            f_offset_px = 0.0

            if d_res.decision != "ABSENT" and cb_conf >= 0.45:
                foveal_invoked = True
                f_res = foveal_model.refine_prediction(
                    screen_frame_rgb=raw_np,
                    query=query,
                    coarse_point_norm=(cb_norm_x, cb_norm_y),
                    crop_size=96,
                    confidence_threshold=0.45,
                )
                f_offset_px = f_res.offset_px
                b_ymin_c, b_xmin_c, b_ymax_c, b_xmax_c = f_res.refined_bbox_norm
                box_w_c = max(1e-3, (b_xmax_c - b_xmin_c) * 96)
                box_h_c = max(1e-3, (b_ymax_c - b_ymin_c) * 96)
                is_wide = (box_w_c / box_h_c) >= 2.0

                if f_res.foveal_confidence >= 0.70 and f_res.offset_px <= 6.0:
                    if is_wide or f_res.offset_px <= 3.0:
                        foveal_accepted = True
                        cc_norm_x, cc_norm_y = f_res.refined_coord_norm
                        cc_conf = max(cb_conf, f_res.foveal_confidence)

            cc_px_x = cc_norm_x * (raw_w - 1)
            cc_px_y = cc_norm_y * (raw_h - 1)
            err_c = math.hypot(cc_px_x - gt_px_x, cc_px_y - gt_px_y)
            errors_c.append(err_c)
            in_box_c = (b_xmin <= cc_norm_x <= b_xmax) and (b_ymin <= cc_norm_y <= b_ymax)
            if in_box_c:
                in_box_c_count += 1

            # Differences
            diff_a_b = math.hypot(ca_px_x - cb_px_x, ca_px_y - cb_px_y)
            diff_b_c = math.hypot(cb_px_x - cc_px_x, cb_px_y - cc_px_y)

            # Failure classification
            fail_category, fail_reason = classify_failure(
                err_px=err_b,
                in_box=in_box_b,
                gt_box_px=box_px,
                pred_px=(cb_px_x, cb_px_y),
                gt_px=(gt_px_x, gt_px_y),
                candidate_peaks=d_res.candidates,
                raw_size=(raw_w, raw_h),
                other_points=other_points,
                target_query=query,
                foveal_invoked=foveal_invoked,
                foveal_accepted=foveal_accepted,
                err_b_px=err_b,
                err_c_px=err_c,
            )

            category_counts[fail_category] = category_counts.get(fail_category, 0) + 1

            forensic_records.append({
                "sample_index": target_counter,
                "frame_id": frame_id,
                "app_name": app_name,
                "ui_category": ui_category,
                "target_id": target_id,
                "query": query,
                "element_kind": element_kind,
                "raw_dimensions": [raw_w, raw_h],
                "model_input_dimensions": [256, 256],
                "coordinate_transform": "Bilinear stretch (256, 256) -> original canvas",
                "gt_point_norm": [round(gt_norm_x, 4), round(gt_norm_y, 4)],
                "gt_point_px": [round(gt_px_x, 2), round(gt_px_y, 2)],
                "gt_box_px": [round(v, 2) for v in box_px],
                "box_dimensions_px": [round(box_w_px, 1), round(box_h_px, 1)],
                "aspect_ratio": aspect_ratio,
                "arm_a_pred_px": [round(ca_px_x, 2), round(ca_px_y, 2)],
                "arm_a_err_px": round(err_a, 2),
                "arm_a_in_box": bool(in_box_a),
                "arm_a_conf": round(ca_conf, 3),
                "arm_b_pred_px": [round(cb_px_x, 2), round(cb_px_y, 2)],
                "arm_b_err_px": round(err_b, 2),
                "arm_b_in_box": bool(in_box_b),
                "arm_b_conf": round(cb_conf, 3),
                "arm_b_margin": round(d_res.margin, 3),
                "arm_b_is_ambiguous": bool(d_res.is_ambiguous),
                "arm_c_pred_px": [round(cc_px_x, 2), round(cc_px_y, 2)],
                "arm_c_err_px": round(err_c, 2),
                "arm_c_in_box": bool(in_box_c),
                "arm_c_conf": round(cc_conf, 3),
                "diff_a_b_px": round(diff_a_b, 2),
                "diff_b_c_px": round(diff_b_c, 2),
                "foveal_invoked": foveal_invoked,
                "foveal_accepted": foveal_accepted,
                "foveal_offset_px": round(f_offset_px, 2),
                "candidate_peaks": cand_peaks,
                "failure_category": fail_category,
                "failure_evidence": fail_reason,
            })

    total_samples = len(forensic_records)
    print(f"Processed {total_samples} held-out targets.")

    # 3. Save JSON and CSV reports
    json_path = out_path / "per_sample_forensics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(forensic_records, f, indent=2)

    csv_path = out_path / "per_sample_forensics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_index", "frame_id", "app_name", "ui_category", "target_id",
            "query", "element_kind", "aspect_ratio",
            "arm_a_err_px", "arm_a_in_box", "arm_b_err_px", "arm_b_in_box",
            "arm_c_err_px", "arm_c_in_box", "arm_b_margin", "diff_a_b_px",
            "diff_b_c_px", "foveal_accepted", "failure_category", "failure_evidence"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in forensic_records:
            writer.writerow(r)

    print(f"Saved forensic records to {json_path} and {csv_path}")

    # 4. Generate Visual Outlier Gallery for Top 16 Outliers
    print("Generating Visual Outlier Gallery for worst localization errors...")
    # Sort by Arm B error descending
    sorted_by_err = sorted(forensic_records, key=lambda x: x["arm_b_err_px"], reverse=True)
    top_outliers = sorted_by_err[:16]

    gallery_manifest = []

    for rank, rec in enumerate(top_outliers, start=1):
        fr_id = rec["frame_id"]
        # Find frame
        matching_fr = next(fr for fr in test_frames if fr["frame_id"] == fr_id)
        img_p = data_dir / matching_fr["file_path"]
        base_img = Image.open(img_p).convert("RGB")
        draw_img = base_img.copy()
        draw = ImageDraw.Draw(draw_img)

        # Coordinates
        gt_x, gt_y = rec["gt_point_px"]
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = rec["gt_box_px"]
        pa_x, pa_y = rec["arm_a_pred_px"]
        pb_x, pb_y = rec["arm_b_pred_px"]
        pc_x, pc_y = rec["arm_c_pred_px"]

        # Draw GT BBox (Green)
        draw.rectangle([gt_xmin, gt_ymin, gt_xmax, gt_ymax], outline=(0, 230, 0), width=2)
        # Draw GT centroid (Green circle)
        draw.ellipse([gt_x - 3, gt_y - 3, gt_x + 3, gt_y + 3], fill=(0, 255, 0), outline=(0, 100, 0))

        # Draw Arm A (Blue Cross)
        draw.line([pa_x - 4, pa_y, pa_x + 4, pa_y], fill=(0, 120, 255), width=2)
        draw.line([pa_x, pa_y - 4, pa_x, pa_y + 4], fill=(0, 120, 255), width=2)

        # Draw Arm B (Orange Cross)
        draw.line([pb_x - 5, pb_y, pb_x + 5, pb_y], fill=(255, 140, 0), width=2)
        draw.line([pb_x, pb_y - 5, pb_x, pb_y + 5], fill=(255, 140, 0), width=2)

        # Draw Arm C (Red Cross)
        draw.line([pc_x - 4, pc_y - 4, pc_x + 4, pc_y + 4], fill=(255, 0, 0), width=2)
        draw.line([pc_x - 4, pc_y + 4, pc_x + 4, pc_y - 4], fill=(255, 0, 0), width=2)

        # Crop 128x128 window around midpoint of GT and Arm B
        mid_x = (gt_x + pb_x) / 2.0
        mid_y = (gt_y + pb_y) / 2.0
        crop_x1 = max(0, int(mid_x - 64))
        crop_y1 = max(0, int(mid_y - 64))
        crop_x2 = min(base_img.width, crop_x1 + 128)
        crop_y2 = min(base_img.height, crop_y1 + 128)
        if crop_x2 - crop_x1 < 128 and crop_x1 > 0:
            crop_x1 = max(0, crop_x2 - 128)
        if crop_y2 - crop_y1 < 128 and crop_y1 > 0:
            crop_y1 = max(0, crop_y2 - 128)

        patch = draw_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        patch_resized = patch.resize((256, 256), Image.NEAREST)

        import re
        safe_query = re.sub(r'[^a-zA-Z0-9_\-]', '_', rec["query"])[:25]
        outlier_filename = f"outlier_{rank:02d}_{fr_id}_{safe_query}.png"
        outlier_path = gallery_dir / outlier_filename
        patch_resized.save(outlier_path)

        gallery_manifest.append({
            "rank": rank,
            "filename": outlier_filename,
            "path": str(outlier_path),
            "frame_id": fr_id,
            "query": rec["query"],
            "category": rec["ui_category"],
            "err_a": rec["arm_a_err_px"],
            "err_b": rec["arm_b_err_px"],
            "err_c": rec["arm_c_err_px"],
            "in_box_b": rec["arm_b_in_box"],
            "failure_category": rec["failure_category"],
            "failure_evidence": rec["failure_evidence"],
        })

    # Write OUTLIER_GALLERY.md
    gallery_md_p = out_path / "OUTLIER_GALLERY.md"
    with open(gallery_md_p, "w", encoding="utf-8") as f:
        f.write("# AVCS Outlier Error Gallery (Top 16 Long-Tail Errors)\n\n")
        f.write("Visual diagnostics showing Ground Truth (Green box/dot), Arm A (Blue cross), Arm B (Orange cross), and Arm C (Red cross).\n\n")
        f.write("| Rank | Query | Category | Arm A Err | Arm B Err | In Box? | Failure Diagnosis |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for it in gallery_manifest:
            f.write(f"| #{it['rank']} | **{it['query']}** | `{it['category']}` | {it['err_a']} px | **{it['err_b']} px** | {it['in_box_b']} | **{it['failure_category']}**: {it['failure_evidence']} |\n")
        f.write("\n\n## Visual Crops\n\n")
        for it in gallery_manifest:
            f.write(f"### Rank #{it['rank']}: {it['query']} (`{it['category']}`)\n")
            f.write(f"- **Error (Arm B):** {it['err_b']} px (In Box: `{it['in_box_b']}`)\n")
            f.write(f"- **Category:** `{it['failure_category']}`\n")
            f.write(f"- **Diagnosis:** {it['failure_evidence']}\n\n")
            f.write(f"![Outlier #{it['rank']}](gallery/{it['filename']})\n\n---\n\n")

    # 5. Metric Audit & Distribution
    def compute_distribution(errors: List[float], in_boxes: int) -> Dict[str, Any]:
        n = len(errors)
        return {
            "total_samples": n,
            "in_box_count": in_boxes,
            "in_box_pct": round(in_boxes / n * 100.0, 2),
            "median_p50_px": round(float(np.percentile(errors, 50, method="linear")), 2),
            "p75_px": round(float(np.percentile(errors, 75, method="linear")), 2),
            "p90_px": round(float(np.percentile(errors, 90, method="linear")), 2),
            "p95_px": round(float(np.percentile(errors, 95, method="linear")), 2),
            "p99_px": round(float(np.percentile(errors, 99, method="linear")), 2),
            "max_px": round(float(np.max(errors)), 2),
            "mean_px": round(float(np.mean(errors)), 2),
            "pct_le_1px": round(sum(1 for e in errors if e <= 1.0) / n * 100.0, 2),
            "pct_le_2px": round(sum(1 for e in errors if e <= 2.0) / n * 100.0, 2),
            "pct_le_3px": round(sum(1 for e in errors if e <= 3.0) / n * 100.0, 2),
            "pct_le_5px": round(sum(1 for e in errors if e <= 5.0) / n * 100.0, 2),
            "pct_le_10px": round(sum(1 for e in errors if e <= 10.0) / n * 100.0, 2),
            "pct_le_15px": round(sum(1 for e in errors if e <= 15.0) / n * 100.0, 2),
            "pct_le_20px": round(sum(1 for e in errors if e <= 20.0) / n * 100.0, 2),
            "count_le_3px": sum(1 for e in errors if e <= 3.0),
        }

    dist_a = compute_distribution(errors_a, in_box_a_count)
    dist_b = compute_distribution(errors_b, in_box_b_count)
    dist_c = compute_distribution(errors_c, in_box_c_count)

    summary = {
        "sprint": "AVCS Milestone 3.2 Error Forensics & Metric Audit",
        "checkpoints": {
            "coarse": {"path": coarse_checkpoint, "sha256": coarse_sha},
            "reranker": {"path": reranker_checkpoint, "sha256": reranker_sha},
            "foveal": {"path": foveal_checkpoint, "sha256": foveal_sha},
        },
        "total_held_out_samples": total_samples,
        "valid_click_region_success_arm_b": {
            "count": valid_region_clicks_b,
            "pct": round(valid_region_clicks_b / total_samples * 100.0, 2),
        },
        "failure_taxonomy_distribution": {
            cat: {"count": count, "pct": round(count / total_samples * 100.0, 2)}
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        },
        "metric_distributions": {
            "arm_a_coarse_v2": dist_a,
            "arm_b_global_disambiguator": dist_b,
            "arm_c_selective_fovea": dist_c,
        },
        "top_outliers": gallery_manifest,
    }

    summary_p = out_path / "forensics_summary.json"
    with open(summary_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Forensics Summary Complete ===")
    print(f"Total Targets Analyzed: {total_samples}")
    print(f"Valid Click Region Success (Point-in-Box): {summary['valid_click_region_success_arm_b']['pct']}% ({valid_region_clicks_b}/{total_samples})")
    print(f"Failure Taxonomy Breakdown:")
    for cat, data in summary["failure_taxonomy_distribution"].items():
        print(f"  - {cat:48s}: {data['count']:3d} ({data['pct']:.1f}%)")
    print(f"\nCentroid Distribution (Arm B):")
    print(f"  P50: {dist_b['median_p50_px']} px | P90: {dist_b['p90_px']} px | P95: {dist_b['p95_px']} px | Max: {dist_b['max_px']} px")
    print(f"  <=1px: {dist_b['pct_le_1px']}% | <=2px: {dist_b['pct_le_2px']}% | <=3px: {dist_b['pct_le_3px']}% | <=5px: {dist_b['pct_le_5px']}% | <=10px: {dist_b['pct_le_10px']}% | <=20px: {dist_b['pct_le_20px']}%")

    return summary


if __name__ == "__main__":
    run_error_forensics_pipeline()
