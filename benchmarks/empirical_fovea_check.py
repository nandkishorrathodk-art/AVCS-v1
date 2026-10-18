"""Empirical reality check and limit-discovery benchmark for FovealAcuityEngine and refine_affordance_with_fovea.

Investigates:
1. Spatial boundary localization error on diverse UI elements (buttons, inputs, tabs, icons, links, labels, containers).
2. Spatial resolution limits: tiny 8x8 icons, dense 12px text labels, low-contrast buttons, borderless elements.
3. Contrast score gating: reliability of rejecting flat/noisy backgrounds vs false rejection of valid elements.
4. Latency breakdown of un-downsampled foveal crop extraction + inspection on CPU across resolutions.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from avcs.grounding import FOVEAL_ELEMENT_TYPES, FovealAcuityEngine, FovealInspection
from avcs.human_eye import HumanEye
from avcs.training.gui_dataset import DesktopGUIGenerator, compute_crop_contrast_score


def calculate_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two [xmin, ymin, xmax, ymax] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-9:
        return 0.0
    return float(inter_area / union_area)


def run_spatial_boundary_evaluation(
    engine: FovealAcuityEngine,
    num_samples_per_kind: int = 100,
) -> Dict[str, Any]:
    """Test 1: Measures spatial boundary localization error across diverse UI element kinds."""
    print("\n" + "=" * 70)
    print("SECTION 1: SPATIAL BOUNDARY LOCALIZATION ERROR ON DIVERSE UI ELEMENTS")
    print("=" * 70)

    gen = DesktopGUIGenerator()
    results_by_kind: Dict[str, Dict[str, Any]] = {}
    all_errors_l2_px: List[float] = []
    all_ious: List[float] = []
    all_x1_err_px: List[float] = []
    all_y1_err_px: List[float] = []
    all_x2_err_px: List[float] = []
    all_y2_err_px: List[float] = []
    all_w_err_px: List[float] = []
    all_h_err_px: List[float] = []

    confusion_matrix = {k: {k2: 0 for k2 in FOVEAL_ELEMENT_TYPES} for k in FOVEAL_ELEMENT_TYPES}

    crop_w, crop_h = 96, 96

    for kind in FOVEAL_ELEMENT_TYPES:
        kind_l2: List[float] = []
        kind_iou: List[float] = []
        kind_x1_err: List[float] = []
        kind_y1_err: List[float] = []
        kind_x2_err: List[float] = []
        kind_y2_err: List[float] = []
        kind_w_err: List[float] = []
        kind_h_err: List[float] = []
        correct_cls = 0

        for i in range(num_samples_per_kind):
            seed = 100000 + i * 17 + FOVEAL_ELEMENT_TYPES.index(kind) * 313
            sample = gen.generate_foveal_element(element_kind=kind, seed=seed)

            # Ground truth: sample.bbox is [ymin, xmin, ymax, xmax]
            gt_ymin, gt_xmin, gt_ymax, gt_xmax = sample.bbox
            gt_box_xyxy = (gt_xmin, gt_ymin, gt_xmax, gt_ymax)
            gt_cx_px = ((gt_xmin + gt_xmax) / 2.0) * crop_w
            gt_cy_px = ((gt_ymin + gt_ymax) / 2.0) * crop_h
            gt_w_px = (gt_xmax - gt_xmin) * crop_w
            gt_h_px = (gt_ymax - gt_ymin) * crop_h

            # Inspect crop
            insp = engine.inspect_crop(sample.crop)

            # Predicted: insp.bbox is [xmin, ymin, xmax, ymax]
            pred_xmin, pred_ymin, pred_xmax, pred_ymax = insp.bbox
            pred_cx_px = ((pred_xmin + pred_xmax) / 2.0) * crop_w
            pred_cy_px = ((pred_ymin + pred_ymax) / 2.0) * crop_h
            pred_w_px = (pred_xmax - pred_xmin) * crop_w
            pred_h_px = (pred_ymax - pred_ymin) * crop_h

            # Errors in native crop pixels
            dx = abs(pred_cx_px - gt_cx_px)
            dy = abs(pred_cy_px - gt_cy_px)
            l2_err = math.hypot(dx, dy)

            x1_err = abs(pred_xmin - gt_xmin) * crop_w
            y1_err = abs(pred_ymin - gt_ymin) * crop_h
            x2_err = abs(pred_xmax - gt_xmax) * crop_w
            y2_err = abs(pred_ymax - gt_ymax) * crop_h
            w_err = abs(pred_w_px - gt_w_px)
            h_err = abs(pred_h_px - gt_h_px)

            iou = calculate_iou(insp.bbox, gt_box_xyxy)

            # Classification
            if insp.element_kind in confusion_matrix[kind]:
                confusion_matrix[kind][insp.element_kind] += 1
            if insp.element_kind == kind:
                correct_cls += 1

            kind_l2.append(l2_err)
            kind_iou.append(iou)
            kind_x1_err.append(x1_err)
            kind_y1_err.append(y1_err)
            kind_x2_err.append(x2_err)
            kind_y2_err.append(y2_err)
            kind_w_err.append(w_err)
            kind_h_err.append(h_err)

        all_errors_l2_px.extend(kind_l2)
        all_ious.extend(kind_iou)
        all_x1_err_px.extend(kind_x1_err)
        all_y1_err_px.extend(kind_y1_err)
        all_x2_err_px.extend(kind_x2_err)
        all_y2_err_px.extend(kind_y2_err)
        all_w_err_px.extend(kind_w_err)
        all_h_err_px.extend(kind_h_err)

        acc = correct_cls / float(num_samples_per_kind)
        results_by_kind[kind] = {
            "num_samples": num_samples_per_kind,
            "accuracy": acc,
            "mean_l2_px": float(np.mean(kind_l2)),
            "median_l2_px": float(np.median(kind_l2)),
            "p95_l2_px": float(np.percentile(kind_l2, 95)),
            "pct_within_2px": float(np.mean([e <= 2.0 for e in kind_l2])),
            "mean_iou": float(np.mean(kind_iou)),
            "median_iou": float(np.median(kind_iou)),
            "p95_iou": float(np.percentile(kind_iou, 95)),
            "pct_iou_gte_05": float(np.mean([i >= 0.5 for i in kind_iou])),
            "pct_iou_gte_07": float(np.mean([i >= 0.7 for i in kind_iou])),
            "mean_x1_err_px": float(np.mean(kind_x1_err)),
            "mean_y1_err_px": float(np.mean(kind_y1_err)),
            "mean_x2_err_px": float(np.mean(kind_x2_err)),
            "mean_y2_err_px": float(np.mean(kind_y2_err)),
            "mean_w_err_px": float(np.mean(kind_w_err)),
            "mean_h_err_px": float(np.mean(kind_h_err)),
        }

    # Summary across all kinds
    overall_summary = {
        "total_samples": len(all_errors_l2_px),
        "mean_l2_px": float(np.mean(all_errors_l2_px)),
        "median_l2_px": float(np.median(all_errors_l2_px)),
        "p95_l2_px": float(np.percentile(all_errors_l2_px, 95)),
        "pct_within_1px": float(np.mean([e <= 1.0 for e in all_errors_l2_px])),
        "pct_within_2px": float(np.mean([e <= 2.0 for e in all_errors_l2_px])),
        "pct_within_3px": float(np.mean([e <= 3.0 for e in all_errors_l2_px])),
        "mean_iou": float(np.mean(all_ious)),
        "median_iou": float(np.median(all_ious)),
        "pct_iou_gte_05": float(np.mean([i >= 0.5 for i in all_ious])),
        "pct_iou_gte_07": float(np.mean([i >= 0.7 for i in all_ious])),
        "pct_iou_gte_085": float(np.mean([i >= 0.85 for i in all_ious])),
        "mean_x1_err_px": float(np.mean(all_x1_err_px)),
        "mean_y1_err_px": float(np.mean(all_y1_err_px)),
        "mean_x2_err_px": float(np.mean(all_x2_err_px)),
        "mean_y2_err_px": float(np.mean(all_y2_err_px)),
        "mean_w_err_px": float(np.mean(all_w_err_px)),
        "mean_h_err_px": float(np.mean(all_h_err_px)),
    }

    print("\n--- Summary by Element Kind (Native 96x96 Foveal Crop) ---")
    header = f"{'Kind':<14} | {'Acc':<6} | {'Mean e_L2':<10} | {'Med e_L2':<10} | {'%<=2px':<8} | {'Mean IoU':<9} | {'%IoU>=0.7':<10} | {'dW (px)':<8} | {'dH (px)':<8}"
    print(header)
    print("-" * len(header))
    for kind, stats in results_by_kind.items():
        print(
            f"{kind:<14} | {stats['accuracy']*100:5.1f}% | {stats['mean_l2_px']:8.2f}px | {stats['median_l2_px']:8.2f}px | "
            f"{stats['pct_within_2px']*100:6.1f}% | {stats['mean_iou']:8.3f} | {stats['pct_iou_gte_07']*100:8.1f}% | "
            f"{stats['mean_w_err_px']:7.2f} | {stats['mean_h_err_px']:7.2f}"
        )
    print("-" * len(header))
    print(
        f"{'OVERALL':<14} | {'-':<6} | {overall_summary['mean_l2_px']:8.2f}px | {overall_summary['median_l2_px']:8.2f}px | "
        f"{overall_summary['pct_within_2px']*100:6.1f}% | {overall_summary['mean_iou']:8.3f} | {overall_summary['pct_iou_gte_07']*100:8.1f}% | "
        f"{overall_summary['mean_w_err_px']:7.2f} | {overall_summary['mean_h_err_px']:7.2f}"
    )

    return {
        "results_by_kind": results_by_kind,
        "overall_summary": overall_summary,
        "confusion_matrix": confusion_matrix,
    }


def run_gaze_jitter_sensitivity(
    engine: FovealAcuityEngine,
    jitter_radii: List[int] = [0, 2, 4, 6, 8, 12, 16, 20],
) -> Dict[str, Any]:
    """Evaluates how initial gaze / affordance centroid offset degrades localization and IoU."""
    print("\n--- Gaze Fixation Centroid Jitter Sensitivity ---")
    gen = DesktopGUIGenerator()
    jitter_results: Dict[int, Dict[str, float]] = {}

    for jitter_px in jitter_radii:
        ious: List[float] = []
        l2_errors: List[float] = []
        correct_kinds = 0
        total = 70

        for i in range(total):
            kind = FOVEAL_ELEMENT_TYPES[i % len(FOVEAL_ELEMENT_TYPES)]
            sample = gen.generate_foveal_element(element_kind=kind, seed=200000 + i)

            # Embed 96x96 crop into a larger 360x640 frame centered at (320, 180)
            H, W = 360, 640
            frame = np.full((H, W, 3), 35, dtype=np.uint8)
            y0, x0 = 180 - 48, 320 - 48
            frame[y0:y0 + 96, x0:x0 + 96] = sample.crop

            # Ground truth bbox in frame coordinates
            gt_ymin, gt_xmin, gt_ymax, gt_xmax = sample.bbox
            frame_gt_x1 = (x0 + gt_xmin * 96) / W
            frame_gt_y1 = (y0 + gt_ymin * 96) / H
            frame_gt_x2 = (x0 + gt_xmax * 96) / W
            frame_gt_y2 = (y0 + gt_ymax * 96) / H
            frame_gt_box = (frame_gt_x1, frame_gt_y1, frame_gt_x2, frame_gt_y2)

            gt_cx_px = (frame_gt_x1 + frame_gt_x2) / 2.0 * W
            gt_cy_px = (frame_gt_y1 + frame_gt_y2) / 2.0 * H

            # Apply random angle jitter
            angle = (i * 1.618) % (2 * math.pi)
            offset_x = jitter_px * math.cos(angle)
            offset_y = jitter_px * math.sin(angle)

            fixation_x = (320.0 + offset_x) / W
            fixation_y = (180.0 + offset_y) / H

            insp = engine.inspect(frame, (fixation_x, fixation_y))

            pred_cx_px = (insp.bbox[0] + insp.bbox[2]) / 2.0 * W
            pred_cy_px = (insp.bbox[1] + insp.bbox[3]) / 2.0 * H

            l2_err = math.hypot(pred_cx_px - gt_cx_px, pred_cy_px - gt_cy_px)
            iou = calculate_iou(insp.bbox, frame_gt_box)

            if insp.element_kind == kind:
                correct_kinds += 1
            ious.append(iou)
            l2_errors.append(l2_err)

        jitter_results[jitter_px] = {
            "mean_l2_px": float(np.mean(l2_errors)),
            "median_l2_px": float(np.median(l2_errors)),
            "p95_l2_px": float(np.percentile(l2_errors, 95)),
            "mean_iou": float(np.mean(ious)),
            "median_iou": float(np.median(ious)),
            "accuracy": float(correct_kinds / total),
        }
        print(
            f"Jitter: {jitter_px:2d} px | Mean e_L2: {np.mean(l2_errors):6.2f} px | "
            f"Median e_L2: {np.median(l2_errors):6.2f} px | Mean IoU: {np.mean(ious):.3f} | Acc: {correct_kinds/total*100:5.1f}%"
        )

    return jitter_results


def run_spatial_resolution_limits(
    engine: FovealAcuityEngine,
) -> Dict[str, Any]:
    """Test 2: Investigates spatial resolution limits: tiny icons, dense text labels, low-contrast buttons, borderless."""
    print("\n" + "=" * 70)
    print("SECTION 2: SPATIAL RESOLUTION LIMITS & CORNER CASES")
    print("=" * 70)

    results: Dict[str, Any] = {}

    # 2A: Tiny Icons Scale Sweep
    print("\n--- 2A. Tiny Icons Scale Sweep (8x8 to 40x40) ---")
    icon_sizes = [8, 10, 12, 14, 16, 20, 24, 32, 40]
    icon_sweep_data = {}

    for sz in icon_sizes:
        ious = []
        l2_errors = []
        contrast_scores = []
        classified_as_icon = 0
        pred_widths = []
        pred_heights = []
        n_trials = 30

        for i in range(n_trials):
            crop = np.full((96, 96, 3), 40, dtype=np.uint8)
            cx, cy = 48, 48
            # Draw icon of size sz x sz
            x1 = cx - sz // 2
            y1 = cy - sz // 2
            x2 = x1 + sz
            y2 = y1 + sz
            # Icon styling: light gray with subtle border
            crop[y1:y2, x1:x2] = (140, 145, 160)
            crop[y1:min(y2, y1+1), x1:x2] = (200, 205, 220)
            crop[max(y1, y2-1):y2, x1:x2] = (200, 205, 220)

            contrast = compute_crop_contrast_score(crop)
            contrast_scores.append(contrast)

            gt_box = (x1 / 96.0, y1 / 96.0, x2 / 96.0, y2 / 96.0)
            insp = engine.inspect_crop(crop)

            pred_w_px = (insp.bbox[2] - insp.bbox[0]) * 96.0
            pred_h_px = (insp.bbox[3] - insp.bbox[1]) * 96.0
            pred_cx_px = (insp.bbox[0] + insp.bbox[2]) / 2.0 * 96.0
            pred_cy_px = (insp.bbox[1] + insp.bbox[3]) / 2.0 * 96.0

            l2_err = math.hypot(pred_cx_px - cx, pred_cy_px - cy)
            iou = calculate_iou(insp.bbox, gt_box)

            if insp.element_kind == "icon":
                classified_as_icon += 1
            ious.append(iou)
            l2_errors.append(l2_err)
            pred_widths.append(pred_w_px)
            pred_heights.append(pred_h_px)

        icon_sweep_data[sz] = {
            "mean_contrast": float(np.mean(contrast_scores)),
            "pct_passes_min_contrast_001": float(np.mean([c >= 0.010 for c in contrast_scores])),
            "mean_iou": float(np.mean(ious)),
            "mean_l2_px": float(np.mean(l2_errors)),
            "icon_accuracy": float(classified_as_icon / n_trials),
            "mean_pred_w_px": float(np.mean(pred_widths)),
            "mean_pred_h_px": float(np.mean(pred_heights)),
            "gt_size_px": sz,
        }
        print(
            f"Icon {sz:2d}x{sz:2d} px | Contrast: {np.mean(contrast_scores):.4f} "
            f"(Gating Pass: {icon_sweep_data[sz]['pct_passes_min_contrast_001']*100:5.1f}%) | "
            f"Pred WxH: {np.mean(pred_widths):4.1f}x{np.mean(pred_heights):4.1f} px | "
            f"IoU: {np.mean(ious):.3f} | e_L2: {np.mean(l2_errors):4.2f}px | Icon Acc: {classified_as_icon/n_trials*100:5.1f}%"
        )
    results["tiny_icons"] = icon_sweep_data

    # 2B: Dense Text Labels and Spacing Limits
    print("\n--- 2B. Dense Text Labels and Vertical Pitch Limits ---")
    text_heights = [8, 10, 12, 14, 16, 20]
    text_height_data = {}

    for th in text_heights:
        ious = []
        vert_errors = []
        contrast_scores = []
        kinds = {}
        n_trials = 30

        for i in range(n_trials):
            crop = np.full((96, 96, 3), 40, dtype=np.uint8)
            cx, cy = 48, 48
            tw = 50
            # Text label of height th
            y1 = cy - th // 2
            y2 = y1 + th
            x1 = cx - tw // 2
            x2 = x1 + tw

            # Simulate text lines inside
            crop[y1:y2, x1:x2] = 45  # subtle text bounding area
            # glyph lines
            crop[cy - max(1, th // 4):cy + max(1, th // 4), x1 + 2:x2 - 2] = (220, 220, 225)

            contrast = compute_crop_contrast_score(crop)
            contrast_scores.append(contrast)

            gt_box = (x1 / 96.0, y1 / 96.0, x2 / 96.0, y2 / 96.0)
            insp = engine.inspect_crop(crop)

            kinds[insp.element_kind] = kinds.get(insp.element_kind, 0) + 1
            pred_y1_px = insp.bbox[1] * 96.0
            pred_y2_px = insp.bbox[3] * 96.0
            vert_err = (abs(pred_y1_px - y1) + abs(pred_y2_px - y2)) / 2.0

            iou = calculate_iou(insp.bbox, gt_box)
            ious.append(iou)
            vert_errors.append(vert_err)

        dominant_kind = max(kinds, key=kinds.get)
        text_height_data[th] = {
            "mean_contrast": float(np.mean(contrast_scores)),
            "mean_iou": float(np.mean(ious)),
            "mean_vert_err_px": float(np.mean(vert_errors)),
            "dominant_kind": dominant_kind,
            "kind_distribution": kinds,
        }
        print(
            f"Text H: {th:2d} px | Contrast: {np.mean(contrast_scores):.4f} | "
            f"Mean IoU: {np.mean(ious):.3f} | Mean Vert Err: {np.mean(vert_errors):4.2f} px | "
            f"Dominant Kind: {dominant_kind} ({kinds[dominant_kind]}/{n_trials})"
        )
    results["text_labels"] = text_height_data

    # Multi-label vertical packing test (e.g. menu items spaced closely)
    print("\n--- 2B.1 Multi-Label Menu Item Spacing Distraction ---")
    spacing_gaps = [2, 4, 6, 8, 12, 16, 24]
    menu_spacing_data = {}
    for gap in spacing_gaps:
        ious = []
        height_dilations = []
        n_trials = 30
        th = 12
        tw = 60
        for i in range(n_trials):
            crop = np.full((96, 96, 3), 40, dtype=np.uint8)
            cx, cy = 48, 48
            # Central target item
            y1 = cy - th // 2
            y2 = y1 + th
            x1 = cx - tw // 2
            x2 = x1 + tw
            crop[cy - 1:cy + 2, x1:x2] = (220, 220, 225)

            # Neighboring items above and below
            y_above_2 = y1 - gap
            y_above_1 = y_above_2 - th
            if y_above_1 >= 0:
                crop[y_above_1 + 4:y_above_2 - 4, x1:x2] = (180, 180, 185)

            y_below_1 = y2 + gap
            y_below_2 = y_below_1 + th
            if y_below_2 < 96:
                crop[y_below_1 + 4:y_below_2 - 4, x1:x2] = (180, 180, 185)

            gt_box = (x1 / 96.0, y1 / 96.0, x2 / 96.0, y2 / 96.0)
            insp = engine.inspect_crop(crop)
            pred_h_px = (insp.bbox[3] - insp.bbox[1]) * 96.0
            height_dilations.append(pred_h_px / th)
            ious.append(calculate_iou(insp.bbox, gt_box))

        menu_spacing_data[gap] = {
            "mean_iou": float(np.mean(ious)),
            "mean_height_ratio": float(np.mean(height_dilations)),
        }
        print(
            f"Item Gap: {gap:2d} px | Mean IoU: {np.mean(ious):.3f} | "
            f"Pred Height / GT Height: {np.mean(height_dilations):.2f}x "
            f"({'Merged neighbors' if np.mean(height_dilations) > 1.8 else 'Isolated target'})"
        )
    results["menu_spacing"] = menu_spacing_data

    # 2C: Low-Contrast Buttons
    print("\n--- 2C. Low-Contrast Button Detection & Localization ---")
    delta_rgbs = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64]
    contrast_sweep_data = {}
    bg_val = 45

    for d_rgb in delta_rgbs:
        contrast_scores = []
        ious = []
        l2_errors = []
        passes_gating = []
        btn_acc = []
        n_trials = 30

        for i in range(n_trials):
            crop = np.full((96, 96, 3), bg_val, dtype=np.uint8)
            bw, bh = 56, 26
            cx, cy = 48, 48
            x1 = cx - bw // 2
            y1 = cy - bh // 2
            x2 = x1 + bw
            y2 = y1 + bh

            # Low contrast button fill and subtle border
            btn_color = bg_val + d_rgb
            border_color = bg_val + int(d_rgb * 1.5)
            crop[y1:y2, x1:x2] = btn_color
            crop[y1:min(y2, y1+1), x1:x2] = border_color
            crop[max(y1, y2-1):y2, x1:x2] = border_color

            contrast = compute_crop_contrast_score(crop)
            contrast_scores.append(contrast)

            gt_box = (x1 / 96.0, y1 / 96.0, x2 / 96.0, y2 / 96.0)
            insp = engine.inspect_crop(crop)

            pred_cx_px = (insp.bbox[0] + insp.bbox[2]) / 2.0 * 96.0
            pred_cy_px = (insp.bbox[1] + insp.bbox[3]) / 2.0 * 96.0
            l2_err = math.hypot(pred_cx_px - cx, pred_cy_px - cy)
            iou = calculate_iou(insp.bbox, gt_box)

            passes_gating.append(contrast >= 0.010)
            btn_acc.append(insp.element_kind == "button")
            ious.append(iou)
            l2_errors.append(l2_err)

        contrast_sweep_data[d_rgb] = {
            "mean_contrast": float(np.mean(contrast_scores)),
            "pct_passes_min_contrast": float(np.mean(passes_gating)),
            "mean_iou": float(np.mean(ious)),
            "mean_l2_px": float(np.mean(l2_errors)),
            "btn_accuracy": float(np.mean(btn_acc)),
        }
        print(
            f"Delta RGB: {d_rgb:2d} | Contrast: {np.mean(contrast_scores):.5f} | "
            f"Passes min_contrast (>=0.010): {np.mean(passes_gating)*100:5.1f}% | "
            f"IoU: {np.mean(ious):.3f} | e_L2: {np.mean(l2_errors):4.2f}px | Btn Acc: {np.mean(btn_acc)*100:5.1f}%"
        )
    results["low_contrast_buttons"] = contrast_sweep_data

    # 2D: Borderless Elements
    print("\n--- 2D. Borderless Elements (Text-Only Buttons, Floating Icons, Frameless Inputs) ---")
    borderless_scenarios = [
        ("borderless_text_button", "button", 50, 16),
        ("borderless_icon", "icon", 24, 24),
        ("borderless_input_underline", "input_field", 64, 24),
    ]
    borderless_results = {}

    for name, expected_kind, bw, bh in borderless_scenarios:
        ious = []
        l2_errors = []
        pred_w_list = []
        pred_h_list = []
        correct_kinds = 0
        n_trials = 30

        for i in range(n_trials):
            crop = np.full((96, 96, 3), 40, dtype=np.uint8)
            cx, cy = 48, 48
            x1 = cx - bw // 2
            y1 = cy - bh // 2
            x2 = x1 + bw
            y2 = y1 + bh

            if name == "borderless_text_button":
                # Only white text glyph line in center, NO button background or border
                crop[cy - 2:cy + 2, x1:x2] = (255, 255, 255)
            elif name == "borderless_icon":
                # Pure icon geometry without container rect: plus sign
                crop[cy - 8:cy + 8, cx - 2:cx + 2] = (200, 205, 220)
                crop[cy - 2:cy + 2, cx - 8:cx + 8] = (200, 205, 220)
            elif name == "borderless_input_underline":
                # Only a bottom underline and a caret, NO box outline
                crop[y2 - 2:y2, x1:x2] = (0, 120, 215)  # blue underline
                crop[cy - 6:cy + 6, x1 + 6:x1 + 8] = (255, 255, 255)  # caret

            gt_box = (x1 / 96.0, y1 / 96.0, x2 / 96.0, y2 / 96.0)
            insp = engine.inspect_crop(crop)

            pred_w = (insp.bbox[2] - insp.bbox[0]) * 96.0
            pred_h = (insp.bbox[3] - insp.bbox[1]) * 96.0
            pred_cx = (insp.bbox[0] + insp.bbox[2]) / 2.0 * 96.0
            pred_cy = (insp.bbox[1] + insp.bbox[3]) / 2.0 * 96.0

            l2_err = math.hypot(pred_cx - cx, pred_cy - cy)
            iou = calculate_iou(insp.bbox, gt_box)

            if insp.element_kind == expected_kind:
                correct_kinds += 1
            ious.append(iou)
            l2_errors.append(l2_err)
            pred_w_list.append(pred_w)
            pred_h_list.append(pred_h)

        borderless_results[name] = {
            "expected_kind": expected_kind,
            "accuracy": float(correct_kinds / n_trials),
            "mean_iou": float(np.mean(ious)),
            "mean_l2_px": float(np.mean(l2_errors)),
            "gt_dims_px": [bw, bh],
            "mean_pred_dims_px": [float(np.mean(pred_w_list)), float(np.mean(pred_h_list))],
        }
        print(
            f"{name:<28} | Exp: {expected_kind:<11} | Acc: {correct_kinds/n_trials*100:5.1f}% | "
            f"IoU: {np.mean(ious):.3f} | e_L2: {np.mean(l2_errors):4.2f}px | "
            f"GT Dims: {bw}x{bh} -> Pred: {np.mean(pred_w_list):.1f}x{np.mean(pred_h_list):.1f} px"
        )
    results["borderless"] = borderless_results

    return results


def run_contrast_gating_evaluation(
    engine: FovealAcuityEngine,
    human_eye: HumanEye,
) -> Dict[str, Any]:
    """Test 3: Evaluates contrast score gating and fail-closed rejection behavior."""
    print("\n" + "=" * 70)
    print("SECTION 3: CONTRAST SCORE GATING & FAIL-CLOSED REJECTION BEHAVIOR")
    print("=" * 70)

    categories = {}

    # Category A: Uniform flat screen
    print("\n--- 3A. Featureless Flat Crops ---")
    flat_colors = [(0, 0, 0), (32, 32, 32), (128, 128, 128), (240, 240, 240), (255, 255, 255)]
    flat_scores = []
    flat_eye_rejected = 0

    for col in flat_colors:
        crop = np.full((96, 96, 3), col, dtype=np.uint8)
        score = compute_crop_contrast_score(crop)
        flat_scores.append(score)
        # Check via refine_affordance_with_fovea
        res = human_eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=crop, min_contrast=0.010)
        if res is None:
            flat_eye_rejected += 1

    categories["flat_uniform"] = {
        "min_score": float(np.min(flat_scores)),
        "max_score": float(np.max(flat_scores)),
        "mean_score": float(np.mean(flat_scores)),
        "rejection_rate_at_001": float(flat_eye_rejected / len(flat_colors)),
    }
    print(
        f"Flat Screen: Contrast min={np.min(flat_scores):.6f}, max={np.max(flat_scores):.6f} | "
        f"Rejection Rate at 0.010: {flat_eye_rejected / len(flat_colors) * 100:.1f}%"
    )

    # Category B: Sensor / Capture Gaussian Noise on Flat Background
    print("\n--- 3B. Flat Screen with Sensor / Gaussian Noise Sweep ---")
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
    noise_results = {}
    for sig in sigmas:
        scores = []
        eye_accepted = 0
        n_trials = 30
        for i in range(n_trials):
            np.random.seed(300000 + i)
            base = np.full((96, 96, 3), 128, dtype=np.float32)
            noise = np.random.normal(0.0, sig, base.shape)
            crop = np.clip(base + noise, 0, 255).astype(np.uint8)
            score = compute_crop_contrast_score(crop)
            scores.append(score)
            res = human_eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=crop, min_contrast=0.010)
            if res is not None:
                eye_accepted += 1

        noise_results[sig] = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "leak_rate_at_001": float(eye_accepted / n_trials),
        }
        print(
            f"Noise sigma={sig:4.1f} LSB | Mean Contrast: {np.mean(scores):.5f} | "
            f"False Acceptance (Leakage past 0.010 gate): {eye_accepted/n_trials*100:5.1f}%"
        )
    categories["sensor_noise"] = noise_results

    # Category C: Smooth Linear Wallpaper Gradients
    print("\n--- 3C. Smooth Desktop Wallpaper Gradients ---")
    slopes = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
    gradient_results = {}
    for slope in slopes:
        scores = []
        eye_accepted = 0
        n_trials = 20
        for i in range(n_trials):
            # Gradient along Y (shape 96x96x3)
            y_ramp = np.linspace(50, 50 + slope * 96, 96, endpoint=False)
            y_ramp_2d = np.tile(y_ramp[:, None], (1, 96))
            crop = np.repeat(y_ramp_2d[:, :, None], 3, axis=-1)
            crop = np.clip(crop, 0, 255).astype(np.uint8)
            score = compute_crop_contrast_score(crop)
            scores.append(score)
            res = human_eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=crop, min_contrast=0.010)
            if res is not None:
                eye_accepted += 1

        gradient_results[slope] = {
            "mean_score": float(np.mean(scores)),
            "leak_rate_at_001": float(eye_accepted / n_trials),
        }
        print(
            f"Gradient slope: {slope:4.2f} px/px | Mean Contrast: {np.mean(scores):.5f} | "
            f"False Acceptance: {eye_accepted/n_trials*100:5.1f}%"
        )
    categories["wallpaper_gradients"] = gradient_results

    # Category D: Legitimate UI Elements vs False Rejection
    print("\n--- 3D. Legitimate UI Elements & False Rejection Risk ---")
    gen = DesktopGUIGenerator()
    ui_scores_by_kind = {}
    false_rejections_by_kind = {}

    for kind in FOVEAL_ELEMENT_TYPES:
        scores = []
        rejections = 0
        n_trials = 50
        for i in range(n_trials):
            sample = gen.generate_foveal_element(element_kind=kind, seed=400000 + i * 13)
            score = sample.contrast_score
            scores.append(score)
            res = human_eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=sample.crop, min_contrast=0.010)
            if res is None:
                rejections += 1

        ui_scores_by_kind[kind] = {
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "false_rejection_rate": float(rejections / n_trials),
        }
        print(
            f"{kind:<14} | Contrast: mean={np.mean(scores):.5f}, min={np.min(scores):.5f}, max={np.max(scores):.5f} | "
            f"False Rejection at 0.010: {rejections/n_trials*100:5.1f}%"
        )
    categories["legitimate_ui"] = ui_scores_by_kind

    # Category E: The Spatial Dilution Paradox on Tiny Elements
    print("\n--- 3E. Spatial Dilution Paradox (High-Contrast Small Glyph vs Crop Area) ---")
    dilution_results = {}
    glyph_sizes = [4, 6, 8, 10, 12, 16, 20, 24]
    for gsz in glyph_sizes:
        # A pure black-and-white (255 delta) tiny square in 96x96 frame
        crop = np.zeros((96, 96, 3), dtype=np.uint8)
        c0 = 48 - gsz // 2
        crop[c0:c0 + gsz, c0:c0 + gsz] = 255
        score = compute_crop_contrast_score(crop)
        res = human_eye.refine_affordance_with_fovea((0.5, 0.5, "clickable", 0.9), frame=crop, min_contrast=0.010)
        rejected = res is None
        dilution_results[gsz] = {
            "contrast_score": score,
            "rejected_by_0010": rejected,
        }
        print(
            f"White-on-Black {gsz:2d}x{gsz:2d} px (Max possible local contrast) | "
            f"Contrast Score: {score:.5f} | Rejected by min_contrast=0.010: {rejected}"
        )
    categories["spatial_dilution"] = dilution_results

    return categories


def run_cpu_latency_benchmarks(
    engine: FovealAcuityEngine,
    warmup: int = 50,
    iterations: int = 500,
) -> Dict[str, Any]:
    """Test 4: High-precision latency breakdown on CPU across resolutions and pipeline stages."""
    print("\n" + "=" * 70)
    print("SECTION 4: CPU LATENCY BENCHMARK & PIPELINE BREAKDOWN")
    print("=" * 70)

    # 4A. Component-by-Component Timing Breakdown on 96x96
    print("\n--- 4A. Pipeline Component Timing Breakdown (96x96 Crop) ---")
    crop = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        engine.inspect_crop(crop)

    t_crop_slice = []
    t_contrast = []
    t_tensor_prep = []
    t_interpolate = []
    t_forward = []
    t_postproc = []
    t_total = []

    dev = engine.device

    for _ in range(iterations):
        t0 = time.perf_counter_ns()

        # 1. Slicing / crop extraction
        c_t0 = time.perf_counter_ns()
        c_crop, c_box = engine.crop_fovea(crop, (0.5, 0.5))
        c_t1 = time.perf_counter_ns()

        # 2. Edge sharpness / contrast
        k_t0 = time.perf_counter_ns()
        gray = np.mean(c_crop, axis=-1)
        gy, gx = np.gradient(gray)
        contrast = float(np.mean(np.hypot(gx, gy)) / 255.0)
        k_t1 = time.perf_counter_ns()

        # 3. Tensor preparation
        tp_t0 = time.perf_counter_ns()
        t_crop = torch.from_numpy(c_crop).float() / 255.0
        t_crop = t_crop.permute(2, 0, 1).unsqueeze(0).to(dev)
        tp_t1 = time.perf_counter_ns()

        # 4. Interpolation
        ti_t0 = time.perf_counter_ns()
        t_in = F.interpolate(t_crop, size=engine.input_size, mode="bilinear", align_corners=False)
        ti_t1 = time.perf_counter_ns()

        # 5. Model forward pass
        mf_t0 = time.perf_counter_ns()
        with torch.inference_mode():
            logits, reg_bbox = engine.head(t_in)
        mf_t1 = time.perf_counter_ns()

        # 6. Post-processing & coord mapping
        pp_t0 = time.perf_counter_ns()
        probs = torch.softmax(logits, dim=-1)[0]
        cat_idx = int(torch.argmax(probs).item())
        pred_box = reg_bbox[0].tolist()
        pp_t1 = time.perf_counter_ns()

        tot_ns = time.perf_counter_ns() - t0

        t_crop_slice.append((c_t1 - c_t0) / 1e6)
        t_contrast.append((k_t1 - k_t0) / 1e6)
        t_tensor_prep.append((tp_t1 - tp_t0) / 1e6)
        t_interpolate.append((ti_t1 - ti_t0) / 1e6)
        t_forward.append((mf_t1 - mf_t0) / 1e6)
        t_postproc.append((pp_t1 - pp_t0) / 1e6)
        t_total.append(tot_ns / 1e6)

    breakdown = {
        "crop_slice_ms": {
            "mean": float(np.mean(t_crop_slice)),
            "p50": float(np.median(t_crop_slice)),
            "p95": float(np.percentile(t_crop_slice, 95)),
        },
        "contrast_calc_ms": {
            "mean": float(np.mean(t_contrast)),
            "p50": float(np.median(t_contrast)),
            "p95": float(np.percentile(t_contrast, 95)),
        },
        "tensor_prep_ms": {
            "mean": float(np.mean(t_tensor_prep)),
            "p50": float(np.median(t_tensor_prep)),
            "p95": float(np.percentile(t_tensor_prep, 95)),
        },
        "interpolate_ms": {
            "mean": float(np.mean(t_interpolate)),
            "p50": float(np.median(t_interpolate)),
            "p95": float(np.percentile(t_interpolate, 95)),
        },
        "forward_head_ms": {
            "mean": float(np.mean(t_forward)),
            "p50": float(np.median(t_forward)),
            "p95": float(np.percentile(t_forward, 95)),
        },
        "postproc_ms": {
            "mean": float(np.mean(t_postproc)),
            "p50": float(np.median(t_postproc)),
            "p95": float(np.percentile(t_postproc, 95)),
        },
        "total_ms": {
            "mean": float(np.mean(t_total)),
            "p50": float(np.median(t_total)),
            "p90": float(np.percentile(t_total, 90)),
            "p95": float(np.percentile(t_total, 95)),
            "p99": float(np.percentile(t_total, 99)),
            "min": float(np.min(t_total)),
            "max": float(np.max(t_total)),
            "std": float(np.std(t_total)),
        },
    }

    print(f"Total Inspection Latency: Mean: {breakdown['total_ms']['mean']:.3f} ms | p50: {breakdown['total_ms']['p50']:.3f} ms | p95: {breakdown['total_ms']['p95']:.3f} ms")
    print(f"  - Crop Extraction (Slice): {breakdown['crop_slice_ms']['mean']*1000:6.1f} us ({breakdown['crop_slice_ms']['mean']/breakdown['total_ms']['mean']*100:4.1f}%)")
    print(f"  - Contrast Gradient Calc:  {breakdown['contrast_calc_ms']['mean']:6.3f} ms ({breakdown['contrast_calc_ms']['mean']/breakdown['total_ms']['mean']*100:4.1f}%)")
    print(f"  - Tensor Conversion & Prep:{breakdown['tensor_prep_ms']['mean']:6.3f} ms ({breakdown['tensor_prep_ms']['mean']/breakdown['total_ms']['mean']*100:4.1f}%)")
    print(f"  - Bilinear Interpolation:  {breakdown['interpolate_ms']['mean']:6.3f} ms ({breakdown['interpolate_ms']['mean']/breakdown['total_ms']['mean']*100:4.1f}%)")
    print(f"  - Neural Net Forward Head: {breakdown['forward_head_ms']['mean']:6.3f} ms ({breakdown['forward_head_ms']['mean']/breakdown['total_ms']['mean']*100:4.1f}%)")
    print(f"  - Post-proc & Coord Map:   {breakdown['postproc_ms']['mean']*1000:6.1f} us ({breakdown['postproc_ms']['mean']/breakdown['total_ms']['mean']*100:4.1f}%)")

    # 4B. Resolution Scaling of End-to-End inspect()
    print("\n--- 4B. Screen Resolution Scaling of Full inspect(frame, centroid) ---")
    resolutions = [
        ("Pre-cropped 96x96", (96, 96)),
        ("SD 640x360", (360, 640)),
        ("HD 1280x720", (720, 1280)),
        ("FHD 1920x1080", (1080, 1920)),
        ("QHD 2560x1440", (1440, 2560)),
        ("4K UHD 3840x2160", (2160, 3840)),
    ]

    res_scaling = {}
    for name, (H, W) in resolutions:
        frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        # Warmup
        for _ in range(10):
            engine.inspect(frame, (0.5, 0.5))

        latencies = []
        n_res_trials = 200 if H <= 1080 else 100
        for _ in range(n_res_trials):
            t0 = time.perf_counter_ns()
            engine.inspect(frame, (0.5, 0.5))
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            latencies.append(dt_ms)

        res_scaling[name] = {
            "resolution": [W, H],
            "crop_shape_pixels": [
                int(min(H, int(max(24, H * engine.radius_frac)) * 2)),
                int(min(W, int(max(24, W * engine.radius_frac)) * 2)),
            ],
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
        rw = int(max(24, W * engine.radius_frac))
        rh = int(max(24, H * engine.radius_frac))
        print(
            f"{name:<18} (Crop: {rh*2:3d}x{rw*2:3d} px) | Mean: {np.mean(latencies):6.2f} ms | "
            f"p50: {np.median(latencies):6.2f} ms | p95: {np.percentile(latencies, 95):6.2f} ms | p99: {np.percentile(latencies, 99):6.2f} ms"
        )

    # 4C. Single-Thread vs Multi-Thread Impact
    print("\n--- 4C. CPU Threading Scaling (FHD 1080p) ---")
    frame_1080 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    orig_threads = torch.get_num_threads()
    thread_scaling = {}
    for th in [1, 2, 4, 8]:
        try:
            torch.set_num_threads(th)
            latencies = []
            for _ in range(100):
                t0 = time.perf_counter_ns()
                engine.inspect(frame_1080, (0.5, 0.5))
                latencies.append((time.perf_counter_ns() - t0) / 1e6)
            thread_scaling[th] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
            }
            print(f"Threads={th:2d} | Mean: {np.mean(latencies):6.2f} ms | p50: {np.median(latencies):6.2f} ms | p95: {np.percentile(latencies, 95):6.2f} ms")
        except Exception as e:
            print(f"Thread test error: {e}")
    torch.set_num_threads(orig_threads)

    return {
        "pipeline_breakdown_96x96": breakdown,
        "resolution_scaling": res_scaling,
        "thread_scaling": thread_scaling,
    }


def main():
    print("=" * 70)
    print("STARTING EMPIRICAL REALITY CHECK: FOVEAL ACUITY ENGINE & REFINE AFFORDANCE")
    print("=" * 70)

    # Initialize FovealAcuityEngine and HumanEye with loaded weights
    fovea = FovealAcuityEngine(device="cpu", load_weights=True)
    assert fovea.weights_loaded, "FovealAcuityEngine failed to load weights!"

    eye = HumanEye(source="synthetic", load_weights=True)
    assert eye.fovea_engine.weights_loaded, "HumanEye fovea_engine failed to load weights!"

    print("[Init] Models successfully loaded with pre-trained weights.")

    t_start = time.time()

    # 1. Spatial boundary localization
    sec1 = run_spatial_boundary_evaluation(fovea, num_samples_per_kind=100)
    sec1_jitter = run_gaze_jitter_sensitivity(fovea)

    # 2. Spatial resolution limits
    sec2 = run_spatial_resolution_limits(fovea)

    # 3. Contrast score gating
    sec3 = run_contrast_gating_evaluation(fovea, eye)

    # 4. Latency profiling
    sec4 = run_cpu_latency_benchmarks(fovea, warmup=30, iterations=300)

    total_time = time.time() - t_start
    print(f"\nAll benchmark sections completed in {total_time:.2f}s.")

    # Save comprehensive results to JSON
    out_dir = os.path.join("benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "empirical_fovea_reality_check.json")

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_s": total_time,
        "section_1_boundary_localization": sec1,
        "section_1_jitter_sensitivity": sec1_jitter,
        "section_2_resolution_limits": sec2,
        "section_3_contrast_gating": sec3,
        "section_4_cpu_latency": sec4,
    }

    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"[Done] Detailed JSON report written to {out_path}")


if __name__ == "__main__":
    main()
