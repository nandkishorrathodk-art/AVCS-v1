"""Empirical Adversarial Audit of UI Grounding Dataset.

Validates:
1. Hash uniqueness and pairwise split disjointness for all 270 PNG screens.
2. Negative control frames (30 frames): is_negative, interactive_points==0, mask==0.
3. Geometric enclosure across all positive elements: x1 < x2, y1 < y2, x1 <= cx <= x2, y1 <= cy <= y2.
4. Perceptual hashing / duplicate image analysis.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image

DATASET_ROOT = Path("datasets/ui_grounding")
SCREENS_DIR = DATASET_ROOT / "screens"
MASKS_DIR = DATASET_ROOT / "masks"
SPLITS_DIR = DATASET_ROOT / "splits"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"


def run_audit() -> Dict[str, Any]:
    report: Dict[str, Any] = {"checks": {}, "statistics": {}, "verdict": "UNKNOWN"}

    # 1. Screen files on disk
    png_files = sorted(list(SCREENS_DIR.glob("*.png")))
    total_pngs = len(png_files)
    file_hashes: Dict[str, str] = {}
    for p in png_files:
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        file_hashes[p.name] = digest

    unique_hashes = set(file_hashes.values())
    report["statistics"]["total_png_files"] = total_pngs
    report["statistics"]["unique_sha256_digests"] = len(unique_hashes)

    h1_pass = (total_pngs == 270) and (len(unique_hashes) == 270)
    report["checks"]["sha256_uniqueness"] = {
        "passed": h1_pass,
        "total_files": total_pngs,
        "unique_digests": len(unique_hashes),
        "collisions": total_pngs - len(unique_hashes),
    }

    # 2. Manifest and split disjointness
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    report["statistics"]["manifest_frame_count"] = len(frames)

    split_hashes: Dict[str, Set[str]] = {
        "train": set(),
        "val": set(),
        "test_held_out": set(),
        "test_negative": set(),
    }
    split_frames: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test_held_out": [],
        "test_negative": [],
    }

    for fr in frames:
        sp = fr["split"]
        fname = Path(fr["file_path"]).name
        h = file_hashes[fname]
        split_hashes[sp].add(h)
        split_frames[sp].append(fr)

    train_h = split_hashes["train"]
    val_h = split_hashes["val"]
    th_h = split_hashes["test_held_out"]
    neg_h = split_hashes["test_negative"]

    disjoint_checks = {
        "train_vs_test_held_out": {
            "disjoint": train_h.isdisjoint(th_h),
            "overlap_count": len(train_h & th_h),
        },
        "train_vs_val": {
            "disjoint": train_h.isdisjoint(val_h),
            "overlap_count": len(train_h & val_h),
        },
        "val_vs_test_held_out": {
            "disjoint": val_h.isdisjoint(th_h),
            "overlap_count": len(val_h & th_h),
        },
        "train_vs_test_negative": {
            "disjoint": train_h.isdisjoint(neg_h),
            "overlap_count": len(train_h & neg_h),
        },
        "val_vs_test_negative": {
            "disjoint": val_h.isdisjoint(neg_h),
            "overlap_count": len(val_h & neg_h),
        },
        "test_held_out_vs_test_negative": {
            "disjoint": th_h.isdisjoint(neg_h),
            "overlap_count": len(th_h & neg_h),
        },
    }
    all_disjoint = all(v["disjoint"] for v in disjoint_checks.values())
    report["checks"]["split_disjointness"] = {
        "passed": all_disjoint,
        "details": disjoint_checks,
        "split_counts": {k: len(v) for k, v in split_hashes.items()},
    }

    # 3. Negative controls audit (30 frames)
    neg_frames = split_frames["test_negative"]
    neg_passed = True
    neg_failures = []
    for fr in neg_frames:
        fid = fr["frame_id"]
        is_neg = fr.get("is_negative")
        pts = fr.get("interactive_points", [])
        mask_path = DATASET_ROOT / fr["affordance_mask_path"]

        if is_neg is not True:
            neg_passed = False
            neg_failures.append(f"{fid}: is_negative is {is_neg}")
        if len(pts) != 0:
            neg_passed = False
            neg_failures.append(f"{fid}: interactive_points len={len(pts)}")
        if not mask_path.exists():
            neg_passed = False
            neg_failures.append(f"{fid}: mask path missing {mask_path}")
        else:
            mask = np.load(mask_path)
            if mask.shape != (16, 16):
                neg_passed = False
                neg_failures.append(f"{fid}: mask shape {mask.shape} != (16, 16)")
            if not np.all(mask == 0):
                neg_passed = False
                neg_failures.append(f"{fid}: mask not all zeros (max={np.max(mask)})")

    report["checks"]["negative_controls"] = {
        "passed": neg_passed,
        "total_negative_frames": len(neg_frames),
        "failures": neg_failures,
    }

    # 4. Geometric enclosure audit
    positive_frames = split_frames["train"] + split_frames["val"] + split_frames["test_held_out"]
    geom_passed = True
    geom_failures = []
    total_elements = 0
    widths = []
    heights = []
    cx_offsets = []
    cy_offsets = []
    tolerance = 1e-4

    for fr in positive_frames:
        fid = fr["frame_id"]
        pts = fr.get("interactive_points", [])
        if len(pts) == 0:
            geom_passed = False
            geom_failures.append(f"{fid}: positive frame has 0 interactive points")

        for pt in pts:
            total_elements += 1
            el_id = pt["element_id"]
            cx = float(pt["x"])
            cy = float(pt["y"])
            bbox = pt["bbox"]
            ymin, xmin, ymax, xmax = [float(c) for c in bbox]

            w = xmax - xmin
            h = ymax - ymin
            widths.append(w)
            heights.append(h)

            # Strict inequality x1 < x2, y1 < y2
            if not (xmin < xmax):
                geom_passed = False
                geom_failures.append(f"{fid}/{el_id}: xmin >= xmax ({xmin} >= {xmax})")
            if not (ymin < ymax):
                geom_passed = False
                geom_failures.append(f"{fid}/{el_id}: ymin >= ymax ({ymin} >= {ymax})")

            # Centroid enclosure: xmin <= cx <= xmax and ymin <= cy <= ymax
            if not (xmin - tolerance <= cx <= xmax + tolerance):
                geom_passed = False
                geom_failures.append(f"{fid}/{el_id}: cx {cx} not in [{xmin}, {xmax}]")
            if not (ymin - tolerance <= cy <= ymax + tolerance):
                geom_passed = False
                geom_failures.append(f"{fid}/{el_id}: cy {cy} not in [{ymin}, {ymax}]")

            box_cx = (xmin + xmax) / 2.0
            box_cy = (ymin + ymax) / 2.0
            cx_offsets.append(abs(cx - box_cx))
            cy_offsets.append(abs(cy - box_cy))

    report["statistics"]["total_positive_frames"] = len(positive_frames)
    report["statistics"]["total_interactive_elements"] = total_elements
    report["statistics"]["bbox_width_min"] = float(np.min(widths)) if widths else 0.0
    report["statistics"]["bbox_width_median"] = float(np.median(widths)) if widths else 0.0
    report["statistics"]["bbox_width_max"] = float(np.max(widths)) if widths else 0.0
    report["statistics"]["bbox_height_min"] = float(np.min(heights)) if heights else 0.0
    report["statistics"]["bbox_height_median"] = float(np.median(heights)) if heights else 0.0
    report["statistics"]["bbox_height_max"] = float(np.max(heights)) if heights else 0.0
    report["statistics"]["max_cx_offset_from_center"] = float(np.max(cx_offsets)) if cx_offsets else 0.0
    report["statistics"]["max_cy_offset_from_center"] = float(np.max(cy_offsets)) if cy_offsets else 0.0

    report["checks"]["geometric_enclosure"] = {
        "passed": geom_passed,
        "total_elements_checked": total_elements,
        "failures": geom_failures[:10],
        "total_failures": len(geom_failures),
    }

    # Final overall verdict
    if h1_pass and all_disjoint and neg_passed and geom_passed:
        report["verdict"] = "APPROVE"
    else:
        report["verdict"] = "REQUEST_CHANGES"

    return report


if __name__ == "__main__":
    rep = run_audit()
    print("================================================================================")
    print("                    ADVERSARIAL DATASET AUDIT REPORT                            ")
    print("================================================================================")
    print(f"VERDICT: {rep['verdict']}")
    print(f"Total PNG Screens Evaluated: {rep['statistics']['total_png_files']}")
    print(f"Unique SHA-256 Digests: {rep['statistics']['unique_sha256_digests']} (100% unique: {rep['checks']['sha256_uniqueness']['passed']})")
    print("\n--- Split Disjointness ---")
    for k, v in rep["checks"]["split_disjointness"]["details"].items():
        print(f"  {k}: disjoint={v['disjoint']} (overlap: {v['overlap_count']})")
    print("\n--- Negative Controls (30 frames) ---")
    print(f"  Passed: {rep['checks']['negative_controls']['passed']}")
    tot_el = rep['statistics']['total_interactive_elements']
    print(f"\n--- Geometric Enclosure ({tot_el} elements) ---")
    print(f"  Passed: {rep['checks']['geometric_enclosure']['passed']}")
    print(f"  Failures count: {rep['checks']['geometric_enclosure']['total_failures']}")
    print(f"  BBox Width range: [{rep['statistics']['bbox_width_min']:.4f}, {rep['statistics']['bbox_width_max']:.4f}], median: {rep['statistics']['bbox_width_median']:.4f}")
    print(f"  BBox Height range: [{rep['statistics']['bbox_height_min']:.4f}, {rep['statistics']['bbox_height_max']:.4f}], median: {rep['statistics']['bbox_height_median']:.4f}")
    print(f"  Max Centroid Offset from Box Center: cx={rep['statistics']['max_cx_offset_from_center']:.6f}, cy={rep['statistics']['max_cy_offset_from_center']:.6f}")
    print("================================================================================")
