"""Evaluate Query-Conditioned Foveal Refinement on test_held_out.

Applies FovealAcuityEngine un-downsampled 1:1 patch extraction and sub-pixel edge alignment
to evaluate P95 centroid error reduction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
from PIL import Image

from avcs.perception.goal_grounder import load_grounding_checkpoint
from avcs.perception.grounding import FovealAcuityEngine


def evaluate_foveal(
    checkpoint_path: str = "runs/weights/candidates/candidate_m3_highres_v2.pt",
    manifest_path: str = "datasets/ui_grounding/manifest.json",
):
    print(f"Loading checkpoint {checkpoint_path}...")
    model = load_grounding_checkpoint(checkpoint_path, device=torch.device("cpu"))
    manifest_p = Path(manifest_path)
    data_dir = manifest_p.parent

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_frames = [fr for fr in manifest["frames"] if fr.get("split") == "test_held_out"]
    fovea = FovealAcuityEngine(device="cpu", load_weights=False, fovea_patch_size=128)

    coarse_errors = []
    refined_errors = []
    improved_count = 0

    for fr in test_frames:
        img_path = data_dir / fr["file_path"]
        pil_raw = Image.open(img_path).convert("RGB")
        raw_w, raw_h = pil_raw.size
        raw_np = np.array(pil_raw)

        # 256x256 input for coarse model
        pil_256 = pil_raw.resize((256, 256), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        for pt in fr.get("interactive_points", []):
            query = pt.get("target_query", "click button")
            gt_x = float(pt["x"])
            gt_y = float(pt["y"])
            kind = pt.get("element_kind", "unknown")
            gt_box = pt.get("bbox", [gt_y - 0.04, gt_x - 0.04, gt_y + 0.04, gt_x + 0.04])

            # 1. Coarse prediction
            res = model.predict_point(img_t, query)
            c_x, c_y = res["point"]
            dx_c = (c_x - gt_x) * raw_w
            dy_c = (c_y - gt_y) * raw_h
            err_c = math.hypot(dx_c, dy_c)
            coarse_errors.append(err_c)

            # 2. Foveal refinement: un-downsampled 1:1 patch around coarse point
            crop, norm_crop_box = fovea.crop_fovea(raw_np, (c_x, c_y))
            # norm_crop_box: (x1/W, y1/H, x2/W, y2/H)
            cb_x1, cb_y1, cb_x2, cb_y2 = norm_crop_box
            crop_w = cb_x2 - cb_x1
            crop_h = cb_y2 - cb_y1

            # Relative coarse point in crop [0, 1]
            rel_cx = (c_x - cb_x1) / max(1e-6, crop_w)
            rel_cy = (c_y - cb_y1) / max(1e-6, crop_h)
            pred_box_rel = (
                max(0.0, rel_cx - 0.15),
                max(0.0, rel_cy - 0.15),
                min(1.0, rel_cx + 0.15),
                min(1.0, rel_cy + 0.15),
            )

            # Refine subpixel boundaries on native-resolution crop
            rx1, ry1, rx2, ry2 = fovea.refine_subpixel_boundaries(crop, pred_box_rel, element_kind=kind)

            # If element has clear edges (width and height > 4px in crop), re-center to element center
            crop_px_w = (rx2 - rx1) * crop.shape[1]
            crop_px_h = (ry2 - ry1) * crop.shape[0]

            if crop_px_w >= 8.0 and crop_px_h >= 6.0:
                rel_ref_x = (rx1 + rx2) / 2.0
                rel_ref_y = (ry1 + ry2) / 2.0
                ref_x = cb_x1 + rel_ref_x * crop_w
                ref_y = cb_y1 + rel_ref_y * crop_h
            else:
                ref_x = c_x
                ref_y = c_y

            dx_r = (ref_x - gt_x) * raw_w
            dy_r = (ref_y - gt_y) * raw_h
            err_r = math.hypot(dx_r, dy_r)
            refined_errors.append(err_r)

            if err_r < err_c:
                improved_count += 1

    print("\n=== FOVEAL ACUITY REFINEMENT RESULTS (316 HELD-OUT SAMPLES) ===")
    print(f"Coarse Median Error:   {np.median(coarse_errors):.2f} px")
    print(f"Refined Median Error:  {np.median(refined_errors):.2f} px")
    print(f"Coarse Mean Error:     {np.mean(coarse_errors):.2f} px")
    print(f"Refined Mean Error:    {np.mean(refined_errors):.2f} px")
    print(f"Coarse P90 Error:      {np.percentile(coarse_errors, 90):.2f} px")
    print(f"Refined P90 Error:     {np.percentile(refined_errors, 90):.2f} px")
    print(f"Coarse P95 Error:      {np.percentile(coarse_errors, 95):.2f} px")
    print(f"Refined P95 Error:     {np.percentile(refined_errors, 95):.2f} px")
    print(f"Improved Samples:      {improved_count}/{len(coarse_errors)} ({improved_count/len(coarse_errors)*100:.1f}%)")


if __name__ == "__main__":
    evaluate_foveal()

