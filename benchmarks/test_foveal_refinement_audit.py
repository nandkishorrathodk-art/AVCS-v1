import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import numpy as np
from PIL import Image
import torch

from avcs.human_eye import HumanEye

print("==================================================================")
print("PRIORITY 2 & 3: AUDITING COARSE VS FOVEAL REFINED GROUNDING")
print("==================================================================")

manifest_path = Path("datasets/ui_grounding/manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

test_frames = [f for f in manifest.get("frames", []) if f.get("split") == "test_held_out"]
print(f"Total held-out test frames: {len(test_frames)}")

eye = HumanEye(load_weights=True)

errors_coarse = []
errors_foveal = []
coarse_hits_15px = 0
foveal_hits_15px = 0
coarse_hits_5px = 0
foveal_hits_5px = 0
total_elements = 0

for frame_info in test_frames:
    img_path = Path("datasets/ui_grounding") / frame_info["file_path"]
    if not img_path.exists():
        continue
    img = Image.open(img_path).convert("RGB")
    frame = np.array(img)
    H, W = frame.shape[:2]

    for pt in frame_info.get("interactive_points", []):
        query = pt.get("target_query", "")
        if not query:
            continue
        gt_x = float(pt["x"]) * W
        gt_y = float(pt["y"]) * H

        # 1. Coarse Grounding via eye.locate
        res = eye.locate(query, frame=frame)
        pred_norm_x, pred_norm_y = res.centroid
        px_cx = pred_norm_x * W
        px_cy = pred_norm_y * H

        err_coarse = np.sqrt((px_cx - gt_x) ** 2 + (px_cy - gt_y) ** 2)
        errors_coarse.append(err_coarse)
        if err_coarse <= 15.0:
            coarse_hits_15px += 1
        if err_coarse <= 5.0:
            coarse_hits_5px += 1

        # 2. Foveal Saccade Refinement
        foveal_res = eye.fovea_engine.inspect(frame, (pred_norm_x, pred_norm_y))
        rf_x, rf_y = foveal_res.refined_centroid
        rf_px = rf_x * W
        rf_py = rf_y * H

        err_fovea = np.sqrt((rf_px - gt_x) ** 2 + (rf_py - gt_y) ** 2)
        errors_foveal.append(err_fovea)
        if err_fovea <= 15.0:
            foveal_hits_15px += 1
        if err_fovea <= 5.0:
            foveal_hits_5px += 1

        total_elements += 1

print(f"\nEvaluated {total_elements} held-out test targets across {len(test_frames)} screens.")

print("\n--- 1. Coarse Neural Grounding (No Fovea) ---")
print(f"  Accuracy (<=15px)       : {coarse_hits_15px / total_elements * 100:.2f}% ({coarse_hits_15px}/{total_elements})")
print(f"  Precision (<=5px)       : {coarse_hits_5px / total_elements * 100:.2f}% ({coarse_hits_5px}/{total_elements})")
print(f"  Median Centroid Error   : {np.median(errors_coarse):.2f} px")
print(f"  Mean Centroid Error     : {np.mean(errors_coarse):.2f} px")
print(f"  P95 Centroid Error      : {np.percentile(errors_coarse, 95):.2f} px")

print("\n--- 2. Foveal Saccade Refined Grounding (Grounder + Fovea) ---")
print(f"  Accuracy (<=15px)       : {foveal_hits_15px / total_elements * 100:.2f}% ({foveal_hits_15px}/{total_elements})")
print(f"  Precision (<=5px)       : {foveal_hits_5px / total_elements * 100:.2f}% ({foveal_hits_5px}/{total_elements})")
print(f"  Median Centroid Error   : {np.median(errors_foveal):.2f} px")
print(f"  Mean Centroid Error     : {np.mean(errors_foveal):.2f} px")
print(f"  P95 Centroid Error      : {np.percentile(errors_foveal, 95):.2f} px")
print("==================================================================")

