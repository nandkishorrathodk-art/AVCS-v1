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
import torch.nn.functional as F

from avcs.human_eye import HumanEye
from avcs.goal_grounder import encode_queries

print("==================================================================")
print("TESTING 2-STAGE HIERARCHICAL FOVEAL REFINEMENT ON HELD-OUT FRAMES")
print("==================================================================")

manifest_path = Path("datasets/ui_grounding/manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

test_frames = [f for f in manifest.get("frames", []) if f.get("split") == "test_held_out"]
print(f"Total held-out test frames: {len(test_frames)}")

eye = HumanEye(load_weights=True)
grounder = eye.grounder
grounder.eval()

errors_coarse = []
errors_2stage = []
coarse_hits_15px = 0
coarse_hits_5px = 0
stage2_hits_15px = 0
stage2_hits_5px = 0
stage2_hits_2px = 0
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

        # 1. Coarse Grounding
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

        # 2. Query-Conditioned Foveal Crop Grounding
        # Crop 96x96 or 64x64 patch centered at (px_cx, px_cy) from un-downsampled frame
        crop_sz = 96
        half = crop_sz // 2
        x0 = max(0, min(W - crop_sz, int(px_cx) - half))
        y0 = max(0, min(H - crop_sz, int(px_cy) - half))
        x1 = x0 + crop_sz
        y1 = y0 + crop_sz

        crop = frame[y0:y1, x0:x1]
        if crop.shape[0] == crop_sz and crop.shape[1] == crop_sz:
            crop_t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            q_tokens = encode_queries([query])
            with torch.inference_mode():
                # Pass crop into grounder
                crop_in = crop_t.unsqueeze(0)
                crop_res = grounder.predict_point(crop_in, query)
                local_pt = crop_res["point"]  # [norm_x, norm_y] in [0, 1] of crop
                local_conf = crop_res["confidence"]

            if local_conf >= 0.35:
                ref_x = x0 + local_pt[0] * crop_sz
                ref_y = y0 + local_pt[1] * crop_sz
            else:
                ref_x = px_cx
                ref_y = px_cy
        else:
            ref_x = px_cx
            ref_y = px_cy

        err_stage2 = np.sqrt((ref_x - gt_x) ** 2 + (ref_y - gt_y) ** 2)
        errors_2stage.append(err_stage2)
        if err_stage2 <= 15.0:
            stage2_hits_15px += 1
        if err_stage2 <= 5.0:
            stage2_hits_5px += 1
        if err_stage2 <= 2.0:
            stage2_hits_2px += 1

        total_elements += 1

print(f"\nEvaluated {total_elements} held-out test targets across {len(test_frames)} screens.")

print("\n--- 1. Coarse Neural Grounding ---")
print(f"  Accuracy (<=15px)       : {coarse_hits_15px / total_elements * 100:.2f}% ({coarse_hits_15px}/{total_elements})")
print(f"  Precision (<=5px)       : {coarse_hits_5px / total_elements * 100:.2f}% ({coarse_hits_5px}/{total_elements})")
print(f"  Median Centroid Error   : {np.median(errors_coarse):.2f} px")
print(f"  Mean Centroid Error     : {np.mean(errors_coarse):.2f} px")
print(f"  P95 Centroid Error      : {np.percentile(errors_coarse, 95):.2f} px")

print("\n--- 2. Two-Stage Query-Conditioned Foveal Grounding ---")
print(f"  Accuracy (<=15px)       : {stage2_hits_15px / total_elements * 100:.2f}% ({stage2_hits_15px}/{total_elements})")
print(f"  Precision (<=5px)       : {stage2_hits_5px / total_elements * 100:.2f}% ({stage2_hits_5px}/{total_elements})")
print(f"  High-Precision (<=2px)  : {stage2_hits_2px / total_elements * 100:.2f}% ({stage2_hits_2px}/{total_elements})")
print(f"  Median Centroid Error   : {np.median(errors_2stage):.2f} px")
print(f"  Mean Centroid Error     : {np.mean(errors_2stage):.2f} px")
print(f"  P95 Centroid Error      : {np.percentile(errors_2stage, 95):.2f} px")
print("==================================================================")

