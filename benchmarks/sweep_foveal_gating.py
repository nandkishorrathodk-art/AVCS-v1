"""Sweeps gating mechanisms for Two-Stage Foveal Grounding.

Runs 1 inference pass across test_held_out, caches (c_res, f_res, gt),
then sweeps gating parameters (offset threshold, coarse confidence threshold, element kinds).
"""
import json
import math
from pathlib import Path
import sys

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from PIL import Image
import numpy as np
import torch
from avcs.perception.goal_grounder import load_grounding_checkpoint
from avcs.perception.foveal_grounder import load_foveal_checkpoint

def main():
    coarse = load_grounding_checkpoint("runs/weights/candidates/candidate_m3_highres_v2.pt")
    foveal = load_foveal_checkpoint("runs/weights/candidates/candidate_m3_foveal_v1.pt")

    with open("datasets/ui_grounding/manifest.json") as f:
        mf = json.load(f)
    test_frames = [fr for fr in mf["frames"] if fr.get("split") == "test_held_out"]

    cached = []
    print("Caching 316 test predictions...")
    for fr in test_frames:
        img_p = Path("datasets/ui_grounding") / fr["file_path"]
        pil = Image.open(img_p).convert("RGB")
        raw_np = np.array(pil)
        pil_256 = pil.resize((256, 256), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        for pt in fr["interactive_points"]:
            q = pt["target_query"]
            gt_x, gt_y = float(pt["x"]), float(pt["y"])
            gt_box = pt.get("bbox", [gt_y - 0.04, gt_x - 0.04, gt_y + 0.04, gt_x + 0.04])
            kind = pt.get("element_kind", "unknown")

            c_res = coarse.predict_point(img_t, q)
            f_res = foveal.refine_prediction(raw_np, q, c_res["point"], crop_size=96, confidence_threshold=0.45)

            err_c = math.hypot((c_res["point"][0] - gt_x) * 255, (c_res["point"][1] - gt_y) * 255)
            err_f = math.hypot((f_res.refined_coord_norm[0] - gt_x) * 255, (f_res.refined_coord_norm[1] - gt_y) * 255)

            cached.append({
                "q": q,
                "kind": kind,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "gt_box": gt_box,
                "c_point": c_res["point"],
                "c_conf": c_res["confidence"],
                "err_c": err_c,
                "f_point": f_res.refined_coord_norm,
                "f_conf": f_res.foveal_confidence,
                "err_f": err_f,
                "offset_px": f_res.offset_px,
            })

    total = len(cached)
    coarse_errors = [c["err_c"] for c in cached]
    print(f"\n--- Coarse Candidate V2 Alone ---")
    print(f"Top-1: {sum(1 for c in cached if ((c['gt_box'][1] <= c['c_point'][0] <= c['gt_box'][3] and c['gt_box'][0] <= c['c_point'][1] <= c['gt_box'][2]) or c['err_c'] <= 15.0) and c['c_conf'] >= 0.55)/total*100:.2f}%")
    print(f"Median Error: {np.median(coarse_errors):.2f} px")
    print(f"P95 Error   : {np.percentile(coarse_errors, 95):.2f} px")
    print(f"<= 3.0 px   : {sum(1 for e in coarse_errors if e <= 3.0)/total*100:.1f}%")

    print("\n--- Sweeping Max Allowed Foveal Offset ---")
    for max_offset in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 25.0, 96.0]:
        errors = []
        hits = 0
        in_box = 0
        samples_le_3 = 0
        improved_tail = 0
        total_tail = 0

        for c in cached:
            # Gating policy: only refine if offset is within bounds
            if c["offset_px"] <= max_offset and c["f_conf"] >= 0.45:
                pred_pt = c["f_point"]
                err = c["err_f"]
                conf = max(c["c_conf"], c["f_conf"])
            else:
                pred_pt = c["c_point"]
                err = c["err_c"]
                conf = c["c_conf"]

            errors.append(err)
            if err <= 3.0:
                samples_le_3 += 1
            b = c["gt_box"]
            is_in_box = (b[1] <= pred_pt[0] <= b[3]) and (b[0] <= pred_pt[1] <= b[2])
            if is_in_box:
                in_box += 1
            if (is_in_box or err <= 15.0) and conf >= 0.55:
                hits += 1
            if c["err_c"] > 3.0:
                total_tail += 1
                if err < c["err_c"]:
                    improved_tail += 1

        med = np.median(errors)
        p95 = np.percentile(errors, 95)
        top1 = hits / total * 100.0
        pct_le3 = samples_le_3 / total * 100.0
        print(f"Max Offset: {max_offset:4.1f}px | Top-1: {top1:5.2f}% | Med: {med:4.2f}px | P95: {p95:5.2f}px | <=3px: {pct_le3:4.1f}% | Tail improved: {improved_tail}/{total_tail}")

    print("\n--- Selective Gating: Only Refine Input Fields & Tabs ---")
    for target_kinds in [["input_field"], ["input_field", "tab"], ["button"]]:
        errors = []
        hits = 0
        samples_le_3 = 0
        for c in cached:
            if c["kind"] in target_kinds and c["offset_px"] <= 6.0 and c["f_conf"] >= 0.45:
                err = c["err_f"]
                conf = max(c["c_conf"], c["f_conf"])
                b = c["gt_box"]
                is_in_box = (b[1] <= c["f_point"][0] <= b[3]) and (b[0] <= c["f_point"][1] <= b[2])
            else:
                err = c["err_c"]
                conf = c["c_conf"]
                b = c["gt_box"]
                is_in_box = (b[1] <= c["c_point"][0] <= b[3]) and (b[0] <= c["c_point"][1] <= b[2])

            errors.append(err)
            if err <= 3.0:
                samples_le_3 += 1
            if (is_in_box or err <= 15.0) and conf >= 0.55:
                hits += 1

        med = np.median(errors)
        p95 = np.percentile(errors, 95)
        top1 = hits / total * 100.0
        print(f"Kinds {target_kinds}: Top-1: {top1:5.2f}% | Med: {med:4.2f}px | P95: {p95:5.2f}px | <=3px: {samples_le_3/total*100:4.1f}%")

if __name__ == "__main__":
    main()

