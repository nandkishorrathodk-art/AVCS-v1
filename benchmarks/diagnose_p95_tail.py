"""Diagnose and rank the P95 localization error tail on test_held_out.

Clusters and categorizes the largest pixel errors to guide foveal refinement and model tuning.
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


def diagnose_tail(
    checkpoint_path: str = "runs/weights/candidates/candidate_m3_highres_v2.pt",
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    top_k: int = 30,
) -> List[Dict[str, Any]]:
    print(f"Loading checkpoint {checkpoint_path}...")
    model = load_grounding_checkpoint(checkpoint_path, device=torch.device("cpu"))
    manifest_p = Path(manifest_path)
    data_dir = manifest_p.parent

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_frames = [fr for fr in manifest["frames"] if fr.get("split") == "test_held_out"]
    records: List[Dict[str, Any]] = []

    for fr in test_frames:
        img_path = data_dir / fr["file_path"]
        with Image.open(img_path) as pil_img:
            arr = np.array(pil_img.resize((256, 256), Image.BILINEAR).convert("RGB"), dtype=np.uint8)
        img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)

        w = fr.get("width", 256)
        h = fr.get("height", 256)
        cat = fr.get("category", "unknown")

        for pt in fr.get("interactive_points", []):
            query = pt.get("target_query", "click button")
            gt_x = float(pt["x"])
            gt_y = float(pt["y"])
            kind = pt.get("element_kind", "unknown")
            gt_box = pt.get("bbox", [gt_y - 0.04, gt_x - 0.04, gt_y + 0.04, gt_x + 0.04])

            res = model.predict_point(img_t, query)
            pred_x, pred_y = res["point"]
            conf = res["confidence"]

            dx = (pred_x - gt_x) * w
            dy = (pred_y - gt_y) * h
            err_px = float(math.hypot(dx, dy))

            ymin, xmin, ymax, xmax = gt_box
            in_box = (xmin <= pred_x <= xmax) and (ymin <= pred_y <= ymax)

            # Failure taxonomy classification
            failure_category = "accurate (<3px)"
            if err_px > 3.0:
                if in_box:
                    failure_category = "in_box_off_center"
                elif kind == "input_field" or "input" in query.lower() or "search" in query.lower():
                    failure_category = "input_field_anchor"
                elif kind in ("tab", "menu_item") or "tab" in query.lower():
                    failure_category = "adjacent_tab_confusion"
                elif kind == "icon" or "icon" in query.lower():
                    failure_category = "icon_spatial_ambiguity"
                elif "row" in query.lower() or "line" in query.lower() or kind == "text_label":
                    failure_category = "dense_text_row"
                else:
                    failure_category = "semantic_mislocalization"

            records.append({
                "query": query,
                "category": cat,
                "kind": kind,
                "file_path": fr["file_path"],
                "gt_xy": (round(gt_x, 4), round(gt_y, 4)),
                "pred_xy": (round(pred_x, 4), round(pred_y, 4)),
                "dx_px": round(dx, 2),
                "dy_px": round(dy, 2),
                "err_px": round(err_px, 2),
                "in_box": in_box,
                "confidence": round(conf, 3),
                "failure_category": failure_category,
            })

    records.sort(key=lambda r: r["err_px"], reverse=True)

    print(f"\nTotal test samples: {len(records)}")
    all_errors = [r["err_px"] for r in records]
    hits = sum(1 for r in records if (r["in_box"] or r["err_px"] <= 5.0) and r["confidence"] >= 0.35)
    print(f"Top-1 Hits: {hits}/{len(records)} ({hits/len(records)*100:.2f}%)")
    print(f"Median error: {np.median(all_errors):.2f} px")
    print(f"Mean error:   {np.mean(all_errors):.2f} px")
    print(f"P90 error:    {np.percentile(all_errors, 90):.2f} px")
    print(f"P95 error:    {np.percentile(all_errors, 95):.2f} px")
    print(f"P99 error:    {np.percentile(all_errors, 99):.2f} px")
    print(f"Max error:    {np.max(all_errors):.2f} px")

    # Cluster summary
    clusters: Dict[str, List[float]] = {}
    for r in records:
        fc = r["failure_category"]
        clusters.setdefault(fc, []).append(r["err_px"])

    print("\n--- Failure Cluster Breakdown ---")
    for fc, errs in sorted(clusters.items(), key=lambda x: -len(x[1])):
        print(f"  {fc:25s}: count={len(errs):3d} ({len(errs)/len(records)*100:4.1f}%), med={np.median(errs):.2f} px, max={np.max(errs):.2f} px")

    print(f"\n--- Top {top_k} Worst Errors ---")
    for i, r in enumerate(records[:top_k], 1):
        print(f"#{i:2d}: {r['err_px']:6.2f} px | in_box={str(r['in_box']):5s} | conf={r['confidence']:.2f} | [{r['failure_category']:22s}] | '{r['query']}' ({r['kind']} in {r['category']})")

    return records


if __name__ == "__main__":
    cp = sys.argv[1] if len(sys.argv) > 1 else "runs/weights/candidates/candidate_m3_highres_v2.pt"
    diagnose_tail(checkpoint_path=cp)
