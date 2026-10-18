import json
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

    improved = []
    degraded = []
    unchanged = []

    for fr in test_frames:
        img_p = Path("datasets/ui_grounding") / fr["file_path"]
        pil = Image.open(img_p).convert("RGB")
        raw_np = np.array(pil)
        pil_256 = pil.resize((256, 256), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(pil_256).transpose(2, 0, 1)).float().div_(255.0)

        for pt in fr["interactive_points"]:
            q = pt["target_query"]
            c_res = coarse.predict_point(img_t, q)
            c_norm = c_res["point"]
            gt_x, gt_y = float(pt["x"]), float(pt["y"])
            err_c = np.hypot((c_norm[0] - gt_x) * 255, (c_norm[1] - gt_y) * 255)
            f_res = foveal.refine_prediction(raw_np, q, c_norm, crop_size=96, confidence_threshold=0.45)
            ref_norm = f_res.refined_coord_norm
            err_ref = np.hypot((ref_norm[0] - gt_x) * 255, (ref_norm[1] - gt_y) * 255)
            delta = err_ref - err_c
            rec = {
                "q": q,
                "cat": fr.get("category"),
                "kind": pt.get("element_kind"),
                "err_c": err_c,
                "err_ref": err_ref,
                "delta": delta,
                "offset": f_res.offset_px,
                "c_conf": c_res["confidence"],
                "f_conf": f_res.foveal_confidence,
            }
            if delta < -0.5:
                improved.append(rec)
            elif delta > 0.5:
                degraded.append(rec)
            else:
                unchanged.append(rec)

    total = len(improved) + len(degraded) + len(unchanged)
    print(f"Total: {total} | Improved: {len(improved)} ({len(improved)/total*100:.1f}%) | Degraded: {len(degraded)} ({len(degraded)/total*100:.1f}%) | Unchanged: {len(unchanged)}")

    print("\n=== Top 8 Improved Cases ===")
    for r in sorted(improved, key=lambda x: x["delta"])[:8]:
        print(f"{r['q'][:25]:<25} | {r['cat']:<16} | Coarse: {r['err_c']:.2f}px -> Refined: {r['err_ref']:.2f}px (gain: {-r['delta']:.2f}px, offset: {r['offset']:.2f}px)")

    print("\n=== Top 8 Degraded Cases ===")
    for r in sorted(degraded, key=lambda x: -x["delta"])[:8]:
        print(f"{r['q'][:25]:<25} | {r['cat']:<16} | Coarse: {r['err_c']:.2f}px -> Refined: {r['err_ref']:.2f}px (loss: {r['delta']:.2f}px, offset: {r['offset']:.2f}px)")

    deg_coarse = [r["err_c"] for r in degraded]
    le3 = sum(1 for e in deg_coarse if e <= 3.0)
    print(f"\nDegraded cases where coarse error was ALREADY <= 3.0px: {le3}/{len(degraded)} ({le3/len(degraded)*100:.1f}%)")
    print(f"Degraded cases where coarse error was ALREADY <= 2.0px: {sum(1 for e in deg_coarse if e <= 2.0)}/{len(degraded)}")

if __name__ == "__main__":
    main()
