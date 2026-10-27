"""Phase 1 scorecard: measure the loop over the whole held-out set.

Zero new model. This runs the shipped checkpoint through the production gate
over *every* query in a split and reports the numbers that decide the next
phase:

  * correct-click / false-click / miss rates (the safety-relevant pair)
  * calibration: can a presence threshold separate present from absent
  * per-layout and per-kind breakdown (where does it fail)
  * near-miss vs wild-miss (is a wrong click adjacent or elsewhere)
  * per-frame latency

    python3 -m avcs.system1.scorecard
    python3 -m avcs.system1.scorecard --out benchmarks/results/system1_scorecard.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import time

import numpy as np
import torch
from PIL import Image

from avcs.control.capability import CapabilityKernel
from avcs.system1.loop import to_tensor
from avcs.system1.model import OneModel, OneModelConfig

KIND_FROM_QUERY = {
    "input field": "input_field",
    "tab": "tab",
    "button": "button",
    "icon": "icon",
}
FLOORS = (0.3, 0.5, 0.7, 0.9)


def load_model(checkpoint: str) -> OneModel:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = OneModel(OneModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def query_kind(query: str) -> str:
    q = query.lower()
    for needle, kind in KIND_FROM_QUERY.items():
        if needle in q:
            return kind
    return "other"


def collect(root: str, split: str) -> list[dict]:
    """Every (frame, query) pair in the split, with its ground truth."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(root, "annotations", split, "*.json"))):
        ann = json.loads(open(fp).read())
        regions = ann.get("regions", [])
        for q in ann.get("queries", []):
            target_box = None
            tn = q.get("target_point_norm")
            if tn is not None:
                for r in regions:
                    x0, y0, x1, y1 = r["bbox_xyxy_norm"]
                    if x0 <= tn[0] <= x1 and y0 <= tn[1] <= y1:
                        target_box = (x0, y0, x1, y1)
                        break
            rows.append(
                {
                    "frame_id": ann["frame_id"],
                    "category": ann.get("category", "unknown"),
                    "query": q["query"],
                    "kind": query_kind(q["query"]),
                    "present": bool(q.get("target_presence")),
                    "target_norm": tuple(tn) if tn else None,
                    "target_box": target_box,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/weights/system1_one_model.pt")
    ap.add_argument("--root", default="datasets/cleanroom_system1_v1")
    ap.add_argument("--splits", nargs="*", default=["validation", "test_heldout"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    g = model.config.grid
    report: dict = {"checkpoint": args.checkpoint, "grid": g, "splits": {}}

    for split in args.splits:
        rows = collect(args.root, split)
        frame_cache: dict[str, np.ndarray] = {}

        def frame(fid: str) -> np.ndarray:
            if fid not in frame_cache:
                p = os.path.join(args.root, "frames", split, f"{fid}.png")
                frame_cache[fid] = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            return frame_cache[fid]

        # One forward pass per (frame, query); record everything needed.
        recs = []
        lat = []
        for r in rows:
            x = to_tensor(frame(r["frame_id"]), model.config.input_size).unsqueeze(0)
            with torch.no_grad():
                t0 = time.perf_counter()
                out = model(x, [r["query"]])
                lat.append((time.perf_counter() - t0) * 1000.0)
            presence = float(torch.sigmoid(out.presence_logits.squeeze(-1))[0])
            pred = int(out.point_logits.argmax(dim=-1))
            rec = dict(r)
            rec["presence"] = presence
            rec["pred_cell"] = pred
            rec["pred_norm"] = tuple(out.point_norm()[0].tolist())
            if r["target_norm"] is not None:
                px, py = r["target_norm"]
                tcx = int(np.clip(px * g, 0, g - 1))
                tcy = int(np.clip(py * g, 0, g - 1))
                rec["target_cell"] = tcy * g + tcx
                rec["cell_dist"] = max(abs(pred // g - tcy), abs(pred % g - tcx))
                rec["norm_dist"] = float(
                    ((out.point_norm()[0, 0] - px) ** 2 + (out.point_norm()[0, 1] - py) ** 2) ** 0.5
                )
            if r["target_box"] is not None:
                x0, y0, x1, y1 = r["target_box"]
                rec["in_bbox"] = bool(x0 <= rec["pred_norm"][0] <= x1 and y0 <= rec["pred_norm"][1] <= y1)
            recs.append(rec)

        present = [r for r in recs if r["present"]]
        absent = [r for r in recs if not r["present"]]

        # --- rate table at several presence floors ---------------------- #
        floor_rows = []
        for f in FLOORS:
            allowed_p = [r for r in present if r["presence"] >= f]
            correct = [r for r in allowed_p if r.get("cell_dist") == 0]
            within1 = [r for r in allowed_p if r.get("cell_dist", 99) <= 1]
            in_box = [r for r in allowed_p if r.get("in_bbox")]
            false_click = [r for r in absent if r["presence"] >= f]
            floor_rows.append(
                {
                    "floor": f,
                    "in_bbox": len(in_box),
                    "correct_click": len(correct),
                    "within1": len(within1),
                    "clicked_present": len(allowed_p),
                    "false_click": len(false_click),
                    "missed_present": len(present) - len(allowed_p),
                }
            )

        # --- calibration ------------------------------------------------ #
        pv = [r["presence"] for r in present]
        av = [r["presence"] for r in absent]
        calibration = {
            "present_n": len(pv),
            "absent_n": len(av),
            "present_median": round(statistics.median(pv), 4) if pv else None,
            "absent_median": round(statistics.median(av), 4) if av else None,
            "present_min": round(min(pv), 4) if pv else None,
            "absent_max": round(max(av), 4) if av else None,
            "separable": bool(pv and av and min(pv) > max(av)),
        }

        # --- per-layout and per-kind ------------------------------------ #
        def breakdown(key: str) -> dict:
            buckets: dict[str, dict] = collections.defaultdict(
                lambda: {"n": 0, "correct": 0, "in_box": 0, "errs": []}
            )
            for r in present:
                b = buckets[r[key]]
                b["n"] += 1
                if r.get("cell_dist") == 0:
                    b["correct"] += 1
                if r.get("in_bbox"):
                    b["in_box"] += 1
                if r.get("norm_dist") is not None:
                    b["errs"].append(r["norm_dist"])
            return {
                k: {
                    "n": v["n"],
                    "correct": v["correct"],
                    "in_box": v["in_box"],
                    "acc": round(v["in_box"] / v["n"], 3),
                    "median_err": round(statistics.median(v["errs"]), 3) if v["errs"] else None,
                }
                for k, v in sorted(buckets.items())
            }

        # --- near-miss vs wild-miss (present, gate allowed at 0.5) ------ #
        near = wild = 0
        for r in present:
            if r["presence"] < 0.5:
                continue
            d = r.get("cell_dist", 99)
            if d == 0:
                continue
            if d <= 1:
                near += 1
            else:
                wild += 1

        split_report = {
            "n_pairs": len(recs),
            "n_present": len(present),
            "n_absent": len(absent),
            "floors": floor_rows,
            "calibration": calibration,
            "per_layout": breakdown("category"),
            "per_kind": breakdown("kind"),
            "near_miss": near,
            "wild_miss": wild,
            "latency_ms_min": round(min(lat), 2),
            "latency_ms_median": round(statistics.median(lat), 2),
        }
        report["splits"][split] = split_report

        # ---------------------------- print ----------------------------- #
        print(f"\n{'='*66}\n{split}: {len(recs)} (frame, query) pairs  "
              f"[{len(present)} present, {len(absent)} absent]\n{'='*66}")
        print(f"{'floor':>6} {'in_bbox':>8} {'exact':>8} {'within1':>8} {'clicked':>8} {'falseclick':>11} {'missed':>7}")
        for fr in floor_rows:
            print(f"{fr['floor']:>6.2f} {fr['in_bbox']:>4}/{len(present):<3} "
                  f"{fr['correct_click']:>4}/{len(present):<3} {fr['within1']:>4}/{len(present):<3} "
                  f"{fr['clicked_present']:>8} {fr['false_click']:>6}/{len(absent):<4} {fr['missed_present']:>7}")
        c = calibration
        print(f"\ncalibration: present median {c['present_median']} min {c['present_min']} | "
              f"absent median {c['absent_median']} max {c['absent_max']} | "
              f"separable={c['separable']}")
        print("\nper-layout (present; acc = point inside target bbox):")
        for k, v in split_report["per_layout"].items():
            print(f"  {k:16s} n={v['n']:<4} in_box={v['in_box']:<4} acc={v['acc']:.2f} med_err={v['median_err']}")
        print("per-kind (present):")
        for k, v in split_report["per_kind"].items():
            print(f"  {k:16s} n={v['n']:<4} in_box={v['in_box']:<4} acc={v['acc']:.2f} med_err={v['median_err']}")
        print(f"\nwrong clicks: near-miss(<=1 cell)={near}  wild-miss(>1 cell)={wild}")
        print(f"latency: min {split_report['latency_ms_min']} ms  median {split_report['latency_ms_median']} ms")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
