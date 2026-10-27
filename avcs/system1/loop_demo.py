"""End-to-end loop demo: real frames, one model, capability gate, virtual click.

    python3 -m avcs.system1.loop_demo --checkpoint runs/weights/system1_one_model.pt

Runs the real perception path (no mock model) over real held-out clean-room
frames and prints each turn's evidence and outcome. Actuation is virtual, so the
"verified" column reports a real photometric delta between consecutive frames
but the click itself moves no cursor. Every gate decision is the production
kernel's, not a demo re-implementation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from avcs.control.capability import CapabilityKernel
from avcs.system1.loop import OneModelLoop, VirtualHand, run_sequence
from avcs.system1.model import OneModel, OneModelConfig


def load_model(checkpoint: str) -> OneModel:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = OneModel(OneModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def load_frame(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def simulate_response(frame: np.ndarray, point_norm: tuple[float, float], radius: float = 0.02) -> np.ndarray:
    """Return a copy of `frame` with a small UI-like change at `point_norm`.

    A stand-in for a real app reacting to the click. The change is placed at the
    *annotation's* true target, not the model's prediction, so whether the loop
    verifies depends on whether the model grounded close enough for the ripple
    window to see it -- the demo cannot pass by construction.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    cy, cx = int(point_norm[1] * (h - 1)), int(point_norm[0] * (w - 1))
    r = max(1, int(radius * min(h, w)))
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = out[y0:y1, x0:x1].astype(np.int16)
    out[y0:y1, x0:x1] = np.clip(patch + 90, 0, 255).astype(np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/weights/system1_one_model.pt")
    ap.add_argument("--split", default="test_heldout")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--root", default="datasets/cleanroom_system1_v1")
    ap.add_argument(
        "--grant",
        nargs="*",
        default=["click"],
        help="mutating ops this run is allowed to perform (default: click). "
        "Pass no values to run fully fail-closed.",
    )
    ap.add_argument("--presence-floor", type=float, default=0.5)
    ap.add_argument(
        "--simulate-response",
        action="store_true",
        help="synthesize a post-click capture with a localized change at the true "
        "target, so the verify stage has something honest to measure",
    )
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    kernel = CapabilityKernel(granted_ops=args.grant or ())
    hand = VirtualHand(1920, 1080)
    loop = OneModelLoop(model, kernel=kernel, hand=hand, presence_floor=args.presence_floor)

    ann_dir = os.path.join(args.root, "annotations", args.split)
    files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))[: args.frames]

    frames, queries, targets = [], [], []
    for fp in files:
        ann = json.loads(open(fp).read())
        q = next((q for q in ann.get("queries", []) if q.get("target_presence")), None)
        if not q:
            continue
        frames.append(load_frame(os.path.join(args.root, "frames", args.split, f"{ann['frame_id']}.png")))
        queries.append(q["query"])
        targets.append(q.get("target_point_norm"))

    post_frames = None
    if args.simulate_response:
        post_frames = [
            simulate_response(f, tuple(t)) if t else None for f, t in zip(frames, targets)
        ]

    print(f"[loop] grant={sorted(args.grant or []) or 'NONE (fail-closed)'}  "
          f"presence_floor={args.presence_floor}  frames={len(frames)}  "
          f"simulate_response={args.simulate_response}\n")
    steps = run_sequence(loop, frames, queries, post_frames=post_frames)

    n_click = n_verified = 0
    for s, tgt in zip(steps, targets):
        err = ""
        if tgt is not None:
            d = ((s.point_norm[0] - tgt[0]) ** 2 + (s.point_norm[1] - tgt[1]) ** 2) ** 0.5
            err = f"  target_err={d:.3f}"
        print(f"turn {s.turn}: query={s.query!r}")
        print(f"  category={s.category}  action={s.action}  presence={s.presence:.3f}")
        print(f"  point={tuple(round(v,3) for v in s.point_norm)}{err}")
        print(f"  gate={'ALLOW' if s.allowed else 'DENY'}  ({s.gate_reason})")
        print(f"  click={s.clicked}  delta={s.visual_delta}  verified={s.verified}")
        print(f"  -> {s.note}\n")
        if s.clicked is not None:
            n_click += 1
        if s.verified:
            n_verified += 1

    print(f"[summary] clicks={n_click}/{len(steps)}  "
          f"verified_changes={n_verified}/{n_click if n_click else 0}")


if __name__ == "__main__":
    main()
