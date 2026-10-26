"""Load a trained one model and ask it about a screen.

This is the whole point of the architecture in one file: load one checkpoint,
run one forward pass, and read every answer -- scene category, what elements are
on screen, what they are, what the scene affords, and where a spoken query
points -- out of the same latent. No detector is called, no rule is evaluated.

    python3 -m avcs.system1.demo --checkpoint runs/weights/system1_one_model.pt \
        --frame datasets/cleanroom_system1_v1/frames/test_heldout/test_heldout_00000.png \
        --query "click Stop button"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from avcs.system1.model import (
    ACTION_NAMES,
    CATEGORY_NAMES,
    ELEMENT_KINDS,
    WORD_NAMES,
    OneModel,
    OneModelConfig,
)


def load(checkpoint: str, device: str = "cpu") -> tuple[OneModel, dict]:
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = OneModelConfig(**blob["config"])
    model = OneModel(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def load_frame(path: str, size: int = 128) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def describe(model: OneModel, image: torch.Tensor, query: str) -> dict:
    out = model(image, [query])
    g = model.config.grid

    obj = torch.sigmoid(out.objectness_logits[0])
    cells = (obj > 0.5).nonzero(as_tuple=False).flatten()
    kinds = out.kind_logits[0].argmax(-1)
    words = out.word_logits[0].argmax(-1)

    elements = []
    for c in cells.tolist():
        cy, cx = divmod(c, g)
        elements.append(
            {
                "kind": ELEMENT_KINDS[int(kinds[c])],
                "word": WORD_NAMES[int(words[c])],
                "cell": (cy, cx),
                "point_norm": (round((cx + 0.5) / g, 3), round((cy + 0.5) / g, 3)),
                "objectness": round(float(obj[c]), 3),
            }
        )

    point = out.point_norm()[0].tolist()
    presence = torch.sigmoid(out.presence_logits[0]).item()
    action = ACTION_NAMES[int(out.action_logits[0].argmax())]
    category = CATEGORY_NAMES[int(out.category_logits[0].argmax())]
    return {
        "category": category,
        "action": action,
        "elements": elements,
        "query": query,
        "query_point_norm": (round(point[0], 3), round(point[1], 3)),
        "query_presence": round(presence, 3),
        "latent_norm": round(float(out.latent.norm()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--query", default="click")
    a = ap.parse_args()

    model, blob = load(a.checkpoint)
    print(f"one model: {model.total_parameters:,} params")
    if "val_metrics" in blob:
        print("val metrics:", blob["val_metrics"])
    if "heldout_metrics" in blob:
        print("held-out metrics:", blob["heldout_metrics"])
    print()

    img = load_frame(a.frame, model.config.input_size)
    report = describe(model, img, a.query)
    print(f"frame     : {a.frame}")
    print(f"category  : {report['category']}")
    print(f"action    : {report['action']}")
    print(f"latent    : |z| = {report['latent_norm']}   (one state, all tasks)")
    print(f"elements  : {len(report['elements'])}")
    for e in report["elements"][:8]:
        print(f"    {e['kind']:<12} '{e['word']}' at {e['point_norm']}  obj={e['objectness']}")
    print(f"query     : {report['query']!r}")
    print(f"  presence: {report['query_presence']}")
    print(f"  point   : {report['query_point_norm']}")


if __name__ == "__main__":
    main()
