"""Dataset for the one model.

Turns each cleanroom annotation into every target the model needs in one go:
the frame's category, a per-cell objectness / kind / word map, the action the
scene affords, and one query with its target cell (or an absence label).

All targets come from the same annotation file, so a single example teaches all
heads simultaneously -- which is what forces the trunk to hold one shared
representation instead of letting each task grow its own features.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from avcs.system1.model import ACTION_NAMES, CATEGORY_NAMES, ELEMENT_KINDS, WORD_NAMES

_KIND_INDEX = {k: i for i, k in enumerate(ELEMENT_KINDS)}
_WORD_INDEX = {w: i for i, w in enumerate(WORD_NAMES)}
_CAT_INDEX = {c: i for i, c in enumerate(CATEGORY_NAMES)}


def _action_from_query(query: str) -> int:
    """The action the task demands, read off the query's verb.

    This corpus only issues ``click`` queries, so the action head has a single
    class to learn here. That is a property of the data, not the model; the head
    is wired to condition on the query so a corpus with more verbs can train it.
    """
    verb = query.strip().split(" ")[0].lower() if query.strip() else "none"
    return ACTION_NAMES.index(verb) if verb in ACTION_NAMES else ACTION_NAMES.index("none")


class System1OneModelDataset(Dataset):
    def __init__(
        self,
        root: str = "datasets/cleanroom_system1_v1",
        split: str = "train",
        grid: int = 16,
        image_size: int = 128,
        include_negative: bool = True,
    ) -> None:
        self.root = root
        self.grid = grid
        self.image_size = image_size
        ann_dir = os.path.join(root, "annotations", split)
        self.files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
        if not self.files:
            raise FileNotFoundError(f"no annotations under {ann_dir}")
        self.frame_dir = os.path.join(root, "frames", split)
        self.include_negative = include_negative

    def __len__(self) -> int:
        return len(self.files)

    def _frame_path(self, frame_id: str) -> str:
        return os.path.join(self.frame_dir, f"{frame_id}.png")

    def __getitem__(self, idx: int) -> dict:
        ann = json.loads(open(self.files[idx]).read())
        g = self.grid
        frame_id = ann["frame_id"]

        img = Image.open(self._frame_path(frame_id)).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        category = _CAT_INDEX.get(ann["category"], _CAT_INDEX["unknown"])

        objectness = torch.zeros(g * g)
        kind = torch.zeros(g * g, dtype=torch.long)
        word = torch.zeros(g * g, dtype=torch.long)

        for r in ann.get("regions", []):
            x0, y0, x1, y1 = r["bbox_xyxy_norm"]
            cx0 = int(np.clip(x0 * g, 0, g - 1))
            cy0 = int(np.clip(y0 * g, 0, g - 1))
            cx1 = int(np.clip(np.ceil(x1 * g), cx0 + 1, g))
            cy1 = int(np.clip(np.ceil(y1 * g), cy0 + 1, g))
            k_idx = _KIND_INDEX.get(r.get("kind", ""), 0)
            w_idx = _WORD_INDEX.get(r.get("label", "<none>"), 0)
            for cy in range(cy0, cy1):
                for cx in range(cx0, cx1):
                    cell = cy * g + cx
                    objectness[cell] = 1.0
                    kind[cell] = k_idx
                    word[cell] = w_idx

        # Pick one query. Prefer a positive one so the point term has signal,
        # but keep absence examples so the presence head learns to say "no".
        queries = ann.get("queries", [])
        positives = [q for q in queries if q.get("target_presence")]
        negatives = [q for q in queries if not q.get("target_presence")]
        rng = np.random.default_rng(idx)
        if positives and (not negatives or rng.random() < 0.7):
            q = positives[int(rng.integers(len(positives)))]
        elif negatives:
            q = negatives[int(rng.integers(len(negatives)))]
        else:
            q = {"query": "click", "target_presence": False, "target_point_norm": None}

        present = 1.0 if q.get("target_presence") else 0.0
        point_cell = 0
        if present and q.get("target_point_norm"):
            px, py = q["target_point_norm"]
            cx = int(np.clip(px * g, 0, g - 1))
            cy = int(np.clip(py * g, 0, g - 1))
            point_cell = cy * g + cx

        action = torch.tensor(_action_from_query(q["query"]))

        return {
            "image": image,
            "category": torch.tensor(category),
            "objectness": objectness,
            "kind": kind,
            "word": word,
            "action": action,
            "query": q["query"],
            "present": torch.tensor(present),
            "point_cell": torch.tensor(point_cell),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "category": torch.stack([b["category"] for b in batch]),
        "objectness": torch.stack([b["objectness"] for b in batch]),
        "kind": torch.stack([b["kind"] for b in batch]),
        "word": torch.stack([b["word"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "query": [b["query"] for b in batch],
        "present": torch.stack([b["present"] for b in batch]),
        "point_cell": torch.stack([b["point_cell"] for b in batch]),
    }
