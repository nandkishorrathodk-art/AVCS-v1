"""Hash-verified dataset reader for the isolated System 1 clean-room corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from avcs.dataset.cleanroom_v1 import ANNOTATION_SCHEMA
from avcs.dataset.cleanroom_v1.generator import (
    CANVAS_SIZE,
    SUPPORTED_SPLITS,
    _sha256,
    validate_dataset,
)
from avcs.perception.goal_grounder import encode_queries
from avcs.perception.system1_v1 import KIND_TO_IDX
from avcs.perception.visual_pilot import UI_CATEGORIES
from avcs.perception.word_recognizer import (
    CANONICAL_CROP_HEIGHT,
    CANONICAL_CROP_WIDTH,
    WORD_TO_ID,
)

LEVELS = {"p2": 64, "p3": 32, "p4": 16}


def _empty_proposals() -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, size in LEVELS.items():
        result[f"{name}_aff"] = torch.zeros((size, size), dtype=torch.long)
        result[f"{name}_box"] = torch.zeros((4, size, size), dtype=torch.float32)
        result[f"{name}_kind"] = torch.zeros((size, size), dtype=torch.long)
    return result


def _level_for(box_xyxy: list[int]) -> str:
    x1, y1, x2, y2 = box_xyxy
    longest_side = max(x2 - x1, y2 - y1)
    if longest_side < 64:
        return "p2"
    if longest_side <= 96:
        return "p3"
    return "p4"


def _proposal_targets(regions: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    targets = _empty_proposals()
    for region in regions:
        level = _level_for(region["bbox_xyxy_px"])
        size = LEVELS[level]
        x1, y1, x2, y2 = region["bbox_xyxy_norm"]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        grid_x = min(size - 1, max(0, int(center_x * size)))
        grid_y = min(size - 1, max(0, int(center_y * size)))
        if targets[f"{level}_aff"][grid_y, grid_x].item() == 1:
            raise ValueError("Two annotated controls collide in a proposal-grid cell")
        targets[f"{level}_aff"][grid_y, grid_x] = 1
        targets[f"{level}_box"][:, grid_y, grid_x] = torch.tensor(
            [y1, x1, y2, x2], dtype=torch.float32
        )
        targets[f"{level}_kind"][grid_y, grid_x] = KIND_TO_IDX[region["kind"]]
    return targets


def _crop_tensor(image: Image.Image, box: tuple[int, int, int, int]) -> torch.Tensor:
    crop = image.crop(box).resize(
        (CANONICAL_CROP_WIDTH, CANONICAL_CROP_HEIGHT),
        Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(crop, dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)


class CleanroomSystem1Dataset(Dataset):
    """Frame-query samples with all supervision derived from generated labels."""

    def __init__(
        self,
        corpus_dir: str | Path,
        split: str,
        validate_hashes: bool = True,
    ) -> None:
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unknown clean-room split: {split!r}")
        self.corpus_dir = Path(corpus_dir).resolve()
        self.split = split
        if validate_hashes:
            validate_dataset(self.corpus_dir)
        self.manifest_path = self.corpus_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest_sha256 = _sha256(self.manifest_path)
        self.category_to_id = {name: index for index, name in enumerate(UI_CATEGORIES)}

        self.samples: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for record in self.manifest["files"]:
            if record["split"] != split:
                continue
            annotation_path = self.corpus_dir / record["annotation"]["relative_path"]
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("schema") != ANNOTATION_SCHEMA:
                raise ValueError(f"Unsupported annotation schema: {annotation_path}")
            if record["category"] not in self.category_to_id:
                raise ValueError(f"Category is unsupported by AVCSSystem1V1: {record['category']}")
            for query in annotation["queries"]:
                self.samples.append((record, annotation, query))
        if not self.samples:
            raise ValueError(f"Clean-room split is empty: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, annotation, query = self.samples[index]
        image_path = self.corpus_dir / record["image"]["relative_path"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image_pixels = np.asarray(image, dtype=np.uint8).copy()
        image_tensor = torch.from_numpy(image_pixels).permute(2, 0, 1).float().div_(255.0)

        proposal = _proposal_targets(annotation["regions"])
        positive = bool(query["target_presence"])
        target_point = query["target_point_norm"] if positive else [0.0, 0.0]

        region_by_id = {region["region_id"]: region for region in annotation["regions"]}
        target_box = (
            region_by_id[query["target_region_id"]]["bbox_xyxy_norm"]
            if positive
            else [0.0, 0.0, 0.0, 0.0]
        )
        crop_list: list[torch.Tensor] = []
        word_ids: list[int] = []
        for region in annotation["regions"]:
            x1, y1, x2, y2 = region["bbox_xyxy_px"]
            crop_list.append(_crop_tensor(image, (x1, y1, x2, y2)))
            word_label = region["word_class"]
            if word_label not in WORD_TO_ID:
                raise ValueError(f"Word label is unsupported by System 1 vocabulary: {word_label}")
            word_ids.append(WORD_TO_ID[word_label])

        unknown = annotation["unknown_word"]
        ux1, uy1, ux2, uy2 = unknown["bbox_xyxy_px"]
        crop_list.append(_crop_tensor(image, (ux1, uy1, ux2, uy2)))
        word_ids.append(WORD_TO_ID["<none>"])

        while len(crop_list) < 5:
            crop_list.append(_crop_tensor(image, (18, 44, 66, 66)))
            word_ids.append(WORD_TO_ID["<none>"])
        if len(crop_list) != 5:
            raise ValueError("Clean-room frames must provide exactly four controls and one unknown-word crop")

        if positive:
            target_region = region_by_id[query["target_region_id"]]
            target_box = target_region["bbox_xyxy_norm"]
            if not (
                target_box[0] <= target_point[0] <= target_box[2]
                and target_box[1] <= target_point[1] <= target_box[3]
            ):
                raise ValueError("Grounding point is outside its renderer-produced target box")

        return {
            "image": image_tensor,
            "query": query["query"],
            "category": torch.tensor(self.category_to_id[record["category"]], dtype=torch.long),
            "target_present": torch.tensor(float(positive), dtype=torch.float32),
            "target_point": torch.tensor(target_point, dtype=torch.float32),
            "target_box": torch.tensor(target_box, dtype=torch.float32),
            "p2_aff": proposal["p2_aff"],
            "p2_box": proposal["p2_box"],
            "p2_kind": proposal["p2_kind"],
            "p3_aff": proposal["p3_aff"],
            "p3_box": proposal["p3_box"],
            "p3_kind": proposal["p3_kind"],
            "p4_aff": proposal["p4_aff"],
            "p4_box": proposal["p4_box"],
            "p4_kind": proposal["p4_kind"],
            "word_crops": torch.stack(crop_list),
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "unknown_word_index": torch.tensor(len(annotation["regions"]), dtype=torch.long),
            "frame_id": record["frame_id"],
            "split": self.split,
            "layout_family": record["layout_family"],
            "target_region_id": query["target_region_id"],
        }


def encode_cleanroom_queries(queries: list[str]) -> torch.Tensor:
    """Encode generated natural-language queries with the System 1 byte encoder."""
    if not queries:
        raise ValueError("A batch must contain at least one target query")
    return encode_queries(queries)


def image_dimensions() -> tuple[int, int]:
    """Return fixed corpus dimensions for consistency checks."""
    return CANVAS_SIZE
