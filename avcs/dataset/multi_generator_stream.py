"""Multi-Generator DataStream with Arm D2 Balanced Sampling and Collation.

Provides balanced multi-generator data loading across synthetic generators
(`datasets/ui_grounding` and `datasets/cleanroom_system1_v1`) for training and evaluating
class-agnostic UI objectness models.

Key Capabilities:
- Arm D2 Weighted Sampling: Uses `WeightedRandomSampler` with inverse-frequency weights
  so each batch draws 50% UI Grounding and 50% Cleanroom in expectation, with total epoch
  size equal to N_ui + N_cleanroom to eliminate generator starvation.
- Strict Layout Leakage Isolation: Enforces disjoint layout families and sessions between
  train, validation, and test splits.
- Unified Collation: Standardizes outputs into `UnifiedBatchDict` matching PROJECT.md §6.2.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from avcs.dataset.unified_manifest import (
    GENERATOR_CLEANROOM,
    GENERATOR_UI_GROUNDING,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
    SPLIT_TRAIN,
    SPLIT_VAL,
    UnifiedAnnotationFrame,
    UnifiedAnnotationManifest,
    _normalize_split_name,
    build_combined_manifest,
    load_cleanroom_manifest,
    load_ui_grounding_manifest,
)

GENERATOR_ID_UI: int = 0
GENERATOR_ID_CLEANROOM: int = 1
GENERATOR_NAME_TO_ID: Dict[str, int] = {
    GENERATOR_UI_GROUNDING: GENERATOR_ID_UI,
    GENERATOR_CLEANROOM: GENERATOR_ID_CLEANROOM,
}
GENERATOR_ID_TO_NAME: Dict[int, str] = {
    GENERATOR_ID_UI: GENERATOR_UI_GROUNDING,
    GENERATOR_ID_CLEANROOM: GENERATOR_CLEANROOM,
}


class UnifiedBatchDict(TypedDict):
    """Canonical batch dictionary conforming to PROJECT.md §6.2 contract."""

    images: torch.Tensor  # [B, 3, 128, 128] float32 in [0.0, 1.0]
    target_boxes: torch.Tensor  # [B, max_objs, 4] normalized [x1, y1, x2, y2]
    target_centroids: torch.Tensor  # [B, max_objs, 2] normalized (cx, cy)
    box_mask: torch.Tensor  # [B, max_objs] bool (True for valid objects)
    has_target: torch.Tensor  # [B] float32 (1.0 for UI screens, 0.0 for blanks)
    generator_id: torch.Tensor  # [B] int64 (0: ui_grounding, 1: cleanroom_v1)


def calculate_arm_d2_weights(
    ui_count: int,
    cleanroom_count: int,
) -> Tuple[float, float]:
    """Calculate inverse-frequency weights for Arm D2 balanced sampling.

    Ensures each generator contributes 50% of the sample draws per epoch
    over a total epoch draw count of (ui_count + cleanroom_count).
    """
    total = ui_count + cleanroom_count
    if total == 0:
        return 0.0, 0.0
    w_ui = total / (2.0 * ui_count) if ui_count > 0 else 0.0
    w_cleanroom = total / (2.0 * cleanroom_count) if cleanroom_count > 0 else 0.0
    return w_ui, w_cleanroom


class MultiGeneratorDataset(Dataset):
    """High-throughput multi-generator PyTorch Dataset for UI Objectness.

    Pre-caches normalized geometries and optionally pre-loads resized image tensors
    in RAM for sub-millisecond CPU epoch iteration.
    """

    def __init__(
        self,
        manifest: Optional[UnifiedAnnotationManifest] = None,
        ui_manifest_path: Union[str, Path] = "datasets/ui_grounding/manifest.json",
        cleanroom_dir: Union[str, Path] = "datasets/cleanroom_system1_v1",
        split: str = SPLIT_TRAIN,
        target_image_size: Tuple[int, int] = (128, 128),
        preload_images: bool = True,
    ) -> None:
        super().__init__()
        self.split = _normalize_split_name(split)
        self.target_image_size = target_image_size
        self.preload_images = preload_images

        # Load or filter manifest
        if manifest is None:
            combined = build_combined_manifest(
                ui_manifest_path=ui_manifest_path,
                cleanroom_dir=cleanroom_dir,
            )
            self.manifest = combined.filter_split(self.split)
        else:
            self.manifest = manifest.filter_split(self.split)

        self.frames: List[UnifiedAnnotationFrame] = list(self.manifest.frames)

        # Image cache in RAM: frame_id -> torch.Tensor [3, H, W]
        self._image_cache: Dict[str, torch.Tensor] = {}

        # Precompute per-sample metadata and pre-load images if enabled
        self._samples: List[Dict[str, Any]] = []
        for frame in self.frames:
            gen_id = GENERATOR_NAME_TO_ID.get(frame.generator_source, GENERATOR_ID_UI)
            is_neg = frame.is_negative_frame or len(frame.objects) == 0

            # Normalized boxes [x1, y1, x2, y2]
            if is_neg:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                centroids = torch.zeros((0, 2), dtype=torch.float32)
                has_target = 0.0
            else:
                boxes_list = [list(obj.bbox_xyxy_norm) for obj in frame.objects]
                centroids_list = [list(obj.centroid_xy_norm) for obj in frame.objects]
                boxes = torch.tensor(boxes_list, dtype=torch.float32)
                centroids = torch.tensor(centroids_list, dtype=torch.float32)
                has_target = 1.0

            sample_entry = {
                "frame_id": frame.frame_id,
                "generator_id": gen_id,
                "generator_source": frame.generator_source,
                "image_path": frame.image_path,
                "target_boxes": boxes,
                "target_centroids": centroids,
                "has_target": float(has_target),
                "is_negative": is_neg,
                "layout_template_id": frame.layout_template_id or "",
                "split": frame.split,
            }
            self._samples.append(sample_entry)

            if self.preload_images:
                self._load_image_tensor(frame.frame_id, frame.image_path)

    def _load_image_tensor(self, frame_id: str, image_path: str) -> torch.Tensor:
        """Load, resize, and normalize an image to [3, H, W] float32 in [0, 1]."""
        if frame_id in self._image_cache:
            return self._image_cache[frame_id]

        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        with Image.open(p) as img:
            img_rgb = img.convert("RGB")
            if img_rgb.size != self.target_image_size:
                img_resized = img_rgb.resize(
                    self.target_image_size, Image.Resampling.BILINEAR
                )
            else:
                img_resized = img_rgb

            arr = np.asarray(img_resized, dtype=np.float32) / 255.0
            # [H, W, 3] -> [3, H, W]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        if self.preload_images:
            self._image_cache[frame_id] = tensor
        return tensor

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        image_tensor = self._load_image_tensor(sample["frame_id"], sample["image_path"])

        return {
            "image": image_tensor,
            "target_boxes": sample["target_boxes"],
            "target_centroids": sample["target_centroids"],
            "has_target": sample["has_target"],
            "generator_id": sample["generator_id"],
            "frame_id": sample["frame_id"],
            "split": sample["split"],
            "layout_template_id": sample["layout_template_id"],
            "is_negative": sample["is_negative"],
        }

    @property
    def ui_count(self) -> int:
        """Number of UI Grounding frames in the dataset split."""
        return sum(1 for s in self._samples if s["generator_id"] == GENERATOR_ID_UI)

    @property
    def cleanroom_count(self) -> int:
        """Number of Cleanroom frames in the dataset split."""
        return sum(1 for s in self._samples if s["generator_id"] == GENERATOR_ID_CLEANROOM)

    def get_sample_weights(self) -> List[float]:
        """Compute per-sample weights according to Arm D2 balanced sampling."""
        w_ui, w_cleanroom = calculate_arm_d2_weights(self.ui_count, self.cleanroom_count)
        return [
            w_ui if s["generator_id"] == GENERATOR_ID_UI else w_cleanroom
            for s in self._samples
        ]


def unified_batch_collate(batch: List[Dict[str, Any]]) -> UnifiedBatchDict:
    """Collate sample dictionaries into a canonical UnifiedBatchDict.

    Handles variable numbers of objects per frame with boolean masking and padding.
    """
    b_size = len(batch)
    if b_size == 0:
        raise ValueError("Cannot collate an empty batch")

    # Stack images: [B, 3, 128, 128]
    images = torch.stack([item["image"] for item in batch], dim=0)

    # Determine maximum number of objects in this batch
    max_objs = max(len(item["target_boxes"]) for item in batch)

    if max_objs == 0:
        target_boxes = torch.zeros((b_size, 0, 4), dtype=torch.float32)
        target_centroids = torch.zeros((b_size, 0, 2), dtype=torch.float32)
        box_mask = torch.zeros((b_size, 0), dtype=torch.bool)
    else:
        target_boxes = torch.zeros((b_size, max_objs, 4), dtype=torch.float32)
        target_centroids = torch.zeros((b_size, max_objs, 2), dtype=torch.float32)
        box_mask = torch.zeros((b_size, max_objs), dtype=torch.bool)

        for i, item in enumerate(batch):
            k = len(item["target_boxes"])
            if k > 0:
                target_boxes[i, :k, :] = item["target_boxes"]
                target_centroids[i, :k, :] = item["target_centroids"]
                box_mask[i, :k] = True

    has_target = torch.tensor(
        [float(item["has_target"]) for item in batch], dtype=torch.float32
    )
    generator_id = torch.tensor(
        [int(item["generator_id"]) for item in batch], dtype=torch.int64
    )

    return {
        "images": images,
        "target_boxes": target_boxes,
        "target_centroids": target_centroids,
        "box_mask": box_mask,
        "has_target": has_target,
        "generator_id": generator_id,
    }


def create_multi_generator_dataloader(
    dataset: MultiGeneratorDataset,
    batch_size: int = 16,
    sampling_policy: str = "balanced",  # "balanced" (Arm D2) or "natural"
    seed: int = 7,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader with optional Arm D2 balanced sampling."""
    if sampling_policy == "balanced":
        weights = dataset.get_sample_weights()
        num_samples = len(dataset)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=unified_batch_collate,
            drop_last=drop_last,
        )
    elif sampling_policy == "natural":
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            collate_fn=unified_batch_collate,
            drop_last=drop_last,
        )
    elif sampling_policy == "sequential":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=unified_batch_collate,
            drop_last=drop_last,
        )
    else:
        raise ValueError(
            f"Unknown sampling policy: {sampling_policy!r}. Must be 'balanced', 'natural', or 'sequential'."
        )


class MultiGeneratorDataStream:
    """High-level DataStream encapsulating dataset, sampler, and dataloader."""

    def __init__(
        self,
        manifest: Optional[UnifiedAnnotationManifest] = None,
        ui_manifest_path: Union[str, Path] = "datasets/ui_grounding/manifest.json",
        cleanroom_dir: Union[str, Path] = "datasets/cleanroom_system1_v1",
        split: str = SPLIT_TRAIN,
        batch_size: int = 16,
        target_image_size: Tuple[int, int] = (128, 128),
        sampling_policy: str = "balanced",
        seed: int = 7,
        num_workers: int = 0,
        preload_images: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.split = _normalize_split_name(split)
        self.batch_size = batch_size
        self.sampling_policy = sampling_policy
        self.seed = seed

        self.dataset = MultiGeneratorDataset(
            manifest=manifest,
            ui_manifest_path=ui_manifest_path,
            cleanroom_dir=cleanroom_dir,
            split=self.split,
            target_image_size=target_image_size,
            preload_images=preload_images,
        )

        self.dataloader = create_multi_generator_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampling_policy=self.sampling_policy,
            seed=self.seed,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> Iterator[UnifiedBatchDict]:
        return iter(self.dataloader)

    def get_batch(self) -> UnifiedBatchDict:
        """Fetch the first batch for testing or inspection."""
        for batch in self.dataloader:
            return batch
        raise StopIteration("DataStream is empty")

    @property
    def ui_count(self) -> int:
        return self.dataset.ui_count

    @property
    def cleanroom_count(self) -> int:
        return self.dataset.cleanroom_count

    @property
    def total_samples(self) -> int:
        return len(self.dataset)


def verify_layout_leakage_isolation(
    train_manifest: UnifiedAnnotationManifest,
    test_manifest: UnifiedAnnotationManifest,
) -> Dict[str, Any]:
    """Verify strict isolation of layout templates, families, sessions, and image hashes.

    Ensures zero test leakage across train and test splits.
    """
    report: Dict[str, Any] = {
        "isolated": True,
        "cleanroom_isolated": True,
        "ui_grounding_isolated": True,
        "image_hashes_isolated": True,
        "cleanroom_train_families": [],
        "cleanroom_test_families": [],
        "ui_train_sessions": [],
        "ui_test_sessions": [],
        "collisions": [],
    }

    # 1. Cleanroom layout family isolation
    clean_train = [f for f in train_manifest.frames if f.generator_source == GENERATOR_CLEANROOM]
    clean_test = [f for f in test_manifest.frames if f.generator_source == GENERATOR_CLEANROOM]

    clean_train_families = {f.layout_template_id for f in clean_train if f.layout_template_id}
    clean_test_families = {f.layout_template_id for f in clean_test if f.layout_template_id}

    report["cleanroom_train_families"] = sorted(list(clean_train_families))
    report["cleanroom_test_families"] = sorted(list(clean_test_families))

    overlap_families = clean_train_families & clean_test_families
    if overlap_families:
        report["cleanroom_isolated"] = False
        report["isolated"] = False
        report["collisions"].append(f"Cleanroom layout family collision: {overlap_families}")

    # 2. UI Grounding session isolation
    ui_train = [f for f in train_manifest.frames if f.generator_source == GENERATOR_UI_GROUNDING]
    ui_test = [f for f in test_manifest.frames if f.generator_source == GENERATOR_UI_GROUNDING]

    ui_train_sessions = {f.metadata.get("session_id", "") for f in ui_train if f.metadata.get("session_id")}
    ui_test_sessions = {f.metadata.get("session_id", "") for f in ui_test if f.metadata.get("session_id")}

    report["ui_train_sessions"] = sorted(list(ui_train_sessions))
    report["ui_test_sessions"] = sorted(list(ui_test_sessions))

    overlap_sessions = ui_train_sessions & ui_test_sessions
    if overlap_sessions:
        report["ui_grounding_isolated"] = False
        report["isolated"] = False
        report["collisions"].append(f"UI Grounding session collision: {overlap_sessions}")

    # 3. Image SHA-256 hash isolation
    def _compute_hashes(frames: List[UnifiedAnnotationFrame]) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for fr in frames:
            p = Path(fr.image_path)
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                hashes[fr.frame_id] = h
        return hashes

    train_hashes = _compute_hashes(train_manifest.frames)
    test_hashes = _compute_hashes(test_manifest.frames)

    train_hash_set = set(train_hashes.values())
    test_hash_set = set(test_hashes.values())
    hash_overlap = train_hash_set & test_hash_set

    if hash_overlap:
        report["image_hashes_isolated"] = False
        report["isolated"] = False
        report["collisions"].append(f"Image SHA-256 collisions detected: {len(hash_overlap)} hashes overlap")

    return report
