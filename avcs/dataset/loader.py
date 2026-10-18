"""High-throughput PyTorch DataLoader and Dataset for UI Grounding."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from avcs.goal_grounder import MAX_QUERY_BYTES, encode_queries


class GroundingSample(dict):
    """Dictionary representing a single grounding sample, supporting 6-tuple unpacking:
    
    (image_tensor, query_str, centroid_norm, bbox_norm, label, is_negative) = sample
    """

    def __iter__(self) -> Iterator[Any]:
        yield self["image"]
        yield self["query_str"]
        yield self["target_centroid"]
        yield self["target_bbox"]
        yield self["label"]
        yield self["is_negative"]

    @property
    def image(self) -> torch.Tensor:
        return self["image"]

    @property
    def query_str(self) -> str:
        return self["query_str"]

    @property
    def target_centroid(self) -> torch.Tensor:
        return self["target_centroid"]

    @property
    def target_bbox(self) -> torch.Tensor:
        return self["target_bbox"]

    @property
    def label(self) -> torch.Tensor:
        return self["label"]

    @property
    def is_negative(self) -> bool:
        return bool(self["is_negative"])

    def __getitem__(self, key: str) -> Any:
        if key == "target_heatmap" and "target_heatmap" not in self:
            c = self["target_centroid"]
            cx, cy = float(c[0]), float(c[1])
            has_t = bool(self["has_target"].item() > 0.5)
            hm_h, hm_w = (64, 64)
            if not has_t:
                self["target_heatmap"] = torch.zeros((hm_h, hm_w), dtype=torch.float32)
            else:
                yy, xx = torch.meshgrid(
                    (torch.arange(hm_h, dtype=torch.float32) + 0.5) / hm_h,
                    (torch.arange(hm_w, dtype=torch.float32) + 0.5) / hm_w,
                    indexing="ij",
                )
                dist = (xx - cx).square() * (hm_w**2) + (yy - cy).square() * (hm_h**2)
                raw_heat = torch.exp(-dist / (2.0 * 1.5**2))
                self["target_heatmap"] = raw_heat / raw_heat.sum().clamp_min(1e-7)
        return super().__getitem__(key)

    @property
    def target_heatmap(self) -> torch.Tensor:
        return self["target_heatmap"]


class UIGroundingDataset(Dataset):
    """High-throughput PyTorch Dataset for UI Visual Grounding & Affordance detection.
    
    Features:
    - Preloaded in-memory image frame cache (270 frames is ~53 MB in RAM).
    - Pre-encoded query tokens and target geometries.
    - Deferred / SIMD vectorized batch Gaussian heatmap generation in collate_fn.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path] = "datasets/ui_grounding/manifest.json",
        data_dir: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (64, 64),
        heatmap_sigma: float = 1.5,
        expand_element_queries: bool = False,
        include_negative_queries: bool = False,
        preload_frames: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_dir = Path(data_dir) if data_dir else self.manifest_path.parent
        self.split = split
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.expand_element_queries = expand_element_queries
        self.include_negative_queries = include_negative_queries
        self.preload_frames = preload_frames

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest_data = json.load(f)

        raw_frames = self.manifest_data.get("frames", [])
        if self.split:
            raw_frames = [fr for fr in raw_frames if fr.get("split") == self.split]
        self.frames = raw_frames

        # In-memory image frame cache: file_path -> torch.Tensor [3, H, W] float32 in [0, 1]
        self._frame_cache: Dict[str, torch.Tensor] = {}
        if self.preload_frames:
            for fr in self.frames:
                img_rel = fr["file_path"]
                if img_rel not in self._frame_cache:
                    self._frame_cache[img_rel] = self._load_image_tensor(img_rel)

        # Build indexed sample records with pre-parsed tensors
        self._sample_records: List[GroundingSample] = []
        for fr in self.frames:
            is_neg = fr.get("is_negative", False)
            pts = fr.get("interactive_points", [])
            neg_queries = fr.get("negative_queries", ["blank desktop", "search bar"])
            img_rel = fr["file_path"]
            img_t = self._frame_cache.get(img_rel) if self.preload_frames else None

            if is_neg or not pts:
                # Negative sample
                if self.include_negative_queries and neg_queries:
                    for nq in neg_queries:
                        self._sample_records.append(self._create_grounding_sample(fr, None, nq, True, img_rel, img_t))
                else:
                    q = neg_queries[0] if neg_queries else "blank background"
                    self._sample_records.append(self._create_grounding_sample(fr, None, q, True, img_rel, img_t))
            else:
                if self.expand_element_queries:
                    # Every interactive point is an independent query sample
                    for pt in pts:
                        q = pt.get("target_query", f"click {pt.get('element_kind', 'button')}")
                        self._sample_records.append(self._create_grounding_sample(fr, pt, q, False, img_rel, img_t))
                    if self.include_negative_queries and neg_queries:
                        for nq in neg_queries:
                            self._sample_records.append(self._create_grounding_sample(fr, None, nq, True, img_rel, img_t))
                else:
                    # Canonical frame sample using primary element
                    pt = pts[0]
                    q = pt.get("target_query", f"click {pt.get('element_kind', 'button')}")
                    self._sample_records.append(self._create_grounding_sample(fr, pt, q, False, img_rel, img_t))

    def _load_image_tensor(self, img_rel: str) -> torch.Tensor:
        """Reads and normalizes single frame to [3, H, W] float32 in [0, 1]."""
        img_full = self.data_dir / img_rel
        with Image.open(img_full) as pil_img:
            if pil_img.size != self.image_size:
                pil_img = pil_img.resize(self.image_size, Image.BILINEAR)
            img_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        # CHW uint8 -> float32 [0.0, 1.0]
        return torch.from_numpy(np.ascontiguousarray(img_arr.transpose(2, 0, 1))).float().div_(255.0)

    def _create_grounding_sample(
        self,
        fr: Dict[str, Any],
        pt: Optional[Dict[str, Any]],
        query_str: str,
        is_neg: bool,
        img_rel: str,
        img_tensor: Optional[torch.Tensor],
    ) -> GroundingSample:
        """Pre-computes coordinate tensors and query tokens at init time."""
        if is_neg or pt is None:
            centroid_norm = torch.tensor([-1.0, -1.0], dtype=torch.float32)
            bbox_norm = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)
            label = torch.tensor(0, dtype=torch.long)
            has_target = torch.tensor(0.0, dtype=torch.float32)
        else:
            cx = float(pt.get("x", 0.5))
            cy = float(pt.get("y", 0.5))
            centroid_norm = torch.tensor([cx, cy], dtype=torch.float32)

            bb = pt.get("bbox", [cy - 0.05, cx - 0.05, cy + 0.05, cx + 0.05])
            bbox_norm = torch.tensor(bb, dtype=torch.float32)

            act = pt.get("action_type", "clickable")
            label_val = 2 if act == "typeable" else 1
            label = torch.tensor(label_val, dtype=torch.long)
            has_target = torch.tensor(1.0, dtype=torch.float32)

        try:
            query_tokens = encode_queries([query_str or "action"])[0]
        except Exception:
            query_tokens = torch.zeros(MAX_QUERY_BYTES, dtype=torch.long)

        data: Dict[str, Any] = {
            "_img_rel": img_rel,
            "query_str": query_str,
            "target_centroid": centroid_norm,
            "target_bbox": bbox_norm,
            "label": label,
            "is_negative": is_neg,
            "has_target": has_target,
            "query_tokens": query_tokens,
            "frame_id": fr.get("frame_id", ""),
            "category": fr.get("category", ""),
        }
        if img_tensor is not None:
            data["image"] = img_tensor

        return GroundingSample(data)

    def __len__(self) -> int:
        return len(self._sample_records)

    def __getitem__(self, idx: int) -> GroundingSample:
        sample = self._sample_records[idx]
        if "image" not in sample:
            img_rel = sample["_img_rel"]
            if self.preload_frames and img_rel in self._frame_cache:
                sample["image"] = self._frame_cache[img_rel]
            else:
                img_t = self._load_image_tensor(img_rel)
                if self.preload_frames:
                    self._frame_cache[img_rel] = img_t
                sample["image"] = img_t
        return sample


# Module-level coordinate meshgrid cache for SIMD batch collation
_GRID_CACHE: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_coordinate_grid(hm_h: int, hm_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieves or caches 1xHmXWm coordinate grids."""
    key = (hm_h, hm_w)
    if key not in _GRID_CACHE:
        yy, xx = torch.meshgrid(
            (torch.arange(hm_h, dtype=torch.float32) + 0.5) / hm_h,
            (torch.arange(hm_w, dtype=torch.float32) + 0.5) / hm_w,
            indexing="ij",
        )
        _GRID_CACHE[key] = (xx.unsqueeze(0), yy.unsqueeze(0))
    return _GRID_CACHE[key]


def ui_grounding_collate_fn(
    batch: List[GroundingSample],
    heatmap_size: Tuple[int, int] = (64, 64),
    sigma_cells: float = 1.5,
) -> Dict[str, Any]:
    """Vectorized SIMD batch collation for high-throughput CPU training."""
    B = len(batch)
    image_tensor = torch.stack([item["image"] for item in batch], dim=0)
    query_strs = [item["query_str"] for item in batch]
    target_centroid = torch.stack([item["target_centroid"] for item in batch], dim=0)
    target_bbox = torch.stack([item["target_bbox"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    is_negatives = torch.tensor([item["is_negative"] for item in batch], dtype=torch.bool)
    has_target = torch.stack([item["has_target"] for item in batch], dim=0)
    query_tokens = torch.stack([item["query_tokens"] for item in batch], dim=0)

    # SIMD Vectorized batch Gaussian heatmap generation
    Hm, Wm = heatmap_size
    grid_x, grid_y = _get_coordinate_grid(Hm, Wm)

    px = target_centroid[:, 0].view(B, 1, 1)
    py = target_centroid[:, 1].view(B, 1, 1)
    inv_2_sigma_sq = (Wm * Hm) / (2.0 * sigma_cells**2)

    dx = grid_x - px
    dy = grid_y - py
    dx.square_()
    dy.square_()
    dx.add_(dy).mul_(-inv_2_sigma_sq)
    target_heatmap = dx.exp_()
    sums = target_heatmap.sum(dim=(-2, -1), keepdim=True).clamp_min_(1e-7)
    target_heatmap.div_(sums).mul_(has_target.view(B, 1, 1))

    return {
        "image": image_tensor,
        "query_str": query_strs,
        "target_centroid": target_centroid,
        "target_bbox": target_bbox,
        "label": labels,
        "is_negative": is_negatives,
        "has_target": has_target,
        "query_tokens": query_tokens,
        "target_heatmap": target_heatmap,
    }


def make_ui_grounding_dataloader(
    dataset: UIGroundingDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    heatmap_size: Tuple[int, int] = (64, 64),
    sigma_cells: float = 1.5,
) -> DataLoader:
    """Instantiates a tuned CPU DataLoader with zero-copy vectorized collation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda b: ui_grounding_collate_fn(b, heatmap_size, sigma_cells),
    )


# Backward-compatibility aliases
FastUIGroundingDataset = UIGroundingDataset
fast_ui_grounding_collate_fn = ui_grounding_collate_fn
make_fast_ui_grounding_dataloader = make_ui_grounding_dataloader
