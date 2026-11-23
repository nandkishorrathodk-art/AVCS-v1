"""High-throughput PyTorch DataLoader and Dataset for UI Grounding."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

MAX_QUERY_BYTES = 128


def encode_queries(queries: Sequence[str], max_length: int = MAX_QUERY_BYTES) -> torch.Tensor:
    tensors = []
    for q in queries:
        b = (q or "").encode("utf-8")[:max_length]
        arr = np.frombuffer(b, dtype=np.uint8)
        padded = np.zeros(max_length, dtype=np.uint8)
        padded[: len(arr)] = arr
        tensors.append(torch.from_numpy(padded).long())
    return torch.stack(tensors, dim=0)


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


# ============================================================================
# Phase 2: LeRobot v3.0 Pure-RGB Canonical Schemas, Cleaner & QA Doctor
# ============================================================================

@dataclass
class LeRobotV3TrajectoryEpisode:
    """Canonical Parquet episode schema for pure-RGB VLA desktop control.

    Stores dense multi-modal steps with zero-B-frame synchronization
    and strict pre-action frame binding timestamps.
    """
    episode_id: str
    step_index: int
    video_frame_index: int
    frame_present_ns: int
    action_event_ns: int
    screen_width: int
    screen_height: int
    dpi_scale: float = 1.0
    os_platform: str = "windows"
    task_instruction: str = ""
    subgoal_instruction: Optional[str] = None
    cursor_pre_xy: Tuple[float, float] = (0.5, 0.5)
    action_kind: int = 0  # 0:left_click, 1:right_click, 2:double_click, 3:move, 4:drag, 5:scroll, 6:hotkey, 7:type, 8:wait, 9:stop
    target_xy: Tuple[float, float] = (0.5, 0.5)
    drag_trajectory_xy: List[Tuple[float, float]] = field(default_factory=list)
    scroll_delta: Tuple[float, float] = (0.0, 0.0)
    hotkey_ids: List[int] = field(default_factory=list)
    typed_text: str = ""
    pixel_diff_ratio: float = 0.0
    image_bytes: bytes = b""
    image_tensor: Optional[torch.Tensor] = None

    @property
    def latency_ns(self) -> int:
        return max(0, self.action_event_ns - self.frame_present_ns)

    @property
    def latency_ms(self) -> float:
        return self.latency_ns / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "video_frame_index": self.video_frame_index,
            "frame_present_ns": self.frame_present_ns,
            "action_event_ns": self.action_event_ns,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "dpi_scale": self.dpi_scale,
            "os_platform": self.os_platform,
            "task_instruction": self.task_instruction,
            "subgoal_instruction": self.subgoal_instruction,
            "cursor_pre_xy": list(self.cursor_pre_xy),
            "action_kind": self.action_kind,
            "target_xy": list(self.target_xy),
            "drag_trajectory_xy": [list(pt) for pt in self.drag_trajectory_xy],
            "scroll_delta": list(self.scroll_delta),
            "hotkey_ids": list(self.hotkey_ids),
            "typed_text": self.typed_text,
            "pixel_diff_ratio": self.pixel_diff_ratio,
            "image_bytes": self.image_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LeRobotV3TrajectoryEpisode:
        return cls(
            episode_id=str(data.get("episode_id", "")),
            step_index=int(data.get("step_index", 0)),
            video_frame_index=int(data.get("video_frame_index", 0)),
            frame_present_ns=int(data.get("frame_present_ns", 0)),
            action_event_ns=int(data.get("action_event_ns", 0)),
            screen_width=int(data.get("screen_width", 1280)),
            screen_height=int(data.get("screen_height", 800)),
            dpi_scale=float(data.get("dpi_scale", 1.0)),
            os_platform=str(data.get("os_platform", "windows")),
            task_instruction=str(data.get("task_instruction", "")),
            subgoal_instruction=data.get("subgoal_instruction"),
            cursor_pre_xy=tuple(data.get("cursor_pre_xy", (0.5, 0.5))),
            action_kind=int(data.get("action_kind", 0)),
            target_xy=tuple(data.get("target_xy", (0.5, 0.5))),
            drag_trajectory_xy=[tuple(pt) for pt in data.get("drag_trajectory_xy", [])],
            scroll_delta=tuple(data.get("scroll_delta", (0.0, 0.0))),
            hotkey_ids=list(data.get("hotkey_ids", [])),
            typed_text=str(data.get("typed_text", "")),
            pixel_diff_ratio=float(data.get("pixel_diff_ratio", 0.0)),
            image_bytes=bytes(data.get("image_bytes", b"")),
        )


class TrajectoryCleaner:
    """Curation engine for desktop demonstrations: idle collapse, jitter smoothing, typing coalescing, misclick quarantine."""

    @staticmethod
    def collapse_idle_frames(
        steps: List[LeRobotV3TrajectoryEpisode],
        min_pixel_diff: float = 0.005,
        max_idle_wait_steps: int = 3,
    ) -> List[LeRobotV3TrajectoryEpisode]:
        """Drops consecutive idle wait frames where screen did not change (L1 delta < min_pixel_diff)."""
        if not steps:
            return []
        cleaned: List[LeRobotV3TrajectoryEpisode] = []
        idle_wait_count = 0
        for step in steps:
            # Action kind 8 is WAIT
            if step.action_kind == 8 and step.pixel_diff_ratio < min_pixel_diff:
                idle_wait_count += 1
                if idle_wait_count <= max_idle_wait_steps:
                    cleaned.append(step)
            else:
                idle_wait_count = 0
                cleaned.append(step)
        return cleaned

    @staticmethod
    def smooth_jitter(
        steps: List[LeRobotV3TrajectoryEpisode],
        min_distance_px: float = 4.0,
        screen_w: int = 1920,
        screen_h: int = 1080,
    ) -> List[LeRobotV3TrajectoryEpisode]:
        """Filters out sub-4px mouse micro-jitter movements unless followed by click or drag."""
        if not steps:
            return []
        cleaned: List[LeRobotV3TrajectoryEpisode] = []
        min_norm_dist_sq = (min_distance_px / max(1, screen_w)) ** 2 + (min_distance_px / max(1, screen_h)) ** 2
        for i, step in enumerate(steps):
            if step.action_kind == 3:  # MOVE
                next_is_click = (i + 1 < len(steps)) and steps[i + 1].action_kind in (0, 1, 2, 4)
                if not next_is_click:
                    dx = step.target_xy[0] - step.cursor_pre_xy[0]
                    dy = step.target_xy[1] - step.cursor_pre_xy[1]
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < min_norm_dist_sq:
                        continue
            cleaned.append(step)
        return cleaned

    @staticmethod
    def coalesce_typing(
        steps: List[LeRobotV3TrajectoryEpisode],
        max_keystroke_gap_ns: int = 300_000_000,  # 300ms
    ) -> List[LeRobotV3TrajectoryEpisode]:
        """Coalesces rapid consecutive single-character keystrokes into unified typed_text actions."""
        if not steps:
            return []
        cleaned: List[LeRobotV3TrajectoryEpisode] = []
        for step in steps:
            if step.action_kind == 7 and cleaned and cleaned[-1].action_kind == 7:  # TYPE
                prev = cleaned[-1]
                time_delta = step.action_event_ns - prev.action_event_ns
                if 0 <= time_delta <= max_keystroke_gap_ns:
                    merged_text = prev.typed_text + step.typed_text
                    cleaned[-1] = LeRobotV3TrajectoryEpisode(
                        episode_id=prev.episode_id,
                        step_index=prev.step_index,
                        video_frame_index=prev.video_frame_index,
                        frame_present_ns=prev.frame_present_ns,
                        action_event_ns=step.action_event_ns,
                        screen_width=prev.screen_width,
                        screen_height=prev.screen_height,
                        dpi_scale=prev.dpi_scale,
                        os_platform=prev.os_platform,
                        task_instruction=prev.task_instruction,
                        subgoal_instruction=prev.subgoal_instruction,
                        cursor_pre_xy=prev.cursor_pre_xy,
                        action_kind=7,
                        target_xy=prev.target_xy,
                        drag_trajectory_xy=prev.drag_trajectory_xy,
                        scroll_delta=prev.scroll_delta,
                        hotkey_ids=prev.hotkey_ids,
                        typed_text=merged_text,
                        pixel_diff_ratio=max(prev.pixel_diff_ratio, step.pixel_diff_ratio),
                        image_bytes=prev.image_bytes,
                        image_tensor=prev.image_tensor,
                    )
                    continue
            cleaned.append(step)
        return cleaned

    @staticmethod
    def quarantine_misclicks(
        steps: List[LeRobotV3TrajectoryEpisode],
        undo_window_ns: int = 1_500_000_000,  # 1.5s
    ) -> Tuple[List[LeRobotV3TrajectoryEpisode], int]:
        """Quarantines misclicks followed by Ctrl+Z / undo within 1.5s."""
        if not steps:
            return [], 0
        quarantine_indices = set()
        for i, step in enumerate(steps):
            is_undo = (step.action_kind == 6) and (step.typed_text.lower() in ("ctrl+z", "undo") or 29 in step.hotkey_ids)
            if is_undo and i > 0:
                for j in range(i - 1, -1, -1):
                    prior = steps[j]
                    if step.action_event_ns - prior.action_event_ns <= undo_window_ns:
                        if prior.action_kind in (0, 1, 2, 4):
                            quarantine_indices.add(j)
                            quarantine_indices.add(i)
                            break  # Undo targets the nearest preceding action
                    else:
                        break
        cleaned = [step for i, step in enumerate(steps) if i not in quarantine_indices]
        return cleaned, len(quarantine_indices)

    @classmethod
    def clean_trajectory(cls, steps: List[LeRobotV3TrajectoryEpisode]) -> List[LeRobotV3TrajectoryEpisode]:
        """Runs full 4-stage curation pipeline on an episode trajectory."""
        s1 = cls.collapse_idle_frames(steps)
        s2 = cls.smooth_jitter(s1)
        s3 = cls.coalesce_typing(s2)
        s4, _ = cls.quarantine_misclicks(s3)
        for i, step in enumerate(s4):
            step.step_index = i
        return s4


class QADoctor:
    """Automated Quality Assurance Doctor for LeRobot v3.0 pure-RGB episodes.

    Enforces <5ms frame-to-event synchronization, monotonic clocks, coordinate
    validity, and patches upstream LeRobot v3.0 bugs (Issues #3513 and #3080).
    """

    @staticmethod
    def validate_episode(steps: List[LeRobotV3TrajectoryEpisode]) -> Dict[str, Any]:
        """Audits an episode against the Phase 2 contract."""
        issues: List[str] = []
        if not steps:
            return {"valid": False, "issues": ["Empty episode"], "p95_sync_error_ms": 0.0}

        sync_errors_ns: List[int] = []
        prev_present_ns = 0
        prev_event_ns = 0

        for idx, step in enumerate(steps):
            # 1. Pre-action binding check: frame_present_ns must be <= action_event_ns - 2ms (2e6 ns)
            min_margin = 2_000_000
            diff = step.action_event_ns - step.frame_present_ns
            if diff < min_margin:
                issues.append(f"Step {idx}: frame-to-event margin violation: {diff}ns < {min_margin}ns (post-action leakage risk)")
            sync_errors_ns.append(max(0, diff))

            # 2. Monotonic clock check
            if idx > 0:
                if step.frame_present_ns < prev_present_ns:
                    issues.append(f"Step {idx}: frame_present_ns non-monotonic ({step.frame_present_ns} < {prev_present_ns})")
                if step.action_event_ns < prev_event_ns:
                    issues.append(f"Step {idx}: action_event_ns non-monotonic ({step.action_event_ns} < {prev_event_ns})")

            prev_present_ns = step.frame_present_ns
            prev_event_ns = step.action_event_ns

            # 3. Coordinate bounds check [0.0, 1.0]
            for coord_name, (cx, cy) in [("cursor_pre_xy", step.cursor_pre_xy), ("target_xy", step.target_xy)]:
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    issues.append(f"Step {idx}: {coord_name} coordinates ({cx}, {cy}) outside [0, 1] unit square")

            # 4. Action kind validity (0..9)
            if not (0 <= step.action_kind <= 9):
                issues.append(f"Step {idx}: invalid action_kind {step.action_kind} (must be 0..9)")

        p95_sync_error_ms = 0.0
        if sync_errors_ns:
            sorted_errs = sorted(sync_errors_ns)
            p95_sync_error_ms = sorted_errs[int(0.95 * (len(sorted_errs) - 1))] / 1_000_000.0

        return {
            "valid": len(issues) == 0,
            "step_count": len(steps),
            "issues": issues,
            "p95_sync_error_ms": p95_sync_error_ms,
        }

    @staticmethod
    def audit_dataset(dataset_episodes: List[List[LeRobotV3TrajectoryEpisode]]) -> Dict[str, Any]:
        """Audits an entire collection of demonstration episodes."""
        total = len(dataset_episodes)
        passed = 0
        all_issues: Dict[int, List[str]] = {}
        all_step_sync_errors_ns: List[int] = []

        for i, ep in enumerate(dataset_episodes):
            report = QADoctor.validate_episode(ep)
            if report["valid"]:
                passed += 1
            else:
                all_issues[i] = report["issues"]
            for step in ep:
                all_step_sync_errors_ns.append(max(0, step.action_event_ns - step.frame_present_ns))

        sorted_errs = sorted(all_step_sync_errors_ns)
        p95_global = (sorted_errs[int(0.95 * (len(sorted_errs) - 1))] / 1_000_000.0) if sorted_errs else 0.0

        return {
            "total_episodes": total,
            "passed_episodes": passed,
            "pass_rate": passed / max(1, total),
            "p95_sync_error_ms": p95_global,
            "corrupted_episodes": total - passed,
            "quarantined_issues": all_issues,
        }

    @staticmethod
    def patch_lerobot_3513(frame_indices: Sequence[int], total_frames: int) -> List[int]:
        """Patches LeRobot v3.0 Issue #3513 (IndexError on jittered video timestamps).

        Clamps computed frame indices strictly to valid video range [0, total_frames - 1].
        """
        if total_frames <= 0:
            return [0] * len(frame_indices)
        return [max(0, min(int(idx), total_frames - 1)) for idx in frame_indices]

    @staticmethod
    def patch_lerobot_3080(
        global_frame_idx: int,
        episode_start_frame: int,
        episode_length: int,
    ) -> int:
        """Patches LeRobot v3.0 Issue #3080 (global vs local video indexing mismatch).

        Maps global multi-episode video timeline index to local episode frame index.
        """
        local_idx = global_frame_idx - episode_start_frame
        return max(0, min(local_idx, max(0, episode_length - 1)))


class LeRobotV3Dataset(Dataset):
    """Patched LeRobot v3.0 Dataset loader for pure-RGB desktop control trajectories.

    Features:
    - Mitigates Issue #3513 via timestamp clamping.
    - Mitigates Issue #3080 via local episode frame resolution.
    """

    def __init__(
        self,
        episodes: List[List[LeRobotV3TrajectoryEpisode]],
        target_size: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.episodes = episodes
        self.target_size = target_size
        self._flattened_steps: List[LeRobotV3TrajectoryEpisode] = []
        for ep in self.episodes:
            self._flattened_steps.extend(ep)

    def __len__(self) -> int:
        return len(self._flattened_steps)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Clamped indexing mitigating Issue #3513
        safe_idx = QADoctor.patch_lerobot_3513([idx], len(self._flattened_steps))[0]
        step = self._flattened_steps[safe_idx]

        target_h, target_w = self.target_size
        if step.image_tensor is not None:
            image_t = step.image_tensor
        elif step.image_bytes:
            try:
                import io
                from PIL import Image
                with Image.open(io.BytesIO(step.image_bytes)) as pil_img:
                    if pil_img.size != (target_w, target_h):
                        pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
                    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                    image_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)
            except Exception:
                image_t = torch.zeros((3, target_h, target_w), dtype=torch.float32)
        else:
            image_t = torch.zeros((3, target_h, target_w), dtype=torch.float32)

        return {
            "step_index": step.step_index,
            "video_frame_index": step.video_frame_index,
            "image": image_t,
            "action_kind": torch.tensor(step.action_kind, dtype=torch.long),
            "target_xy": torch.tensor(step.target_xy, dtype=torch.float32),
            "cursor_pre_xy": torch.tensor(step.cursor_pre_xy, dtype=torch.float32),
            "task_instruction": step.task_instruction,
            "latency_ms": torch.tensor(step.latency_ms, dtype=torch.float32),
        }


def make_lerobot_v3_dataloader(
    dataset: LeRobotV3Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Instantiates a high-throughput PyTorch DataLoader for LeRobot v3.0 pure-RGB trajectories."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def generate_golden_diagnostic_dataset(
    num_episodes: int = 500,
    steps_per_episode: int = 10,
    base_timestamp_ns: int = 1_000_000_000,
) -> List[List[LeRobotV3TrajectoryEpisode]]:
    """Generates a reproducible 500-episode Golden Diagnostic Dataset.

    Every episode satisfies Exit Gate 2 invariants:
    - Pre-action frame binding margin: action_event_ns - frame_present_ns >= 2ms (2,000,000ns)
    - P95 sync error strictly < 5.0ms (bounded between 2.2ms and 4.7ms)
    - Monotonic timestamps for frames and actions
    - Valid coordinates in [0.0, 1.0]
    - Valid action kinds in [0, 9]
    """
    dataset: List[List[LeRobotV3TrajectoryEpisode]] = []
    actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for ep_idx in range(num_episodes):
        episode: List[LeRobotV3TrajectoryEpisode] = []
        ep_id = f"golden_diagnostic_ep_{ep_idx:04d}"
        ep_start_ns = base_timestamp_ns + ep_idx * 1_000_000_000

        for step_idx in range(steps_per_episode):
            # 30 FPS video frame intervals (~33.3ms)
            frame_ns = ep_start_ns + step_idx * 33_333_333
            # Sync latency between 2.2ms and 4.7ms (guarantees >= 2ms safety margin and < 5.0ms P95)
            sync_latency_ns = 2_200_000 + ((ep_idx * 17 + step_idx * 31) % 2500) * 1000
            action_ns = frame_ns + sync_latency_ns

            act_kind = actions[step_idx % len(actions)]
            cx = round(0.1 + ((ep_idx * 13 + step_idx * 7) % 80) / 100.0, 4)
            cy = round(0.1 + ((ep_idx * 19 + step_idx * 11) % 80) / 100.0, 4)

            episode.append(
                LeRobotV3TrajectoryEpisode(
                    episode_id=ep_id,
                    step_index=step_idx,
                    video_frame_index=step_idx,
                    frame_present_ns=frame_ns,
                    action_event_ns=action_ns,
                    screen_width=1920,
                    screen_height=1080,
                    dpi_scale=1.0,
                    os_platform="windows",
                    task_instruction=f"Diagnostic validation task {ep_idx}",
                    subgoal_instruction=f"Step {step_idx} execution",
                    cursor_pre_xy=(cx, cy),
                    action_kind=act_kind,
                    target_xy=(cx, cy),
                    drag_trajectory_xy=[(cx, cy)] if act_kind == 4 else [],
                    scroll_delta=(0.0, 120.0) if act_kind == 5 else (0.0, 0.0),
                    typed_text="hello" if act_kind == 7 else "",
                    pixel_diff_ratio=0.02,
                )
            )
        dataset.append(episode)

    return dataset


# --------------------------------------------------------------------------- #
# Phase 4 Data Mixture (OS-Atlas / ShowUI-desktop / AgentNet & DAgger)
# --------------------------------------------------------------------------- #


ACTION_NAME_TO_INDEX: Dict[str, int] = {
    "left_click": 0,
    "right_click": 1,
    "double_click": 2,
    "mouse_move": 3,
    "move": 3,
    "hover": 3,
    "cursor_move": 3,
    "triple_click": 3,
    "drag_start": 4,
    "drag_end": 4,
    "drag": 4,
    "left_click_drag": 4,
    "scroll": 5,
    "mouse_scroll": 5,
    "wheel": 5,
    "hotkey": 6,
    "key": 6,
    "press": 6,
    "shortcut": 6,
    "type_text": 7,
    "type": 7,
    "input_text": 7,
    "write": 7,
    "press_key": 7,
    "wait": 8,
    "sleep": 8,
    "delay": 8,
    "stop": 9,
    "finish": 9,
    "done": 9,
    "complete": 9,
    "click": 0,
    "mouse_click": 0,
}


def normalize_open_source_action(raw_action: Any) -> int:
    """Normalizes heterogeneous action representations into discrete class index [0, 9]."""
    if isinstance(raw_action, int):
        return max(0, min(9, raw_action))
    if isinstance(raw_action, str):
        cleaned = raw_action.strip().lower().replace("-", "_").replace(" ", "_")
        return ACTION_NAME_TO_INDEX.get(cleaned, 0)
    return 0


def normalize_open_source_point(
    raw_point: Any,
    screen_width: float = 1920.0,
    screen_height: float = 1080.0,
) -> Tuple[float, float]:
    """Normalizes 2D coordinates into [0.0, 1.0] from pixel or float representations."""
    if raw_point is None:
        return (0.5, 0.5)
    if isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
        px, py = float(raw_point[0]), float(raw_point[1])
        if px > 1.0 or py > 1.0:
            return (
                max(0.0, min(1.0, px / max(screen_width, 1.0))),
                max(0.0, min(1.0, py / max(screen_height, 1.0))),
            )
        return (max(0.0, min(1.0, px)), max(0.0, min(1.0, py)))
    return (0.5, 0.5)


class Phase4DataMixtureDataset(Dataset):
    """Unified Phase 4 Data Mixture Dataset supporting:
    1. Golden Demonstration episodes (500-episode diagnostic stream).
    2. Open-source desktop grounding datasets (OS-Atlas, ShowUI-desktop, AgentNet).
    3. DAgger interactive recovery intervention pairs (hunting loop and popup recovery).
    """

    def __init__(
        self,
        golden_episodes: Optional[List[List[LeRobotV3TrajectoryEpisode]]] = None,
        open_source_records: Optional[List[Dict[str, Any]]] = None,
        dagger_interventions: Optional[List[Dict[str, Any]]] = None,
        target_size: Tuple[int, int] = (400, 640),
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.samples: List[Dict[str, Any]] = []

        # 1. Ingest Golden Demonstration episodes
        if golden_episodes:
            for ep in golden_episodes:
                for step in ep:
                    self.samples.append({
                        "source": "golden",
                        "image_tensor": step.image_tensor,
                        "image_bytes": step.image_bytes,
                        "action_kind": step.action_kind,
                        "target_xy": step.target_xy,
                        "cursor_pre_xy": step.cursor_pre_xy,
                        "task_instruction": step.task_instruction or step.subgoal_instruction,
                        "scroll_delta": step.scroll_delta,
                        "drag_trajectory": [(step.target_xy[0], step.target_xy[1])] * 16,
                    })

        # 2. Ingest Open-Source dataset records (OS-Atlas / ShowUI-desktop / AgentNet schemas)
        if open_source_records:
            for rec in open_source_records:
                source = rec.get("dataset", rec.get("source", "open_source"))
                # Coordinate extraction supporting point, bbox, click_point, coords
                raw_pt = rec.get("point") or rec.get("click_point") or rec.get("coords") or rec.get("target_coords")
                if raw_pt is None and "bbox" in rec and len(rec["bbox"]) >= 4:
                    b = rec["bbox"]
                    raw_pt = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                norm_pt = normalize_open_source_point(raw_pt)

                # Instruction extraction
                instr = (
                    rec.get("instruction")
                    or rec.get("query")
                    or rec.get("goal")
                    or rec.get("task_instruction")
                    or "Execute desktop control step"
                )

                # Action extraction
                act_raw = rec.get("action") or rec.get("action_type") or rec.get("action_kind", 0)
                act_idx = normalize_open_source_action(act_raw)

                # Cursor pre-xy
                pre_xy = normalize_open_source_point(rec.get("cursor_pre_xy", norm_pt))

                self.samples.append({
                    "source": str(source),
                    "image_tensor": rec.get("image_tensor", rec.get("image")),
                    "image_bytes": rec.get("image_bytes"),
                    "action_kind": act_idx,
                    "target_xy": norm_pt,
                    "cursor_pre_xy": pre_xy,
                    "task_instruction": str(instr),
                    "scroll_delta": rec.get("scroll_delta", (0.0, 0.0)),
                    "drag_trajectory": rec.get("drag_trajectory", [(norm_pt[0], norm_pt[1])] * 16),
                })

        # 3. Ingest DAgger recovery intervention pairs
        if dagger_interventions:
            for interv in dagger_interventions:
                raw_act = interv.get("action_type", interv.get("action_kind", 0))
                act_idx = normalize_open_source_action(raw_act)
                norm_pt = normalize_open_source_point(interv.get("target_coords", (0.5, 0.5)))
                self.samples.append({
                    "source": "dagger",
                    "image_tensor": interv.get("image_tensor", interv.get("image")),
                    "image_bytes": interv.get("image_bytes"),
                    "action_kind": act_idx,
                    "target_xy": norm_pt,
                    "cursor_pre_xy": normalize_open_source_point(interv.get("cursor_xy", (0.5, 0.5))),
                    "task_instruction": str(interv.get("goal", interv.get("reason", "Recover from failure"))),
                    "scroll_delta": (0.0, 0.0),
                    "drag_trajectory": [(norm_pt[0], norm_pt[1])] * 16,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        target_h, target_w = self.target_size

        img_raw = item.get("image_tensor")
        if isinstance(img_raw, torch.Tensor):
            if img_raw.dim() == 3:
                if img_raw.shape[1] != target_h or img_raw.shape[2] != target_w:
                    img_tensor = torch.nn.functional.interpolate(
                        img_raw.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                else:
                    img_tensor = img_raw
            else:
                img_tensor = torch.zeros((3, target_h, target_w), dtype=torch.float32)
        elif item.get("image_bytes"):
            try:
                import io
                from PIL import Image
                with Image.open(io.BytesIO(item["image_bytes"])) as pil_img:
                    if pil_img.size != (target_w, target_h):
                        pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
                    arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                    img_tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)
            except Exception:
                img_tensor = torch.zeros((3, target_h, target_w), dtype=torch.float32)
        else:
            img_tensor = torch.zeros((3, target_h, target_w), dtype=torch.float32)

        raw_traj = item.get("drag_trajectory") or [(0.5, 0.5)] * 16
        if isinstance(raw_traj, torch.Tensor):
            traj_tensor = raw_traj.float()
        else:
            traj_tensor = torch.tensor(list(raw_traj)[:16], dtype=torch.float32)
            if traj_tensor.shape[0] < 16:
                padding = traj_tensor[-1:].repeat(16 - traj_tensor.shape[0], 1)
                traj_tensor = torch.cat([traj_tensor, padding], dim=0)

        s_delta = item.get("scroll_delta", (0.0, 0.0))
        if isinstance(s_delta, torch.Tensor):
            scroll_tensor = s_delta.float()
        else:
            scroll_tensor = torch.tensor(list(s_delta)[:2], dtype=torch.float32)

        return {
            "image": img_tensor,
            "action_kind": torch.tensor(item["action_kind"], dtype=torch.long),
            "target_xy": torch.tensor(item["target_xy"], dtype=torch.float32),
            "cursor_pre_xy": torch.tensor(item["cursor_pre_xy"], dtype=torch.float32),
            "task_instruction": item["task_instruction"],
            "scroll_delta": scroll_tensor,
            "drag_trajectory": traj_tensor,
            "source": item["source"],
        }


def create_phase4_train_val_dataloaders(
    golden_episodes: Optional[List[List[LeRobotV3TrajectoryEpisode]]] = None,
    open_source_records: Optional[List[Dict[str, Any]]] = None,
    dagger_interventions: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 16,
    target_size: Tuple[int, int] = (400, 640),
    val_split_episodes: int = 50,
) -> Tuple[DataLoader, DataLoader]:
    """Builds high-throughput training and validation DataLoaders.

    Validation is strictly partitioned from Golden Demonstration episodes.
    Training mixes open-source dataset schemas, training golden streams, and DAgger interventions.
    """
    if golden_episodes is None:
        golden_episodes = generate_golden_diagnostic_dataset(num_episodes=100, steps_per_episode=5)

    val_eps = golden_episodes[:val_split_episodes]
    train_golden = golden_episodes[val_split_episodes:]

    train_ds = Phase4DataMixtureDataset(
        golden_episodes=train_golden,
        open_source_records=open_source_records,
        dagger_interventions=dagger_interventions,
        target_size=target_size,
    )

    val_ds = Phase4DataMixtureDataset(
        golden_episodes=val_eps,
        target_size=target_size,
    )

    def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        action_kinds = torch.stack([b["action_kind"] for b in batch], dim=0)
        target_xys = torch.stack([b["target_xy"] for b in batch], dim=0)
        cursor_pre_xys = torch.stack([b["cursor_pre_xy"] for b in batch], dim=0)
        instructions = [b["task_instruction"] for b in batch]
        scroll_deltas = torch.stack([b["scroll_delta"] for b in batch], dim=0)
        drag_trajectories = torch.stack([b["drag_trajectory"] for b in batch], dim=0)
        sources = [b["source"] for b in batch]

        return {
            "image": images,
            "action_kind": action_kinds,
            "target_xy": target_xys,
            "cursor_pre_xy": cursor_pre_xys,
            "task_instruction": instructions,
            "scroll_delta": scroll_deltas,
            "drag_trajectory": drag_trajectories,
            "source": sources,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader

