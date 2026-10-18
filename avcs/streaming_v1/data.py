"""Recording-grouped RGB sequence format and deterministic procedural fixtures."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SEQUENCE_SCHEMA = "avcs.streaming.sequence.v1"
SEQUENCE_SCHEMA_V2 = "avcs.streaming.sequence.v2"
SEMANTIC_EVENT_LABELS = (
    "stable",
    "pixel_noise_only",
    "motion",
    "appearance",
    "disappearance",
    "occlusion",
    "scene_transition",
    "unexpected_event",
)
_SPLITS = {"train", "validation", "test"}
_ANNOTATION_ARRAYS = {
    "change_masks",
    "motion_grid_displacement",
    "motion_mask",
    "semantic_event_labels",
    "object_id_masks",
    "occlusion_masks",
}
_V2_ANNOTATION_STATUSES = {
    "change_masks",
    "motion_grid_displacement",
    "motion_mask",
    "object_identity",
    "occlusion_events",
    "semantic_event_labels",
    "object_id_masks",
    "occlusion_masks",
}
_LEGACY_GENERATOR_SHA256 = {
    "84d9f09a30afc83df2619473fc761122c061a3545c27cccae8fcceac906a1937",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SequenceRecord:
    recording_id: str
    split: str
    path: str
    sha256: str
    source_id: str
    frame_count: int
    width: int
    height: int
    annotations: dict[str, str]
    provenance: dict[str, Any]


def validate_manifest(root: str | Path) -> dict[str, Any]:
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    schema = manifest.get("schema")
    if schema not in {SEQUENCE_SCHEMA, SEQUENCE_SCHEMA_V2}:
        raise ValueError("unsupported streaming sequence manifest schema.")
    if schema == SEQUENCE_SCHEMA_V2 and manifest.get("semantic_event_labels") != list(
        SEMANTIC_EVENT_LABELS
    ):
        raise ValueError("v2 manifest semantic event label taxonomy is missing or invalid.")
    generator_hash = manifest.get("generation", {}).get("generator_sha256")
    if (
        generator_hash is not None
        and generator_hash != sha256_file(Path(__file__).resolve())
        and generator_hash not in _LEGACY_GENERATOR_SHA256
    ):
        raise ValueError("procedural sequence generator source hash differs from manifest.")
    recordings = manifest.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        raise ValueError("manifest requires at least one recording.")
    seen_ids: set[str] = set()
    split_sources: dict[str, set[str]] = {split: set() for split in _SPLITS}
    split_recordings: dict[str, set[str]] = {split: set() for split in _SPLITS}
    for row in recordings:
        if not isinstance(row, dict):
            raise ValueError("recording entry must be an object.")
        record = SequenceRecord(**row)
        if record.recording_id in seen_ids:
            raise ValueError(f"duplicate recording id: {record.recording_id}")
        seen_ids.add(record.recording_id)
        if record.split not in _SPLITS:
            raise ValueError(f"unsupported split: {record.split}")
        if not record.source_id or not re.fullmatch(r"[0-9a-f]{64}", record.sha256):
            raise ValueError("recording source id or SHA-256 is invalid.")
        if not isinstance(record.annotations, dict) or not isinstance(record.provenance, dict):
            raise ValueError("recording annotations and provenance must be objects.")
        if schema == SEQUENCE_SCHEMA_V2:
            if set(record.annotations) != _V2_ANNOTATION_STATUSES:
                raise ValueError(
                    f"{record.recording_id}: v2 requires an explicit status for every annotation."
                )
            source_hash = record.provenance.get("source_sha256")
            if (
                not isinstance(source_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
                or source_hash != record.source_id
            ):
                raise ValueError(
                    f"{record.recording_id}: source_id must match verified source SHA-256."
                )
        if any(
            status not in {"available", "not_collected", "ambiguous"}
            for status in record.annotations.values()
        ):
            raise ValueError(f"{record.recording_id}: invalid annotation availability status.")
        relative = Path(record.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("recording paths must stay inside the dataset root.")
        recording_path = (directory / relative).resolve()
        if directory not in recording_path.parents or not recording_path.is_file():
            raise ValueError(f"recording file is missing or escapes dataset root: {record.path}")
        if sha256_file(recording_path) != record.sha256:
            raise ValueError(f"recording SHA-256 mismatch: {record.recording_id}")
        with np.load(recording_path, allow_pickle=False) as arrays:
            frames = arrays.get("frames")
            timestamps = arrays.get("timestamps")
            if (
                frames is None or frames.ndim != 4 or frames.shape[-1] != 3
                or frames.dtype != np.uint8
            ):
                raise ValueError(f"{record.recording_id}: frames must be uint8 (T,H,W,3).")
            if frames.shape[0] != record.frame_count or frames.shape[1:3] != (record.height, record.width):
                raise ValueError(f"{record.recording_id}: frame dimensions do not match manifest.")
            if timestamps is None or timestamps.shape != (record.frame_count,):
                raise ValueError(f"{record.recording_id}: timestamps must have one value per frame.")
            if not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
                raise ValueError(f"{record.recording_id}: timestamps must be finite and strictly increasing.")
            for name, label in record.annotations.items():
                if label == "available" and name not in arrays:
                    raise ValueError(f"{record.recording_id}: declared annotation {name!r} is missing.")
                if name not in _ANNOTATION_ARRAYS:
                    if label == "available":
                        raise ValueError(
                            f"{record.recording_id}: unsupported available annotation {name!r}."
                        )
                    continue
                if label != "available" and name in arrays:
                    raise ValueError(
                        f"{record.recording_id}: unavailable annotation {name!r} "
                        "must not be stored as a training target."
                    )
                if label != "available":
                    continue
                shape = {
                    "change_masks": (record.frame_count - 1, record.height, record.width),
                    "motion_grid_displacement": (
                        record.frame_count - 1,
                        record.height,
                        record.width,
                        2,
                    ),
                    "motion_mask": (record.frame_count - 1, record.height, record.width),
                    "semantic_event_labels": (
                        record.frame_count - 1,
                        len(SEMANTIC_EVENT_LABELS),
                    ),
                    "object_id_masks": (record.frame_count, record.height, record.width),
                    "occlusion_masks": (record.frame_count, record.height, record.width),
                }[name]
                values = arrays[name]
                if values.shape != shape:
                    raise ValueError(
                        f"{record.recording_id}: annotation {name!r} has an invalid shape."
                    )
                if name in {"change_masks", "motion_mask", "occlusion_masks"} and (
                    values.dtype not in {np.dtype(np.uint8), np.dtype(np.bool_)}
                    or not np.isin(values, (0, 1)).all()
                ):
                    raise ValueError(f"{record.recording_id}: {name} must be a binary mask.")
                if name == "motion_grid_displacement" and (
                    not np.issubdtype(values.dtype, np.number)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"{record.recording_id}: motion labels must be finite numeric values.")
                if name == "semantic_event_labels" and (
                    values.dtype != np.dtype(np.int8)
                    or not np.isin(values, (-1, 0, 1)).all()
                ):
                    raise ValueError(
                        f"{record.recording_id}: event labels must be int8 -1/0/1 "
                        "(unknown/negative/positive)."
                    )
                if name == "object_id_masks" and (
                    not np.issubdtype(values.dtype, np.signedinteger)
                    or (values < -1).any()
                ):
                    raise ValueError(
                        f"{record.recording_id}: object IDs must be signed integers "
                        "with -1 reserved for unknown pixels."
                    )
        split_sources[record.split].add(record.source_id)
        split_recordings[record.split].add(record.recording_id)
    if any(not split_recordings[split] for split in _SPLITS):
        raise ValueError("train, validation, and test each require at least one recording.")
    for first in _SPLITS:
        for second in _SPLITS:
            if first < second and split_sources[first] & split_sources[second]:
                raise ValueError(f"source leakage across {first}/{second} splits.")
    return {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "recording_counts": {split: len(split_recordings[split]) for split in sorted(_SPLITS)},
        "source_counts": {split: len(split_sources[split]) for split in sorted(_SPLITS)},
        "rights_status": manifest.get("rights", {}).get("status", "UNRECORDED"),
        "recording_count": len(recordings),
    }


class SequenceDataset(Dataset[dict[str, Any]]):
    """Load a whole timestamped recording; split assignment is source-level."""

    def __init__(self, root: str | Path, split: str) -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}.")
        self.root = Path(root).resolve()
        self.validation = validate_manifest(self.root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.records = [
            SequenceRecord(**row) for row in manifest["recordings"] if row["split"] == split
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with np.load(self.root / record.path, allow_pickle=False) as arrays:
            result: dict[str, Any] = {
                "recording_id": record.recording_id,
                "source_id": record.source_id,
                "split": record.split,
                "frames": torch.from_numpy(arrays["frames"].copy()).permute(0, 3, 1, 2).float() / 255.0,
                "timestamps": torch.from_numpy(arrays["timestamps"].copy()).double(),
                "annotations": record.annotations,
            }
            for name in record.annotations:
                if record.annotations[name] == "available":
                    result[name] = torch.from_numpy(arrays[name].copy())
            return result


def generate_procedural_sequences(
    output_dir: str | Path,
    *,
    seed: int = 731,
    train_recordings: int = 12,
    validation_recordings: int = 4,
    test_recordings: int = 4,
    frames_per_recording: int = 24,
    width: int = 64,
    height: int = 64,
    fps: float = 10.0,
) -> dict[str, Any]:
    """Generate deterministic moving-shape sequences for pipeline experiments only."""
    if min(train_recordings, validation_recordings, test_recordings) < 1:
        raise ValueError("each source-level split needs at least one recording.")
    if frames_per_recording < 4 or min(width, height) < 32 or fps <= 0:
        raise ValueError("need >=4 frames, dimensions >=32, and positive fps.")
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts = {
        "train": train_recordings,
        "validation": validation_recordings,
        "test": test_recordings,
    }
    records: list[dict[str, Any]] = []
    global_index = 0
    for split, count in counts.items():
        for _ in range(count):
            local_seed = int(rng.integers(0, 2**31 - 1))
            sequence_rng = np.random.default_rng(local_seed)
            recording_id = f"{split}-{global_index:04d}"
            global_index += 1
            source_id = hashlib.sha256(f"avcs-streaming-v1-source-{local_seed}".encode()).hexdigest()
            frames, change_masks, motion, motion_mask = _make_recording(
                sequence_rng, frames_per_recording, width, height
            )
            timestamps = np.arange(frames_per_recording, dtype=np.float64) / fps
            path = Path("recordings") / f"{recording_id}.npz"
            full_path = root / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                full_path,
                frames=frames,
                timestamps=timestamps,
                change_masks=change_masks,
                motion_grid_displacement=motion,
                motion_mask=motion_mask,
            )
            records.append({
                "recording_id": recording_id,
                "split": split,
                "path": str(path),
                "sha256": sha256_file(full_path),
                "source_id": source_id,
                "frame_count": frames_per_recording,
                "width": width,
                "height": height,
                "annotations": {
                    "change_masks": "available",
                    "motion_grid_displacement": "available",
                    "motion_mask": "available",
                    "object_identity": "not_collected",
                    "occlusion_events": "not_collected",
                },
                "provenance": {
                    "kind": "procedurally_generated",
                    "generator_seed": local_seed,
                    "external_assets": False,
                    "rights_evidence": "project-owner review and signed manifest-bound approval pending",
                },
            })
    manifest = {
        "schema": SEQUENCE_SCHEMA,
        "dataset_version": "avcs-streaming-v1-synthetic-0.1.0",
        "created_by": "avcs.streaming_v1.data.generate_procedural_sequences",
        "generation": {
            "seed": seed,
            "recordings_per_split": counts,
            "frames_per_recording": frames_per_recording,
            "width": width,
            "height": height,
            "nominal_fps": fps,
            "scene": "textured background, translating colored rectangle, occasional occluding bar",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "rights": {
            "status": "PENDING_OWNER_ATTESTATION",
            "external_assets": False,
            "internal_training": "PENDING",
            "distribution": "NOT_AUTHORIZED",
        },
        "annotation_policy": {
            "change_masks": "pixel-difference masks generated from known procedural frames",
            "motion_grid_displacement": "ground-truth translating-object displacement in normalized grid units",
            "missing_or_ambiguous": "represented explicitly in each recording annotation-status map",
        },
        "recordings": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_manifest(root)


def _make_recording(
    rng: np.random.Generator,
    frames_count: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    background = rng.integers(15, 75, size=(height, width, 3), dtype=np.uint8)
    texture = rng.integers(0, 9, size=(height, width, 1), dtype=np.uint8)
    background = np.clip(background[None, :, :, :] + texture[None, :, :, :], 0, 255).astype(np.uint8)
    frames = np.repeat(background, frames_count, axis=0)
    side = int(rng.integers(8, 15))
    x = int(rng.integers(0, width - side))
    y = int(rng.integers(0, height - side))
    dx = int(rng.choice([-2, -1, 1, 2]))
    dy = int(rng.choice([-2, -1, 1, 2]))
    color = rng.integers(150, 255, size=3, dtype=np.uint8)
    bar_y = int(rng.integers(height // 3, 2 * height // 3))
    bar_x0 = int(rng.integers(0, width // 3))
    bar_x1 = min(width, bar_x0 + int(rng.integers(12, 25)))
    positions: list[tuple[int, int]] = []
    for t in range(frames_count):
        frames[t, y:min(height, y + side), x:min(width, x + side)] = color
        positions.append((x, y))
        if frames_count // 3 <= t < frames_count // 2:
            frames[t, bar_y:min(height, bar_y + 4), bar_x0:bar_x1] = (230, 220, 40)
        x += dx
        y += dy
        if x < 0 or x + side >= width:
            dx = -dx
            x = min(max(x, 0), width - side)
        if y < 0 or y + side >= height:
            dy = -dy
            y = min(max(y, 0), height - side)
    change_masks = np.any(frames[1:] != frames[:-1], axis=-1)
    motion = np.zeros((frames_count - 1, height, width, 2), dtype=np.float32)
    motion_mask = np.zeros((frames_count - 1, height, width), dtype=np.uint8)
    for t, ((x0, y0), (x1, y1)) in enumerate(zip(positions[:-1], positions[1:])):
        dx_actual = x1 - x0
        dy_actual = y1 - y0
        motion[t, y1:y1 + side, x1:x1 + side, 0] = -2.0 * dx_actual / max(1, width - 1)
        motion[t, y1:y1 + side, x1:x1 + side, 1] = -2.0 * dy_actual / max(1, height - 1)
        visible_x1, visible_x2 = x1, min(width, x1 + side)
        visible_y1, visible_y2 = y1, min(height, y1 + side)
        motion_mask[t, visible_y1:visible_y2, visible_x1:visible_x2] = 1
        if frames_count // 3 <= t + 1 < frames_count // 2 and y1 <= bar_y + 4 and y1 + side > bar_y:
            occlude_y1, occlude_y2 = max(y1, bar_y), min(y1 + side, bar_y + 4)
            motion_mask[t, occlude_y1:occlude_y2, max(x1, bar_x0):min(x1 + side, bar_x1)] = 0
    return frames, change_masks.astype(np.uint8), motion, motion_mask
