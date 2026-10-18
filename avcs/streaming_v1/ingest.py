"""Import permission-documented timestamped RGB sequences into the dataset format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from avcs.streaming_v1.data import (
    SEMANTIC_EVENT_LABELS,
    SEQUENCE_SCHEMA_V2,
    sha256_file,
    validate_manifest,
)

INGEST_INDEX_SCHEMA = "avcs.streaming.ingest-index.v2"
_ARRAY_NAMES = {
    "frames",
    "timestamps",
    "change_masks",
    "motion_grid_displacement",
    "motion_mask",
    "semantic_event_labels",
    "object_id_masks",
    "occlusion_masks",
}
_ANNOTATION_SHAPES = {
    "change_masks": lambda shape: (shape[0] - 1, *shape[1:3]),
    "motion_mask": lambda shape: (shape[0] - 1, *shape[1:3]),
    "motion_grid_displacement": lambda shape: (shape[0] - 1, *shape[1:3], 2),
    "semantic_event_labels": lambda shape: (
        shape[0] - 1,
        len(SEMANTIC_EVENT_LABELS),
    ),
    "object_id_masks": lambda shape: shape,
    "occlusion_masks": lambda shape: shape,
}
_MASK_ANNOTATIONS = {"change_masks", "motion_mask", "occlusion_masks"}
_ANNOTATION_STATUSES = {"available", "not_collected", "ambiguous"}
_ANNOTATION_NAMES = {
    "change_masks",
    "motion_grid_displacement",
    "motion_mask",
    "object_identity",
    "occlusion_events",
    "semantic_event_labels",
    "object_id_masks",
    "occlusion_masks",
}


def ingest_sequence_index(
    index_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Copy source-verified, timestamped RGB NPZs into a pending-rights manifest.

    The original source is hashed in place and is not copied. A separate signed
    approval for the resulting manifest is still required before training.
    """
    source_index = Path(index_path).resolve()
    if not source_index.is_file():
        raise FileNotFoundError(source_index)
    source_root = source_index.parent
    index = json.loads(source_index.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("schema") != INGEST_INDEX_SCHEMA:
        raise ValueError(f"ingest index schema must be {INGEST_INDEX_SCHEMA!r}.")
    version = index.get("dataset_version")
    rows = index.get("recordings")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("ingest index requires a non-empty dataset_version.")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ingest index requires at least one recording.")

    prepared: list[tuple[dict[str, Any], Path, dict[str, np.ndarray]]] = []
    seen_ids: set[str] = set()
    split_sources: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each ingest recording must be an object.")
        required = {
            "recording_id",
            "source_id",
            "split",
            "path",
            "source_path",
            "annotations",
            "provenance",
        }
        if set(row) != required:
            raise ValueError("ingest recording has missing or unexpected fields.")
        recording_id = row["recording_id"]
        if (
            not isinstance(recording_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", recording_id)
            or recording_id in seen_ids
        ):
            raise ValueError("recording_id must be unique and path-safe.")
        seen_ids.add(recording_id)
        if (
            not isinstance(row["source_id"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["source_id"])
        ):
            raise ValueError(f"{recording_id}: source_id must be a lowercase source SHA-256.")
        if not isinstance(row["split"], str) or row["split"] not in split_sources:
            raise ValueError(f"{recording_id}: split must be train, validation, or test.")
        split_sources[row["split"]].add(row["source_id"])
        if not isinstance(row["path"], str):
            raise ValueError(f"{recording_id}: source path must be a relative string.")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("ingest source paths must stay inside the index directory.")
        source_path = (source_root / relative).resolve()
        if source_root not in source_path.parents or not source_path.is_file():
            raise ValueError(f"ingest source file is missing or escapes index directory: {relative}")
        if not isinstance(row["source_path"], str):
            raise ValueError(f"{recording_id}: original source path must be a relative string.")
        source_relative = Path(row["source_path"])
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise ValueError("original source paths must stay inside the index directory.")
        original_path = (source_root / source_relative).resolve()
        if source_root not in original_path.parents or not original_path.is_file():
            raise ValueError(
                f"{recording_id}: original source file is missing or escapes index directory."
            )
        source_digest = sha256_file(original_path)
        if row["source_id"] != source_digest:
            raise ValueError(
                f"{recording_id}: source_id must equal SHA-256 of the original source file."
            )

        provenance = row["provenance"]
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("rights_evidence"), str)
            or not provenance["rights_evidence"].strip()
            or not isinstance(provenance.get("permitted_uses"), list)
            or "internal_training" not in provenance["permitted_uses"]
        ):
            raise PermissionError(
                f"{recording_id}: record documented internal-training rights before ingestion."
            )
        annotations = row["annotations"]
        if (
            not isinstance(annotations, dict)
            or set(annotations) != _ANNOTATION_NAMES
            or any(
                not isinstance(status, str) or status not in _ANNOTATION_STATUSES
                for status in annotations.values()
            )
        ):
            raise ValueError(f"{recording_id}: annotation states must be explicit.")
        with np.load(source_path, allow_pickle=False) as archive:
            if not {"frames", "timestamps"}.issubset(archive.files):
                raise ValueError(f"{recording_id}: NPZ requires frames and timestamps.")
            arrays = {
                name: archive[name].copy()
                for name in archive.files
                if name in _ARRAY_NAMES
            }
        frames = arrays["frames"]
        timestamps = arrays["timestamps"]
        if (
            frames.dtype != np.uint8
            or frames.ndim != 4
            or frames.shape[0] < 2
            or frames.shape[-1] != 3
            or min(frames.shape[1:3]) < 32
        ):
            raise ValueError(f"{recording_id}: frames must be uint8 (T>=2,H>=32,W>=32,3).")
        if (
            timestamps.shape != (frames.shape[0],)
            or timestamps.dtype != np.float64
            or not np.isfinite(timestamps).all()
            or (np.diff(timestamps) <= 0).any()
        ):
            raise ValueError(f"{recording_id}: timestamps must be finite and strictly increasing.")
        for name, status in annotations.items():
            if status == "available":
                if name not in arrays:
                    raise ValueError(f"{recording_id}: {name!r} is declared available but absent.")
                if name in _ANNOTATION_SHAPES and arrays[name].shape != _ANNOTATION_SHAPES[name](frames.shape):
                    raise ValueError(f"{recording_id}: {name!r} has an invalid temporal shape.")
                if name in _MASK_ANNOTATIONS and (
                    arrays[name].dtype not in {np.dtype(np.uint8), np.dtype(np.bool_)}
                    or not np.isin(arrays[name], (0, 1)).all()
                ):
                    raise ValueError(f"{recording_id}: {name!r} must be a binary mask.")
                if name == "semantic_event_labels" and (
                    arrays[name].dtype != np.dtype(np.int8)
                    or not np.isin(arrays[name], (-1, 0, 1)).all()
                ):
                    raise ValueError(
                        f"{recording_id}: semantic event labels must be int8 "
                        "(-1 unknown, 0 absent, 1 present)."
                    )
                if name == "object_id_masks" and (
                    not np.issubdtype(arrays[name].dtype, np.signedinteger)
                    or (arrays[name] < -1).any()
                ):
                    raise ValueError(
                        f"{recording_id}: object ID masks must be signed integers "
                        "with -1 reserved for unknown pixels."
                    )
                if name == "motion_grid_displacement" and (
                    not np.issubdtype(arrays[name].dtype, np.number)
                    or not np.isfinite(arrays[name]).all()
                ):
                    raise ValueError(f"{recording_id}: motion labels must be finite numeric values.")
        row = {
            **row,
            "provenance": {
                **provenance,
                "source_sha256": source_digest,
            },
        }
        prepared.append((row, source_path, arrays))

    if any(not split_sources[split] for split in split_sources):
        raise ValueError("ingest index requires train, validation, and test sources.")
    for first, first_sources in split_sources.items():
        for second, second_sources in split_sources.items():
            if first < second and first_sources & second_sources:
                raise ValueError(f"source leakage across {first}/{second} splits.")

    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row, _source_path, arrays in prepared:
        relative = Path("recordings") / f"{row['recording_id']}.npz"
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        frames = arrays["frames"]
        records.append({
            "recording_id": row["recording_id"],
            "source_id": row["source_id"],
            "split": row["split"],
            "path": str(relative),
            "sha256": sha256_file(target),
            "frame_count": int(frames.shape[0]),
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "annotations": row["annotations"],
            "provenance": row["provenance"],
        })

    manifest = {
        "schema": SEQUENCE_SCHEMA_V2,
        "dataset_version": version,
        "created_by": "avcs.streaming_v1.ingest.ingest_sequence_index",
        "semantic_event_labels": list(SEMANTIC_EVENT_LABELS),
        "rights": {
            "status": "PENDING_OWNER_ATTESTATION",
            "internal_training": "PENDING",
            "distribution": "NOT_AUTHORIZED",
            "source_rights_documented": True,
        },
        "annotation_policy": index.get("annotation_policy", {}),
        "recordings": records,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validate_manifest(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(ingest_sequence_index(args.index, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
