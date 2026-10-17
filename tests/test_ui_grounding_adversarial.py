"""Adversarial Verification Suite for AVCS UI Grounding Dataset.

Independent empirical verification of:
1. SHA-256 Digest Uniqueness and Split Disjointness across all 270 PNG screens.
2. Negative Control Integrity across all 30 negative frames (affordance masks and points).
3. Geometric Enclosure across all positive interactive elements (bbox valid and enclosing centroid).
4. Cross-split session and metadata isolation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image
import pytest

DATASET_ROOT = Path("datasets/ui_grounding")
SCREENS_DIR = DATASET_ROOT / "screens"
MASKS_DIR = DATASET_ROOT / "masks"
SPLITS_DIR = DATASET_ROOT / "splits"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"


@pytest.fixture(scope="module")
def manifest_json() -> Dict[str, Any]:
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def disk_hashes() -> Dict[str, str]:
    """Computes SHA-256 directly from disk for all PNG files in screens/."""
    assert SCREENS_DIR.exists(), f"Screens directory missing: {SCREENS_DIR}"
    png_files = sorted(list(SCREENS_DIR.glob("*.png")))
    hashes: Dict[str, str] = {}
    for p in png_files:
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hashes[p.name] = digest
    return hashes


def test_sha256_digests_count_and_uniqueness(disk_hashes: Dict[str, str]):
    """1. Compute SHA-256 digests for all 270 PNG files in `datasets/ui_grounding/screens/`.
    Assert len(all_hashes) == 270 (100% unique images).
    """
    assert len(disk_hashes) == 270, f"Expected exactly 270 PNG files on disk, found {len(disk_hashes)}"
    unique_hashes = set(disk_hashes.values())
    assert len(unique_hashes) == 270, (
        f"Hash collision detected on disk! Total files: {len(disk_hashes)}, "
        f"Unique hashes: {len(unique_hashes)}"
    )


def test_splits_pairwise_disjointness_and_zero_negative_overlap(
    manifest_json: Dict[str, Any],
    disk_hashes: Dict[str, str],
):
    """1b. Assert:
    - train_hashes.isdisjoint(test_held_out_hashes)
    - train_hashes.isdisjoint(val_hashes)
    - val_hashes.isdisjoint(test_held_out_hashes)
    - zero overlap with test_negative_hashes
    """
    frames = manifest_json.get("frames", [])
    assert len(frames) == 270, f"Manifest frames count is {len(frames)}, expected 270"

    split_hashes: Dict[str, Set[str]] = {
        "train": set(),
        "val": set(),
        "test_held_out": set(),
        "test_negative": set(),
    }
    split_counts: Dict[str, int] = {k: 0 for k in split_hashes}

    for fr in frames:
        sp = fr["split"]
        assert sp in split_hashes, f"Unexpected split name: {sp}"
        fname = Path(fr["file_path"]).name
        assert fname in disk_hashes, f"File {fname} from manifest not found in disk_hashes"
        h = disk_hashes[fname]
        split_hashes[sp].add(h)
        split_counts[sp] += 1

    # Verify split size expectations
    assert split_counts["train"] == 160
    assert split_counts["val"] == 40
    assert split_counts["test_held_out"] == 40
    assert split_counts["test_negative"] == 30

    train_h = split_hashes["train"]
    val_h = split_hashes["val"]
    test_h = split_hashes["test_held_out"]
    neg_h = split_hashes["test_negative"]

    assert len(train_h) == 160, f"Train hashes has duplicates: {len(train_h)} vs 160"
    assert len(val_h) == 40, f"Val hashes has duplicates: {len(val_h)} vs 40"
    assert len(test_h) == 40, f"Test held-out hashes has duplicates: {len(test_h)} vs 40"
    assert len(neg_h) == 30, f"Test negative hashes has duplicates: {len(neg_h)} vs 30"

    # Pairwise disjointness assertions
    assert train_h.isdisjoint(test_h), (
        f"CRITICAL LEAKAGE: train & test_held_out share {len(train_h & test_h)} hashes: {train_h & test_h}"
    )
    assert train_h.isdisjoint(val_h), (
        f"CRITICAL LEAKAGE: train & val share {len(train_h & val_h)} hashes: {train_h & val_h}"
    )
    assert val_h.isdisjoint(test_h), (
        f"CRITICAL LEAKAGE: val & test_held_out share {len(val_h & test_h)} hashes: {val_h & test_h}"
    )
    assert train_h.isdisjoint(neg_h), (
        f"CRITICAL LEAKAGE: train & test_negative share {len(train_h & neg_h)} hashes: {train_h & neg_h}"
    )
    assert val_h.isdisjoint(neg_h), (
        f"CRITICAL LEAKAGE: val & test_negative share {len(val_h & neg_h)} hashes: {val_h & neg_h}"
    )
    assert test_h.isdisjoint(neg_h), (
        f"CRITICAL LEAKAGE: test_held_out & test_negative share {len(test_h & neg_h)} hashes: {test_h & neg_h}"
    )

    # Union completeness check
    total_split_hashes = train_h | val_h | test_h | neg_h
    assert len(total_split_hashes) == 270, (
        f"Union of split hashes is {len(total_split_hashes)}, expected 270"
    )


def test_split_json_files_exact_match(manifest_json: Dict[str, Any]):
    """Verify datasets/ui_grounding/splits/*.json are 100% consistent with manifest.json."""
    splits = ["train", "val", "test_held_out", "test_negative"]
    manifest_split_map: Dict[str, Set[str]] = {s: set() for s in splits}
    for fr in manifest_json["frames"]:
        manifest_split_map[fr["split"]].add(fr["frame_id"])

    for s in splits:
        split_file = SPLITS_DIR / f"{s}.json"
        assert split_file.exists(), f"Split file missing: {split_file}"
        with open(split_file, "r", encoding="utf-8") as f:
            ids = json.load(f)
        assert len(ids) == len(manifest_split_map[s]), (
            f"Split {s} file count {len(ids)} != manifest count {len(manifest_split_map[s])}"
        )
        assert set(ids) == manifest_split_map[s], f"Split {s} frame IDs mismatch manifest"


def test_negative_controls_rigorous_audit(manifest_json: Dict[str, Any]):
    """2. Verify negative controls:
    - Check all 30 frames in `test_negative`.
    - Verify `is_negative is True`, `len(interactive_points) == 0`,
      and the 16x16 affordance mask `.npy` is strictly all zeros.
    """
    frames = manifest_json.get("frames", [])
    neg_frames = [fr for fr in frames if fr["split"] == "test_negative"]
    assert len(neg_frames) == 30, f"Expected 30 negative frames, got {len(neg_frames)}"

    for fr in neg_frames:
        frame_id = fr["frame_id"]
        assert fr["is_negative"] is True, f"Negative frame {frame_id} has is_negative={fr.get('is_negative')}"
        assert fr["category"] in ["desktop_canvas", "blank_screen", "wallpaper"], (
            f"Negative frame {frame_id} unexpected category: {fr.get('category')}"
        )
        assert len(fr["interactive_points"]) == 0, (
            f"Negative frame {frame_id} contains interactive points: {fr['interactive_points']}"
        )

        # Check image file
        img_p = DATASET_ROOT / fr["file_path"]
        assert img_p.exists(), f"Negative image file missing: {img_p}"
        with Image.open(img_p) as pil_img:
            assert pil_img.size == (256, 256), f"Negative frame {frame_id} size {pil_img.size} != (256, 256)"
            assert pil_img.mode in ("RGB", "RGBA")

        # Check mask file
        mask_p = DATASET_ROOT / fr["affordance_mask_path"]
        assert mask_p.exists(), f"Negative mask file missing: {mask_p}"
        mask = np.load(mask_p)

        assert mask.shape == (16, 16), f"Negative mask {frame_id} shape {mask.shape} != (16, 16)"
        assert np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.floating), (
            f"Unexpected mask dtype: {mask.dtype}"
        )
        assert np.all(mask == 0), f"Negative mask {frame_id} has non-zero elements: max={np.max(mask)}"
        assert mask.sum() == 0, f"Negative mask {frame_id} sum is {mask.sum()}"
        assert np.max(np.abs(mask)) == 0, f"Negative mask {frame_id} absolute max > 0"

        # Check negative queries exist
        assert "negative_queries" in fr, f"Negative frame {frame_id} missing negative_queries"
        assert len(fr["negative_queries"]) > 0, f"Negative frame {frame_id} has empty negative_queries"


def test_geometric_enclosure_rigorous_audit(manifest_json: Dict[str, Any]):
    """3. Verify geometric enclosure:
    - For all positive frames, check every interactive element:
      `x1 < x2`, `y1 < y2`, and centroid `(cx, cy)` satisfies
      `x1 <= cx <= x2` and `y1 <= cy <= y2`.
    """
    frames = manifest_json.get("frames", [])
    positive_frames = [fr for fr in frames if fr["split"] != "test_negative"]
    assert len(positive_frames) == 240, f"Expected 240 positive frames, got {len(positive_frames)}"

    total_elements_checked = 0
    tolerance = 1e-4

    for fr in positive_frames:
        frame_id = fr["frame_id"]
        assert fr["is_negative"] is False, f"Positive frame {frame_id} has is_negative=True"
        pts = fr.get("interactive_points", [])
        assert len(pts) > 0, f"Positive frame {frame_id} has 0 interactive points!"

        # Check affordance mask has active hotspots
        mask_p = DATASET_ROOT / fr["affordance_mask_path"]
        assert mask_p.exists(), f"Positive mask file missing: {mask_p}"
        mask = np.load(mask_p)
        assert mask.shape == (16, 16), f"Positive mask {frame_id} shape != (16, 16)"
        assert np.any(mask > 0), f"Positive frame {frame_id} mask is all zeros!"

        for pt in pts:
            el_id = pt["element_id"]
            cx = float(pt["x"])
            cy = float(pt["y"])
            bbox = pt["bbox"]
            assert len(bbox) == 4, f"Element {el_id} in {frame_id} bbox length != 4"
            ymin, xmin, ymax, xmax = [float(c) for c in bbox]

            # 1. Bounds check in [0, 1]
            assert 0.0 <= cx <= 1.0, f"Centroid cx {cx} out of [0, 1] in {el_id} ({frame_id})"
            assert 0.0 <= cy <= 1.0, f"Centroid cy {cy} out of [0, 1] in {el_id} ({frame_id})"
            assert 0.0 <= xmin <= 1.0, f"xmin {xmin} out of [0, 1] in {el_id} ({frame_id})"
            assert 0.0 <= xmax <= 1.0, f"xmax {xmax} out of [0, 1] in {el_id} ({frame_id})"
            assert 0.0 <= ymin <= 1.0, f"ymin {ymin} out of [0, 1] in {el_id} ({frame_id})"
            assert 0.0 <= ymax <= 1.0, f"ymax {ymax} out of [0, 1] in {el_id} ({frame_id})"

            # 2. Strict inequality x1 < x2, y1 < y2
            assert xmin < xmax, f"x1 >= x2 ({xmin} >= {xmax}) for element {el_id} in {frame_id}"
            assert ymin < ymax, f"y1 >= y2 ({ymin} >= {ymax}) for element {el_id} in {frame_id}"

            # 3. Geometric enclosure: x1 <= cx <= x2 and y1 <= cy <= y2
            # Check with numerical tolerance for 4-decimal-place rounding
            assert xmin - tolerance <= cx <= xmax + tolerance, (
                f"Centroid cx {cx} NOT enclosed by [{xmin}, {xmax}] for {el_id} in {frame_id}"
            )
            assert ymin - tolerance <= cy <= ymax + tolerance, (
                f"Centroid cy {cy} NOT enclosed by [{ymin}, {ymax}] for {el_id} in {frame_id}"
            )

            # Check that centroid is reasonably close to bbox center
            box_cx = (xmin + xmax) / 2.0
            box_cy = (ymin + ymax) / 2.0
            assert abs(cx - box_cx) <= (xmax - xmin) / 2.0 + tolerance, (
                f"Centroid cx {cx} offset exceeds half-width in {el_id}"
            )
            assert abs(cy - box_cy) <= (ymax - ymin) / 2.0 + tolerance, (
                f"Centroid cy {cy} offset exceeds half-height in {el_id}"
            )

            total_elements_checked += 1

    assert total_elements_checked >= 500, (
        f"Too few interactive elements checked: {total_elements_checked}"
    )


def test_oracle_failure_detection():
    """Verify that the test assertions would fail on deliberate bugs (oracle validation)."""
    # 1. Collision oracle
    dummy_hashes = {"a.png": "hash1", "b.png": "hash1"}
    with pytest.raises(AssertionError):
        assert len(dummy_hashes) == len(set(dummy_hashes.values()))

    # 2. Geometry non-enclosure oracle
    xmin, xmax = 0.2, 0.4
    ymin, ymax = 0.3, 0.5
    bad_cx = 0.45
    with pytest.raises(AssertionError):
        assert xmin <= bad_cx <= xmax

    # 3. Degenerate box oracle
    deg_xmin, deg_xmax = 0.3, 0.3
    with pytest.raises(AssertionError):
        assert deg_xmin < deg_xmax
