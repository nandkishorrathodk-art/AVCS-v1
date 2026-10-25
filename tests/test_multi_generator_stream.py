"""Comprehensive Unit and Verification Tests for Multi-Generator Stream & Unified Manifest.

Verifies:
1. Unified Draft-07 annotation manifest schema compliance and validation.
2. Canonical loaders for `datasets/ui_grounding` and `datasets/cleanroom_system1_v1`.
3. Dual coordinate box consistency (eliminates coordinate inversion bugs).
4. Collation into canonical `UnifiedBatchDict` matching PROJECT.md §6.2.
5. Arm D2 balanced sampling (50% UI / 50% Cleanroom expected per batch, epoch = N_ui + N_cleanroom).
6. Zero template, layout family, session, and image hash leakage between train and test splits.
7. Explicit negative / blank frame handling (has_target == 0.0, zero bounding boxes).
8. Serialization and round-trip fidelity.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from avcs.dataset.multi_generator_stream import (
    GENERATOR_ID_CLEANROOM,
    GENERATOR_ID_UI,
    MultiGeneratorDataset,
    MultiGeneratorDataStream,
    UnifiedBatchDict,
    calculate_arm_d2_weights,
    create_multi_generator_dataloader,
    unified_batch_collate,
    verify_layout_leakage_isolation,
)
from avcs.dataset.unified_manifest import (
    DRAFT_07_UNIFIED_MANIFEST_SCHEMA,
    GENERATOR_CLEANROOM,
    GENERATOR_UI_GROUNDING,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
    SPLIT_TRAIN,
    SPLIT_VAL,
    UnifiedAnnotationFrame,
    UnifiedAnnotationManifest,
    UnifiedObjectAnnotation,
    build_combined_manifest,
    load_cleanroom_manifest,
    load_ui_grounding_manifest,
    load_unified_manifest,
    validate_unified_manifest_dict,
)

UI_MANIFEST_PATH = Path("datasets/ui_grounding/manifest.json")
CLEANROOM_DIR = Path("datasets/cleanroom_system1_v1")


@pytest.fixture(scope="module")
def combined_manifest() -> UnifiedAnnotationManifest:
    """Fixture loading the combined manifest for testing."""
    return build_combined_manifest(
        ui_manifest_path=UI_MANIFEST_PATH,
        cleanroom_dir=CLEANROOM_DIR,
    )


# =========================================================================
# 1. Unified Annotation Manifest Schema & Validation Tests
# =========================================================================

def test_draft_07_schema_properties():
    """Verify Draft-07 schema dictionary definition and core properties."""
    schema = DRAFT_07_UNIFIED_MANIFEST_SCHEMA
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["title"] == "AVCS_Unified_UI_Objectness_Manifest"
    assert schema["version"] == "2.0.0"
    assert "frames" in schema["required"]
    assert "coordinate_convention" in schema["required"]


def test_validate_unified_manifest_dict_success(combined_manifest: UnifiedAnnotationManifest):
    """Combined manifest must validate cleanly against Draft-07 schema checks."""
    manifest_dict = combined_manifest.to_dict()
    is_valid, errors = validate_unified_manifest_dict(manifest_dict)
    assert is_valid is True, f"Validation failed with errors: {errors}"
    assert len(errors) == 0


def test_validate_unified_manifest_dict_catches_errors():
    """Validator must detect corrupted structures, coordinate inversions, and negative frame violations."""
    # 1. Missing required field
    bad_manifest = {"title": "Test", "frames": []}
    is_valid, errors = validate_unified_manifest_dict(bad_manifest)
    assert is_valid is False
    assert any("Missing required manifest field" in e for e in errors)

    # 2. Inverted coordinates (x1 > x2)
    corrupted = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AVCS_Unified_UI_Objectness_Manifest",
        "version": "2.0.0",
        "coordinate_convention": "normalized_xyxy_origin_top_left",
        "frames": [
            {
                "frame_id": "frame_corrupted",
                "generator_source": "ui_grounding",
                "image_path": "screens/test.png",
                "width": 256,
                "height": 256,
                "split": "train",
                "is_negative_frame": False,
                "objects": [
                    {
                        "object_id": "obj_01",
                        "bbox_xyxy_norm": [0.8, 0.1, 0.2, 0.4],  # x1 (0.8) > x2 (0.2)
                        "bbox_yxyx_norm": [0.1, 0.8, 0.4, 0.2],
                        "centroid_xy_norm": [0.5, 0.25],
                        "is_actionable": True,
                        "objectness_label": "clickable",
                        "validity_state": "VALID",
                        "element_kind": "button",
                    }
                ],
            }
        ],
    }
    is_valid, errors = validate_unified_manifest_dict(corrupted)
    assert is_valid is False
    assert any("inverted bbox_xyxy_norm" in e for e in errors)

    # 3. Dual box mismatch
    mismatched = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AVCS_Unified_UI_Objectness_Manifest",
        "version": "2.0.0",
        "coordinate_convention": "normalized_xyxy_origin_top_left",
        "frames": [
            {
                "frame_id": "frame_mismatched",
                "generator_source": "ui_grounding",
                "image_path": "screens/test.png",
                "width": 256,
                "height": 256,
                "split": "train",
                "is_negative_frame": False,
                "objects": [
                    {
                        "object_id": "obj_01",
                        "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4],
                        "bbox_yxyx_norm": [0.9, 0.1, 0.4, 0.3],  # y1 is 0.9 instead of 0.2
                        "centroid_xy_norm": [0.2, 0.3],
                        "is_actionable": True,
                        "objectness_label": "clickable",
                        "validity_state": "VALID",
                        "element_kind": "button",
                    }
                ],
            }
        ],
    }
    is_valid, errors = validate_unified_manifest_dict(mismatched)
    assert is_valid is False
    assert any("dual box mismatch" in e for e in errors)

    # 4. Negative frame with objects
    neg_with_objs = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AVCS_Unified_UI_Objectness_Manifest",
        "version": "2.0.0",
        "coordinate_convention": "normalized_xyxy_origin_top_left",
        "frames": [
            {
                "frame_id": "frame_neg_bad",
                "generator_source": "ui_grounding",
                "image_path": "screens/test.png",
                "width": 256,
                "height": 256,
                "split": "test_negative",
                "is_negative_frame": True,
                "negative_type": "wallpaper",
                "objects": [
                    {
                        "object_id": "obj_illegal",
                        "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4],
                        "bbox_yxyx_norm": [0.2, 0.1, 0.4, 0.3],
                        "centroid_xy_norm": [0.2, 0.3],
                        "is_actionable": True,
                        "objectness_label": "clickable",
                        "validity_state": "VALID",
                        "element_kind": "button",
                    }
                ],
            }
        ],
    }
    is_valid, errors = validate_unified_manifest_dict(neg_with_objs)
    assert is_valid is False
    assert any("must be empty" in e for e in errors)


# =========================================================================
# 2. Generator Loaders & Manifest Conversion Tests
# =========================================================================

def test_load_ui_grounding_manifest():
    """Verify loading and normalization of `datasets/ui_grounding`."""
    manifest = load_ui_grounding_manifest(UI_MANIFEST_PATH)
    assert len(manifest) == 270

    split_counts = manifest.get_split_counts()
    assert split_counts[SPLIT_TRAIN] == 160
    assert split_counts[SPLIT_VAL] == 40
    assert split_counts[SPLIT_TEST_HELD_OUT] == 40
    assert split_counts[SPLIT_TEST_NEGATIVE] == 30

    gen_counts = manifest.get_generator_counts()
    assert gen_counts[GENERATOR_UI_GROUNDING] == 270

    # Verify first train frame
    f0 = manifest[0]
    assert f0.generator_source == GENERATOR_UI_GROUNDING
    assert f0.is_negative_frame is False
    assert len(f0.objects) > 0

    # Spot-check object attributes
    obj = f0.objects[0]
    assert len(obj.bbox_xyxy_norm) == 4
    assert len(obj.bbox_yxyx_norm) == 4
    assert len(obj.centroid_xy_norm) == 2
    assert obj.is_actionable in (True, False)
    assert obj.objectness_label in ("clickable", "typeable", "scrollable", "static")
    assert obj.validity_state == "VALID"

    # Spot-check negative frames
    neg_frames = [f for f in manifest.frames if f.split == SPLIT_TEST_NEGATIVE]
    assert len(neg_frames) == 30
    for nf in neg_frames:
        assert nf.is_negative_frame is True
        assert len(nf.objects) == 0
        assert nf.negative_type is not None


def test_load_cleanroom_manifest():
    """Verify loading and normalization of `datasets/cleanroom_system1_v1`."""
    manifest = load_cleanroom_manifest(CLEANROOM_DIR, validate_hashes=False)
    assert len(manifest) == 112

    split_counts = manifest.get_split_counts()
    assert split_counts[SPLIT_TRAIN] == 64
    assert split_counts[SPLIT_VAL] == 16
    assert split_counts[SPLIT_TEST_HELD_OUT] == 16
    assert split_counts[SPLIT_TEST_NEGATIVE] == 16

    gen_counts = manifest.get_generator_counts()
    assert gen_counts[GENERATOR_CLEANROOM] == 112

    # Verify positive frame
    f0 = manifest[0]
    assert f0.generator_source == GENERATOR_CLEANROOM
    assert f0.is_negative_frame is False
    assert len(f0.objects) == 4  # Cleanroom frames contain exactly 4 controls

    # Spot check cleanroom element kinds
    kinds = {obj.element_kind for obj in f0.objects}
    assert kinds == {"button", "input_field", "tab", "icon"}

    # Spot-check negative frames
    neg_frames = [f for f in manifest.frames if f.split == SPLIT_TEST_NEGATIVE]
    assert len(neg_frames) == 16
    for nf in neg_frames:
        assert nf.is_negative_frame is True
        assert len(nf.objects) == 0
        assert nf.negative_type == "empty_desktop"


def test_build_combined_manifest(combined_manifest: UnifiedAnnotationManifest):
    """Verify total aggregation of both generators in the combined manifest."""
    assert len(combined_manifest) == 382  # 270 + 112

    split_counts = combined_manifest.get_split_counts()
    assert split_counts[SPLIT_TRAIN] == 224       # 160 + 64
    assert split_counts[SPLIT_VAL] == 56         # 40 + 16
    assert split_counts[SPLIT_TEST_HELD_OUT] == 56 # 40 + 16
    assert split_counts[SPLIT_TEST_NEGATIVE] == 46 # 30 + 16

    gen_counts = combined_manifest.get_generator_counts()
    assert gen_counts[GENERATOR_UI_GROUNDING] == 270
    assert gen_counts[GENERATOR_CLEANROOM] == 112


def test_roundtrip_serialization(combined_manifest: UnifiedAnnotationManifest):
    """Manifest must serialize to JSON and deserialize back with bit-exact structure."""
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        combined_manifest.to_json(tmp_path)
        loaded = UnifiedAnnotationManifest.from_json(tmp_path)
        assert len(loaded) == len(combined_manifest)
        assert loaded.schema_version == combined_manifest.schema_version
        assert loaded.coordinate_convention == combined_manifest.coordinate_convention

        # Spot check random frame
        orig_f = combined_manifest[10]
        load_f = loaded[10]
        assert orig_f.frame_id == load_f.frame_id
        assert orig_f.generator_source == load_f.generator_source
        assert len(orig_f.objects) == len(load_f.objects)
        for o1, o2 in zip(orig_f.objects, load_f.objects):
            assert o1.bbox_xyxy_norm == o2.bbox_xyxy_norm
            assert o1.bbox_yxyx_norm == o2.bbox_yxyx_norm
            assert o1.centroid_xy_norm == o2.centroid_xy_norm
            assert o1.element_kind == o2.element_kind
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# =========================================================================
# 3. Dual Coordinate Box Consistency Tests
# =========================================================================

def test_dual_coordinate_box_consistency(combined_manifest: UnifiedAnnotationManifest):
    """Assert for all objects that bbox_xyxy_norm and bbox_yxyx_norm are consistent."""
    total_objects_tested = 0
    for frame in combined_manifest.frames:
        if frame.is_negative_frame:
            continue
        for obj in frame.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy_norm
            y1_c, x1_c, y2_c, x2_c = obj.bbox_yxyx_norm

            # Coordinates must be bounded in [0.0, 1.0]
            assert 0.0 <= x1 <= 1.0, f"x1 out of bounds in {obj.object_id}: {x1}"
            assert 0.0 <= y1 <= 1.0, f"y1 out of bounds in {obj.object_id}: {y1}"
            assert 0.0 <= x2 <= 1.0, f"x2 out of bounds in {obj.object_id}: {x2}"
            assert 0.0 <= y2 <= 1.0, f"y2 out of bounds in {obj.object_id}: {y2}"

            # Box ordering
            assert x1 <= x2, f"Inverted X box in {obj.object_id}: {x1} > {x2}"
            assert y1 <= y2, f"Inverted Y box in {obj.object_id}: {y1} > {y2}"

            # Dual coordinate identity
            assert abs(x1 - x1_c) < 1e-5
            assert abs(y1 - y1_c) < 1e-5
            assert abs(x2 - x2_c) < 1e-5
            assert abs(y2 - y2_c) < 1e-5

            # Centroid inside or on boundary of box
            cx, cy = obj.centroid_xy_norm
            assert x1 - 1e-4 <= cx <= x2 + 1e-4, f"Centroid cx ({cx}) outside box [{x1}, {x2}] in {obj.object_id}"
            assert y1 - 1e-4 <= cy <= y2 + 1e-4, f"Centroid cy ({cy}) outside box [{y1}, {y2}] in {obj.object_id}"

            total_objects_tested += 1

    assert total_objects_tested > 1000, f"Expected >1000 objects tested, got {total_objects_tested}"


# =========================================================================
# 4. Collation & Batch Dict Shape Tests
# =========================================================================

def test_unified_batch_collate_shapes(combined_manifest: UnifiedAnnotationManifest):
    """Assert UnifiedBatchDict shapes adhere strictly to PROJECT.md §6.2."""
    dataset = MultiGeneratorDataset(
        manifest=combined_manifest,
        split=SPLIT_TRAIN,
        target_image_size=(128, 128),
        preload_images=False,
    )
    assert len(dataset) == 224

    # Sample a batch of 8
    samples = [dataset[i] for i in range(8)]
    batch = unified_batch_collate(samples)

    assert isinstance(batch, dict)
    assert set(batch.keys()) == {
        "images",
        "target_boxes",
        "target_centroids",
        "box_mask",
        "has_target",
        "generator_id",
    }

    # 1. Images: [B, 3, 128, 128] float32 in [0.0, 1.0]
    assert batch["images"].shape == (8, 3, 128, 128)
    assert batch["images"].dtype == torch.float32
    assert 0.0 <= batch["images"].min().item()
    assert batch["images"].max().item() <= 1.0

    # 2. Target boxes: [B, max_objs, 4]
    max_objs = batch["target_boxes"].shape[1]
    assert max_objs >= 4  # Cleanroom has 4, UI has up to 11
    assert batch["target_boxes"].shape == (8, max_objs, 4)
    assert batch["target_boxes"].dtype == torch.float32

    # 3. Target centroids: [B, max_objs, 2]
    assert batch["target_centroids"].shape == (8, max_objs, 2)
    assert batch["target_centroids"].dtype == torch.float32

    # 4. Box mask: [B, max_objs] bool
    assert batch["box_mask"].shape == (8, max_objs)
    assert batch["box_mask"].dtype == torch.bool

    # 5. Has target: [B] float32
    assert batch["has_target"].shape == (8,)
    assert batch["has_target"].dtype == torch.float32
    assert torch.all(batch["has_target"] == 1.0)  # Train split contains all positive screens

    # 6. Generator id: [B] int64
    assert batch["generator_id"].shape == (8,)
    assert batch["generator_id"].dtype == torch.int64
    for gid in batch["generator_id"].tolist():
        assert gid in (GENERATOR_ID_UI, GENERATOR_ID_CLEANROOM)


# =========================================================================
# 5. Arm D2 Generator Batch Balance Tests
# =========================================================================

def test_arm_d2_sampling_weights():
    """Verify mathematical calculation of Arm D2 inverse-frequency weights."""
    ui_count = 160
    clean_count = 64
    total = 224

    w_ui, w_clean = calculate_arm_d2_weights(ui_count, clean_count)

    # Expected: w_ui = total / (2 * 160) = 224 / 320 = 0.70
    assert abs(w_ui - (224.0 / (2.0 * 160.0))) < 1e-6
    # Expected: w_clean = total / (2 * 64) = 224 / 128 = 1.75
    assert abs(w_clean - (224.0 / (2.0 * 64.0))) < 1e-6

    # Ratio of total mass should be 1:1
    mass_ui = ui_count * w_ui
    mass_clean = clean_count * w_clean
    assert abs(mass_ui - mass_clean) < 1e-5


def test_arm_d2_batch_balance_over_multiple_batches(combined_manifest: UnifiedAnnotationManifest):
    """Verify that Arm D2 WeightedRandomSampler draws ~50% UI and ~50% Cleanroom over 20 batches."""
    stream = MultiGeneratorDataStream(
        manifest=combined_manifest,
        split=SPLIT_TRAIN,
        batch_size=16,
        sampling_policy="balanced",
        seed=42,
        preload_images=False,
    )

    assert stream.ui_count == 160
    assert stream.cleanroom_count == 64
    assert stream.total_samples == 224

    ui_draws = 0
    cleanroom_draws = 0
    total_draws = 0

    # Draw across 2 full epochs (2 * 14 batches = 28 batches)
    for epoch in range(2):
        for batch in stream:
            gids = batch["generator_id"].tolist()
            ui_draws += gids.count(GENERATOR_ID_UI)
            cleanroom_draws += gids.count(GENERATOR_ID_CLEANROOM)
            total_draws += len(gids)

    ui_pct = (ui_draws / total_draws) * 100.0
    clean_pct = (cleanroom_draws / total_draws) * 100.0

    # Under balanced sampling, expect ~50% +/- 8% over 448 draws
    assert 42.0 <= ui_pct <= 58.0, f"UI draw percentage {ui_pct:.1f}% deviates from 50% target"
    assert 42.0 <= clean_pct <= 58.0, f"Cleanroom draw percentage {clean_pct:.1f}% deviates from 50% target"


# =========================================================================
# 6. Zero Template / Layout Leakage Isolation Tests
# =========================================================================

def test_zero_layout_template_leakage(combined_manifest: UnifiedAnnotationManifest):
    """Assert zero template, layout family, session ID, and image hash leakage across splits."""
    train_manifest = combined_manifest.filter_split(SPLIT_TRAIN)
    test_manifest = combined_manifest.filter_split(SPLIT_TEST_HELD_OUT)

    report = verify_layout_leakage_isolation(train_manifest, test_manifest)
    assert report["isolated"] is True, f"Data leakage detected: {report['collisions']}"
    assert report["cleanroom_isolated"] is True
    assert report["ui_grounding_isolated"] is True
    assert report["image_hashes_isolated"] is True

    # Check cleanroom layout families explicitly
    clean_train_families = set(report["cleanroom_train_families"])
    clean_test_families = set(report["cleanroom_test_families"])
    assert clean_train_families == {"dialog", "grid", "sidebar", "toolbar"}
    assert clean_test_families == {"split_panel", "stacked"}
    assert clean_train_families.isdisjoint(clean_test_families)

    # Check UI grounding sessions explicitly
    ui_train_sessions = set(report["ui_train_sessions"])
    ui_test_sessions = set(report["ui_test_sessions"])
    assert len(ui_train_sessions) >= 5
    assert len(ui_test_sessions) >= 2
    assert ui_train_sessions.isdisjoint(ui_test_sessions)


# =========================================================================
# 7. Blank & Negative Frame Handling Tests
# =========================================================================

def test_blank_negative_frame_handling(combined_manifest: UnifiedAnnotationManifest):
    """Verify test_negative frames output has_target == 0.0 and empty target boxes."""
    stream = MultiGeneratorDataStream(
        manifest=combined_manifest,
        split=SPLIT_TEST_NEGATIVE,
        batch_size=16,
        sampling_policy="sequential",
        preload_images=False,
    )

    assert stream.total_samples == 46  # 30 UI + 16 Cleanroom

    total_negatives_evaluated = 0
    for batch in stream:
        b_size = len(batch["has_target"])
        total_negatives_evaluated += b_size

        # Every frame in negative split must have has_target == 0.0
        assert torch.all(batch["has_target"] == 0.0), "Found positive has_target in negative split"

        # Box mask must be all False
        assert torch.all(batch["box_mask"] == False), "Found True in box_mask for negative frames"

        # Target boxes shape must have max_objs == 0
        assert batch["target_boxes"].shape == (b_size, 0, 4)
        assert batch["target_centroids"].shape == (b_size, 0, 2)

    assert total_negatives_evaluated == 46
