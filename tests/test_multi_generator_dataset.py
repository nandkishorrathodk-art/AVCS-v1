"""Comprehensive Verification Test Suite for Multi-Generator Dataset & Harmonization.

Part of StreamingVisualCortexV2 Milestone M2: Multi-Generator Dataset Pipeline.

Verifies:
1. Unified Draft-07 schema compliance and 5 generic UI classes:
   Class 0: Button
   Class 1: Input box
   Class 2: Icon/toggle
   Class 3: Dialog/popup
   Class 4: Text area
2. Dual bounding box representations (bbox_xyxy_norm vs bbox_yxyx_norm) and centroid consistency.
3. Generator harmonization across `ui_grounding` (270 frames) and `cleanroom_system1_v1` (112 frames).
4. Strict leak-free split isolation (zero template, layout family, session, or hash overlap between train and test).
5. Arm D2 balanced sampling (~50/50 generator balance across batches).
6. Hard-negative screen injection (15-20% blank screens in training batches).
7. Procedural temporal sequence generation (Head 1): spatial change heatmaps (128x128) and binary spinner labels.
8. Pre/post action observation pair generation (Head 3): pre/post frames, action locations, response labels in {0.0, 1.0}.
9. Photometric color inversion (50%) and brightness/contrast/gamma augmentation suite.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from avcs.dataset.unified.generator_harmonizer import (
    GENERATOR_CLEANROOM,
    GENERATOR_UI_GROUNDING,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
    SPLIT_TRAIN,
    SPLIT_VAL,
    harmonize_all_generators,
    harmonize_cleanroom,
    harmonize_ui_grounding,
    verify_leak_free_split_isolation,
)
from avcs.dataset.unified.schema import (
    CLASS_BUTTON,
    CLASS_DIALOG_POPUP,
    CLASS_ICON_TOGGLE,
    CLASS_INPUT_BOX,
    CLASS_TEXT_AREA,
    DRAFT_07_UNIFIED_SCHEMA,
    NUM_CLASSES,
    UI_CLASS_ID_TO_NAME,
    UI_CLASS_NAME_TO_ID,
    UI_CLASS_NAMES,
    UnifiedFrame,
    UnifiedManifest,
    UnifiedObject,
    map_legacy_kind_to_class,
    normalize_split_name,
    validate_unified_schema,
)
from avcs.dataset.unified.temporal_sampler import (
    ActionObservationPair,
    TemporalSequence,
    generate_action_pair,
    generate_temporal_sequence,
    sample_action_pair_batch,
    sample_temporal_batch,
)
from avcs.training.multi_generator_dataset import (
    GENERATOR_ID_CLEANROOM,
    GENERATOR_ID_UI,
    ArmD2MultiGeneratorDataset,
    apply_photometric_augmentations,
    calculate_arm_d2_weights,
    collate_multi_generator_batch,
    create_arm_d2_dataloader,
    generate_hard_negative_screen,
)

UI_MANIFEST_PATH = Path("datasets/ui_grounding/manifest.json")
CLEANROOM_DIR = Path("datasets/cleanroom_system1_v1")


@pytest.fixture(scope="module")
def harmonized_manifest() -> UnifiedManifest:
    """Fixture loading the harmonized 5-class multi-generator manifest."""
    return harmonize_all_generators(
        ui_manifest_path=UI_MANIFEST_PATH,
        cleanroom_dir=CLEANROOM_DIR,
    )


# =========================================================================
# 1. Unified Schema & 5-Class Distribution Tests
# =========================================================================
def test_draft_07_schema_structure():
    """Verify Draft-07 schema root properties and enum definitions."""
    schema = DRAFT_07_UNIFIED_SCHEMA
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["title"] == "AVCS_Unified_5Class_UI_Manifest"
    assert schema["version"] == "2.0.0"
    assert "frames" in schema["required"]
    assert "coordinate_convention" in schema["required"]


def test_5_class_taxonomy_constants():
    """Verify generic 5 UI classes constants and mappings."""
    assert NUM_CLASSES == 5
    assert len(UI_CLASS_NAMES) == 5
    assert UI_CLASS_NAMES == ("button", "input_box", "icon_toggle", "dialog_popup", "text_area")

    assert UI_CLASS_ID_TO_NAME[CLASS_BUTTON] == "button"
    assert UI_CLASS_ID_TO_NAME[CLASS_INPUT_BOX] == "input_box"
    assert UI_CLASS_ID_TO_NAME[CLASS_ICON_TOGGLE] == "icon_toggle"
    assert UI_CLASS_ID_TO_NAME[CLASS_DIALOG_POPUP] == "dialog_popup"
    assert UI_CLASS_ID_TO_NAME[CLASS_TEXT_AREA] == "text_area"

    # Test legacy kind mapper
    assert map_legacy_kind_to_class("button") == (CLASS_BUTTON, "button")
    assert map_legacy_kind_to_class("input_field") == (CLASS_INPUT_BOX, "input_box")
    assert map_legacy_kind_to_class("icon") == (CLASS_ICON_TOGGLE, "icon_toggle")
    assert map_legacy_kind_to_class("tab") == (CLASS_BUTTON, "button")
    assert map_legacy_kind_to_class("modal_dialog") == (CLASS_DIALOG_POPUP, "dialog_popup")
    assert map_legacy_kind_to_class("text_area") == (CLASS_TEXT_AREA, "text_area")


def test_harmonized_manifest_5_class_distribution(harmonized_manifest: UnifiedManifest):
    """Manifest must contain all 5 generic UI classes with verified positive counts."""
    assert len(harmonized_manifest) == 382  # 270 UI + 112 Cleanroom

    class_counts = harmonized_manifest.get_class_counts()
    print("Harmonized 5-Class Distribution:", class_counts)

    # Assert all 5 classes have positive coverage
    for cname in UI_CLASS_NAMES:
        assert cname in class_counts, f"Missing class {cname} in class_counts"
        assert class_counts[cname] > 0, f"Class {cname} has zero count! Must be populated."

    assert class_counts["button"] >= 1000
    assert class_counts["input_box"] >= 300
    assert class_counts["icon_toggle"] >= 200
    assert class_counts["dialog_popup"] >= 30
    assert class_counts["text_area"] >= 300


def test_dual_bounding_box_and_centroid_consistency(harmonized_manifest: UnifiedManifest):
    """Every object must exhibit exact dual bounding box and centroid consistency."""
    tolerance = 1e-4
    total_objects = 0

    for frame in harmonized_manifest.frames:
        if frame.is_negative_frame:
            assert len(frame.objects) == 0
            continue

        for obj in frame.objects:
            total_objects += 1
            x1, y1, x2, y2 = obj.bbox_xyxy_norm
            y1_c, x1_c, y2_c, x2_c = obj.bbox_yxyx_norm
            cx, cy = obj.centroid_xy_norm

            # Coordinates in range [0, 1]
            assert 0.0 <= x1 <= 1.0 + tolerance
            assert 0.0 <= y1 <= 1.0 + tolerance
            assert 0.0 <= x2 <= 1.0 + tolerance
            assert 0.0 <= y2 <= 1.0 + tolerance
            assert x1 <= x2 + tolerance
            assert y1 <= y2 + tolerance

            # Dual coordinate identity
            assert abs(x1 - x1_c) <= tolerance
            assert abs(y1 - y1_c) <= tolerance
            assert abs(x2 - x2_c) <= tolerance
            assert abs(y2 - y2_c) <= tolerance

            # Centroid within bounds
            assert 0.0 <= cx <= 1.0 + tolerance
            assert 0.0 <= cy <= 1.0 + tolerance

    assert total_objects > 2500, f"Expected >2500 objects across 382 frames, found {total_objects}"


def test_validate_unified_schema_success(harmonized_manifest: UnifiedManifest):
    """Harmonized manifest must pass Draft-07 schema validation with 0 errors."""
    m_dict = harmonized_manifest.to_dict()
    is_valid, errors = validate_unified_schema(m_dict)
    assert is_valid is True, f"Schema validation failed: {errors}"
    assert len(errors) == 0


def test_validate_unified_schema_catches_invalid_classes_and_coords():
    """Schema validator must catch corrupted data, invalid class IDs, and coordinate inversions."""
    # 1. Invalid class_id 99
    bad_obj_manifest = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AVCS_Unified_5Class_UI_Manifest",
        "version": "2.0.0",
        "coordinate_convention": "normalized_xyxy_origin_top_left",
        "frames": [
            {
                "frame_id": "test_01",
                "generator_source": "ui_grounding",
                "image_path": "path/to/img.png",
                "width": 128,
                "height": 128,
                "split": "train",
                "is_negative_frame": False,
                "objects": [
                    {
                        "object_id": "obj_01",
                        "class_id": 99,  # Invalid
                        "class_name": "button",
                        "bbox_xyxy_norm": [0.1, 0.1, 0.5, 0.5],
                        "bbox_yxyx_norm": [0.1, 0.1, 0.5, 0.5],
                        "centroid_xy_norm": [0.3, 0.3],
                        "is_actionable": True,
                        "validity_state": "VALID",
                    }
                ],
            }
        ],
    }
    is_valid, errors = validate_unified_schema(bad_obj_manifest)
    assert is_valid is False
    assert any("invalid class_id" in e for e in errors)

    # 2. Inverted coordinates x1 > x2
    inverted_manifest = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AVCS_Unified_5Class_UI_Manifest",
        "version": "2.0.0",
        "coordinate_convention": "normalized_xyxy_origin_top_left",
        "frames": [
            {
                "frame_id": "test_02",
                "generator_source": "ui_grounding",
                "image_path": "path/to/img.png",
                "width": 128,
                "height": 128,
                "split": "train",
                "is_negative_frame": False,
                "objects": [
                    {
                        "object_id": "obj_02",
                        "class_id": 0,
                        "class_name": "button",
                        "bbox_xyxy_norm": [0.8, 0.1, 0.2, 0.5],  # Inverted x
                        "bbox_yxyx_norm": [0.1, 0.8, 0.5, 0.2],
                        "centroid_xy_norm": [0.5, 0.3],
                        "is_actionable": True,
                        "validity_state": "VALID",
                    }
                ],
            }
        ],
    }
    is_valid, errors = validate_unified_schema(inverted_manifest)
    assert is_valid is False
    assert any("inverted bbox_xyxy_norm" in e for e in errors)


# =========================================================================
# 2. Split Isolation Tests
# =========================================================================
def test_leak_free_split_isolation(harmonized_manifest: UnifiedManifest):
    """Templates and layout families in test_held_out must be strictly disjoint from train."""
    audit = verify_leak_free_split_isolation(harmonized_manifest)

    assert audit["is_leak_free"] is True, f"Leakage detected: {audit}"
    assert len(audit["cleanroom_train_test_overlap"]) == 0
    assert len(audit["ui_train_test_overlap"]) == 0
    assert audit["hash_train_test_overlap_count"] == 0
    assert audit["hash_train_val_overlap_count"] == 0

    # Verify expected template partition
    assert set(audit["cleanroom_train_templates"]) == {"dialog", "grid", "sidebar", "toolbar"}
    assert set(audit["cleanroom_test_templates"]) == {"split_panel", "stacked"}


def test_split_sample_counts(harmonized_manifest: UnifiedManifest):
    """Verify sample counts per canonical split."""
    counts = harmonized_manifest.get_split_counts()
    assert counts[SPLIT_TRAIN] == 224  # 160 UI + 64 Cleanroom
    assert counts[SPLIT_VAL] == 56  # 40 UI + 16 Cleanroom
    assert counts[SPLIT_TEST_HELD_OUT] == 56  # 40 UI + 16 Cleanroom
    assert counts[SPLIT_TEST_NEGATIVE] == 46  # 30 UI + 16 Cleanroom


# =========================================================================
# 3. Arm D2 Balanced Sampling & Collation Tests
# =========================================================================
def test_arm_d2_sampling_weights():
    """Verify Arm D2 sampling weight calculations for exact inverse-frequency balancing."""
    w_ui, w_cr = calculate_arm_d2_weights(ui_count=160, cleanroom_count=64)
    total = 160 + 64
    assert w_ui == total / (2.0 * 160)  # 224 / 320 = 0.7
    assert w_cr == total / (2.0 * 64)  # 224 / 128 = 1.75
    # Expected draws
    assert (160 * w_ui) == (64 * w_cr) == total / 2.0


def test_arm_d2_batch_balance_over_multiple_batches(harmonized_manifest: UnifiedManifest):
    """Arm D2 DataLoader must draw approximately 50% UI Grounding and 50% Cleanroom over time."""
    loader = create_arm_d2_dataloader(
        manifest=harmonized_manifest,
        split=SPLIT_TRAIN,
        batch_size=16,
        hard_negative_prob=0.0,  # disable negative injection for pure generator balance check
        rng_seed=42,
    )

    total_ui = 0
    total_cr = 0
    epochs_to_run = 2

    for _ in range(epochs_to_run):
        for batch in loader:
            gen_ids = batch["generator_id"].tolist()
            total_ui += sum(1 for g in gen_ids if g == GENERATOR_ID_UI)
            total_cr += sum(1 for g in gen_ids if g == GENERATOR_ID_CLEANROOM)

    total_samples = total_ui + total_cr
    ui_prop = total_ui / total_samples
    cr_prop = total_cr / total_samples

    print(f"Arm D2 Empirical Draws: UI={total_ui} ({ui_prop:.2%}), CR={total_cr} ({cr_prop:.2%})")

    # Over 320 samples, 50% target should be within [0.40, 0.60] with 99.9% binomial confidence
    assert 0.40 <= ui_prop <= 0.60, f"UI draw proportion {ui_prop:.2%} outside [0.40, 0.60]"
    assert 0.40 <= cr_prop <= 0.60, f"CR draw proportion {cr_prop:.2%} outside [0.40, 0.60]"


def test_collate_multi_generator_batch_shapes():
    """Verify unified batch collation outputs matching the V2 model contract."""
    sample_ui = {
        "frame_id": "ui_01",
        "image": torch.rand(3, 128, 128),
        "target_boxes": torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]], dtype=torch.float32),
        "target_classes": torch.tensor([0, 1], dtype=torch.int64),
        "target_centroids": torch.tensor([[0.25, 0.25], [0.65, 0.65]], dtype=torch.float32),
        "has_target": 1.0,
        "generator_id": GENERATOR_ID_UI,
        "is_negative": False,
    }
    sample_blank = {
        "frame_id": "neg_01",
        "image": torch.rand(3, 128, 128),
        "target_boxes": torch.zeros((0, 4), dtype=torch.float32),
        "target_classes": torch.zeros((0,), dtype=torch.int64),
        "target_centroids": torch.zeros((0, 2), dtype=torch.float32),
        "has_target": 0.0,
        "generator_id": GENERATOR_ID_CLEANROOM,
        "is_negative": True,
    }

    batch = collate_multi_generator_batch([sample_ui, sample_blank])

    assert batch["images"].shape == (2, 3, 128, 128)
    assert batch["target_boxes"].shape == (2, 2, 4)
    assert batch["target_classes"].shape == (2, 2)
    assert batch["target_centroids"].shape == (2, 2, 2)
    assert batch["box_mask"].shape == (2, 2)
    assert batch["has_target"].shape == (2,)
    assert batch["generator_id"].shape == (2,)
    assert batch["is_negative"].shape == (2,)

    # UI sample has 2 boxes valid
    assert batch["box_mask"][0].tolist() == [True, True]
    assert batch["has_target"][0].item() == 1.0
    # Blank sample has 0 boxes valid
    assert batch["box_mask"][1].tolist() == [False, False]
    assert batch["has_target"][1].item() == 0.0
    assert batch["target_classes"][1].tolist() == [-1, -1]


# =========================================================================
# 4. Hard-Negative Injection Tests
# =========================================================================
def test_hard_negative_injection_rate(harmonized_manifest: UnifiedManifest):
    """Training dataset must inject 15-20% blank screens and static wallpapers."""
    target_rate = 0.175
    dataset = ArmD2MultiGeneratorDataset(
        manifest=harmonized_manifest,
        split=SPLIT_TRAIN,
        apply_augmentation=False,
        hard_negative_prob=target_rate,
        rng_seed=123,
    )

    num_samples = 400
    neg_count = 0

    for _ in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        if sample["is_negative"]:
            neg_count += 1
            # Negative frames must have 0 target boxes and has_target == 0.0
            assert sample["target_boxes"].shape[0] == 0
            assert sample["target_classes"].shape[0] == 0
            assert sample["has_target"] == 0.0

    empirical_rate = neg_count / num_samples
    print(f"Hard-negative injection empirical rate: {empirical_rate:.2%} (target: {target_rate:.2%})")

    # In 400 trials, 17.5% expected lies cleanly within [0.12, 0.25]
    assert 0.12 <= empirical_rate <= 0.25


def test_generate_hard_negative_screen_properties():
    """Verify generated hard negative screens (solid, gradient, wallpaper)."""
    for _ in range(10):
        screen = generate_hard_negative_screen((128, 128))
        assert screen.shape == (3, 128, 128)
        assert screen.dtype == torch.float32
        assert screen.min() >= 0.0
        assert screen.max() <= 1.0


# =========================================================================
# 5. Temporal Sequence Generation Tests (Head 1)
# =========================================================================
def test_temporal_sequence_generation():
    """Test procedural temporal sequences for spinner, moving, and static modes."""
    # 1. Spinner sequence
    spinner_seq = generate_temporal_sequence(mode="spinner", sequence_length=4)
    assert spinner_seq.frames.shape == (4, 3, 128, 128)
    assert spinner_seq.change_heatmap.shape == (1, 128, 128)
    assert spinner_seq.spinner_label == 1.0
    assert spinner_seq.change_heatmap.max().item() > 0.0  # Visual delta present

    # 2. Moving sequence
    moving_seq = generate_temporal_sequence(mode="moving", sequence_length=4)
    assert moving_seq.frames.shape == (4, 3, 128, 128)
    assert moving_seq.change_heatmap.shape == (1, 128, 128)
    assert moving_seq.spinner_label == 0.0  # Not a spinner
    assert moving_seq.change_heatmap.max().item() > 0.0  # Movement delta present

    # 3. Static sequence
    static_seq = generate_temporal_sequence(mode="static", sequence_length=4)
    assert static_seq.frames.shape == (4, 3, 128, 128)
    assert static_seq.change_heatmap.shape == (1, 128, 128)
    assert static_seq.spinner_label == 0.0
    assert static_seq.change_heatmap.max().item() == 0.0  # Zero change for static screens


def test_sample_temporal_batch():
    """Verify temporal batch sampler shapes and labels."""
    batch = sample_temporal_batch(batch_size=8, sequence_length=4)
    assert batch["sequences"].shape == (8, 4, 3, 128, 128)
    assert batch["change_heatmaps"].shape == (8, 1, 128, 128)
    assert batch["spinner_labels"].shape == (8, 1)

    labels = batch["spinner_labels"].squeeze(1).tolist()
    assert all(lbl in (0.0, 1.0) for lbl in labels)
    assert batch["sequences"].min() >= 0.0
    assert batch["sequences"].max() <= 1.0


# =========================================================================
# 6. Post-Action Visual Response Pair Tests (Head 3)
# =========================================================================
def test_action_pair_generation():
    """Test pre/post action pair generation for response (1.0) vs no response (0.0)."""
    # 1. Action with visible response
    pos_pair = generate_action_pair(has_response=True)
    assert pos_pair.pre_frame.shape == (3, 128, 128)
    assert pos_pair.post_frame.shape == (3, 128, 128)
    assert pos_pair.response_label == 1.0
    assert 0.0 <= pos_pair.action_location_norm[0] <= 1.0
    assert 0.0 <= pos_pair.action_location_norm[1] <= 1.0

    # Optical change must be greater than zero
    delta = (pos_pair.post_frame - pos_pair.pre_frame).abs().sum().item()
    assert delta > 0.1, f"Expected visual delta for response=1.0, got {delta}"

    # 2. Action without response (miss / static background)
    neg_pair = generate_action_pair(has_response=False)
    assert neg_pair.pre_frame.shape == (3, 128, 128)
    assert neg_pair.post_frame.shape == (3, 128, 128)
    assert neg_pair.response_label == 0.0
    neg_delta = (neg_pair.post_frame - neg_pair.pre_frame).abs().sum().item()
    assert neg_delta == 0.0, f"Expected zero delta for response=0.0, got {neg_delta}"


def test_sample_action_pair_batch():
    """Verify action pair batch sampler shapes and labels."""
    batch = sample_action_pair_batch(batch_size=8, positive_ratio=0.5)
    assert batch["pre_frames"].shape == (8, 3, 128, 128)
    assert batch["post_frames"].shape == (8, 3, 128, 128)
    assert batch["action_locations"].shape == (8, 2)
    assert batch["response_labels"].shape == (8, 1)

    labels = batch["response_labels"].squeeze(1).tolist()
    assert all(lbl in (0.0, 1.0) for lbl in labels)


# =========================================================================
# 7. Photometric Augmentation & Inversion Pipeline Tests
# =========================================================================
def test_photometric_inversion_exactness():
    """Verify exact photometric color inversion arithmetic."""
    dummy_img = torch.tensor([[[0.1, 0.4], [0.7, 0.9]]], dtype=torch.float32).repeat(3, 1, 1)
    inv = apply_photometric_augmentations(
        dummy_img,
        inversion_prob=1.0,
        brightness_range=(0.0, 0.0),
        contrast_range=(1.0, 1.0),
        gamma_range=(1.0, 1.0),
    )
    expected = 1.0 - dummy_img
    max_diff = (inv - expected).abs().max().item()
    assert max_diff < 1e-6


def test_photometric_inversion_probability():
    """Verify 50% inversion probability over multiple trials."""
    dummy_img = torch.full((3, 64, 64), 0.2, dtype=torch.float32)
    inversions = 0
    num_trials = 200

    for _ in range(num_trials):
        out = apply_photometric_augmentations(
            dummy_img,
            inversion_prob=0.5,
            brightness_range=(0.0, 0.0),
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
        )
        if out.mean().item() > 0.5:
            inversions += 1

    emp_p = inversions / num_trials
    print(f"Empirical inversion probability: {emp_p:.2%}")
    assert 0.38 <= emp_p <= 0.62


def test_photometric_augmentation_bounds_and_invariance():
    """Verify brightness, contrast, and gamma jitter remain within [0.0, 1.0]."""
    dummy_img = torch.rand(3, 128, 128)
    for _ in range(20):
        augmented = apply_photometric_augmentations(dummy_img, inversion_prob=0.5)
        assert augmented.shape == (3, 128, 128)
        assert augmented.min() >= 0.0
        assert augmented.max() <= 1.0
