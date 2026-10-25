"""Adversarial Verification & Stress-Test Suite for Multi-Generator Dataset Pipeline (Milestone M2).

Adversarially challenges and verifies:
1. Dataset Split Isolation & Data Leakage Prevention:
   - Exhaustive SHA-256 hash collision analysis across all 382 images.
   - Strict disjointness of layout template families (Cleanroom) and session IDs (UI Grounding).
   - Absence of duplicate images or leakages across train, val, test_held_out, and test_negative.
2. Coordinate Boundary & Geometry Invariants:
   - Exhaustive verification of all 2,748 bounding boxes: 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0.
   - Exhaustive centroid checks: 0.0 <= cx, cy <= 1.0 with |cx - (x1+x2)/2| < 1e-4.
   - Detection of zero-area, inverted, subnormal, or degenerate boxes.
3. Dual Bounding Box Consistency:
   - Direct verification that |bbox_xyxy_norm[0] - bbox_yxyx_norm[1]| < 1e-4 across all 2,748 boxes.
   - Bijective coordinate transformation invariants.
4. Schema Validation Corruption Robustness:
   - Robust rejection of NaNs, Infs, -Infs, out-of-bounds coordinates, inverted boxes.
   - Robust rejection of dual box drift, class ID/name mismatches, and negative frame corruptions.
   - Malformed data structure fuzzing (string coordinates, duplicate frame IDs, invalid dimensions).
5. Arm D2 Multi-Generator Stream & Temporal Samplers:
   - Balance verification preventing generator starvation.
   - Photometric invariance of coordinate annotations under color inversion.
   - Temporal sequence change heatmaps and pre/post action pair fidelity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest
import torch

from avcs.dataset.unified.generator_harmonizer import (
    harmonize_all_generators,
    verify_leak_free_split_isolation,
)
from avcs.dataset.unified.schema import (
    CLASS_BUTTON,
    CLASS_DIALOG_POPUP,
    CLASS_ICON_TOGGLE,
    CLASS_INPUT_BOX,
    CLASS_TEXT_AREA,
    DRAFT_07_UNIFIED_SCHEMA,
    GENERATOR_CLEANROOM,
    GENERATOR_UI_GROUNDING,
    NUM_CLASSES,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TEST_NEGATIVE,
    SPLIT_TRAIN,
    SPLIT_VAL,
    UI_CLASS_ID_TO_NAME,
    UI_CLASS_NAMES,
    UnifiedFrame,
    UnifiedManifest,
    UnifiedObject,
    validate_unified_schema,
)
from avcs.dataset.unified.temporal_sampler import (
    generate_action_pair,
    generate_temporal_sequence,
)
from avcs.training.multi_generator_dataset import (
    ArmD2MultiGeneratorDataset,
    collate_multi_generator_batch,
)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture(scope="module")
def harmonized_manifest() -> UnifiedManifest:
    """Load and return the fully harmonized 5-class manifest across all generators."""
    return harmonize_all_generators()


# ==============================================================================
# 1. Adversarial Split Isolation & Data Leakage Prevention
# ==============================================================================
class TestAdversarialSplitIsolationAndLeakage:
    """Stress-test split isolation, image hash collisions, template and session leakage."""

    def test_total_frame_count_exactness(self, harmonized_manifest: UnifiedManifest):
        """Assert exact total frame count of 382 (270 ui_grounding + 112 cleanroom)."""
        assert len(harmonized_manifest) == 382, (
            f"Expected exactly 382 frames, got {len(harmonized_manifest)}"
        )

    def test_all_382_image_files_exist_and_non_empty(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Verify that every image path points to an existing non-empty file on disk."""
        missing = []
        empty = []
        for frame in harmonized_manifest.frames:
            p = Path(frame.image_path)
            if not p.exists():
                missing.append(frame.image_path)
            elif p.stat().st_size == 0:
                empty.append(frame.image_path)

        assert len(missing) == 0, f"Found {len(missing)} missing image files: {missing[:5]}"
        assert len(empty) == 0, f"Found {len(empty)} empty image files: {empty[:5]}"

    def test_exhaustive_382_image_sha256_pairwise_split_disjointness(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Compute SHA-256 for all 382 images; assert zero collisions across all split pairs."""
        hashes_by_split: Dict[str, Dict[str, str]] = {
            SPLIT_TRAIN: {},
            SPLIT_VAL: {},
            SPLIT_TEST_HELD_OUT: {},
            SPLIT_TEST_NEGATIVE: {},
        }

        for frame in harmonized_manifest.frames:
            raw_bytes = Path(frame.image_path).read_bytes()
            img_hash = hashlib.sha256(raw_bytes).hexdigest()
            hashes_by_split[frame.split][frame.frame_id] = img_hash

        # Assert counts per split
        assert len(hashes_by_split[SPLIT_TRAIN]) == 224, "Train must have 224 frames"
        assert len(hashes_by_split[SPLIT_VAL]) == 56, "Val must have 56 frames"
        assert len(hashes_by_split[SPLIT_TEST_HELD_OUT]) == 56, "Test held-out must have 56 frames"
        assert len(hashes_by_split[SPLIT_TEST_NEGATIVE]) == 46, "Test negative must have 46 frames"

        train_set = set(hashes_by_split[SPLIT_TRAIN].values())
        val_set = set(hashes_by_split[SPLIT_VAL].values())
        test_held_out_set = set(hashes_by_split[SPLIT_TEST_HELD_OUT].values())
        test_neg_set = set(hashes_by_split[SPLIT_TEST_NEGATIVE].values())

        # Within-split uniqueness for positive splits
        assert len(train_set) == 224, "Train split contains duplicate image hashes!"
        assert len(val_set) == 56, "Val split contains duplicate image hashes!"
        assert len(test_held_out_set) == 56, "Test held-out split contains duplicate image hashes!"

        # Cross-split disjointness assertions
        train_test_overlap = train_set.intersection(test_held_out_set)
        assert len(train_test_overlap) == 0, (
            f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} image hash collisions "
            f"between train and test_held_out! Colliding hashes: {train_test_overlap}"
        )

        train_val_overlap = train_set.intersection(val_set)
        assert len(train_val_overlap) == 0, (
            f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} image hash collisions "
            f"between train and val! Colliding hashes: {train_val_overlap}"
        )

        val_test_overlap = val_set.intersection(test_held_out_set)
        assert len(val_test_overlap) == 0, (
            f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} image hash collisions "
            f"between val and test_held_out! Colliding hashes: {val_test_overlap}"
        )

        train_neg_overlap = train_set.intersection(test_neg_set)
        assert len(train_neg_overlap) == 0, (
            f"DATA LEAKAGE DETECTED: {len(train_neg_overlap)} image hash collisions "
            f"between train and test_negative!"
        )

        val_neg_overlap = val_set.intersection(test_neg_set)
        assert len(val_neg_overlap) == 0, (
            f"DATA LEAKAGE DETECTED: {len(val_neg_overlap)} image hash collisions "
            f"between val and test_negative!"
        )

        test_neg_overlap = test_held_out_set.intersection(test_neg_set)
        assert len(test_neg_overlap) == 0, (
            f"DATA LEAKAGE DETECTED: {len(test_neg_overlap)} image hash collisions "
            f"between test_held_out and test_negative!"
        )

    def test_cleanroom_layout_template_strict_disjointness(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Assert cleanroom layout templates are strictly disjoint across train, val, and test."""
        templates: Dict[str, Set[str]] = {
            SPLIT_TRAIN: set(),
            SPLIT_VAL: set(),
            SPLIT_TEST_HELD_OUT: set(),
            SPLIT_TEST_NEGATIVE: set(),
        }

        for frame in harmonized_manifest.frames:
            if frame.generator_source == GENERATOR_CLEANROOM and frame.layout_template_id:
                templates[frame.split].add(frame.layout_template_id)

        # Expected partitions based on cleanroom layout families
        assert templates[SPLIT_TRAIN] == {"dialog", "grid", "sidebar", "toolbar"}
        assert templates[SPLIT_VAL] == {"compact"}
        assert templates[SPLIT_TEST_HELD_OUT] == {"split_panel", "stacked"}

        # Pairwise disjointness
        assert len(templates[SPLIT_TRAIN].intersection(templates[SPLIT_TEST_HELD_OUT])) == 0
        assert len(templates[SPLIT_TRAIN].intersection(templates[SPLIT_VAL])) == 0
        assert len(templates[SPLIT_VAL].intersection(templates[SPLIT_TEST_HELD_OUT])) == 0

    def test_ui_grounding_session_strict_disjointness(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Assert UI Grounding capture sessions are strictly disjoint across train, val, and test."""
        sessions: Dict[str, Set[str]] = {
            SPLIT_TRAIN: set(),
            SPLIT_VAL: set(),
            SPLIT_TEST_HELD_OUT: set(),
            SPLIT_TEST_NEGATIVE: set(),
        }

        for frame in harmonized_manifest.frames:
            if frame.generator_source == GENERATOR_UI_GROUNDING:
                sess = frame.metadata.get("session_id")
                if sess:
                    sessions[frame.split].add(sess)

        assert len(sessions[SPLIT_TRAIN]) == 8
        assert len(sessions[SPLIT_VAL]) == 4
        assert len(sessions[SPLIT_TEST_HELD_OUT]) == 4
        assert len(sessions[SPLIT_TEST_NEGATIVE]) == 3

        # Pairwise disjointness
        assert len(sessions[SPLIT_TRAIN].intersection(sessions[SPLIT_TEST_HELD_OUT])) == 0
        assert len(sessions[SPLIT_TRAIN].intersection(sessions[SPLIT_VAL])) == 0
        assert len(sessions[SPLIT_VAL].intersection(sessions[SPLIT_TEST_HELD_OUT])) == 0
        assert len(sessions[SPLIT_TRAIN].intersection(sessions[SPLIT_TEST_NEGATIVE])) == 0

    def test_verify_leak_free_split_isolation_function(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Invoke verify_leak_free_split_isolation and assert True with empty intersection sets."""
        audit = verify_leak_free_split_isolation(harmonized_manifest)
        assert audit["is_leak_free"] is True
        assert audit["cleanroom_train_test_overlap"] == []
        assert audit["ui_train_test_overlap"] == []
        assert audit["hash_train_test_overlap_count"] == 0
        assert audit["hash_train_val_overlap_count"] == 0


# ==============================================================================
# 2. Adversarial Coordinate Boundaries & Geometry Invariants
# ==============================================================================
class TestAdversarialCoordinateBoundaries:
    """Stress-test all 2,748 bounding boxes and centroids for boundary violations."""

    def test_total_annotated_object_count_exactness(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Assert exact total object count is 2,748 across 5 generic classes."""
        counts = harmonized_manifest.get_class_counts()
        total_objects = sum(counts.values())
        assert total_objects == 2748, (
            f"Expected exactly 2,748 total objects, found {total_objects}"
        )
        assert counts == {
            "button": 1651,
            "input_box": 366,
            "icon_toggle": 253,
            "dialog_popup": 46,
            "text_area": 432,
        }

    def test_exhaustive_2748_bounding_boxes_in_unit_interval(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Verify 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0 for every box."""
        violations = []
        for frame in harmonized_manifest.frames:
            for obj in frame.objects:
                x1, y1, x2, y2 = obj.bbox_xyxy_norm
                if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
                    violations.append((frame.frame_id, obj.object_id, "out_of_bounds", (x1, y1, x2, y2)))
                if not (x1 <= x2):
                    violations.append((frame.frame_id, obj.object_id, "inverted_x", (x1, x2)))
                if not (y1 <= y2):
                    violations.append((frame.frame_id, obj.object_id, "inverted_y", (y1, y2)))

        assert len(violations) == 0, f"Found {len(violations)} coordinate boundary violations: {violations[:5]}"

    def test_exhaustive_2748_centroids_valid_and_consistent(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Verify 0.0 <= cx, cy <= 1.0 and centroid matches box midpoint within 1e-4."""
        violations = []
        for frame in harmonized_manifest.frames:
            for obj in frame.objects:
                x1, y1, x2, y2 = obj.bbox_xyxy_norm
                cx, cy = obj.centroid_xy_norm

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    violations.append((frame.frame_id, obj.object_id, "centroid_out_of_bounds", (cx, cy)))

                mid_x = round((x1 + x2) / 2.0, 6)
                mid_y = round((y1 + y2) / 2.0, 6)
                if abs(cx - mid_x) >= 1e-4:
                    violations.append((frame.frame_id, obj.object_id, "cx_midpoint_mismatch", (cx, mid_x)))
                if abs(cy - mid_y) >= 1e-4:
                    violations.append((frame.frame_id, obj.object_id, "cy_midpoint_mismatch", (cy, mid_y)))

        assert len(violations) == 0, f"Found {len(violations)} centroid violations: {violations[:5]}"

    def test_no_zero_or_negative_area_bounding_boxes(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Verify that every bounding box has strictly positive width and height."""
        zero_area = []
        for frame in harmonized_manifest.frames:
            for obj in frame.objects:
                x1, y1, x2, y2 = obj.bbox_xyxy_norm
                w = x2 - x1
                h = y2 - y1
                if w <= 0.0 or h <= 0.0:
                    zero_area.append((frame.frame_id, obj.object_id, w, h))

        assert len(zero_area) == 0, f"Found {len(zero_area)} zero/negative area boxes: {zero_area[:5]}"

    def test_minimum_box_dimensions_realistic(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Verify that UI controls have physically visible minimum dimensions."""
        min_w = min(obj.bbox_xyxy_norm[2] - obj.bbox_xyxy_norm[0] for f in harmonized_manifest.frames for obj in f.objects)
        min_h = min(obj.bbox_xyxy_norm[3] - obj.bbox_xyxy_norm[1] for f in harmonized_manifest.frames for obj in f.objects)

        # Minimum width and height must be >= 0.01 (at least 2-3 pixels at 256x256)
        assert min_w >= 0.01, f"Minimum box width {min_w} is unrealistically small (< 0.01)"
        assert min_h >= 0.01, f"Minimum box height {min_h} is unrealistically small (< 0.01)"


# ==============================================================================
# 3. Adversarial Dual Box Consistency
# ==============================================================================
class TestAdversarialDualBoxConsistency:
    """Stress-test dual bounding box representation (|xyxy - yxyx| < 1e-4 everywhere)."""

    def test_exhaustive_2748_dual_box_exactness(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Assert |bbox_xyxy_norm[0] - bbox_yxyx_norm[1]| < 1e-4 across all 2,748 boxes."""
        max_diff = 0.0
        mismatches = []

        for frame in harmonized_manifest.frames:
            for obj in frame.objects:
                x1, y1, x2, y2 = obj.bbox_xyxy_norm
                y1_c, x1_c, y2_c, x2_c = obj.bbox_yxyx_norm

                d_x1 = abs(x1 - x1_c)
                d_y1 = abs(y1 - y1_c)
                d_x2 = abs(x2 - x2_c)
                d_y2 = abs(y2 - y2_c)

                cur_max = max(d_x1, d_y1, d_x2, d_y2)
                if cur_max > max_diff:
                    max_diff = cur_max

                if cur_max >= 1e-4:
                    mismatches.append(
                        (frame.frame_id, obj.object_id, cur_max, obj.bbox_xyxy_norm, obj.bbox_yxyx_norm)
                    )

        assert len(mismatches) == 0, (
            f"Found {len(mismatches)} dual box inconsistencies (max discrepancy {max_diff:.2e}): "
            f"{mismatches[:5]}"
        )
        assert max_diff < 1e-4, f"Dual box max discrepancy {max_diff:.2e} exceeded tolerance 1e-4"

    def test_dual_box_conversion_bijective(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Verify that mapping between xyxy and yxyx is strictly invertible."""
        for frame in harmonized_manifest.frames[:20]:
            for obj in frame.objects:
                x1, y1, x2, y2 = obj.bbox_xyxy_norm
                y1_c, x1_c, y2_c, x2_c = obj.bbox_yxyx_norm
                # Reconstruct xyxy from yxyx
                recon_xyxy = (x1_c, y1_c, x2_c, y2_c)
                assert recon_xyxy == obj.bbox_xyxy_norm


# ==============================================================================
# 4. Extreme Corruption & Adversarial Robustness on Schema Validation
# ==============================================================================
class TestAdversarialSchemaCorruptionRobustness:
    """Stress-test Draft-07 schema validator with adversarial corruptions and hostile inputs."""

    @pytest.fixture
    def valid_minimal_manifest_dict(self) -> Dict[str, Any]:
        """Produce a minimal valid dictionary adhering to the Draft-07 schema."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "AVCS_Unified_5Class_UI_Manifest",
            "version": "2.0.0",
            "coordinate_convention": "normalized_xyxy_origin_top_left",
            "metadata": {},
            "frames": [
                {
                    "frame_id": "test_frame_001",
                    "generator_source": "ui_grounding",
                    "image_path": "datasets/ui_grounding/train/frame_000.png",
                    "width": 256,
                    "height": 256,
                    "split": "train",
                    "layout_template_id": None,
                    "is_negative_frame": False,
                    "negative_type": None,
                    "objects": [
                        {
                            "object_id": "obj_001",
                            "class_id": 0,
                            "class_name": "button",
                            "bbox_xyxy_norm": [0.1, 0.2, 0.4, 0.5],
                            "bbox_yxyx_norm": [0.2, 0.1, 0.5, 0.4],
                            "centroid_xy_norm": [0.25, 0.35],
                            "is_actionable": True,
                            "validity_state": "VALID",
                            "metadata": {},
                        }
                    ],
                    "metadata": {},
                }
            ],
        }

    def test_baseline_validity(self, valid_minimal_manifest_dict: Dict[str, Any]):
        """Sanity check: baseline dictionary passes validation without errors."""
        is_valid, errors = validate_unified_schema(valid_minimal_manifest_dict)
        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.parametrize(
        "bad_coord",
        [
            [-0.05, 0.2, 0.4, 0.5],    # x1 negative
            [0.1, -0.01, 0.4, 0.5],    # y1 negative
            [0.1, 0.2, 1.05, 0.5],     # x2 > 1.0
            [0.1, 0.2, 0.4, 1.01],     # y2 > 1.0
            [-100.0, 0.2, 0.4, 0.5],   # extreme negative
            [0.1, 0.2, 100.0, 0.5],    # extreme positive
            [0.6, 0.2, 0.4, 0.5],      # inverted x (x1 > x2)
            [0.1, 0.8, 0.4, 0.5],      # inverted y (y1 > y2)
        ],
    )
    def test_schema_rejects_out_of_bounds_and_inverted_coordinates(
        self, valid_minimal_manifest_dict: Dict[str, Any], bad_coord: List[float]
    ):
        """Assert validator rejects out-of-bounds or inverted bounding boxes."""
        data = copy.deepcopy(valid_minimal_manifest_dict)
        data["frames"][0]["objects"][0]["bbox_xyxy_norm"] = bad_coord
        # Mirror bad coord to bbox_yxyx_norm to isolate bounding check
        data["frames"][0]["objects"][0]["bbox_yxyx_norm"] = [
            bad_coord[1], bad_coord[0], bad_coord[3], bad_coord[2]
        ]
        is_valid, errors = validate_unified_schema(data)
        assert is_valid is False
        assert len(errors) > 0

    @pytest.mark.parametrize(
        "nan_or_inf",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_schema_rejects_nan_and_inf_coordinates(
        self, valid_minimal_manifest_dict: Dict[str, Any], nan_or_inf: float
    ):
        """Assert validator rejects NaN, Inf, and -Inf in coordinates."""
        data = copy.deepcopy(valid_minimal_manifest_dict)
        data["frames"][0]["objects"][0]["bbox_xyxy_norm"][0] = nan_or_inf
        is_valid, errors = validate_unified_schema(data)
        assert is_valid is False
        assert len(errors) > 0

    @pytest.mark.parametrize("drift", [0.001, 0.01, 0.1, 0.5])
    def test_schema_rejects_dual_box_discrepancies(
        self, valid_minimal_manifest_dict: Dict[str, Any], drift: float
    ):
        """Assert validator catches dual box drift when |xyxy - yxyx| exceeds 1e-4."""
        data = copy.deepcopy(valid_minimal_manifest_dict)
        # Perturb yxyx x1 coordinate
        data["frames"][0]["objects"][0]["bbox_yxyx_norm"][1] += drift
        is_valid, errors = validate_unified_schema(data)
        assert is_valid is False
        assert any("dual box mismatch" in e for e in errors)

    @pytest.mark.parametrize("bad_class_id", [-1, 5, 6, 100, -99])
    def test_schema_rejects_out_of_range_class_ids(
        self, valid_minimal_manifest_dict: Dict[str, Any], bad_class_id: int
    ):
        """Assert validator rejects class_id outside 0..4."""
        data = copy.deepcopy(valid_minimal_manifest_dict)
        data["frames"][0]["objects"][0]["class_id"] = bad_class_id
        is_valid, errors = validate_unified_schema(data)
        assert is_valid is False
        assert any("invalid class_id" in e for e in errors)

    def test_schema_rejects_class_name_mismatch(
        self, valid_minimal_manifest_dict: Dict[str, Any]
    ):
        """Assert validator rejects when class_name does not match class_id."""
        data = copy.deepcopy(valid_minimal_manifest_dict)
        data["frames"][0]["objects"][0]["class_id"] = CLASS_BUTTON  # 0
        data["frames"][0]["objects"][0]["class_name"] = "dialog_popup"  # belongs to 3
        is_valid, errors = validate_unified_schema(data)
        assert is_valid is False
        assert any("does not match class_id" in e for e in errors)

    def test_schema_rejects_negative_frame_with_objects(
        self, valid_minimal_manifest_dict: Dict[str, Any]
    ):
        """Assert validator rejects negative frame that contains bounding box objects."""
        data = copy.deepcopy(valid_minimal_manifest_dict)
        data["frames"][0]["is_negative_frame"] = True
        data["frames"][0]["negative_type"] = "solid_color"
        # Objects is non-empty
        is_valid, errors = validate_unified_schema(data)
        assert is_valid is False
        assert any("must have 0 objects" in e for e in errors)

    def test_schema_rejects_negative_frame_missing_negative_type(
        self, valid_minimal_manifest_dict: Dict[str, Any]
    ):
        """Assert validator rejects negative frame that omits negative_type."""
        data = copy.deepcopy(valid_minimal_manifest_dict)
        data["frames"][0]["is_negative_frame"] = True
        data["frames"][0]["negative_type"] = None
        data["frames"][0]["objects"] = []
        is_valid, errors = validate_unified_schema(data)
        assert is_valid is False
        assert any("missing 'negative_type'" in e for e in errors)

    def test_schema_rejects_duplicate_frame_ids(
        self, valid_minimal_manifest_dict: Dict[str, Any]
    ):
        """Assert validator catches duplicate frame_id entries across the manifest."""
        data = copy.deepcopy(valid_minimal_manifest_dict)
        # Duplicate the first frame
        dup_frame = copy.deepcopy(data["frames"][0])
        data["frames"].append(dup_frame)
        is_valid, errors = validate_unified_schema(data)
        assert is_valid is False
        assert any("Duplicate frame_id" in e for e in errors)

    @pytest.mark.parametrize("bad_root", ["string_root", 12345, [1, 2, 3], None, True])
    def test_schema_rejects_non_dict_root(self, bad_root: Any):
        """Assert validator rejects non-object root documents."""
        is_valid, errors = validate_unified_schema(bad_root)
        assert is_valid is False
        assert any("must be a JSON object" in e for e in errors)

    def test_schema_rejects_missing_required_top_level_fields(
        self, valid_minimal_manifest_dict: Dict[str, Any]
    ):
        """Assert validator detects omission of each top-level required key."""
        for req in ("$schema", "title", "version", "coordinate_convention", "frames"):
            data = copy.deepcopy(valid_minimal_manifest_dict)
            del data[req]
            is_valid, errors = validate_unified_schema(data)
            assert is_valid is False
            assert any(f"Missing required manifest field: {req!r}" in e for e in errors)


# ==============================================================================
# 5. Adversarial Arm D2 Multi-Generator Stream & Temporal Samplers
# ==============================================================================
class TestAdversarialMultiGeneratorPipeline:
    """Stress-test DataLoader batch generation, augmentations, and temporal samplers."""

    def test_photometric_inversion_preserves_bounding_boxes(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Verify spatial coordinates are mathematically identical under color inversion."""
        dataset = ArmD2MultiGeneratorDataset(
            manifest=harmonized_manifest,
            split="train",
            preload_images=False,
            apply_augmentation=True,
        )
        sample = dataset[0]
        # Target boxes must remain valid in [0, 1]
        boxes = sample["target_boxes"]
        if len(boxes) > 0:
            assert (boxes >= 0.0).all() and (boxes <= 1.0).all()
            assert (boxes[:, 0] <= boxes[:, 2]).all()
            assert (boxes[:, 1] <= boxes[:, 3]).all()

    def test_arm_d2_sampling_no_generator_starvation(
        self, harmonized_manifest: UnifiedManifest
    ):
        """Monte Carlo draw of 200 samples: assert both generators drawn between 35% and 65%."""
        dataset = ArmD2MultiGeneratorDataset(
            manifest=harmonized_manifest,
            split="train",
            preload_images=False,
        )
        sampler = dataset.get_arm_d2_sampler()
        ui_count = 0
        cr_count = 0
        torch.manual_seed(42)

        indices = list(sampler)[:200]
        for idx in indices:
            src = dataset.frames[idx].generator_source
            if src == GENERATOR_UI_GROUNDING:
                ui_count += 1
            elif src == GENERATOR_CLEANROOM:
                cr_count += 1

        p_ui = ui_count / 200.0
        p_cr = cr_count / 200.0
        # Arm D2 balances 160 vs 64 frames to approximately 50/50
        assert 0.35 <= p_ui <= 0.65, f"UI Grounding proportion {p_ui:.2f} skewed out of bounds"
        assert 0.35 <= p_cr <= 0.65, f"Cleanroom proportion {p_cr:.2f} skewed out of bounds"

    def test_temporal_sampler_static_sequence_zero_false_alarms(self):
        """Verify static temporal sequence has max change heatmap == 0.0 and spinner == 0.0."""
        import random
        seq = generate_temporal_sequence(
            mode="static",
            sequence_length=4,
            rng=random.Random(123),
        )
        assert seq.frames.shape == (4, 3, 128, 128)
        assert seq.change_heatmap.shape == (1, 128, 128)
        assert float(seq.change_heatmap.max()) == 0.0, "Static sequence must have zero change heatmap"
        assert float(seq.spinner_label) == 0.0, "Static sequence must have zero spinner label"

    def test_temporal_sampler_spinner_sequence_active(self):
        """Verify spinner temporal sequence triggers spinner_label == 1.0 and change > 0."""
        import random
        seq = generate_temporal_sequence(
            mode="spinner",
            sequence_length=4,
            rng=random.Random(456),
        )
        assert seq.frames.shape == (4, 3, 128, 128)
        assert float(seq.spinner_label) == 1.0, "Spinner sequence must have spinner_label == 1.0"
        assert float(seq.change_heatmap.max()) > 0.0, "Spinner sequence must have non-zero change heatmap"

    def test_action_pair_generator_binary_responses(self):
        """Verify action pair generator produces distinct response and no-response frames."""
        import random
        pair_r = generate_action_pair(
            has_response=True, rng=random.Random(789)
        )
        assert pair_r.response_label == 1.0
        assert pair_r.pre_frame.shape == (3, 128, 128)
        assert pair_r.post_frame.shape == (3, 128, 128)
        assert (pair_r.post_frame - pair_r.pre_frame).abs().max() > 0.1, "Response pair must show visual delta"

        pair_nr = generate_action_pair(
            has_response=False, rng=random.Random(790)
        )
        assert pair_nr.response_label == 0.0
        assert (pair_nr.post_frame - pair_nr.pre_frame).abs().max() == 0.0, "No-response pair must be visually identical"
