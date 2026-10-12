from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from avcs.temporal_cortex import TemporalVisualCortex
from avcs.training import compare_temporal_regions
from avcs.training.train_temporal import FramePairBuffer, train_temporal_cortex


def _moving_frame(position: int) -> np.ndarray:
    frame = np.full((64, 96, 3), 24, dtype=np.uint8)
    x = 8 + (position % 24)
    frame[20:36, x:x + 16] = 230
    return frame


def test_crop_changed_region_returns_matched_padded_crops():
    previous = np.full((64, 96, 3), 20, dtype=np.uint8)
    current = previous.copy()
    current[20:28, 40:48] = 220

    region = compare_temporal_regions.crop_changed_region(
        previous,
        current,
        pixel_threshold=8,
        padding_fraction=0.25,
    )

    assert region is not None
    previous_crop, current_crop, changed_fraction, roi_fraction, bounds = region
    assert previous_crop.shape == current_crop.shape
    assert previous_crop.shape[0] >= 32
    assert previous_crop.shape[1] >= 32
    assert changed_fraction == pytest.approx(64 / (64 * 96))
    assert roi_fraction < 1.0
    assert not np.array_equal(previous_crop, current_crop)
    assert bounds == (28, 8, 60, 40)


def test_crop_changed_region_skips_unchanged_pairs_and_validates_inputs():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert compare_temporal_regions.crop_changed_region(
        frame,
        frame.copy(),
        pixel_threshold=8,
        padding_fraction=0.25,
    ) is None

    with pytest.raises(ValueError, match="pixel_threshold"):
        compare_temporal_regions.crop_changed_region(
            frame,
            frame,
            pixel_threshold=256,
            padding_fraction=0.25,
        )


def test_crop_changed_region_respects_minimum_size_at_screen_edges():
    previous = np.zeros((64, 64, 3), dtype=np.uint8)
    current = previous.copy()
    current[:4, :4] = 255

    region = compare_temporal_regions.crop_changed_region(
        previous,
        current,
        pixel_threshold=8,
        padding_fraction=0,
        minimum_size=32,
    )

    assert region is not None
    previous_crop, current_crop, _, roi_fraction, _ = region
    assert previous_crop.shape == current_crop.shape == (32, 32, 3)
    assert roi_fraction == pytest.approx(0.25)


def test_static_control_crop_matches_dimensions_and_has_no_changed_pixels():
    previous = np.zeros((64, 96, 3), dtype=np.uint8)
    current = previous.copy()
    current[20:28, 40:48] = 255
    region = compare_temporal_regions.crop_changed_region(
        previous,
        current,
        pixel_threshold=8,
        padding_fraction=0.25,
    )
    assert region is not None
    active_previous, _, _, _, bounds = region

    control = compare_temporal_regions.crop_static_control_region(
        previous,
        current,
        active_bounds=bounds,
        crop_shape=active_previous.shape[:2],
        pixel_threshold=8,
    )

    assert control is not None
    previous_control, current_control, changed_fraction = control
    assert previous_control.shape == current_control.shape == active_previous.shape
    assert changed_fraction == 0
    assert np.array_equal(previous_control, current_control)


def test_static_control_crop_fails_closed_when_no_nonoverlapping_area_fits():
    previous = np.zeros((64, 64, 3), dtype=np.uint8)
    current = previous.copy()
    current[0, 0] = 255

    control = compare_temporal_regions.crop_static_control_region(
        previous,
        current,
        active_bounds=(0, 0, 64, 64),
        crop_shape=(32, 32),
        pixel_threshold=8,
    )

    assert control is None


def test_static_control_search_finds_static_area_between_edge_candidates():
    previous = np.zeros((96, 96, 3), dtype=np.uint8)
    current = previous.copy()
    current[32:40, 32:40] = 255

    control = compare_temporal_regions.crop_static_control_region(
        previous,
        current,
        active_bounds=(32, 32, 64, 64),
        crop_shape=(24, 24),
        pixel_threshold=8,
    )

    assert control is not None
    previous_control, current_control, changed_fraction = control
    assert previous_control.shape == current_control.shape == (24, 24, 3)
    assert changed_fraction == 0


def test_fixed_active_patch_is_square_centroid_centered_and_clamped():
    previous = np.zeros((64, 96, 3), dtype=np.uint8)
    current = previous.copy()
    current[1:5, 2:6] = 255
    current[3:7, 4:8] = 255

    region = compare_temporal_regions.crop_fixed_changed_region(
        previous,
        current,
        patch_size=32,
        pixel_threshold=8,
    )

    assert region is not None
    previous_patch, current_patch, changed_fraction, patch_changed_fraction, bounds = region
    assert previous_patch.shape == current_patch.shape == (32, 32, 3)
    assert bounds == (0, 0, 32, 32)
    assert changed_fraction == pytest.approx(28 / (64 * 96))
    assert patch_changed_fraction == pytest.approx(28 / (32 * 32))


def test_fixed_active_patch_rejects_size_larger_than_canvas():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="patch_size"):
        compare_temporal_regions.crop_fixed_changed_region(
            frame,
            frame,
            patch_size=48,
            pixel_threshold=8,
        )


def _make_checkpoint(path):
    buffer = FramePairBuffer(capacity=4, frame_size=(64, 64))
    for index in range(4):
        buffer.append(_moving_frame(index), _moving_frame(index + 1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = TemporalVisualCortex(feature_dim=16, latent_dim=16)
    train_temporal_cortex(
        model,
        buffer,
        epochs=1,
        batch_size=2,
        model_init_seed=31,
        checkpoint_path=path,
    )
    return model


def test_region_comparison_uses_same_capture_and_saves_metrics_only(
    monkeypatch,
    tmp_path,
):
    checkpoint_path = tmp_path / "temporal.pt"
    _make_checkpoint(checkpoint_path)

    class FakeCapture:
        def __init__(self):
            self.position = 0

        def open(self):
            return self

        def grab(self):
            frame = _moving_frame(self.position)
            self.position += 1
            return frame

        def close(self):
            return None

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "avcs.capture_adapter.Win32GDICaptureAdapter",
        FakeCapture,
    )
    report_path = tmp_path / "metrics.json"
    args = compare_temporal_regions.build_parser().parse_args(
        [
            "--confirm-live-capture",
            "--checkpoint", str(checkpoint_path),
            "--duration-seconds", "1",
            "--target-hz", "30",
            "--capacity", "8",
            "--frame-size", "64",
            "--fixed-patch-size", "32",
            "--batch-size", "2",
            "--report", str(report_path),
        ]
    )

    report = compare_temporal_regions.run_region_comparison(args)

    capture = report["capture"]
    assert capture["captured_pairs"] >= 2
    assert capture["changed_pairs"] >= 2
    assert capture["paired_triplets"] == capture["changed_pairs"]
    assert report["paired_full_frame"]["evaluated_pairs"] == capture["paired_triplets"]
    assert report["changed_region_crop"]["evaluated_pairs"] == capture["paired_triplets"]
    assert report["matched_control_crop"]["evaluated_pairs"] == capture["paired_triplets"]
    assert report["region_detection"]["coordinates_saved"] is False
    assert report["region_detection"]["control_changed_pixel_fraction_max"] == 0
    assert report["region_detection"]["roi_area_fraction_mean"] == pytest.approx(0.25)
    assert report["raw_frames_saved"] is False
    assert report_path.is_file()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["capture"]["changed_pairs"] == capture["changed_pairs"]
    assert "input-view sensitivity" in saved["interpretation"]
    assert "active_minus_control_same_pairs" in saved
    assert saved["session_summary"]["evidence_unit"] == "single_capture_session"
    assert saved["session_summary"]["paired_triplets"] == capture["paired_triplets"]
    assert saved["session_summary"]["between_session_uncertainty"] is None
    assert saved["session_summary"]["components"]["photometric"]["active_minus_control"] == (
        saved["active_minus_control_same_pairs"]["photometric"]
    )


def test_region_comparison_requires_explicit_live_capture(tmp_path):
    args = compare_temporal_regions.build_parser().parse_args(
        ["--checkpoint", str(tmp_path / "missing.pt")]
    )

    with pytest.raises(ValueError, match="opt-in"):
        compare_temporal_regions.run_region_comparison(args)
