from __future__ import annotations

import sys
import time

import numpy as np
import pytest
import torch

from avcs.temporal_cortex import (
    TemporalVisualCortex,
    temporal_loss_for_frames,
    temporal_ssl_loss,
)
from avcs.training import collect_temporal, evaluate_temporal
from avcs.training.train_temporal import (
    FramePairBuffer,
    compare_temporal_cortex_on_held_out_stream,
    evaluate_temporal_cortex,
    train_temporal_cortex,
)


def test_temporal_model_returns_latent_motion_and_surprise():
    model = TemporalVisualCortex(feature_dim=32, latent_dim=48)
    previous = torch.rand(2, 3, 64, 80)
    current = torch.rand(2, 3, 64, 80)

    output = model(previous, current)

    assert output["latent"].shape == (2, 48)
    assert output["predicted_latent"].shape == (2, 48)
    assert output["motion"].shape == (2, 2, 8, 10)
    assert output["surprise"].shape == (2, 1, 64, 80)
    assert torch.isfinite(output["latent"]).all()
    assert torch.isfinite(output["motion"]).all()
    assert torch.all(output["motion"].abs() <= model.max_displacement)


def test_temporal_loss_backpropagates_without_labels():
    model = TemporalVisualCortex(feature_dim=32, latent_dim=48)
    previous = torch.rand(3, 3, 160, 192)
    current = torch.rand(3, 3, 160, 192)

    losses = temporal_loss_for_frames(model, previous, current)
    losses["total"].backward()

    assert {"prediction", "variance", "decorrelation", "photometric", "motion_smoothness"} <= losses.keys()
    assert torch.isfinite(losses["total"])
    assert model.encoder[0][0].weight.grad is not None


def test_temporal_loss_requires_multiple_pairs():
    model = TemporalVisualCortex(feature_dim=16, latent_dim=16)
    with pytest.raises(ValueError, match="batch of at least two"):
        temporal_loss_for_frames(model, torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))


def test_change_weighting_emphasizes_changed_pixels_without_zeroing_static_pixels():
    source = torch.zeros(2, 3, 8, 8)
    target = source.clone()
    target[0, :, :4, :4] = 1.0
    outputs = {
        "latent": torch.randn(2, 4),
        "predicted_latent": torch.randn(2, 4),
        "warped_previous": source,
        "photometric_source": source,
        "photometric_target": target,
        "motion": torch.zeros(2, 2, 4, 4),
    }

    uniform = temporal_ssl_loss(outputs, change_weight=0.0)
    weighted = temporal_ssl_loss(outputs, change_threshold=0.1, change_weight=4.0)

    assert torch.isclose(uniform["changed_pixel_fraction"], torch.tensor(0.125))
    assert torch.isclose(weighted["changed_pixel_fraction"], uniform["changed_pixel_fraction"])
    assert weighted["photometric"] > uniform["photometric"]
    assert torch.isfinite(weighted["total"])
    with pytest.raises(ValueError, match="change_weight"):
        temporal_ssl_loss(outputs, change_weight=-1.0)
    with pytest.raises(ValueError, match="change_threshold"):
        temporal_ssl_loss(outputs, change_threshold=1.1)


def test_frame_pair_buffer_is_bounded_and_trains_in_memory(tmp_path):
    buffer = FramePairBuffer(capacity=3, frame_size=(32, 32))
    for shift in range(4):
        previous = np.zeros((48, 64, 3), dtype=np.uint8)
        current = np.zeros_like(previous)
        previous[10:20, 10 + shift:20 + shift] = 200
        current[10:20, 11 + shift:21 + shift] = 200
        buffer.append(previous, current)

    assert len(buffer) == 3
    before, after = buffer.batch([0, 1], device=torch.device("cpu"))
    assert before.shape == after.shape == (2, 3, 32, 32)
    assert before.dtype == after.dtype == torch.float32
    model = TemporalVisualCortex(feature_dim=16, latent_dim=16)
    checkpoint_path = tmp_path / "temporal.pt"
    history = train_temporal_cortex(
        model,
        buffer,
        epochs=1,
        batch_size=2,
        seed=4,
        model_init_seed=17,
        checkpoint_path=checkpoint_path,
    )
    assert len(history) == 1
    assert history[0]["batches"] == 1.0
    assert np.isfinite(history[0]["total"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["model"] == "TemporalVisualCortex"
    assert checkpoint["evaluation_status"] == "not_evaluated"
    assert checkpoint["training_config"]["training_pairs"] == 3
    assert checkpoint["training_config"]["model_init_seed"] == 17
    assert checkpoint["data_provenance"] == {}


def test_held_out_evaluation_compares_matched_random_and_trained_models(tmp_path):
    training_buffer = FramePairBuffer(capacity=4, frame_size=(32, 32))
    held_out_buffer = FramePairBuffer(capacity=5, frame_size=(32, 32))
    for index in range(4):
        frame_a = np.full((32, 48, 3), index * 20, dtype=np.uint8)
        frame_b = np.full((32, 48, 3), index * 20 + 5, dtype=np.uint8)
        training_buffer.append(frame_a, frame_b)
    for index in range(5):
        frame_a = np.full((48, 32, 3), index * 15, dtype=np.uint8)
        frame_b = np.full((48, 32, 3), index * 15 + 7, dtype=np.uint8)
        held_out_buffer.append(frame_a, frame_b)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        model = TemporalVisualCortex(feature_dim=16, latent_dim=16)
    checkpoint_path = tmp_path / "stream_a.pt"
    train_temporal_cortex(
        model,
        training_buffer,
        epochs=1,
        batch_size=2,
        seed=5,
        model_init_seed=23,
        checkpoint_path=checkpoint_path,
    )

    report = compare_temporal_cortex_on_held_out_stream(
        checkpoint_path,
        held_out_buffer,
        batch_size=2,
    )

    assert report["held_out_pairs"] == 5
    assert report["raw_frames_saved"] is False
    assert "not evidence" in report["interpretation"]
    assert set(report["trained_minus_random"]) >= {
        "predicted_motion_magnitude_norm_mean",
        "surprise_mean",
        "latent_std_mean",
        "latent_covariance_offdiagonal_rms",
    }
    for key in ("random_initialization", "trained_checkpoint"):
        metrics = report[key]
        assert metrics["evaluated_pairs"] == 5
        assert np.isfinite(metrics["latent_std_mean"])
        assert np.isfinite(metrics["latent_std_min"])
        assert np.isfinite(metrics["latent_covariance_offdiagonal_rms"])
        assert np.isfinite(metrics["latent_covariance_offdiagonal_frobenius"])
        assert metrics["predicted_motion_magnitude_norm_mean"] >= 0
        assert metrics["predicted_motion_magnitude_norm_p95"] >= 0
        assert metrics["surprise_mean"] >= 0
        assert metrics["surprise_per_pair_mean_p95"] >= 0
    assert set(report["trained_minus_random"]) >= {
        "total", "prediction", "variance", "decorrelation",
        "photometric", "motion_smoothness",
    }


def test_held_out_baseline_requires_reproducible_initialization_seed(tmp_path):
    buffer = FramePairBuffer(capacity=2, frame_size=(32, 32))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    buffer.append(frame, frame)
    buffer.append(frame, frame)
    model = TemporalVisualCortex(feature_dim=16, latent_dim=16)
    checkpoint_path = tmp_path / "unseeded.pt"
    train_temporal_cortex(
        model,
        buffer,
        epochs=1,
        batch_size=2,
        checkpoint_path=checkpoint_path,
    )

    with pytest.raises(TypeError, match="model_init_seed"):
        compare_temporal_cortex_on_held_out_stream(checkpoint_path, buffer)


def test_held_out_live_capture_requires_explicit_consent(tmp_path):
    args = evaluate_temporal.build_parser().parse_args(
        [
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--regime", "static_micro_motion",
        ]
    )

    with pytest.raises(ValueError, match="opt-in"):
        evaluate_temporal.run_held_out_evaluation(args)


def test_evaluation_reports_latent_health_for_all_pairs():
    buffer = FramePairBuffer(capacity=3, frame_size=(32, 32))
    for value in (0, 64, 128):
        previous = np.full((32, 48, 3), value, dtype=np.uint8)
        current = np.full((32, 48, 3), value + 1, dtype=np.uint8)
        buffer.append(previous, current)
    model = TemporalVisualCortex(feature_dim=16, latent_dim=16)

    metrics = evaluate_temporal_cortex(model, buffer, batch_size=2)

    assert metrics["evaluated_pairs"] == 3
    assert metrics["latent_std_mean"] >= 0
    assert metrics["latent_std_min"] >= 0
    assert metrics["latent_covariance_offdiagonal_rms"] >= 0
    assert metrics["predicted_motion_magnitude_norm_mean"] >= 0
    assert metrics["predicted_motion_magnitude_norm_p95"] >= 0
    assert metrics["surprise_mean"] >= 0
    assert metrics["surprise_per_pair_mean_p95"] >= 0


@pytest.mark.parametrize(
    ("source_shape", "content_bounds"),
    [
        ((32, 64), (0, 16, 64, 48)),
        ((64, 32), (16, 0, 48, 64)),
    ],
)
def test_frame_pair_buffer_letterboxes_without_distorting_aspect_ratio(
    source_shape,
    content_bounds,
):
    height, width = source_shape
    frame = np.full((height, width, 3), (255, 32, 16), dtype=np.uint8)
    buffer = FramePairBuffer(capacity=2, frame_size=(64, 64))

    buffer.append(frame, frame)
    previous, _ = buffer.batch([0], device=torch.device("cpu"))
    image = (previous[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    left, top, right, bottom = content_bounds

    assert image.shape == (64, 64, 3)
    assert np.all(image[top:bottom, left:right] == (255, 32, 16))
    if top:
        assert np.all(image[:top] == 0)
        assert np.all(image[bottom:] == 0)
    if left:
        assert np.all(image[:, :left] == 0)
        assert np.all(image[:, right:] == 0)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32, 4), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.float32),
    ],
)
def test_frame_pair_buffer_rejects_non_rgb_uint8(frame):
    buffer = FramePairBuffer(frame_size=(32, 32))
    with pytest.raises(ValueError, match="uint8 RGB"):
        buffer.append(frame, frame)


def test_model_rejects_invalid_frame_shapes_and_types():
    model = TemporalVisualCortex(feature_dim=16, latent_dim=16)
    with pytest.raises(ValueError, match="identical RGB"):
        model(torch.rand(2, 3, 32, 32), torch.rand(2, 1, 32, 32))
    with pytest.raises(ValueError, match="floating-point"):
        model(torch.zeros(2, 3, 32, 32, dtype=torch.uint8), torch.zeros(2, 3, 32, 32, dtype=torch.uint8))


def test_live_collector_is_explicitly_gated_and_reports_capture(monkeypatch):
    args = collect_temporal.build_parser().parse_args(["--checkpoint", "unused.pt"])
    with pytest.raises(ValueError, match="opt-in"):
        collect_temporal.run_live_training(args)

    class FakeCapture:
        def open(self):
            return self

        def grab(self):
            return np.full((48, 64, 3), int(time.monotonic() * 1000) % 255, dtype=np.uint8)

        def close(self):
            return None

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "avcs.capture_adapter.Win32GDICaptureAdapter",
        FakeCapture,
    )
    buffer = FramePairBuffer(capacity=8, frame_size=(32, 32))
    metrics = collect_temporal.collect_live_pairs(
        buffer,
        duration_seconds=0.2,
        target_hz=20,
        progress_interval=1,
    )
    assert len(buffer) >= 2
    assert metrics["pairs"] == len(buffer)
    assert metrics["capture_calls"] == len(buffer) + 1
    assert metrics["achieved_pair_hz"] > 0


def test_live_collector_pairs_samples_at_configured_stride(monkeypatch):
    class FakeCapture:
        def __init__(self):
            self.count = 0

        def open(self):
            return self

        def grab(self):
            self.count += 1
            return np.full((48, 64, 3), self.count, dtype=np.uint8)

        def close(self):
            return None

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("avcs.capture_adapter.Win32GDICaptureAdapter", FakeCapture)
    buffer = FramePairBuffer(capacity=8, frame_size=(32, 32))

    metrics = collect_temporal.collect_live_pairs(
        buffer,
        duration_seconds=0.22,
        target_hz=30,
        pair_stride=2,
        progress_interval=1,
    )

    assert metrics["pair_stride"] == 2
    assert metrics["nominal_pair_interval_ms"] == pytest.approx(2000 / 30)
    assert metrics["capture_calls"] == metrics["sampled_frames"]
    assert metrics["pairs"] == (metrics["sampled_frames"] - 1) // 2
    assert len(buffer) == metrics["pairs"]
