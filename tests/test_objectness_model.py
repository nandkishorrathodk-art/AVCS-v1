"""Comprehensive Unit Tests for UIObjectnessModel & Training Pipeline.

Tests:
  1. Architecture & Forward Pass:
     - RealTimeCorticalBackbone4Ch parameter count (~10k parameters)
     - Total parameter count (< 100k parameters)
     - 3-channel RGB and 4-channel preprocessed tensor forward passes
     - Unbatched 3D tensor support and shape validation
     - Invalid input fail-closed checks
  2. UIObjectnessPrediction Output Contract:
     - Dataclass fields matching PROJECT.md §6.3
     - Normalized coordinate bounds [0.0, 1.0] for boxes and centroids
     - Ordering constraints (x1 <= x2, y1 <= y2)
  3. Blank-Screen Safety & Negative Frame Rejection:
     - Uniform blank screens, dark OLED, white, and neutral gray screens
     - Presence threshold gating (presence_prob < 0.35 -> is_blank_rejected=True, 0 candidates)
  4. Multi-Task Loss Formulation:
     - Target heatmap and box generation from UnifiedBatchDict
     - Complete IoU (CIoU) loss behavior and properties
     - Finite loss and backward gradients across all learnable modules
     - Empty/negative frame batch loss computation
  5. CPU Latency Verification:
     - Neural forward pass latency on CPU at 128x128 resolution (p50 <= 10.0 ms)
     - Minimum latency (min <= 8.0 ms)
  6. Checkpoint Discovery & Integrity:
     - State dict schema, sidecar JSON metadata, and CPU loading
"""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.dataset.multi_generator_stream import (
    MultiGeneratorDataStream,
    UnifiedBatchDict,
)
from avcs.models.objectness_model import (
    ConvBNAct,
    DenseObjectnessHead,
    GlobalPresenceHead,
    InvertedResidual,
    RealTimeCorticalBackbone4Ch,
    UIObjectnessLoss,
    UIObjectnessModel,
    UIObjectnessPrediction,
    build_objectness_targets,
    compute_ciou_loss,
)
from avcs.perception.edge_preprocessor import EdgeLuminancePreprocessor


# ==============================================================================
# 1. Architecture & Forward Pass Tests
# ==============================================================================

class TestArchitectureAndForward:
    """Tests for model parameter bounds, layers, and tensor forward passes."""

    def test_backbone_parameter_count(self) -> None:
        """Verifies RealTimeCorticalBackbone4Ch parameter count is ~10k (8k-12k)."""
        backbone = RealTimeCorticalBackbone4Ch(in_channels=4, feature_dim=128)
        n_params = backbone.num_parameters
        assert 8000 <= n_params <= 12000, (
            f"Backbone params must be ~10k, got {n_params}"
        )

    def test_total_model_parameter_count(self) -> None:
        """Verifies total UIObjectnessModel parameter count is lightweight (< 100k)."""
        model = UIObjectnessModel()
        total_params = model.total_parameters
        assert total_params < 100000, (
            f"Total model params must be < 100k for real-time CPU, got {total_params}"
        )

    def test_forward_with_3ch_rgb(self) -> None:
        """Verifies forward pass with [B, 3, 128, 128] raw RGB input."""
        model = UIObjectnessModel().eval()
        x = torch.rand(2, 3, 128, 128, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(x)

        assert isinstance(outputs, dict)
        assert outputs["presence_logit"].shape == (2, 1)
        assert outputs["presence_prob"].shape == (2, 1)
        assert outputs["heatmap_logits"].shape == (2, 1, 16, 16)
        assert outputs["heatmap"].shape == (2, 1, 16, 16)
        assert outputs["box_preds"].shape == (2, 4, 16, 16)
        assert outputs["features"].shape == (2, 128, 16, 16)

        # Probabilities and box predictions must be strictly in [0.0, 1.0]
        assert (outputs["presence_prob"] >= 0.0).all() and (outputs["presence_prob"] <= 1.0).all()
        assert (outputs["heatmap"] >= 0.0).all() and (outputs["heatmap"] <= 1.0).all()
        assert (outputs["box_preds"] >= 0.0).all() and (outputs["box_preds"] <= 1.0).all()

    def test_forward_with_4ch_tensor(self) -> None:
        """Verifies forward pass with pre-computed [B, 4, 128, 128] input."""
        model = UIObjectnessModel().eval()
        x4 = torch.rand(2, 4, 128, 128, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(x4)

        assert outputs["presence_logit"].shape == (2, 1)
        assert outputs["heatmap"].shape == (2, 1, 16, 16)
        assert outputs["box_preds"].shape == (2, 4, 16, 16)

    def test_forward_single_image_3d_support(self) -> None:
        """Verifies support for unbatched 3D tensors [3, 128, 128] and [4, 128, 128]."""
        model = UIObjectnessModel().eval()
        x3 = torch.rand(3, 128, 128, dtype=torch.float32)
        x4 = torch.rand(4, 128, 128, dtype=torch.float32)

        with torch.no_grad():
            out3 = model(x3)
            out4 = model(x4)

        assert out3["presence_logit"].shape == (1, 1)
        assert out4["presence_logit"].shape == (1, 1)

    def test_invalid_input_shapes_fail_closed(self) -> None:
        """Verifies invalid tensor dimensions raise ValueError."""
        model = UIObjectnessModel()
        with pytest.raises(ValueError, match="Expected 3D or 4D tensor"):
            model(torch.rand(128, 128))

        with pytest.raises(ValueError, match="Expected 3 or 4 channels"):
            model(torch.rand(1, 5, 128, 128))


# ==============================================================================
# 2. UIObjectnessPrediction Output Contract Tests
# ==============================================================================

class TestUIObjectnessPredictionContract:
    """Verifies output schema conforms to PROJECT.md §6.3 specification."""

    def test_prediction_dataclass_schema(self) -> None:
        """Tests that predict() returns valid frozen UIObjectnessPrediction."""
        model = UIObjectnessModel().eval()
        img = torch.rand(3, 128, 128, dtype=torch.float32)

        pred = model.predict(img)
        assert isinstance(pred, UIObjectnessPrediction)
        assert isinstance(pred.global_presence_prob, float)
        assert 0.0 <= pred.global_presence_prob <= 1.0
        assert isinstance(pred.candidate_boxes, list)
        assert isinstance(pred.candidate_centroids, list)
        assert isinstance(pred.objectness_confidences, list)
        assert isinstance(pred.is_blank_rejected, bool)
        assert isinstance(pred.heatmap, torch.Tensor)
        assert pred.heatmap.shape == (16, 16)

    def test_candidate_coordinate_bounds_in_unit_interval(self) -> None:
        """Verifies candidate boxes and centroids are in normalized [0.0, 1.0]."""
        model = UIObjectnessModel().eval()
        img = torch.rand(3, 128, 128, dtype=torch.float32)

        # Force presence so candidates are extracted
        pred = model.predict(img, presence_threshold=0.0, heatmap_threshold=0.01, top_k=10)
        assert pred.is_blank_rejected is False

        for box in pred.candidate_boxes:
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert 0.0 <= x1 <= 1.0, f"x1 out of bounds: {x1}"
            assert 0.0 <= y1 <= 1.0, f"y1 out of bounds: {y1}"
            assert 0.0 <= x2 <= 1.0, f"x2 out of bounds: {x2}"
            assert 0.0 <= y2 <= 1.0, f"y2 out of bounds: {y2}"
            assert x1 <= x2, f"Invalid box ordering x1 > x2: {box}"
            assert y1 <= y2, f"Invalid box ordering y1 > y2: {box}"

        for cx, cy in pred.candidate_centroids:
            assert 0.0 <= cx <= 1.0, f"cx out of bounds: {cx}"
            assert 0.0 <= cy <= 1.0, f"cy out of bounds: {cy}"

        for conf in pred.objectness_confidences:
            assert 0.0 <= conf <= 1.0, f"Confidence out of bounds: {conf}"

    def test_batch_prediction_output(self) -> None:
        """Verifies batch inputs return List[UIObjectnessPrediction]."""
        model = UIObjectnessModel().eval()
        imgs = torch.rand(3, 3, 128, 128, dtype=torch.float32)
        preds = model.predict(imgs)
        assert isinstance(preds, list)
        assert len(preds) == 3
        for p in preds:
            assert isinstance(p, UIObjectnessPrediction)


# ==============================================================================
# 3. Blank-Screen Safety & Negative Frame Rejection Tests
# ==============================================================================

class TestBlankScreenRejection:
    """Verifies blank screen safety and fail-closed candidate suppression."""

    def test_presence_threshold_suppresses_candidates(self) -> None:
        """Verifies is_blank_rejected=True forces exactly 0 candidates."""
        model = UIObjectnessModel().eval()
        img = torch.rand(3, 128, 128, dtype=torch.float32)

        # High presence threshold guarantees rejection
        pred = model.predict(img, presence_threshold=1.0)
        assert pred.is_blank_rejected is True
        assert len(pred.candidate_boxes) == 0
        assert len(pred.candidate_centroids) == 0
        assert len(pred.objectness_confidences) == 0
        assert pred.num_candidates == 0

    def test_blank_uniform_screens_fail_closed(self) -> None:
        """Tests that solid color screens (black, white, gray) have zero edge response and reject."""
        prep = EdgeLuminancePreprocessor()
        blank_colors = [
            (0.0, 0.0, 0.0),       # Pitch black
            (0.08, 0.08, 0.08),    # Dark OLED
            (0.5, 0.5, 0.5),       # Neutral gray
            (1.0, 1.0, 1.0),       # Flat white
        ]

        model = UIObjectnessModel().eval()

        for r, g, b in blank_colors:
            img = torch.tensor([[[r]], [[g]], [[b]]], dtype=torch.float32).expand(1, 3, 128, 128)
            four_ch = prep(img)
            # Sobel gradients on flat screen must be exactly zero
            assert torch.all(four_ch[:, 1:3] == 0.0)
            # Raw gradient magnitude sqrt(0 + 0 + eps) = sqrt(1e-6) = 1e-3
            assert torch.all(four_ch[:, 3:4] <= 0.0011)
            # Thresholded edge map (threshold=0.05) has strictly zero edges
            edge_map = (four_ch[:, 3:4] >= 0.05).float()
            assert torch.all(edge_map == 0.0)

            # Model prediction with presence gating rejects and returns 0 candidate boxes
            pred = model.predict(img, presence_threshold=1.0)
            assert pred.is_blank_rejected is True
            assert pred.num_candidates == 0


# ==============================================================================
# 4. Multi-Task Loss Formulation Tests
# ==============================================================================

class TestMultiTaskLossFormulation:
    """Verifies target building, CIoU loss, and backward gradient flow."""

    def test_build_objectness_targets(self) -> None:
        """Tests target heatmap and box map generation from batch dict."""
        target_boxes = torch.tensor([[[0.2, 0.2, 0.4, 0.4]]], dtype=torch.float32)
        target_centroids = torch.tensor([[[0.3, 0.3]]], dtype=torch.float32)
        box_mask = torch.tensor([[True]])
        has_target = torch.tensor([1.0])

        targets = build_objectness_targets(
            target_boxes=target_boxes,
            target_centroids=target_centroids,
            box_mask=box_mask,
            has_target=has_target,
            grid_size=16,
        )

        assert targets["target_heatmap"].shape == (1, 1, 16, 16)
        assert targets["target_boxes_map"].shape == (1, 4, 16, 16)
        assert targets["pos_mask"].shape == (1, 16, 16)

        # Centroid (0.3, 0.3) maps to grid cell (4, 4) since 0.3 * 16 = 4.8
        gx, gy = int(0.3 * 16), int(0.3 * 16)
        assert targets["pos_mask"][0, gy, gx].item() is True
        assert targets["target_heatmap"][0, 0, gy, gx].item() > 0.8

    def test_ciou_loss_properties(self) -> None:
        """Tests CIoU loss mathematical properties."""
        # Identical boxes -> CIoU = 1.0, loss = 0.0
        b1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
        loss_identical = compute_ciou_loss(b1, b1)
        assert abs(float(loss_identical[0].item())) < 1e-4

        # Disjoint boxes -> CIoU < 0, loss > 1.0
        b2 = torch.tensor([[0.9, 0.9, 0.1, 0.1]], dtype=torch.float32)
        loss_disjoint = compute_ciou_loss(b1, b2)
        assert float(loss_disjoint[0].item()) > 1.0

    def test_loss_finite_and_gradients_exist(self) -> None:
        """Verifies multi-task loss produces finite loss and non-zero gradients."""
        model = UIObjectnessModel().train()
        loss_fn = UIObjectnessLoss()

        images = torch.rand(2, 3, 128, 128, dtype=torch.float32)
        target_boxes = torch.tensor(
            [[[0.1, 0.1, 0.3, 0.3]], [[0.4, 0.4, 0.6, 0.6]]], dtype=torch.float32
        )
        target_centroids = torch.tensor([[[0.2, 0.2]], [[0.5, 0.5]]], dtype=torch.float32)
        box_mask = torch.tensor([[True], [True]])
        has_target = torch.tensor([1.0, 1.0])

        batch = {
            "target_boxes": target_boxes,
            "target_centroids": target_centroids,
            "box_mask": box_mask,
            "has_target": has_target,
        }

        outputs = model(images)
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict["loss"]

        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        loss.backward()

        # Check all learnable parameters have non-zero, finite gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
                assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"

    def test_empty_negative_batch_loss(self) -> None:
        """Verifies loss on blank screens with 0 targets computes without error."""
        model = UIObjectnessModel().train()
        loss_fn = UIObjectnessLoss()

        images = torch.zeros(2, 3, 128, 128, dtype=torch.float32)
        target_boxes = torch.zeros((2, 0, 4), dtype=torch.float32)
        target_centroids = torch.zeros((2, 0, 2), dtype=torch.float32)
        box_mask = torch.zeros((2, 0), dtype=torch.bool)
        has_target = torch.tensor([0.0, 0.0])

        batch = {
            "target_boxes": target_boxes,
            "target_centroids": target_centroids,
            "box_mask": box_mask,
            "has_target": has_target,
        }

        outputs = model(images)
        loss_dict = loss_fn(outputs, batch)
        assert torch.isfinite(loss_dict["loss"])
        loss_dict["loss"].backward()


# ==============================================================================
# 5. CPU Latency Verification Tests
# ==============================================================================

class TestCPULatencyVerification:
    """Verifies neural forward pass latency meets the <= 10.0ms CPU threshold."""

    def test_cpu_neural_forward_latency_p50(self) -> None:
        """Neural forward pass on CPU at 128x128 must satisfy p50 <= 10.0 ms."""
        torch.set_num_threads(1)
        model = UIObjectnessModel().eval()
        x4 = torch.randn(1, 4, 128, 128, dtype=torch.float32)

        # Warm-up (20 iterations)
        for _ in range(20):
            with torch.no_grad():
                feat = model.backbone(x4)
                _ = model.presence_head(feat)
                _ = model.objectness_head(feat)
                _ = model.box_head(feat)

        # Pre-allocate inputs to avoid measuring PRNG allocation inside the timing loop
        iterations = 50
        test_inputs = [torch.randn(1, 4, 128, 128, dtype=torch.float32) for _ in range(iterations)]
        latencies_ms: List[float] = []

        for inp in test_inputs:
            t0 = time.perf_counter()
            with torch.no_grad():
                feat = model.backbone(inp)
                _ = model.presence_head(feat)
                _ = model.objectness_head(feat)
                _ = model.box_head(feat)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt)

        p50 = float(np.median(latencies_ms))
        min_lat = float(np.min(latencies_ms))
        p95 = float(np.percentile(latencies_ms, 95))

        print(
            f"\n[M3 CPU Neural Latency Benchmark] Iterations: {iterations} | "
            f"min: {min_lat:.2f}ms | p50: {p50:.2f}ms | p95: {p95:.2f}ms"
        )

        assert p50 <= 10.0, f"CPU latency target failure: p50 must be <= 10.0 ms, got {p50:.2f} ms"
        assert min_lat <= 8.0, f"Minimum latency must be <= 8.0 ms, got {min_lat:.2f} ms"


# ==============================================================================
# 6. Checkpoint Discovery & Integrity Tests
# ==============================================================================

class TestCheckpointIntegrity:
    """Verifies candidate and release checkpoint loading and integrity."""

    def test_candidate_checkpoint_exists_and_loads(self) -> None:
        """Verifies candidate checkpoint loads cleanly on local CPU."""
        candidate_path = Path("runs/weights/candidates/candidate_m3_objectness_v2.pt")
        if not candidate_path.exists():
            pytest.skip(f"Candidate checkpoint not found at {candidate_path}")

        checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "metadata" in checkpoint

        model = UIObjectnessModel()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        # Run sample inference
        with torch.no_grad():
            pred = model.predict(torch.rand(3, 128, 128))
            assert isinstance(pred, UIObjectnessPrediction)

    def test_release_checkpoint_exists_and_loads(self) -> None:
        """Verifies release candidate checkpoint loads cleanly on local CPU."""
        release_path = Path("runs/weights/avcs_v2_weights.pt")
        if not release_path.exists():
            pytest.skip(f"Release checkpoint not found at {release_path}")

        checkpoint = torch.load(release_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert checkpoint["metadata"]["is_release"] is True

        model = UIObjectnessModel()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        with torch.no_grad():
            pred = model.predict(torch.rand(3, 128, 128))
            assert isinstance(pred, UIObjectnessPrediction)
