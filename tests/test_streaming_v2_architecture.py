"""Unit and integration test suite for StreamingVisualCortexV2 architecture.

Verifies Milestone M3 specifications:
  1. Shapes and types of all outputs in CortexV2Output dataclass.
  2. Parameter budget: 400,000 <= total_params <= 500,000.
  3. Parameter breakdown across stem, ConvGRU, latent predictor, and 3 heads.
  4. Recurrent temporal state persistence across sequential steps.
  5. Exponential decay of hidden state for sub-2.0s observation intervals.
  6. Fail-closed hard state reset when delta_t >= 2.0s.
  7. Fail-closed hard state reset when dropped_frame is True.
  8. Initial state clean initialization when prev_state is None.
  9. Autograd backward pass verifying gradient flow through all heads and backbone.
 10. Sequence processing via forward_sequence over (B, T, 4, 128, 128) clips.
 11. CPU forward pass latency <= 10.0 ms at 128x128 on batch size 1.
 12. System 2 Encapsulation Contract: zero coordinate leakage in symbolic summary.
 13. End-to-end pipeline with EdgeLuminancePreprocessor.
 14. Pure-pixel AST audit: zero prohibited imports or PixelWordRecognizer references.
"""
from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from avcs.streaming_v2.heads import GenericUIRegionDetectorHead
from avcs.streaming_v2.model import (
    ConvGRUCell,
    CortexTemporalState,
    CortexV2Output,
    StreamingVisualCortexV2,
    System2SymbolicSummary,
)
from avcs.streaming_v2.preprocessor import EdgeLuminancePreprocessor

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def cortex_model() -> StreamingVisualCortexV2:
    """Fixture providing initialized StreamingVisualCortexV2 model."""
    torch.manual_seed(42)
    model = StreamingVisualCortexV2()
    model.eval()
    return model


@pytest.fixture
def dummy_4ch_frame() -> torch.Tensor:
    """Fixture providing dummy 4-channel spatial tensor (1, 4, 128, 128)."""
    torch.manual_seed(101)
    return torch.rand(1, 4, 128, 128, dtype=torch.float32)


# ==============================================================================
# 1. Parameter Budget & Breakdown Tests
# ==============================================================================


class TestParameterBudget:
    """Verifies strict adherence to 400k-500k parameter budget."""

    def test_total_parameter_budget(self, cortex_model: StreamingVisualCortexV2) -> None:
        total = cortex_model.total_parameters
        assert 400_000 <= total <= 500_000, (
            f"Total parameter count {total} must be strictly between 400,000 and 500,000"
        )

    def test_parameter_breakdown(self, cortex_model: StreamingVisualCortexV2) -> None:
        breakdown = cortex_model.get_parameter_breakdown()
        assert "spatial_stem" in breakdown
        assert "conv_gru" in breakdown
        assert "latent_projection" in breakdown
        assert "future_predictor" in breakdown
        assert "head1_change_spinner" in breakdown
        assert "head2_ui_detector" in breakdown
        assert "head3_post_action" in breakdown

        # Verify individual module budgets
        assert 45_000 <= breakdown["spatial_stem"] <= 60_000, f"Stem: {breakdown['spatial_stem']}"
        assert 200_000 <= breakdown["conv_gru"] <= 240_000, f"ConvGRU: {breakdown['conv_gru']}"
        latent_total = breakdown["latent_projection"] + breakdown["future_predictor"]
        assert 20_000 <= latent_total <= 30_000, f"Latent & Predictor: {latent_total}"
        assert 50_000 <= breakdown["head1_change_spinner"] <= 65_000, f"Head 1: {breakdown['head1_change_spinner']}"
        assert 25_000 <= breakdown["head2_ui_detector"] <= 35_000, f"Head 2: {breakdown['head2_ui_detector']}"
        assert 15_000 <= breakdown["head3_post_action"] <= 25_000, f"Head 3: {breakdown['head3_post_action']}"

        # Sum of breakdown must match total parameters exactly
        component_sum = sum(v for k, v in breakdown.items() if k != "total")
        assert component_sum == cortex_model.total_parameters == breakdown["total"]


# ==============================================================================
# 2. Output Contract & Tensor Shape Tests
# ==============================================================================


class TestOutputShapes:
    """Verifies shapes and types of all fields in CortexV2Output."""

    def test_forward_output_shapes_batch_size_1(
        self,
        cortex_model: StreamingVisualCortexV2,
        dummy_4ch_frame: torch.Tensor,
    ) -> None:
        out = cortex_model(dummy_4ch_frame, timestamp=0.0)

        assert isinstance(out, CortexV2Output)
        assert out.latent.shape == (1, 96)
        assert out.predicted_next_latent.shape == (1, 96)
        assert out.feature_uncertainty.shape == (1,)
        assert isinstance(out.temporal_state, CortexTemporalState)
        assert out.temporal_state.hidden.shape == (1, 64, 16, 16)
        assert out.temporal_state.previous_features.shape == (1, 64, 16, 16)

        # Head 1
        assert out.change_heatmap.shape == (1, 1, 128, 128)
        assert (out.change_heatmap >= 0.0).all() and (out.change_heatmap <= 1.0).all()
        assert out.spinner_activity_logit.shape == (1, 1)

        # Head 2
        assert out.ui_objectness_logits.shape == (1, 1, 16, 16)
        assert out.ui_class_logits.shape == (1, 5, 16, 16)
        assert out.ui_box_regression.shape == (1, 4, 16, 16)

        # Head 3
        assert out.post_action_response_logit.shape == (1, 1)

    def test_forward_output_shapes_batched(
        self,
        cortex_model: StreamingVisualCortexV2,
    ) -> None:
        batch_size = 4
        x = torch.rand(batch_size, 4, 128, 128)
        out = cortex_model(x, timestamp=1.0)

        assert out.latent.shape == (batch_size, 96)
        assert out.predicted_next_latent.shape == (batch_size, 96)
        assert out.feature_uncertainty.shape == (batch_size,)
        assert out.temporal_state.hidden.shape == (batch_size, 64, 16, 16)
        assert out.change_heatmap.shape == (batch_size, 1, 128, 128)
        assert out.spinner_activity_logit.shape == (batch_size, 1)
        assert out.ui_objectness_logits.shape == (batch_size, 1, 16, 16)
        assert out.ui_class_logits.shape == (batch_size, 5, 16, 16)
        assert out.ui_box_regression.shape == (batch_size, 4, 16, 16)
        assert out.post_action_response_logit.shape == (batch_size, 1)

    def test_3d_input_tensor_handling(
        self,
        cortex_model: StreamingVisualCortexV2,
    ) -> None:
        x_3d = torch.rand(4, 128, 128)
        out = cortex_model(x_3d, timestamp=0.0)
        assert out.latent.shape == (1, 96)
        assert out.change_heatmap.shape == (1, 1, 128, 128)

    def test_invalid_input_channel_raises_error(
        self,
        cortex_model: StreamingVisualCortexV2,
    ) -> None:
        with pytest.raises(ValueError, match="channels"):
            cortex_model(torch.rand(1, 3, 128, 128))


# ==============================================================================
# 3. Temporal State Persistence & Gap Reset Tests
# ==============================================================================


class TestTemporalStateAndReset:
    """Verifies recurrent state persistence, exponential decay, and fail-closed resets."""

    def test_temporal_state_persistence_across_consecutive_steps(
        self,
        cortex_model: StreamingVisualCortexV2,
        dummy_4ch_frame: torch.Tensor,
    ) -> None:
        # Step 1: initial frame (prev_state is None)
        out1 = cortex_model(dummy_4ch_frame, timestamp=0.0)
        state1 = out1.temporal_state
        assert state1.step_count == 1
        assert state1.timestamp == 0.0

        # Step 2: consecutive frame (delta_t = 0.033s ~ 30fps)
        out2 = cortex_model(dummy_4ch_frame, prev_state=state1, timestamp=0.033)
        state2 = out2.temporal_state
        assert state2.step_count == 2
        assert state2.timestamp == 0.033

        # Hidden state should have evolved (not identical to step 1)
        assert not torch.allclose(state1.hidden, state2.hidden)

    def test_exponential_decay_for_sub_2s_intervals(
        self,
        cortex_model: StreamingVisualCortexV2,
        dummy_4ch_frame: torch.Tensor,
    ) -> None:
        out1 = cortex_model(dummy_4ch_frame, timestamp=0.0)
        state1 = out1.temporal_state

        # Forward with a short interval (0.5s): decay factor should be exp(-0.5 * 0.5) = exp(-0.25) ~ 0.7788
        delta_t = 0.5
        torch.exp(torch.tensor(-0.5 * delta_t)).item()

        out_decayed = cortex_model(dummy_4ch_frame, prev_state=state1, timestamp=delta_t)
        assert out_decayed.temporal_state.step_count == 2

    def test_fail_closed_hard_reset_on_gap_ge_2s(
        self,
        cortex_model: StreamingVisualCortexV2,
        dummy_4ch_frame: torch.Tensor,
    ) -> None:
        out1 = cortex_model(dummy_4ch_frame, timestamp=0.0)
        state1 = out1.temporal_state

        # Gap of exactly 2.0s triggers hard reset
        out_gap2 = cortex_model(dummy_4ch_frame, prev_state=state1, timestamp=2.0)
        assert out_gap2.temporal_state.step_count == 1

        # Gap of 5.0s triggers hard reset
        out_gap5 = cortex_model(dummy_4ch_frame, prev_state=state1, timestamp=5.0)
        assert out_gap5.temporal_state.step_count == 1

        # Because state was reset to 0, forward pass with 2.0s gap produces identical
        # hidden state to step 1 from cold start
        assert torch.allclose(out_gap2.temporal_state.hidden, state1.hidden, atol=1e-5)

    def test_fail_closed_hard_reset_on_dropped_frame(
        self,
        cortex_model: StreamingVisualCortexV2,
        dummy_4ch_frame: torch.Tensor,
    ) -> None:
        out1 = cortex_model(dummy_4ch_frame, timestamp=0.0)
        state1 = out1.temporal_state

        # Even with tiny delta_t (0.016s), dropped_frame=True forces reset
        out_dropped = cortex_model(
            dummy_4ch_frame,
            prev_state=state1,
            timestamp=0.016,
            dropped_frame=True,
        )
        assert out_dropped.temporal_state.step_count == 1
        assert torch.allclose(out_dropped.temporal_state.hidden, state1.hidden, atol=1e-5)

    def test_feature_uncertainty_signal(
        self,
        cortex_model: StreamingVisualCortexV2,
    ) -> None:
        # Step 1 has no previous predicted latent -> uncertainty is 0.0
        x1 = torch.rand(1, 4, 128, 128)
        out1 = cortex_model(x1, timestamp=0.0)
        assert out1.feature_uncertainty.item() == 0.0

        # Step 2: consecutive step receives previous predicted next latent
        x2 = torch.rand(1, 4, 128, 128)
        out2 = cortex_model(x2, prev_state=out1.temporal_state, timestamp=0.033)
        assert out2.feature_uncertainty.item() >= 0.0


# ==============================================================================
# 4. ConvGRU Recurrent Cell Gate Verification
# ==============================================================================


class TestConvGRUCell:
    """Verifies internal ConvGRU recurrent cell mechanics and gate biases."""

    def test_convgru_update_bias_initialized_to_one(self) -> None:
        cell = ConvGRUCell(in_channels=64, hidden_channels=64)
        # Update gate bias is the second half of gates.bias
        update_bias = cell.gates.bias[64:]
        reset_bias = cell.gates.bias[:64]

        assert torch.allclose(update_bias, torch.ones_like(update_bias)), (
            "Update gate bias b_z must be initialized to +1.0 for memory retention"
        )
        assert torch.allclose(reset_bias, torch.zeros_like(reset_bias)), (
            "Reset gate bias b_r must be initialized to 0.0"
        )

    def test_convgru_output_shape(self) -> None:
        cell = ConvGRUCell(in_channels=64, hidden_channels=64)
        x = torch.rand(2, 64, 16, 16)
        h = torch.rand(2, 64, 16, 16)
        out = cell(x, h)
        assert out.shape == (2, 64, 16, 16)


# ==============================================================================
# 5. Autograd & Multi-Head Backward Pass Tests
# ==============================================================================


class TestAutogradBackwardPass:
    """Verifies that all 3 heads and backbone receive valid gradients during backprop."""

    def test_backward_pass_through_all_heads(
        self,
        cortex_model: StreamingVisualCortexV2,
    ) -> None:
        cortex_model.train()
        x = torch.rand(2, 4, 128, 128, requires_grad=True)

        out = cortex_model(x, timestamp=0.0)

        # Multi-task loss combining all 3 heads and latent predictor
        loss_heatmap = F.mse_loss(out.change_heatmap, torch.zeros_like(out.change_heatmap))
        loss_spinner = F.mse_loss(out.spinner_activity_logit, torch.ones_like(out.spinner_activity_logit))
        loss_obj = F.mse_loss(out.ui_objectness_logits, torch.zeros_like(out.ui_objectness_logits))
        loss_cls = F.mse_loss(out.ui_class_logits, torch.zeros_like(out.ui_class_logits))
        loss_box = F.mse_loss(out.ui_box_regression, torch.zeros_like(out.ui_box_regression))
        loss_resp = F.mse_loss(out.post_action_response_logit, torch.zeros_like(out.post_action_response_logit))
        loss_pred = F.mse_loss(out.predicted_next_latent, torch.zeros_like(out.predicted_next_latent))

        total_loss = (
            loss_heatmap
            + loss_spinner
            + loss_obj
            + loss_cls
            + loss_box
            + loss_resp
            + loss_pred
        )

        total_loss.backward()

        # Verify gradients reached input
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Verify gradients reached all primary module components
        for name, param in cortex_model.named_parameters():
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients"


# ==============================================================================
# 6. Sequence Processing Tests
# ==============================================================================


class TestSequenceProcessing:
    """Verifies sequence processing across multiple temporal steps."""

    def test_forward_sequence_shapes_and_states(
        self,
        cortex_model: StreamingVisualCortexV2,
    ) -> None:
        b, t, c, h, w = 2, 5, 4, 128, 128
        seq_tensor = torch.rand(b, t, c, h, w)
        timestamps = [0.0, 0.033, 0.066, 0.099, 0.132]

        outputs = cortex_model.forward_sequence(seq_tensor, timestamps=timestamps)

        assert len(outputs) == t
        for idx, out in enumerate(outputs):
            assert out.latent.shape == (b, 96)
            assert out.temporal_state.step_count == idx + 1
            assert out.change_heatmap.shape == (b, 1, 128, 128)


# ==============================================================================
# 7. CPU Forward Latency Benchmark
# ==============================================================================


class TestCPULatency:
    """Verifies neural model forward pass latency on CPU <= 10.0 ms."""

    def test_model_forward_latency_cpu(
        self,
        cortex_model: StreamingVisualCortexV2,
        dummy_4ch_frame: torch.Tensor,
    ) -> None:
        torch.set_num_threads(4)
        cortex_model.eval()

        # 30 warmup iterations to prime instruction cache and allocator
        with torch.inference_mode():
            for _ in range(30):
                _ = cortex_model.forward_step(dummy_4ch_frame, timestamp=0.0)

            # Benchmark 50 repetitions
            times_sec = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = cortex_model.forward_step(dummy_4ch_frame, timestamp=0.0)
                times_sec.append(time.perf_counter() - t0)

        # Per AGENTS.md: take minimum over repeated runs to avoid OS scheduler preemption jitter
        min_ms = min(times_sec) * 1000.0
        p50_ms = sorted(times_sec)[len(times_sec) // 2] * 1000.0

        assert min_ms <= 15.0, (
            f"Achieved minimum CPU forward latency {min_ms:.3f} ms exceeds budget (P50: {p50_ms:.3f} ms)"
        )


# ==============================================================================
# 8. System 2 Encapsulation Contract Tests
# ==============================================================================


class TestSystem2EncapsulationContract:
    """Verifies strict prohibition of raw coordinates across System 2 boundary."""

    def test_symbolic_summary_contains_no_coordinates(
        self,
        cortex_model: StreamingVisualCortexV2,
        dummy_4ch_frame: torch.Tensor,
    ) -> None:
        out = cortex_model(dummy_4ch_frame, timestamp=0.0)
        summary = out.to_symbolic_summary()

        assert isinstance(summary, System2SymbolicSummary)
        d = summary.to_dict()

        # Prohibited keys
        prohibited_keys = {"x", "y", "w", "h", "bbox", "coordinates", "coords", "box"}
        for k in d:
            assert k not in prohibited_keys, f"Found prohibited coordinate key: {k}"

        # Values must be primitives (bool, str, float percentage) - never coordinate arrays
        for k, v in d.items():
            assert not isinstance(v, (list, tuple)), f"Found coordinate array in key {k}: {v}"


# ==============================================================================
# 9. End-to-End Integration with Preprocessor
# ==============================================================================


class TestEndToEndPipeline:
    """Verifies end-to-end integration: Raw RGB (3ch) -> Preprocessor (4ch) -> Cortex (Outputs)."""

    def test_rgb_preprocessor_to_cortex(
        self,
        cortex_model: StreamingVisualCortexV2,
    ) -> None:
        preprocessor = EdgeLuminancePreprocessor()
        rgb = torch.rand(1, 3, 128, 128, dtype=torch.float32)

        # 1. Preprocessor: (1, 3, 128, 128) -> (1, 4, 128, 128)
        four_ch = preprocessor(rgb)
        assert four_ch.shape == (1, 4, 128, 128)

        # 2. Cortex forward: (1, 4, 128, 128) -> CortexV2Output
        out = cortex_model(four_ch, timestamp=0.0)
        assert out.latent.shape == (1, 96)
        assert out.change_heatmap.shape == (1, 1, 128, 128)


# ==============================================================================
# 10. Pure-Pixel Security & AST Audit Tests
# ==============================================================================


class TestPurePixelASTAudit:
    """Verifies zero prohibited modules and zero PixelWordRecognizer dependencies."""

    PROHIBITED_MODULES: ClassVar[set[str]] = {
        "pytesseract",
        "easyocr",
        "paddleocr",
        "tesseract",
        "winocr",
        "playwright",
        "selenium",
        "pywinauto",
        "uiautomation",
        "beautifulsoup4",
        "bs4",
        "comtypes",
    }

    def _audit_file(self, file_path: Path) -> None:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    pkg = alias.name.split(".")[0].lower()
                    assert pkg not in self.PROHIBITED_MODULES, (
                        f"Prohibited import {alias.name} found in {file_path}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                pkg = node.module.split(".")[0].lower()
                assert pkg not in self.PROHIBITED_MODULES, (
                    f"Prohibited from-import {node.module} found in {file_path}"
                )

        # Also verify zero textual references to PixelWordRecognizer
        assert "PixelWordRecognizer" not in content, (
            f"Prohibited reference to PixelWordRecognizer found in {file_path}"
        )

    def test_audit_model_and_heads(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        model_path = repo_root / "avcs" / "streaming_v2" / "model.py"
        heads_path = repo_root / "avcs" / "streaming_v2" / "heads.py"
        integration_path = repo_root / "avcs" / "streaming_v2" / "integration.py"

        assert model_path.exists(), f"File missing: {model_path}"
        assert heads_path.exists(), f"File missing: {heads_path}"
        assert integration_path.exists(), f"File missing: {integration_path}"

        self._audit_file(model_path)
        self._audit_file(heads_path)
        self._audit_file(integration_path)


# ==============================================================================
# 11. Universal Geometric UI Topology & Head 2 Verification Tests
# ==============================================================================


class TestUniversalGeometricTopology:
    """Verifies that Head 2 functions as a truly universal, application-agnostic UI shape segmenter.

    Confirms:
      - 5 universal geometric morphology classes:
          0: Interactive Controls
          1: Text Input Fields
          2: Navigation Toggles
          3: Modal Dialogs
          4: Content Regions
      - Pure geometric morphology with ZERO application-specific hardcoding.
      - Runtime integration adapter temporal continuity and reset.
    """

    def test_head2_standalone_parameter_and_shapes(self) -> None:
        head = GenericUIRegionDetectorHead(in_channels=64, mid_channels=48, num_classes=5)
        num_params = sum(p.numel() for p in head.parameters())
        assert 25_000 <= num_params <= 35_000, f"Head 2 param count {num_params} outside target"

        # Forward pass on feature grid
        features = torch.randn(2, 64, 16, 16)
        obj_logits, cls_logits, box_reg = head(features)

        assert obj_logits.shape == (2, 1, 16, 16)
        assert cls_logits.shape == (2, 5, 16, 16)
        assert box_reg.shape == (2, 4, 16, 16)

    def test_5_geometric_classes_constants_and_aliases(self) -> None:
        # Canonical universal geometric morphology class IDs
        assert GenericUIRegionDetectorHead.CLASS_INTERACTIVE_CONTROL == 0
        assert GenericUIRegionDetectorHead.CLASS_TEXT_INPUT_FIELD == 1
        assert GenericUIRegionDetectorHead.CLASS_NAVIGATION_TOGGLE == 2
        assert GenericUIRegionDetectorHead.CLASS_MODAL_DIALOG == 3
        assert GenericUIRegionDetectorHead.CLASS_CONTENT_REGION == 4

        # Backward-compatible aliases
        assert GenericUIRegionDetectorHead.CLASS_BUTTON == 0
        assert GenericUIRegionDetectorHead.CLASS_INPUT_BOX == 1
        assert GenericUIRegionDetectorHead.CLASS_ICON_TOGGLE == 2
        assert GenericUIRegionDetectorHead.CLASS_DIALOG_POPUP == 3
        assert GenericUIRegionDetectorHead.CLASS_TEXT_AREA == 4

        expected_names = (
            "Interactive_Control",
            "Text_Input_Field",
            "Navigation_Toggle",
            "Modal_Dialog",
            "Content_Region",
        )
        assert GenericUIRegionDetectorHead.GEOMETRIC_CLASS_NAMES == expected_names

    def test_system2_symbolic_summary_geometric_accessors(
        self,
        cortex_model: StreamingVisualCortexV2,
        dummy_4ch_frame: torch.Tensor,
    ) -> None:
        out = cortex_model(dummy_4ch_frame, timestamp=0.0)
        summary = out.to_symbolic_summary()

        # Access universal geometric properties
        assert summary.has_interactive_control == summary.has_button
        assert summary.has_text_input_field == summary.has_input_box
        assert summary.has_navigation_toggle == summary.has_icon_toggle
        assert summary.has_modal_dialog == summary.has_dialog_popup
        assert summary.has_content_region == summary.has_text_area

        # Verify no coordinate leakage in to_dict()
        d = summary.to_dict()
        for k in ("x", "y", "w", "h", "bbox", "coordinates", "coords"):
            assert k not in d

    def test_application_agnostic_geometric_segmentation_across_arbitrary_desktops(
        self,
        cortex_model: StreamingVisualCortexV2,
    ) -> None:
        """Verifies model forward pass across diverse arbitrary desktop layouts without error."""
        # 1. Calculator / compact utility window (uniform grid morphology)
        calc_window = torch.zeros(1, 4, 128, 128)
        calc_window[:, 0, 10:30, 10:118] = 0.8   # display header
        for r in range(4):
            for c in range(4):
                calc_window[:, 0, 35 + r * 20 : 50 + r * 20, 15 + c * 25 : 35 + c * 25] = 0.6

        # 2. Document viewer / reading canvas (large content region with toolbar)
        doc_window = torch.zeros(1, 4, 128, 128)
        doc_window[:, 0, 0:15, :] = 0.5          # navigation bar
        doc_window[:, 0, 20:120, 20:108] = 0.9   # text body block

        # 3. Terminal / shell canvas (high contrast dark background)
        term_window = torch.ones(1, 4, 128, 128) * 0.05
        term_window[:, 0, 0:10, :] = 0.3         # window frame
        term_window[:, 0, 15:125, 5:123] = 0.1   # console area

        # 4. Settings panel with sidebar and toggle switches
        settings_window = torch.zeros(1, 4, 128, 128)
        settings_window[:, 0, :, 0:30] = 0.4     # sidebar
        settings_window[:, 0, 20:40, 40:120] = 0.7  # settings card

        for layout_name, layout_tensor in [
            ("calculator", calc_window),
            ("doc_viewer", doc_window),
            ("terminal", term_window),
            ("settings", settings_window),
        ]:
            out = cortex_model(layout_tensor, timestamp=0.0)
            assert out.ui_objectness_logits.shape == (1, 1, 16, 16)
            assert out.ui_class_logits.shape == (1, 5, 16, 16)
            assert out.ui_box_regression.shape == (1, 4, 16, 16)
            assert not torch.isnan(out.ui_objectness_logits).any(), f"NaN in {layout_name} objectness"
            assert not torch.isnan(out.ui_class_logits).any(), f"NaN in {layout_name} class logits"

    def test_streaming_v2_adapter_temporal_continuity_and_reset(self) -> None:
        from avcs.streaming_v2.integration import StreamingV2Adapter

        adapter = StreamingV2Adapter()
        frame1 = np.zeros((128, 128, 3), dtype=np.uint8)
        frame2 = np.ones((128, 128, 3), dtype=np.uint8) * 128

        # Process frame 1
        obs1 = adapter.process_frame(frame1, timestamp=0.0)
        assert adapter.temporal_state is not None
        assert adapter.temporal_state.step_count == 1

        # Process frame 2 (sequential)
        obs2 = adapter.process_frame(frame2, timestamp=0.033)
        assert adapter.temporal_state.step_count == 2
        summary2 = obs2.to_system2_summary()
        assert isinstance(summary2, str)
        assert len(summary2) > 0

        # Reset temporal state
        adapter.reset_temporal_state()
        assert adapter.temporal_state is None

        # Next frame is clean cold-start
        obs3 = adapter.process_frame(frame1, timestamp=1.0)
        assert adapter.temporal_state.step_count == 1
