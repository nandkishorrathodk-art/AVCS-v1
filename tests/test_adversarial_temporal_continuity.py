"""Adversarial Stress Test Harness for StreamingVisualCortexV2.

Milestone M3 Empirical Challenge Suite:
Challenging temporal state continuity, numerical stability, and state gap reset boundaries:
  1. Boundary testing on delta_t:
     - delta_t = 1.99s (exponential decay, non-zero state, step increment)
     - delta_t = 2.00s (hard reset, exactly zero state, step reset to 1)
     - delta_t = 2.01s (hard reset, exactly zero state, step reset to 1)
     - Micro-boundary sweep around 2.00s transition
     - Clock skew handling (negative delta_t)
  2. Long sequence stability:
     - Unroll 100 consecutive steps static sequence
     - Unroll 100 consecutive steps dynamic adversarial sequence
     - Verify zero NaN, zero Inf, bounded activations (|h_t| <= 1.0), no exploding/vanishing
     - Extreme inputs (all zeros, all ones, alternating flash stress)
  3. Dropped frame flag:
     - Verify dropped_frame=True forces instant zero reset even when delta_t is tiny (0.016s)
     - Sub-millisecond intervals (0.0001s)
     - Consecutive dropped frames
  4. Optical feature delta:
     - Verify |f_t - f_{t-1}| behavior on static sequences (delta goes to exactly 0)
     - Dynamic sequences (delta > 0)
     - Spatial correspondence of localized changes
     - Post-action response head discrimination
  5. Backpropagation through time (BPTT):
     - Gradient propagation across multi-step temporal rollouts (T=3, 5, 10)
     - Input gradient reachability all the way to initial frame x_0
     - Multi-task gradient flow to all heads and future predictor
     - Gradient boundedness (no exploding gradients)
     - Truncation of BPTT across hard reset boundaries (delta_t >= 2.0s)
     - Truncation of BPTT via state.detach()
  6. Cross-sample batch isolation:
     - Zero cross-sample contamination between batch items with different temporal histories
"""
from __future__ import annotations

import math

import pytest
import torch

from avcs.streaming_v2.model import (
    CortexV2Output,
    StreamingVisualCortexV2,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def eval_model() -> StreamingVisualCortexV2:
    """Provide a deterministic eval-mode StreamingVisualCortexV2 model."""
    torch.manual_seed(1337)
    model = StreamingVisualCortexV2()
    model.eval()
    return model


@pytest.fixture
def train_model() -> StreamingVisualCortexV2:
    """Provide a deterministic train-mode StreamingVisualCortexV2 model."""
    torch.manual_seed(1337)
    model = StreamingVisualCortexV2()
    model.train()
    return model


def make_static_frame(seed: int = 42) -> torch.Tensor:
    """Generate a reproducible static 4-channel tensor (1, 4, 128, 128)."""
    torch.manual_seed(seed)
    return torch.rand(1, 4, 128, 128, dtype=torch.float32)


def make_patch_frame(base: torch.Tensor, patch_val: float, top: int, left: int, size: int = 24) -> torch.Tensor:
    """Create a new frame by modifying a localized rectangular patch."""
    frame = base.clone()
    frame[:, :, top : top + size, left : left + size] = patch_val
    return frame


# ==============================================================================
# 1. Boundary Testing on delta_t (Gap Reset Dynamics)
# ==============================================================================


class TestDeltaTBoundaryDynamics:
    """Adversarially challenge the 2.00s temporal reset cliff-edge."""

    def test_sub_2s_boundary_decay_1_99(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify delta_t = 1.99s applies exponential decay without hard reset."""
        frame1 = make_static_frame(101)
        frame2 = make_static_frame(102)

        # Warm up 5 steps at 30 fps (delta_t ~ 0.0333s)
        state = None
        t = 0.0
        for _ in range(5):
            out = eval_model(frame1, prev_state=state, timestamp=t)
            state = out.temporal_state
            t += 0.0333

        assert state.step_count == 5
        t_warm = state.timestamp

        # Step 6 with delta_t = 1.99s
        t_next = t_warm + 1.99
        out_1_99 = eval_model(frame2, prev_state=state, timestamp=t_next)
        state_1_99 = out_1_99.temporal_state

        # Must NOT hard reset: step_count increments to 6
        assert state_1_99.step_count == 6, f"Expected step_count 6, got {state_1_99.step_count}"
        assert state_1_99.timestamp == pytest.approx(t_next, rel=1e-5)

        # Theoretical decay factor: exp(-0.5 * 1.99) ~ 0.36972
        expected_decay = math.exp(-0.5 * 1.99)
        assert 0.369 < expected_decay < 0.370

        # Output hidden state must differ from a cold start on frame2
        cold_out = eval_model(frame2, prev_state=None, timestamp=t_next)
        diff_from_cold = torch.norm(state_1_99.hidden - cold_out.temporal_state.hidden).item()
        assert diff_from_cold > 1e-3, (
            f"Hidden state after delta_t=1.99s unexpectedly matched cold start (diff: {diff_from_cold})"
        )

        # Feature uncertainty must be computed from previous predicted latent (non-zero)
        assert out_1_99.feature_uncertainty.item() > 0.0, "Feature uncertainty must be non-zero for delta_t=1.99s"

    def test_cliff_edge_boundary_2_00(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify delta_t = 2.00s triggers exact fail-closed hard reset to zero prior state."""
        frame1 = make_static_frame(101)
        frame2 = make_static_frame(102)

        # Warm up 5 steps
        state = None
        t = 0.0
        for _ in range(5):
            out = eval_model(frame1, prev_state=state, timestamp=t)
            state = out.temporal_state
            t += 0.0333

        t_warm = state.timestamp

        # Step 6 with delta_t = 2.00s (boundary threshold)
        t_next = t_warm + 2.00
        out_2_00 = eval_model(frame2, prev_state=state, timestamp=t_next)
        state_2_00 = out_2_00.temporal_state

        # Hard reset: step_count reset to 1
        assert state_2_00.step_count == 1, (
            f"Expected step_count reset to 1 at delta_t=2.00s, got {state_2_00.step_count}"
        )

        # Hidden state must match cold start on frame2 exactly
        cold_out = eval_model(frame2, prev_state=None, timestamp=t_next)
        max_diff = torch.max(torch.abs(state_2_00.hidden - cold_out.temporal_state.hidden)).item()
        assert max_diff < 1e-5, f"Hidden state at delta_t=2.00s differs from cold start: max_diff={max_diff}"

        # Feature uncertainty must be exactly 0.0 after hard reset
        assert out_2_00.feature_uncertainty.item() == 0.0, "Feature uncertainty must be 0.0 on reset"

    def test_super_2s_boundary_2_01(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify delta_t = 2.01s triggers hard reset identical to delta_t = 2.00s."""
        frame1 = make_static_frame(101)
        frame2 = make_static_frame(102)

        # Warm up
        state = None
        t = 0.0
        for _ in range(5):
            out = eval_model(frame1, prev_state=state, timestamp=t)
            state = out.temporal_state
            t += 0.0333

        t_warm = state.timestamp

        # Step 6 with delta_t = 2.01s
        t_2_01 = t_warm + 2.01
        out_2_01 = eval_model(frame2, prev_state=state, timestamp=t_2_01)
        state_2_01 = out_2_01.temporal_state

        assert state_2_01.step_count == 1
        assert out_2_01.feature_uncertainty.item() == 0.0

        # Step 6 with delta_t = 2.00s on identical input
        t_2_00 = t_warm + 2.00
        out_2_00 = eval_model(frame2, prev_state=state, timestamp=t_2_00)

        # Both reset outputs must be identical in hidden state, latents, and heads
        hidden_diff = torch.max(torch.abs(state_2_01.hidden - out_2_00.temporal_state.hidden)).item()
        latent_diff = torch.max(torch.abs(out_2_01.latent - out_2_00.latent)).item()
        heatmap_diff = torch.max(torch.abs(out_2_01.change_heatmap - out_2_00.change_heatmap)).item()

        assert hidden_diff < 1e-6, f"Hidden difference between 2.00s and 2.01s: {hidden_diff}"
        assert latent_diff < 1e-6, f"Latent difference between 2.00s and 2.01s: {latent_diff}"
        assert heatmap_diff < 1e-6, f"Heatmap difference between 2.00s and 2.01s: {heatmap_diff}"

    @pytest.mark.parametrize(
        ("delta_t", "expected_reset", "expected_step"),
        [
            (1.990, False, 6),
            (1.999, False, 6),
            (1.9999, False, 6),
            (2.0000, True, 1),
            (2.0001, True, 1),
            (2.0100, True, 1),
            (10.000, True, 1),
        ],
    )
    def test_micro_boundary_sweep_around_threshold(
        self,
        eval_model: StreamingVisualCortexV2,
        delta_t: float,
        expected_reset: bool,
        expected_step: int,
    ) -> None:
        """Micro-sweep verifying sharp cliff-edge transition at exactly 2.0000s."""
        frame = make_static_frame(42)

        # Warm up 5 steps
        state = None
        t = 0.0
        for _ in range(5):
            out = eval_model(frame, prev_state=state, timestamp=t)
            state = out.temporal_state
            t += 0.0333

        t_next = state.timestamp + delta_t
        out = eval_model(frame, prev_state=state, timestamp=t_next)

        assert out.temporal_state.step_count == expected_step, (
            f"At delta_t={delta_t:.4f}s, expected step_count {expected_step}, got {out.temporal_state.step_count}"
        )
        if expected_reset:
            assert out.feature_uncertainty.item() == 0.0
        else:
            assert out.feature_uncertainty.item() >= 0.0

    def test_negative_delta_t_clock_skew(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify model handles clock skew / backward timestamps safely without crashing or NaNs."""
        frame = make_static_frame(42)

        out1 = eval_model(frame, timestamp=10.0)
        state1 = out1.temporal_state
        assert state1.step_count == 1

        # Clock jumped backward by 2 seconds: timestamp = 8.0 < 10.0
        out_skew = eval_model(frame, prev_state=state1, timestamp=8.0)

        # Code clamps delta_t = max(delta_t, 0.0) = 0.0 -> decay_factor = exp(0) = 1.0
        assert out_skew.temporal_state.step_count == 2
        assert not torch.isnan(out_skew.temporal_state.hidden).any()
        assert not torch.isinf(out_skew.temporal_state.hidden).any()
        assert not torch.isnan(out_skew.latent).any()


# ==============================================================================
# 2. Long Sequence Stability (100 Steps Unrolling)
# ==============================================================================


class TestLongSequenceStability:
    """Stress-test numerical stability, activation boundedness, and vanishing/exploding behavior."""

    def test_100_step_static_unroll_stability(self, eval_model: StreamingVisualCortexV2) -> None:
        """Unroll 100 consecutive steps on static frame. Verify 0 NaN, 0 Inf, bounded activations."""
        frame = make_static_frame(777)
        state = None
        hidden_norms: list[float] = []

        for step in range(100):
            t = step * (1.0 / 30.0)  # 30 fps
            out = eval_model(frame, prev_state=state, timestamp=t)
            state = out.temporal_state

            # 1. Verify zero NaN and zero Inf across all 11 output tensors
            assert not torch.isnan(out.latent).any(), f"NaN in latent at step {step}"
            assert not torch.isinf(out.latent).any(), f"Inf in latent at step {step}"
            assert not torch.isnan(out.predicted_next_latent).any(), f"NaN in predicted_next_latent at step {step}"
            assert not torch.isnan(out.feature_uncertainty).any(), f"NaN in feature_uncertainty at step {step}"
            assert not torch.isnan(state.hidden).any(), f"NaN in hidden state at step {step}"
            assert not torch.isinf(state.hidden).any(), f"Inf in hidden state at step {step}"
            assert not torch.isnan(out.change_heatmap).any(), f"NaN in change_heatmap at step {step}"
            assert not torch.isnan(out.spinner_activity_logit).any(), f"NaN in spinner_activity at step {step}"
            assert not torch.isnan(out.ui_objectness_logits).any(), f"NaN in ui_objectness at step {step}"
            assert not torch.isnan(out.ui_class_logits).any(), f"NaN in ui_class at step {step}"
            assert not torch.isnan(out.ui_box_regression).any(), f"NaN in ui_box_regression at step {step}"
            assert not torch.isnan(out.post_action_response_logit).any(), f"NaN in post_action_logit at step {step}"

            # 2. Verify bounded hidden activation: |h_t| <= 1.0 (inherent property of ConvGRU candidate tanh)
            max_abs_h = torch.max(torch.abs(state.hidden)).item()
            assert max_abs_h <= 1.0001, f"Hidden activation {max_abs_h} exceeded bound 1.0 at step {step}"

            # 3. Track L2 norm to verify no vanishing state
            h_norm = torch.norm(state.hidden).item()
            hidden_norms.append(h_norm)
            assert h_norm > 0.1, f"Hidden state collapsed / vanished (norm={h_norm}) at step {step}"

            # 4. Step counter monotonic increase
            assert state.step_count == step + 1

        # Verify stability over the entire sequence: hidden norm should stabilize
        assert len(hidden_norms) == 100
        # Final norm vs step 20 norm should not explode (ratio < 5.0)
        ratio = hidden_norms[-1] / (hidden_norms[20] + 1e-6)
        assert 0.2 < ratio < 5.0, f"Hidden state unstable: ratio step 100 to step 20 is {ratio:.3f}"

    def test_100_step_dynamic_adversarial_unroll(self, eval_model: StreamingVisualCortexV2) -> None:
        """Unroll 100 steps of high-variance dynamic frames. Verify stability under rapid transitions."""
        state = None
        torch.manual_seed(999)

        for step in range(100):
            # Dynamic frame: random high-variance image with moving shapes
            frame = torch.rand(1, 4, 128, 128, dtype=torch.float32)
            # Add sudden high-contrast square
            x_pos = (step * 7) % 100
            y_pos = (step * 11) % 100
            frame[:, :, y_pos : y_pos + 20, x_pos : x_pos + 20] = (step % 2) * 1.0

            t = step * 0.01667  # 60 fps
            out = eval_model(frame, prev_state=state, timestamp=t)
            state = out.temporal_state

            # Verify no NaNs or Infs
            assert not torch.isnan(state.hidden).any(), f"NaN in dynamic unroll at step {step}"
            assert not torch.isinf(state.hidden).any(), f"Inf in dynamic unroll at step {step}"
            assert not torch.isnan(out.latent).any()

            # Bounded activations
            max_abs_h = torch.max(torch.abs(state.hidden)).item()
            assert max_abs_h <= 1.0001, f"Hidden activation exceeded 1.0 at dynamic step {step}"

    def test_extreme_inputs_stability(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify model handles extreme input values: all zeros, all ones, and alternating flashes."""
        state = None
        t = 0.0

        # Phase 1: 25 steps of all zeros
        zero_frame = torch.zeros(1, 4, 128, 128, dtype=torch.float32)
        for _ in range(25):
            out = eval_model(zero_frame, prev_state=state, timestamp=t)
            state = out.temporal_state
            t += 0.0333
            assert not torch.isnan(state.hidden).any()

        # Phase 2: 25 steps of all ones
        ones_frame = torch.ones(1, 4, 128, 128, dtype=torch.float32)
        for _ in range(25):
            out = eval_model(ones_frame, prev_state=state, timestamp=t)
            state = out.temporal_state
            t += 0.0333
            assert not torch.isnan(state.hidden).any()

        # Phase 3: 25 steps of alternating flash (0.0 vs 1.0)
        for step in range(25):
            flash_frame = zero_frame if (step % 2 == 0) else ones_frame
            out = eval_model(flash_frame, prev_state=state, timestamp=t)
            state = out.temporal_state
            t += 0.0333
            assert not torch.isnan(state.hidden).any()
            assert torch.max(torch.abs(state.hidden)).item() <= 1.0001


# ==============================================================================
# 3. Dropped Frame Flag (Instant Fail-Closed Reset)
# ==============================================================================


class TestDroppedFrameReset:
    """Verify that dropped_frame=True overrides observation interval and forces instant zero reset."""

    def test_dropped_frame_with_tiny_delta_t(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify dropped_frame=True forces reset even with delta_t = 0.01667s (60 fps)."""
        frame1 = make_static_frame(11)
        frame2 = make_static_frame(12)

        # Warm up 5 steps
        state = None
        t = 0.0
        for _ in range(5):
            out = eval_model(frame1, prev_state=state, timestamp=t)
            state = out.temporal_state
            t += 0.01667

        assert state.step_count == 5

        # Step 6 with tiny delta_t (0.01667s) BUT dropped_frame=True
        t_next = state.timestamp + 0.01667
        out_drop = eval_model(frame2, prev_state=state, timestamp=t_next, dropped_frame=True)
        state_drop = out_drop.temporal_state

        # Must force instant reset: step_count == 1
        assert state_drop.step_count == 1, (
            f"dropped_frame=True failed to reset step_count: got {state_drop.step_count}"
        )

        # Output hidden state must be identical to cold start on frame2
        cold_out = eval_model(frame2, prev_state=None, timestamp=t_next)
        diff = torch.max(torch.abs(state_drop.hidden - cold_out.temporal_state.hidden)).item()
        assert diff < 1e-5, f"Dropped frame output differs from cold start: diff={diff}"

        # Feature uncertainty must be 0.0
        assert out_drop.feature_uncertainty.item() == 0.0

    @pytest.mark.parametrize("tiny_dt", [0.0001, 0.001, 0.005, 0.016, 0.5])
    def test_dropped_frame_micro_delta_t_parameterized(
        self,
        eval_model: StreamingVisualCortexV2,
        tiny_dt: float,
    ) -> None:
        """Verify dropped_frame=True overrides any sub-second interval."""
        frame = make_static_frame(33)
        out1 = eval_model(frame, timestamp=0.0)
        state1 = out1.temporal_state

        out_drop = eval_model(frame, prev_state=state1, timestamp=tiny_dt, dropped_frame=True)
        assert out_drop.temporal_state.step_count == 1
        assert out_drop.feature_uncertainty.item() == 0.0

    def test_consecutive_dropped_frames(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify sequence of multiple consecutive dropped frames stays at step_count=1 without accumulation."""
        frame = make_static_frame(44)
        state = None
        t = 0.0

        for _ in range(5):
            out = eval_model(frame, prev_state=state, timestamp=t, dropped_frame=True)
            state = out.temporal_state
            t += 0.0333
            assert state.step_count == 1
            assert out.feature_uncertainty.item() == 0.0


# ==============================================================================
# 4. Optical Feature Delta (|f_t - f_{t-1}|)
# ==============================================================================


class TestOpticalFeatureDelta:
    """Verify |f_t - f_{t-1}| behavior on static sequences vs dynamic sequences."""

    def test_static_sequence_optical_delta_is_zero(self, eval_model: StreamingVisualCortexV2) -> None:
        """On static sequences, optical feature delta |f_t - f_{t-1}| must be identically zero."""
        frame = make_static_frame(555)

        # Step 1: initial frame
        out1 = eval_model(frame, timestamp=0.0)
        state1 = out1.temporal_state

        # Step 2: identical frame
        out2 = eval_model(frame, prev_state=state1, timestamp=0.033)
        state2 = out2.temporal_state

        # Optical feature delta: |f_t - f_{t-1}|
        f_current = state2.previous_features
        f_previous = state1.previous_features
        f_delta = torch.abs(f_current - f_previous)

        max_delta = torch.max(f_delta).item()
        assert max_delta < 1e-6, f"Expected optical feature delta == 0 on static frame, got {max_delta}"

        # Step 3 to 10: verify delta remains zero
        state = state2
        for step in range(3, 11):
            out = eval_model(frame, prev_state=state, timestamp=step * 0.033)
            state = out.temporal_state
            delta = torch.abs(state.previous_features - f_previous)
            assert torch.max(delta).item() < 1e-6

    def test_dynamic_sequence_optical_delta_positive(self, eval_model: StreamingVisualCortexV2) -> None:
        """On dynamic sequences, optical feature delta must be strictly positive."""
        base_frame = make_static_frame(601)
        # Modified frame: high-contrast 32x32 button in center
        button_frame = base_frame.clone()
        button_frame[:, :, 48:80, 48:80] = 1.0 - button_frame[:, :, 48:80, 48:80]

        out1 = eval_model(base_frame, timestamp=0.0)
        state1 = out1.temporal_state

        out2 = eval_model(button_frame, prev_state=state1, timestamp=0.033)
        state2 = out2.temporal_state

        f_delta = torch.abs(state2.previous_features - state1.previous_features)
        mean_delta = torch.mean(f_delta).item()
        max_delta = torch.max(f_delta).item()

        assert mean_delta > 0.01, f"Expected non-trivial mean feature delta, got {mean_delta}"
        assert max_delta > 0.10, f"Expected non-trivial peak feature delta, got {max_delta}"

        # Change heatmap output must respond to dynamic change compared to static
        out_static = eval_model(base_frame, prev_state=state1, timestamp=0.033)
        heat_diff = torch.max(torch.abs(out2.change_heatmap - out_static.change_heatmap)).item()
        assert heat_diff > 0.005, (
            f"Dynamic change heatmap failed to respond to feature delta (max diff: {heat_diff:.4f})"
        )

    def test_localized_change_spatial_correspondence(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify spatial correspondence: top-left change produces peak delta in top-left feature quadrant."""
        base_frame = torch.zeros(1, 4, 128, 128, dtype=torch.float32)
        # Top-left patch: [8:32, 8:32]
        tl_frame = make_patch_frame(base_frame, patch_val=1.0, top=8, left=8, size=24)

        out1 = eval_model(base_frame, timestamp=0.0)
        out2 = eval_model(tl_frame, prev_state=out1.temporal_state, timestamp=0.033)

        f_delta = torch.abs(out2.temporal_state.previous_features - out1.temporal_state.previous_features)
        # f_delta has shape (1, 64, 16, 16). Spatial grid is 16x16.
        # Top-left quadrant: [:, :, 0:8, 0:8]
        # Bottom-right quadrant: [:, :, 8:16, 8:16]
        tl_energy = f_delta[:, :, :8, :8].mean().item()
        br_energy = f_delta[:, :, 8:, 8:].mean().item()

        assert tl_energy > br_energy * 2.0, (
            f"Expected spatial concentration in top-left: TL={tl_energy:.4f}, BR={br_energy:.4f}"
        )

    def test_post_action_head_discrimination(self, eval_model: StreamingVisualCortexV2) -> None:
        """Verify Head 3 produces different logits for static vs dynamic transitions."""
        frame1 = make_static_frame(1)
        frame2 = make_static_frame(2)

        out1 = eval_model(frame1, timestamp=0.0)
        out_static = eval_model(frame1, prev_state=out1.temporal_state, timestamp=0.033)
        out_dynamic = eval_model(frame2, prev_state=out1.temporal_state, timestamp=0.033)

        logit_static = out_static.post_action_response_logit.item()
        logit_dynamic = out_dynamic.post_action_response_logit.item()

        assert abs(logit_dynamic - logit_static) > 1e-4, (
            f"Post-action response failed to discriminate static ({logit_static}) vs dynamic ({logit_dynamic})"
        )


# ==============================================================================
# 5. Backpropagation Through Time (BPTT) Numerical Stability
# ==============================================================================


class TestBackpropagationThroughTime:
    """Stress-test gradient propagation, numerical explosion, and graph truncation."""

    @pytest.mark.parametrize("seq_len", [3, 5, 10])
    def test_multi_step_bptt_gradient_propagation(
        self,
        train_model: StreamingVisualCortexV2,
        seq_len: int,
    ) -> None:
        """Verify gradients propagate all the way back to x_0 across T-step unrolling."""
        frames = [torch.rand(1, 4, 128, 128, requires_grad=True) for _ in range(seq_len)]
        state = None

        outputs: list[CortexV2Output] = []
        for step, frame in enumerate(frames):
            t = step * 0.033
            out = train_model(frame, prev_state=state, timestamp=t)
            outputs.append(out)
            state = out.temporal_state

        # Multi-task loss combining all 3 specialized heads and the future predictor
        total_loss = torch.tensor(0.0, device=frames[0].device)
        for out in outputs:
            total_loss = (
                total_loss
                + out.change_heatmap.sum()
                + out.spinner_activity_logit.sum()
                + out.ui_objectness_logits.sum()
                + out.ui_class_logits.sum()
                + out.ui_box_regression.sum()
                + out.post_action_response_logit.sum()
                + out.predicted_next_latent.sum()
            )

        total_loss.backward()

        # 1. Gradient MUST reach the very first frame x_0
        assert frames[0].grad is not None, "Gradient did not reach input frame x_0"
        grad_0_norm = torch.norm(frames[0].grad).item()
        assert grad_0_norm > 1e-4, f"Gradient norm at x_0 was zero or sub-epsilon ({grad_0_norm})"

        # 2. Every intermediate frame must have valid, finite gradients
        for idx, frame in enumerate(frames):
            assert frame.grad is not None, f"Frame {idx} missing gradient"
            assert not torch.isnan(frame.grad).any(), f"Frame {idx} grad contains NaN"
            assert not torch.isinf(frame.grad).any(), f"Frame {idx} grad contains Inf"

        # 3. Model parameter gradients must all be valid and finite across all heads
        for name, param in train_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} missing gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"
                assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf grad"

    def test_bptt_gradient_norm_boundedness(self, train_model: StreamingVisualCortexV2) -> None:
        """Verify that gradient norms remain bounded and do not explode exponentially with sequence length."""
        grad_norms: list[float] = []

        for seq_len in [2, 5, 10]:
            train_model.zero_grad()
            frames = [torch.rand(1, 4, 128, 128, requires_grad=True) for _ in range(seq_len)]
            state = None

            for step, frame in enumerate(frames):
                out = train_model(frame, prev_state=state, timestamp=step * 0.033)
                state = out.temporal_state

            loss = state.hidden.sum()
            loss.backward()

            total_norm = 0.0
            for p in train_model.parameters():
                if p.grad is not None:
                    total_norm += torch.norm(p.grad).item() ** 2
            total_norm = math.sqrt(total_norm)
            grad_norms.append(total_norm)

        # Norm at seq_len 10 must not be astronomically larger than at seq_len 2
        ratio = grad_norms[-1] / (grad_norms[0] + 1e-6)
        assert ratio < 100.0, f"Gradient norm exploded across sequence lengths: {grad_norms}"

    def test_bptt_truncation_on_hard_reset(self, train_model: StreamingVisualCortexV2) -> None:
        """Verify that a hard reset (delta_t >= 2.0s) cleanly truncates BPTT across the gap boundary."""
        f0 = torch.rand(1, 4, 128, 128, requires_grad=True)
        f1 = torch.rand(1, 4, 128, 128, requires_grad=True)
        f2 = torch.rand(1, 4, 128, 128, requires_grad=True)
        f3 = torch.rand(1, 4, 128, 128, requires_grad=True)

        # Step 0: t=0.0
        out0 = train_model(f0, timestamp=0.0)
        # Step 1: t=0.033
        out1 = train_model(f1, prev_state=out0.temporal_state, timestamp=0.033)
        # Step 2: t=3.00s -> DELTA_T = 2.967s >= 2.0s (HARD RESET)
        out2 = train_model(f2, prev_state=out1.temporal_state, timestamp=3.00)
        # Step 3: t=3.033s
        out3 = train_model(f3, prev_state=out2.temporal_state, timestamp=3.033)

        # Loss on step 3 using spatial head
        loss = out3.change_heatmap.sum() + out3.ui_objectness_logits.sum()
        loss.backward()

        # Step 2 and 3 must receive gradients
        assert f3.grad is not None and torch.norm(f3.grad) > 0
        assert f2.grad is not None and torch.norm(f2.grad) > 0

        # Step 0 and 1 must NOT receive gradients because hard reset zeroed h_prev
        assert f0.grad is None or torch.norm(f0.grad) == 0.0, (
            "f0 unexpectedly received gradients across hard reset gap"
        )
        assert f1.grad is None or torch.norm(f1.grad) == 0.0, (
            "f1 unexpectedly received gradients across hard reset gap"
        )

    def test_bptt_truncation_with_explicit_detach(self, train_model: StreamingVisualCortexV2) -> None:
        """Verify that state.detach() properly truncates BPTT at the detach boundary."""
        f0 = torch.rand(1, 4, 128, 128, requires_grad=True)
        f1 = torch.rand(1, 4, 128, 128, requires_grad=True)
        f2 = torch.rand(1, 4, 128, 128, requires_grad=True)

        out0 = train_model(f0, timestamp=0.0)
        out1 = train_model(f1, prev_state=out0.temporal_state, timestamp=0.033)

        # Detach state before step 2
        detached_state = out1.temporal_state.detach()
        out2 = train_model(f2, prev_state=detached_state, timestamp=0.066)

        loss = out2.change_heatmap.sum() + out2.ui_objectness_logits.sum()
        loss.backward()

        assert f2.grad is not None and torch.norm(f2.grad) > 0
        assert f0.grad is None or torch.norm(f0.grad) == 0.0
        assert f1.grad is None or torch.norm(f1.grad) == 0.0


# ==============================================================================
# 6. Cross-Sample Batch Isolation
# ==============================================================================


class TestCrossSampleBatchIsolation:
    """Verify that batch processing causes zero cross-sample leakage between independent sequences."""

    def test_batch_sample_independence(self, eval_model: StreamingVisualCortexV2) -> None:
        """Process 2 samples batched vs singly. Output must be identical within float32 tolerance."""
        f_a = make_static_frame(111)
        f_b = make_static_frame(222)
        f_batch = torch.cat([f_a, f_b], dim=0)  # (2, 4, 128, 128)

        # Single runs
        out_a = eval_model(f_a, timestamp=0.0)
        out_b = eval_model(f_b, timestamp=0.0)

        # Batched run
        out_batch = eval_model(f_batch, timestamp=0.0)

        # Compare item 0 with out_a (tolerance 5e-5 for OneDNN / BLAS reassociation)
        diff_latent_a = torch.max(torch.abs(out_batch.latent[0:1] - out_a.latent)).item()
        diff_heat_a = torch.max(torch.abs(out_batch.change_heatmap[0:1] - out_a.change_heatmap)).item()
        diff_obj_a = torch.max(torch.abs(out_batch.ui_objectness_logits[0:1] - out_a.ui_objectness_logits)).item()

        assert diff_latent_a < 5e-5, f"Cross-sample leakage in latent A: {diff_latent_a}"
        assert diff_heat_a < 5e-5, f"Cross-sample leakage in heatmap A: {diff_heat_a}"
        assert diff_obj_a < 5e-5, f"Cross-sample leakage in objectness A: {diff_obj_a}"

        # Compare item 1 with out_b
        diff_latent_b = torch.max(torch.abs(out_batch.latent[1:2] - out_b.latent)).item()
        assert diff_latent_b < 5e-5, f"Cross-sample leakage in latent B: {diff_latent_b}"
