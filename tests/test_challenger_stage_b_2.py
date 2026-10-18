"""Adversarial challenger stress test suite for Stage B: Dual-Model Divergence & Learning Efficiency.

Vectors Tested:
1. Twin parameter divergence (||theta_online - theta_frozen||_F > 0 while frozen twin remains 100% bit-for-bit unchanged).
2. Delta prediction validity across recurring patterns (online model adapts to pattern, delta_pred < 0).
3. 500-step memory stability without graph retention or memory leak (bounded RSS, detached state).
4. Symmetrical state reset without weight corruption (weights strictly preserved, state fully isolated).
"""
from __future__ import annotations

import copy
import math
import psutil
import pytest
import torch
import torch.nn.functional as F

from avcs.streaming_v1.anti_collapse import (
    GradientAuditor,
    StreamingVICRegLoss,
)
from avcs.streaming_v1.learner import (
    DualModelStreamingBenchmark,
    OnlineCortexLearner,
)
from avcs.streaming_v1.model import (
    CortexState,
    create_clean_random_cortex,
)
from avcs.streaming_v1.runtime import StreamingPermissions
from avcs.streaming_v1.sources import SyntheticCaptureSource


# ==============================================================================
# Challenge Vector 1: Twin Parameter Divergence & Frozen Immutability
# ==============================================================================

def test_twin_parameter_divergence_across_submodules():
    """Verify that online learning diverges parameters (||theta_online - theta_frozen||_F > 0)

    across all core functional modules while frozen twin remains 100% bit-for-bit immutable.
    """
    benchmark = DualModelStreamingBenchmark(seed=42, learning_rate=1e-3)

    # 1. Verify initial Frobenius divergence is strictly 0.0
    initial_total_divergence = 0.0
    for p_on, p_fr in zip(benchmark.model_online.parameters(), benchmark.model_frozen.parameters()):
        initial_total_divergence += float(torch.norm(p_on - p_fr).item())
    assert initial_total_divergence == 0.0, "Initial twin parameters must have 0.0 divergence"

    # Clone all frozen reference weights for bit-for-bit verification
    initial_frozen_weights = {
        name: p.clone() for name, p in benchmark.model_frozen.named_parameters()
    }

    # 2. Feed 25 synthetic frames to update online model
    source = SyntheticCaptureSource(width=32, height=32, fps=30.0).open()
    frames = []
    try:
        for _ in range(25):
            rgb = source.grab()
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
    finally:
        source.close()

    report = benchmark.evaluate_stream(frames)
    assert report.total_ticks == 25

    # 3. Compute Frobenius divergence ||theta_online - theta_frozen||_F
    auditor = GradientAuditor()
    module_divergences: dict[str, float] = {}
    total_frobenius_sq = 0.0

    for (name_on, p_on), (name_fr, p_fr) in zip(
        benchmark.model_online.named_parameters(),
        benchmark.model_frozen.named_parameters(),
    ):
        assert name_on == name_fr
        diff = p_on - p_fr
        diff_norm_sq = float(torch.sum(diff ** 2).item())
        total_frobenius_sq += diff_norm_sq

        mod_group = auditor.classify_parameter(name_on)
        module_divergences[mod_group] = module_divergences.get(mod_group, 0.0) + math.sqrt(diff_norm_sq)

    total_frobenius = math.sqrt(total_frobenius_sq)

    # Online model must genuinely diverge
    assert total_frobenius > 0.05, f"Expected Frobenius divergence > 0.05, got {total_frobenius:.6f}"

    # Every core learnable submodule must have non-zero parameter divergence
    for mod_name in ("encoder", "convgru", "latent_predictor"):
        div_val = module_divergences.get(mod_name, 0.0)
        assert div_val > 1e-4, f"Core module '{mod_name}' failed to diverge: {div_val:.6f}"

    # 4. Verify frozen reference model parameters remain 100% bit-for-bit unchanged
    for name, p_fr in benchmark.model_frozen.named_parameters():
        p_init = initial_frozen_weights[name]
        assert torch.equal(p_fr, p_init), (
            f"Frozen parameter '{name}' was corrupted or mutated during benchmark!"
        )
        assert not p_fr.requires_grad, f"Frozen parameter '{name}' must have requires_grad=False"

    assert benchmark.model_frozen.training is False


def test_frozen_twin_immutability_under_adversarial_stream_perturbations():
    """Verify frozen model parameters remain bit-for-bit unchanged even under extreme inputs.

    Adversarial inputs: all-zero frames, saturated 1.0 frames, random uniform noise, and delta spikes.
    """
    benchmark = DualModelStreamingBenchmark(seed=999, learning_rate=2e-3)
    initial_weights = [p.clone() for p in benchmark.model_frozen.parameters()]

    # Adversarial frames
    adv_frames = [
        torch.zeros(1, 3, 32, 32),
        torch.ones(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32) * 1e-4,
        torch.clamp(torch.randn(1, 3, 32, 32) * 5.0, 0.0, 1.0),
        torch.zeros(1, 3, 32, 32),
        torch.ones(1, 3, 32, 32),
    ]

    benchmark.evaluate_stream(adv_frames)

    # Check every parameter tensor for strict equality
    for i, (p_current, p_init) in enumerate(zip(benchmark.model_frozen.parameters(), initial_weights)):
        assert torch.equal(p_current, p_init), f"Parameter index {i} drifted under adversarial stream"


def test_twin_parameters_remain_identical_when_learning_disabled():
    """Verify that when online learning is disabled, divergence stays strictly 0.0."""
    model_online = create_clean_random_cortex(seed=555)
    model_frozen = copy.deepcopy(model_online)

    perms = StreamingPermissions(enable_observation=True, enable_online_learning=False)
    learner = OnlineCortexLearner(model_online, permissions=perms)

    for _ in range(10):
        frame = torch.rand(1, 3, 32, 32)
        learner.process_frame(frame)

    for p_on, p_fr in zip(model_online.parameters(), model_frozen.parameters()):
        assert torch.equal(p_on, p_fr), "When learning is disabled, parameters must not diverge"


# ==============================================================================
# Challenge Vector 2: Delta Prediction Validity Across Recurring Patterns
# ==============================================================================

def test_delta_pred_formula_consistency():
    """Verify delta_pred is strictly online_pre_update_loss_mean - frozen_reference_loss_mean."""
    benchmark = DualModelStreamingBenchmark(seed=777)
    frames = [torch.rand(1, 3, 32, 32) for _ in range(12)]
    report = benchmark.evaluate_stream(frames)

    expected_delta_pred = report.online_pre_update_loss_mean - report.frozen_reference_loss_mean
    assert abs(report.delta_pred - expected_delta_pred) < 1e-7

    expected_improvement = report.frozen_reference_loss_mean - report.online_pre_update_loss_mean
    assert abs(report.prediction_improvement_loss_delta - expected_improvement) < 1e-7
    assert abs(report.prediction_improvement_loss_delta + report.delta_pred) < 1e-7


def test_delta_pred_negative_on_recurring_oscillating_pattern():
    """Stress-test against a synthetic recurring oscillating pattern and verify online model

    adapts to recurring patterns with lower prediction loss than frozen reference (delta_pred < 0).
    """
    # Create an oscillating visual stimulus: a bright bar shifting back and forth periodically
    num_frames = 60
    frames = []
    width, height = 32, 32

    for t in range(num_frames):
        frame = torch.zeros(1, 3, height, width)
        # Bar oscillates with period 10: positions 4 -> 24 -> 4
        phase = (t % 10) / 10.0
        bar_center = int(4 + 20 * (0.5 * (1.0 + math.sin(2.0 * math.pi * phase))))
        bar_center = max(2, min(width - 3, bar_center))
        frame[:, :, :, bar_center - 2 : bar_center + 3] = 0.9
        frames.append(frame)

    benchmark = DualModelStreamingBenchmark(seed=42, learning_rate=1e-3)
    report = benchmark.evaluate_stream(frames)

    # 1. Overall delta_pred must be negative (online outpredicts frozen)
    assert report.delta_pred < 0.0, (
        f"Online model failed to outpredict frozen model on recurring stimulus: "
        f"delta_pred={report.delta_pred:.6f} (online={report.online_pre_update_loss_mean:.6f}, "
        f"frozen={report.frozen_reference_loss_mean:.6f})"
    )

    # 2. Verify adaptation: online loss should improve in the second half vs first half
    online_losses = [m.pre_update_prediction_loss for m in benchmark.learner.history[1:]]
    mid = len(online_losses) // 2
    first_half_mean = float(torch.tensor(online_losses[:mid]).mean().item())
    second_half_mean = float(torch.tensor(online_losses[mid:]).mean().item())

    assert second_half_mean < first_half_mean, (
        f"Expected online adaptation over recurring pattern: second_half ({second_half_mean:.5f}) "
        f"must be lower than first_half ({first_half_mean:.5f})"
    )


def test_pre_update_evaluation_timing_verified_empirically():
    """Verify that pre-update evaluation strictly occurs prior to the optimizer parameter update."""
    model = create_clean_random_cortex(seed=888)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

    f0 = torch.rand(1, 3, 32, 32)
    f1 = torch.rand(1, 3, 32, 32)

    learner.process_frame(f0)
    z0 = learner._prev_latent.clone()
    state_before = CortexState(
        hidden=learner._state.hidden.clone(),
        previous_features=learner._state.previous_features.clone(),
    )

    # Snapshot weights prior to f1 update
    w_before = model.future_predictor[0].weight.clone()

    # Pre-compute expected error using prior (pre-update) weights and state
    with torch.no_grad():
        expected_pred = model.predict_future(z0)
        z1_target, _, _ = model.forward(f1, state_before, delta_t=0.033)
        expected_pre_loss = F.smooth_l1_loss(expected_pred, z1_target).item()

    # Step on f1 (evaluates pre-update loss with t=0 weights, then updates to t=1 weights)
    m1 = learner.process_frame(f1)

    # Check 1: Evaluated loss matches pre-update calculation exactly
    assert abs(m1.pre_update_prediction_loss - expected_pre_loss) < 1e-5

    # Check 2: Weights genuinely updated
    w_after = model.future_predictor[0].weight
    assert not torch.allclose(w_before, w_after)

    # Check 3: Post-update weights yield a different loss, proving pre-update loss was recorded prior to update
    with torch.no_grad():
        post_pred = model.predict_future(z0)
        post_loss = F.smooth_l1_loss(post_pred, z1_target).item()
    assert abs(post_loss - m1.pre_update_prediction_loss) > 1e-4


# ==============================================================================
# Challenge Vector 3: 500-Step Memory Stability Over Stream
# ==============================================================================

def test_500_step_memory_stability_without_graph_retention_or_leak():
    """Run 500 online benchmark ticks and verify RSS memory remains bounded with zero graph retention."""
    process = psutil.Process()

    model = create_clean_random_cortex(seed=1234)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

    # Reusable frame to avoid tensor list overhead
    frame_static = torch.rand(1, 3, 32, 32)

    # Warmup 50 steps
    for _ in range(50):
        learner.process_frame(frame_static)

    # Measure RSS after warmup
    rss_warmup_bytes = process.memory_info().rss
    rss_warmup_mb = rss_warmup_bytes / (1024 * 1024)

    # Run remaining 450 ticks (up to 500)
    for tick in range(50, 500):
        # Slightly modulate frame to ensure active gradient computation, strictly in [0, 1]
        frame = torch.clamp(frame_static * 0.9 + (tick % 5) * 0.02, 0.0, 1.0)
        m = learner.process_frame(frame)

        # Invariant: Computational graph must be detached at every single step
        assert learner._state.hidden.grad_fn is None, f"Tick {tick}: hidden state has attached grad_fn!"
        assert learner._state.previous_features.grad_fn is None, f"Tick {tick}: previous_features has attached grad_fn!"
        assert learner._prev_latent.grad_fn is None, f"Tick {tick}: prev_latent has attached grad_fn!"

        # Invariant: VICReg sliding queue must not grow beyond window_size
        assert len(learner.loss_fn._queue) <= learner.loss_fn.window_size, "VICReg queue overflow!"
        for q_tensor in learner.loss_fn._queue:
            assert q_tensor.grad_fn is None, "VICReg queue tensor retains attached grad_fn graph!"

        # Sanity check: finite losses and gradient norms
        assert math.isfinite(m.pre_update_prediction_loss)
        if m.total_loss is not None:
            assert math.isfinite(m.total_loss)
        if m.grad_norm is not None:
            assert math.isfinite(m.grad_norm)

    rss_final_bytes = process.memory_info().rss
    rss_final_mb = rss_final_bytes / (1024 * 1024)
    rss_growth_mb = rss_final_mb - rss_warmup_mb

    assert len(learner.history) == 500

    # Memory growth across 450 active backprop ticks must remain strictly bounded (< 40 MB)
    assert rss_growth_mb < 40.0, (
        f"Memory leak detected over 500 steps: RSS grew by {rss_growth_mb:.2f} MB "
        f"(warmup={rss_warmup_mb:.2f} MB, final={rss_final_mb:.2f} MB)"
    )


def test_history_records_contain_only_detached_primitives():
    """Verify that OnlineStepMetrics objects store pure scalar floats, avoiding graph retention."""
    model = create_clean_random_cortex(seed=4321)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(model, permissions=perms)

    for _ in range(5):
        learner.process_frame(torch.rand(1, 3, 32, 32))

    for m in learner.history:
        assert isinstance(m.tick, int)
        assert isinstance(m.pre_update_prediction_loss, float)
        assert m.total_loss is None or isinstance(m.total_loss, float)
        assert isinstance(m.latent_variance, float)
        assert m.grad_norm is None or isinstance(m.grad_norm, float)
        assert isinstance(m.updated, bool)
        assert isinstance(m.step_latency_ms, float)


# ==============================================================================
# Challenge Vector 4: Symmetrical State Reset Without Weight Corruption
# ==============================================================================

def test_symmetrical_state_reset_does_not_corrupt_trained_weights():
    """Verify that learner.reset() resets recurrent state without corrupting or resetting model weights."""
    model = create_clean_random_cortex(seed=7777)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

    # Train for 15 steps to mutate weights
    for _ in range(15):
        learner.process_frame(torch.rand(1, 3, 32, 32))

    # Snapshot trained weights
    trained_weights = {name: p.clone() for name, p in model.named_parameters()}

    # Call reset
    learner.reset()

    # 1. State must be cleanly cleared
    assert learner._state is None
    assert learner._prev_latent is None
    assert learner._tick == 0
    assert len(learner.history) == 0
    assert len(learner.loss_fn._queue) == 0

    # 2. Weights must be strictly preserved bit-for-bit
    for name, p in model.named_parameters():
        assert torch.equal(p, trained_weights[name]), (
            f"Reset corrupted weights for parameter '{name}'!"
        )


def test_symmetrical_state_isolation_matches_fresh_model_instance():
    """Verify that resetting state provides 100% symmetrical isolation:

    a reset learner processes a probe frame identically to a fresh, un-streamed learner.
    """
    model = create_clean_random_cortex(seed=8888)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=False)
    learner = OnlineCortexLearner(model, permissions=perms)

    # Feed 10 distinctive frames to accumulate heavy recurrent context
    for _ in range(10):
        learner.process_frame(torch.rand(1, 3, 32, 32) * 0.5 + 0.2)

    assert learner._state is not None
    assert learner._prev_latent is not None

    # Reset learner
    learner.reset()

    # Fresh clone with identical model weights
    fresh_model = copy.deepcopy(model)
    fresh_learner = OnlineCortexLearner(fresh_model, permissions=perms)

    # Probe frame
    probe_frame = torch.rand(1, 3, 32, 32)

    m_reset = learner.process_frame(probe_frame)
    m_fresh = fresh_learner.process_frame(probe_frame)

    # Predictions and losses must match identically
    assert m_reset.pre_update_prediction_loss == m_fresh.pre_update_prediction_loss == 0.0
    assert abs(m_reset.latent_variance - m_fresh.latent_variance) < 1e-6

    # Recurrent state must match bit-for-bit
    assert torch.equal(learner._state.hidden, fresh_learner._state.hidden), "Hidden state leaked context across reset!"
    assert torch.equal(learner._state.previous_features, fresh_learner._state.previous_features)
    assert torch.equal(learner._prev_latent, fresh_learner._prev_latent)


def test_dual_model_benchmark_reset_state_isolation():
    """Verify that DualModelStreamingBenchmark cleans recurrent state between distinct streams."""
    benchmark = DualModelStreamingBenchmark(seed=1212, learning_rate=1e-3)

    stream_a = [torch.rand(1, 3, 32, 32) for _ in range(8)]
    stream_b = [torch.rand(1, 3, 32, 32) for _ in range(8)]

    report_a = benchmark.evaluate_stream(stream_a)
    assert report_a.status == "PASSED"
    assert report_a.frozen_model_parameters_drift < 1e-7

    report_b = benchmark.evaluate_stream(stream_b)
    assert report_b.status == "PASSED"
    assert report_b.frozen_model_parameters_drift < 1e-7
