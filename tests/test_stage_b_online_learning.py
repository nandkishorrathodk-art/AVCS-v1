"""Exhaustive validation suite for Stage B: Online Learning, VICReg Anti-Collapse, and Dual-Model Twin Benchmark."""
from __future__ import annotations

import math

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
    create_clean_random_cortex,
)
from avcs.streaming_v1.runtime import StreamingPermissions
from avcs.streaming_v1.sources import SyntheticCaptureSource

# ==============================================================================
# 1. Pre-Update Evaluation & Timing Tests
# ==============================================================================

def test_pre_update_evaluation_happens_before_weight_update():
    """Verify that pre-update loss is computed on the current frame using PRIOR weights."""
    model = create_clean_random_cortex(seed=123)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

    f0 = torch.rand(1, 3, 32, 32)
    f1 = torch.rand(1, 3, 32, 32)

    # First frame has no pre-update loss (no prior context)
    m0 = learner.process_frame(f0)
    assert m0.pre_update_prediction_loss == 0.0
    assert not m0.updated

    # Capture weights before second frame
    w_before = model.future_predictor[0].weight.clone()

    # Second frame: pre-update loss is computed using w_before
    m1 = learner.process_frame(f1)
    assert m1.pre_update_prediction_loss > 0.0
    assert m1.updated

    # Verify weights actually updated
    w_after = model.future_predictor[0].weight
    assert not torch.allclose(w_before, w_after)


def test_pre_update_evaluation_matches_independent_offline_calculation():
    """Verify that logged pre-update loss strictly matches the error using frozen prior weights."""
    model = create_clean_random_cortex(seed=321)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

    f0 = torch.rand(1, 3, 32, 32)
    f1 = torch.rand(1, 3, 32, 32)

    learner.process_frame(f0)
    z0 = learner._prev_latent.clone()

    # Independent prediction using model state before f1 update
    with torch.no_grad():
        expected_pred = model.predict_future(z0)
        z1_independent, _, _ = model.forward(f1, learner._state, delta_t=0.033)
        expected_loss = F.smooth_l1_loss(expected_pred, z1_independent).item()

    m1 = learner.process_frame(f1)
    assert abs(m1.pre_update_prediction_loss - expected_loss) < 1e-5


# ==============================================================================
# 2. Permission Gates & Fail-Closed Enforcement Tests
# ==============================================================================

def test_permission_gates_prevent_updates_when_disabled():
    """Verify fail-closed permissions: when online learning is False, parameters never mutate."""
    model = create_clean_random_cortex(seed=456)
    # Learning disabled
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=False)
    learner = OnlineCortexLearner(model, permissions=perms)

    f0 = torch.rand(1, 3, 32, 32)
    f1 = torch.rand(1, 3, 32, 32)

    w_initial = model.scale_half[0].weight.clone()

    m0 = learner.process_frame(f0)
    m1 = learner.process_frame(f1)

    assert not m0.updated
    assert not m1.updated
    assert m1.total_loss is None
    assert m1.grad_norm is None

    # Weights remain bit-for-bit identical
    assert torch.equal(w_initial, model.scale_half[0].weight)


def test_observation_permission_fails_closed():
    """Verify observation permission fails closed when enable_observation is False."""
    model = create_clean_random_cortex(seed=789)
    perms = StreamingPermissions(enable_observation=False)
    learner = OnlineCortexLearner(model, permissions=perms)

    f = torch.rand(1, 3, 32, 32)
    with pytest.raises(PermissionError, match="Live observation is disabled"):
        learner.process_frame(f)


def test_learning_permission_assert_learning_allowed_fails_closed():
    """Verify that assert_learning_allowed() raises PermissionError when online learning is False."""
    model = create_clean_random_cortex(seed=987)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=False)
    learner = OnlineCortexLearner(model, permissions=perms)

    with pytest.raises(PermissionError, match="Online parameter learning is disabled"):
        learner.assert_learning_allowed()


def test_learner_default_permissions_fail_closed():
    """Verify that learner initialized with permissions=None defaults to fail closed."""
    model = create_clean_random_cortex(seed=111)
    learner = OnlineCortexLearner(model, permissions=None)

    assert not learner.permissions.enable_observation
    assert not learner.permissions.enable_online_learning
    assert not learner.permissions.authorize_checkpoint_save

    f = torch.rand(1, 3, 32, 32)
    with pytest.raises(PermissionError, match="Live observation is disabled"):
        learner.process_frame(f)


# ==============================================================================
# 3. Anti-Collapse VICReg Variance Regularization Tests
# ==============================================================================

def test_vicreg_variance_guard_penalizes_collapsed_features():
    """Verify that VICReg loss explicitly penalizes zero/near-zero variance representations."""
    loss_fn = StreamingVICRegLoss(lambda_pred=1.0, lambda_var=1.0, target_std=0.5)

    pred = torch.zeros(1, 96)

    # 1. Collapsed target (all features identical: zero variance)
    target_collapsed = torch.ones(1, 96) * 2.0
    _loss_collapsed, metrics_collapsed = loss_fn(pred, target_collapsed)

    # 2. Healthy dispersed target (variance > 0.05)
    target_healthy = torch.randn(1, 96)
    _loss_healthy, metrics_healthy = loss_fn(pred, target_healthy)

    assert metrics_collapsed["var_mean"] < 1e-4
    assert metrics_collapsed["var_loss"] > 0.4
    assert metrics_healthy["var_mean"] > 0.05


def test_streaming_vicreg_sliding_window_maintains_variance_above_threshold():
    """Verify that sliding window maintains Var(z) > 0.05 across streaming ticks."""
    loss_fn = StreamingVICRegLoss(margin_gamma=0.25, min_variance_threshold=0.05, window_size=16)

    torch.manual_seed(42)
    # Feed 20 sequential single-frame latents (B=1)
    for _ in range(20):
        pred = torch.randn(1, 96, requires_grad=True)
        target = torch.randn(1, 96, requires_grad=True) * 0.4  # std 0.4 -> var ~0.16 > 0.05
        loss, metrics = loss_fn(pred, target)
        loss.backward()

    assert metrics["var_mean"] > 0.05
    assert metrics["variance_healthy"]
    assert metrics["min_var"] > 0.02


def test_streaming_vicreg_covariance_decorrelation_penalizes_redundancy():
    """Verify that covariance loss penalizes redundant, collinear feature dimensions."""
    loss_fn = StreamingVICRegLoss(lambda_pred=0.0, lambda_var=0.0, lambda_cov=1.0, window_size=16)

    # 1. Uncorrelated features
    for _ in range(16):
        target_uncorr = torch.randn(1, 96)
        _, met_uncorr = loss_fn(torch.zeros(1, 96), target_uncorr)

    loss_fn.reset()

    # 2. Highly correlated features (repeated dimensions)
    for _ in range(16):
        base_signal = torch.randn(1, 1)
        target_corr = base_signal.repeat(1, 96) + torch.randn(1, 96) * 1e-4
        _, met_corr = loss_fn(torch.zeros(1, 96), target_corr)

    assert met_corr["cov_loss"] > met_uncorr["cov_loss"] * 5.0


def test_online_learner_maintains_non_zero_variance():
    """Verify that running online learning on streaming frames maintains Var(z) >= 0.05."""
    model = create_clean_random_cortex(seed=999)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(model, permissions=perms, learning_rate=5e-4)

    source = SyntheticCaptureSource(width=32, height=32, fps=30.0).open()
    try:
        for _ in range(15):
            rgb = source.grab()
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            learner.process_frame(tensor)
    finally:
        source.close()

    assert len(learner.history) == 15
    final_variance = learner.history[-1].latent_variance
    assert final_variance >= 0.05, f"Expected Var(z) >= 0.05, got {final_variance}"


# ==============================================================================
# 4. Gradient Flow & Module Auditor Tests
# ==============================================================================

def test_gradient_auditor_detects_non_zero_gradients_across_all_modules():
    """Verify that all functional modules (encoder, convgru, latent_predictor) receive non-zero gradients."""
    model = create_clean_random_cortex(seed=555)
    auditor = GradientAuditor(min_grad_norm=1e-5)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(
        model,
        permissions=perms,
        learning_rate=1e-3,
        gradient_auditor=auditor,
    )

    source = SyntheticCaptureSource(width=32, height=32, fps=30.0).open()
    try:
        f0 = torch.from_numpy(source.grab()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        f1 = torch.from_numpy(source.grab()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        learner.process_frame(f0)
        m1 = learner.process_frame(f1)
    finally:
        source.close()

    assert m1.updated
    assert m1.gradient_audit is not None
    report = m1.gradient_audit

    assert report.is_healthy
    assert len(report.zero_grad_modules) == 0
    assert len(report.nan_or_inf_modules) == 0
    assert report.module_norms["encoder"] > 1e-4
    assert report.module_norms["convgru"] > 1e-4
    assert report.module_norms["latent_predictor"] > 1e-4
    assert "Gradient flow healthy" in report.summary


def test_gradient_auditor_detects_zero_gradients_and_fails_health_check():
    """Verify that auditor detects when a module has zero gradient and flags is_healthy=False."""
    model = create_clean_random_cortex(seed=666)
    auditor = GradientAuditor()

    # Artificially populate gradients on predictor only, leaving encoder and convgru at 0
    for name, p in model.named_parameters():
        if "future_predictor" in name:
            p.grad = torch.ones_like(p) * 0.1
        else:
            p.grad = torch.zeros_like(p)

    report = auditor.audit(model)
    assert not report.is_healthy
    assert "encoder" in report.zero_grad_modules
    assert "convgru" in report.zero_grad_modules
    assert "latent_predictor" not in report.zero_grad_modules

    with pytest.raises(RuntimeError, match="Gradient audit failed"):
        auditor.assert_healthy(model)


def test_gradient_auditor_classify_parameters():
    """Verify that parameters are correctly classified into functional module groups."""
    auditor = GradientAuditor()
    assert auditor.classify_parameter("scale_half.0.weight") == "encoder"
    assert auditor.classify_parameter("fuse.0.weight") == "encoder"
    assert auditor.classify_parameter("recurrent.gates.weight") == "convgru"
    assert auditor.classify_parameter("latent_projection.0.weight") == "latent_predictor"
    assert auditor.classify_parameter("future_predictor.2.weight") == "latent_predictor"


# ==============================================================================
# 5. Dual-Model Frozen Twin Benchmark Tests
# ==============================================================================

def test_dual_model_benchmark_initial_weights_identical():
    """Verify that online and frozen twins start with strictly identical initial parameters."""
    benchmark = DualModelStreamingBenchmark(seed=777)
    for p_on, p_fr in zip(benchmark.model_online.parameters(), benchmark.model_frozen.parameters()):
        assert torch.equal(p_on, p_fr), "Initial weights must be bit-for-bit identical"


def test_dual_model_benchmark_frozen_twin_never_drifts():
    """Verify that the frozen reference twin model has strictly ZERO parameter drift (drift == 0.0)."""
    benchmark = DualModelStreamingBenchmark(seed=101)

    # Generate stream of 12 synthetic frames
    source = SyntheticCaptureSource(width=32, height=32, fps=30.0).open()
    frames = []
    try:
        for _ in range(12):
            rgb = source.grab()
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
    finally:
        source.close()

    report = benchmark.evaluate_stream(frames)

    assert report.total_ticks == 12
    assert report.frozen_model_parameters_drift < 1e-7, "Frozen reference model parameters must not drift!"
    assert report.variance_guard_passed
    assert report.gradient_flow_healthy
    assert report.status == "PASSED"
    assert "Stage B Benchmark PASSED" in report.summary


def test_dual_model_benchmark_records_honest_metrics():
    """Verify that benchmark reports honest numerical prediction metrics, gradient health, and delta."""
    benchmark = DualModelStreamingBenchmark(seed=202)

    frames = [torch.rand(1, 3, 32, 32) for _ in range(8)]
    report = benchmark.evaluate_stream(frames)

    assert math.isfinite(report.online_pre_update_loss_mean)
    assert math.isfinite(report.frozen_reference_loss_mean)
    assert math.isfinite(report.delta_pred)
    assert math.isfinite(report.prediction_improvement_loss_delta)
    assert report.final_latent_variance > 0.0
    assert report.mean_gradient_norm > 0.0
    assert report.step_latency_ms_p50 > 0.0
    assert report.throughput_fps > 0.0


def test_measurable_prediction_improvement_on_streaming_pattern():
    """Verify measurable future prediction error reduction (delta_pred < 0) on streaming visual trajectory."""
    source = SyntheticCaptureSource(width=32, height=32, fps=30.0).open()
    frames = []
    try:
        for _ in range(30):
            rgb = source.grab()
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
    finally:
        source.close()

    benchmark = DualModelStreamingBenchmark(seed=42, learning_rate=1e-3)
    report = benchmark.evaluate_stream(frames)

    # delta_pred = online - frozen. Negative delta proves online adaptation on stream!
    assert report.delta_pred < 0.0, (
        f"Expected delta_pred < 0, got {report.delta_pred:.5f} "
        f"(online={report.online_pre_update_loss_mean:.5f}, frozen={report.frozen_reference_loss_mean:.5f})"
    )
    assert report.prediction_improvement_loss_delta > 0.0
    assert report.relative_improvement_pct > 0.0
    assert report.variance_guard_passed
    assert report.gradient_flow_healthy
    assert report.frozen_model_parameters_drift < 1e-7


def test_online_cortex_learner_reset_clears_history_and_state():
    """Verify that learner.reset() cleanly clears recurrent state, previous latent, and step history."""
    model = create_clean_random_cortex(seed=333)
    perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
    learner = OnlineCortexLearner(model, permissions=perms)

    f = torch.rand(1, 3, 32, 32)
    learner.process_frame(f)
    assert len(learner.history) == 1
    assert learner._state is not None
    assert learner._prev_latent is not None

    learner.reset()
    assert len(learner.history) == 0
    assert learner._state is None
    assert learner._prev_latent is None
    assert learner._tick == 0
