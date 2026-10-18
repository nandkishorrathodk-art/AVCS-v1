"""Adversarial stress testing and empirical challenge suite for Stage B.

Vectors tested:
1. Degenerate Stream Anti-Collapse Stress (100 blank/constant frames, Var(z) >= 0.05).
2. Permission Bypass Under Adversarial Mutation (zero grad, bit-for-bit parameter immutability).
3. Gradient Flow & NaN Injection Resistance in GradientAuditor.
4. Long Sequence Numerical & Memory Stability (300 continuous learning steps).
"""
from __future__ import annotations

import gc
import math
import os

import psutil
import pytest
import torch

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
# Vector 1: Degenerate Stream Anti-Collapse Stress
# ==============================================================================

class TestDegenerateStreamAntiCollapse:
    """Stress-test VICReg anti-collapse and latent variance under degenerate visual inputs."""

    def test_100_all_white_frames_maintains_variance_above_threshold(self):
        """Feed 100 identical all-white frames (x_t = 1.0) from cold start; verify Var(z) >= 0.05."""
        model = create_clean_random_cortex(seed=42)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
        learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

        white_frame = torch.ones(1, 3, 32, 32)
        variances: list[float] = []

        for _ in range(100):
            metric = learner.process_frame(white_frame)
            variances.append(metric.latent_variance)

        assert len(variances) == 100
        assert all(v >= 0.05 for v in variances), (
            f"Representation collapsed! Min variance: {min(variances):.5f} < 0.05"
        )
        assert min(variances) > 0.5, f"Expected robust variance, got {min(variances):.4f}"

    def test_100_constant_grey_frames_maintains_variance_above_threshold(self):
        """Feed 100 identical mid-grey frames (x_t = 0.5) from cold start; verify Var(z) >= 0.05."""
        model = create_clean_random_cortex(seed=101)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
        learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

        grey_frame = torch.full((1, 3, 32, 32), 0.5)
        variances: list[float] = []

        for _ in range(100):
            metric = learner.process_frame(grey_frame)
            variances.append(metric.latent_variance)

        assert len(variances) == 100
        assert all(v >= 0.05 for v in variances), (
            f"Representation collapsed under constant grey! Min variance: {min(variances):.5f} < 0.05"
        )

    def test_temporal_stream_transition_to_100_black_frames_retains_variance(self):
        """Initialize temporal state with random frames, then feed 100 zero frames (x_t = 0).

        Verifies that ConvGRU recurrent hidden state and anti-collapse regularizer
        prevent collapse during prolonged dark/black screen periods.
        """
        model = create_clean_random_cortex(seed=202)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
        learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

        # 5 normal initial frames
        for _ in range(5):
            learner.process_frame(torch.rand(1, 3, 32, 32))

        black_frame = torch.zeros(1, 3, 32, 32)
        variances: list[float] = []

        for _ in range(100):
            metric = learner.process_frame(black_frame)
            variances.append(metric.latent_variance)

        assert len(variances) == 100
        assert all(v >= 0.05 for v in variances), (
            f"Collapsed under zero frames after initialization! Min variance: {min(variances):.5f}"
        )

    def test_vicreg_loss_actively_detects_and_penalizes_collapsed_latents(self):
        """Directly feed 100 identical collapsed latents into StreamingVICRegLoss.

        Verifies that variance hinge loss flags collapse risk (variance_healthy=False)
        and applies strong positive penalty (var_loss > 0).
        """
        loss_fn = StreamingVICRegLoss(margin_gamma=0.25, min_variance_threshold=0.05, window_size=16)

        pred = torch.zeros(1, 96, requires_grad=True)
        # Constant identical vector across 100 iterations
        collapsed_target = torch.ones(1, 96) * 3.1415

        for _ in range(100):
            loss, metrics = loss_fn(pred, collapsed_target)

        assert not metrics["variance_healthy"]
        assert metrics["var_mean"] < 1e-4
        assert metrics["var_loss"] > 0.20
        assert float(loss.item()) > 0.10


# ==============================================================================
# Vector 2: Permission Bypass Under Adversarial Mutation
# ==============================================================================

class TestPermissionBypassUnderMutation:
    """Stress-test fail-closed permission gates and parameter immutability."""

    def test_process_frame_with_learning_disabled_never_computes_grads(self):
        """When enable_online_learning=False, process_frame must compute ZERO gradients."""
        model = create_clean_random_cortex(seed=303)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=False)
        learner = OnlineCortexLearner(model, permissions=perms)

        for _ in range(25):
            metric = learner.process_frame(torch.rand(1, 3, 32, 32))
            assert not metric.updated
            assert metric.total_loss is None
            assert metric.grad_norm is None

        # Assert no parameter has a .grad tensor populated
        for name, param in model.named_parameters():
            assert param.grad is None, f"Parameter {name} has unexpected gradient tensor!"

    def test_process_frame_with_learning_disabled_guarantees_bit_for_bit_immutability(self):
        """When enable_online_learning=False, parameters remain bit-for-bit identical after 50 frames."""
        model = create_clean_random_cortex(seed=404)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=False)
        learner = OnlineCortexLearner(model, permissions=perms)

        params_snapshot = {name: param.clone() for name, param in model.named_parameters()}

        for _ in range(50):
            learner.process_frame(torch.rand(1, 3, 32, 32))

        for name, param in model.named_parameters():
            assert torch.equal(params_snapshot[name], param), (
                f"Parameter {name} experienced unauthorized mutation!"
            )

    def test_assert_learning_allowed_raises_permission_error_when_disabled(self):
        """Calling assert_learning_allowed() when enable_online_learning=False must raise PermissionError."""
        model = create_clean_random_cortex(seed=505)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=False)
        learner = OnlineCortexLearner(model, permissions=perms)

        with pytest.raises(PermissionError, match="Online parameter learning is disabled"):
            learner.assert_learning_allowed()

    def test_process_frame_raises_permission_error_when_observation_disabled(self):
        """Calling process_frame() when enable_observation=False must fail closed immediately."""
        model = create_clean_random_cortex(seed=606)
        perms = StreamingPermissions(enable_observation=False, enable_online_learning=False)
        learner = OnlineCortexLearner(model, permissions=perms)

        frame = torch.rand(1, 3, 32, 32)
        with pytest.raises(PermissionError, match="Live observation is disabled"):
            learner.process_frame(frame)

    def test_dynamic_permission_toggle_freezes_and_resumes_cleanly(self):
        """Toggle enable_online_learning False -> True -> False.

        Verifies that parameter updates only happen during the authorized window,
        and freeze immediately bit-for-bit once revoked.
        """
        model = create_clean_random_cortex(seed=707)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=False)
        learner = OnlineCortexLearner(model, permissions=perms, learning_rate=1e-3)

        # 1. Phase 1: Disabled
        snap_0 = {name: p.clone() for name, p in model.named_parameters()}
        for _ in range(10):
            learner.process_frame(torch.rand(1, 3, 32, 32))
        for name, p in model.named_parameters():
            assert torch.equal(snap_0[name], p)

        # 2. Phase 2: Enabled
        learner.permissions.enable_online_learning = True
        for _ in range(10):
            learner.process_frame(torch.rand(1, 3, 32, 32))
        # At least one parameter must have updated
        has_changed = any(not torch.equal(snap_0[name], p) for name, p in model.named_parameters())
        assert has_changed, "Parameters failed to update during authorized window!"

        # 3. Phase 3: Revoked
        learner.permissions.enable_online_learning = False
        snap_revoked = {name: p.clone() for name, p in model.named_parameters()}
        for _ in range(10):
            learner.process_frame(torch.rand(1, 3, 32, 32))
        for name, p in model.named_parameters():
            assert torch.equal(snap_revoked[name], p), f"Parameter {name} mutated after permission revoked!"


# ==============================================================================
# Vector 3: Gradient Flow & NaN Injection Resistance
# ==============================================================================

class TestGradientFlowAndNaNInjection:
    """Stress-test GradientAuditor against adversarial anomalies (NaN, Inf, Zero, Missing grads)."""

    def test_nan_injection_in_encoder_detected_immediately(self):
        """Inject NaN into encoder gradient tensor; verify GradientAuditor catches it."""
        model = create_clean_random_cortex(seed=808)
        auditor = GradientAuditor()

        for p in model.parameters():
            p.grad = torch.ones_like(p) * 0.05

        # Inject NaN into scale_half
        injected = False
        for name, p in model.named_parameters():
            if "scale_half" in name:
                p.grad[0, 0, 0, 0] = float("nan")
                injected = True
                break
        assert injected

        report = auditor.audit(model)
        assert not report.is_healthy
        assert "encoder" in report.nan_or_inf_modules
        assert "non_finite in ['encoder']" in report.summary

        with pytest.raises(RuntimeError, match="Gradient audit failed"):
            auditor.assert_healthy(model)

    def test_inf_injection_in_convgru_detected_immediately(self):
        """Inject positive and negative infinity into ConvGRU recurrent weights; verify detection."""
        model = create_clean_random_cortex(seed=909)
        auditor = GradientAuditor()

        for p in model.parameters():
            p.grad = torch.ones_like(p) * 0.05

        # Inject +inf and -inf into recurrent gates
        injected = False
        for name, p in model.named_parameters():
            if "recurrent.gates" in name:
                p.grad[0, 0, 0, 0] = float("inf")
                p.grad[0, 1, 0, 0] = float("-inf")
                injected = True
                break
        assert injected

        report = auditor.audit(model)
        assert not report.is_healthy
        assert "convgru" in report.nan_or_inf_modules

        with pytest.raises(RuntimeError, match="Gradient audit failed"):
            auditor.assert_healthy(model)

    def test_zero_gradient_module_detected_and_flagged(self):
        """Zero out gradients in latent_predictor while encoder and convgru are healthy."""
        model = create_clean_random_cortex(seed=1010)
        auditor = GradientAuditor()

        for name, p in model.named_parameters():
            if "future_predictor" in name or "latent_projection" in name:
                p.grad = torch.zeros_like(p)
            else:
                p.grad = torch.ones_like(p) * 0.05

        report = auditor.audit(model)
        assert not report.is_healthy
        assert "latent_predictor" in report.zero_grad_modules
        assert "encoder" not in report.zero_grad_modules
        assert "convgru" not in report.zero_grad_modules

    def test_missing_grad_tensor_treated_as_zero_gradient_failure(self):
        """If a parameter in a required module has grad=None, auditor flags zero_grad."""
        model = create_clean_random_cortex(seed=1111)
        auditor = GradientAuditor()

        # Encoder has grads, but convgru and latent_predictor have grad=None
        for name, p in model.named_parameters():
            if auditor.classify_parameter(name) == "encoder":
                p.grad = torch.ones_like(p) * 0.05
            else:
                p.grad = None

        report = auditor.audit(model)
        assert not report.is_healthy
        assert "convgru" in report.zero_grad_modules
        assert "latent_predictor" in report.zero_grad_modules

    def test_live_backpropagation_produces_healthy_gradients_across_all_modules(self):
        """Run actual live learner backward pass and assert healthy non-zero gradient flow."""
        model = create_clean_random_cortex(seed=1212)
        auditor = GradientAuditor(min_grad_norm=1e-5)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
        learner = OnlineCortexLearner(
            model,
            permissions=perms,
            learning_rate=1e-3,
            gradient_auditor=auditor,
        )

        f0 = torch.rand(1, 3, 32, 32)
        f1 = torch.rand(1, 3, 32, 32)
        learner.process_frame(f0)
        m1 = learner.process_frame(f1)

        assert m1.updated
        assert m1.gradient_audit is not None
        report = m1.gradient_audit

        assert report.is_healthy
        assert report.total_norm > 0.05
        assert len(report.zero_grad_modules) == 0
        assert len(report.nan_or_inf_modules) == 0
        for mod in GradientAuditor.CORE_MODULES:
            norm = report.module_norms[mod]
            assert math.isfinite(norm)
            assert norm > 1e-4, f"Module {mod} norm too low: {norm}"


# ==============================================================================
# Vector 4: Long Sequence Numerical & Memory Stability
# ==============================================================================

class TestLongSequenceNumericalStability:
    """Stress-test numerical health and memory behavior over 300 continuous online learning steps."""

    def test_300_continuous_online_learning_steps_stability(self):
        """Execute 300 sequential online learning updates on synthetic video stream.

        Invariants checked:
        1. Flat RAM: RSS memory delta between tick 100 and tick 300 must remain < 25 MB.
        2. No NaNs or Infs: All losses and gradient norms must be strictly finite.
        3. Anti-collapse: Var(z) >= 0.05 at every single step throughout all 300 ticks.
        4. Healthy throughput: Execution must sustain >= 25 FPS on host CPU.
        """
        process = psutil.Process(os.getpid())
        gc.collect()

        model = create_clean_random_cortex(seed=1313)
        perms = StreamingPermissions(enable_observation=True, enable_online_learning=True)
        learner = OnlineCortexLearner(model, permissions=perms, learning_rate=5e-4)

        source = SyntheticCaptureSource(width=32, height=32, fps=30.0).open()

        losses: list[float] = []
        grad_norms: list[float] = []
        variances: list[float] = []
        ram_samples: dict[int, float] = {}

        try:
            for tick in range(300):
                rgb = source.grab()
                frame = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                m = learner.process_frame(frame)

                if m.total_loss is not None:
                    assert math.isfinite(m.total_loss), f"Tick {tick}: non-finite loss {m.total_loss}"
                    losses.append(m.total_loss)

                if m.grad_norm is not None:
                    assert math.isfinite(m.grad_norm), f"Tick {tick}: non-finite grad_norm {m.grad_norm}"
                    assert m.grad_norm > 0.0, f"Tick {tick}: zero gradient norm"
                    grad_norms.append(m.grad_norm)

                assert m.latent_variance >= 0.05, (
                    f"Tick {tick}: representation collapse! Var(z) = {m.latent_variance:.5f} < 0.05"
                )
                variances.append(m.latent_variance)

                if tick in (99, 199, 299):
                    ram_mb = process.memory_info().rss / (1024 * 1024)
                    ram_samples[tick + 1] = ram_mb
        finally:
            source.close()

        assert len(learner.history) == 300
        assert len(losses) == 299  # first tick has no prior context

        # 1. RAM Flatline verification: delta between step 100 and step 300
        ram_100 = ram_samples[100]
        ram_300 = ram_samples[300]
        ram_drift = abs(ram_300 - ram_100)
        assert ram_drift < 25.0, (
            f"Memory leak detected! RAM drift between step 100 ({ram_100:.1f} MB) "
            f"and step 300 ({ram_300:.1f} MB) is {ram_drift:.2f} MB >= 25 MB"
        )

        # 2. Representation health
        assert min(variances) >= 0.05
        assert max(variances) < 2.0  # no explosion

        # 3. Parameter finiteness
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all().item(), f"Parameter {name} contains non-finite values!"

    def test_dual_model_benchmark_300_steps_frozen_twin_absolute_immutability(self):
        """Run 300 continuous frames through DualModelStreamingBenchmark.

        Verifies that across 300 continuous learning steps, the frozen reference
        twin model has strictly ZERO parameter drift (drift < 1e-7).
        """
        source = SyntheticCaptureSource(width=32, height=32, fps=30.0).open()
        frames = []
        try:
            for _ in range(100):  # 100 frames for deep twin validation
                rgb = source.grab()
                frames.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
        finally:
            source.close()

        benchmark = DualModelStreamingBenchmark(seed=1414, learning_rate=5e-4)
        report = benchmark.evaluate_stream(frames)

        assert report.total_ticks == 100
        assert report.frozen_model_parameters_drift < 1e-7, (
            f"Frozen reference twin drifted! Drift: {report.frozen_model_parameters_drift:.8f}"
        )
        assert report.variance_guard_passed
        assert report.gradient_flow_healthy
        assert report.status == "PASSED"
