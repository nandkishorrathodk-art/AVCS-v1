"""Stage B: Online self-supervised learning, VICReg variance guard, and dual-model twin benchmark.

Invariants enforced:
1. Pre-Update Evaluation: Evaluates next-step prediction on incoming frame BEFORE parameter update.
2. Dual-Model Benchmark: Head-to-head comparison of online updating model against an identical frozen twin.
3. Anti-Collapse Guard: Streaming variance regularizer enforcing Var(z) >= 0.05 + non-zero gradient flow.
4. Fail-Closed Controls: StreamingPermissions strictly checked via permissions.assert_learning_allowed().
5. Zero Raw Persistence: All operations execute strictly in ephemeral in-memory tensors; zero frame files to disk.
"""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from avcs.streaming_v1.anti_collapse import (
    GradientAuditor,
    GradientAuditReport,
    StreamingVICRegLoss,
    audit_model_gradients,
)
from avcs.streaming_v1.model import (
    CortexState,
    StreamingVisualCortexV1,
    create_clean_random_cortex,
)
from avcs.streaming_v1.runtime import StreamingPermissions

# Re-export anti-collapse utilities for backwards compatibility
__all__ = [
    "DualModelStreamingBenchmark",
    "GradientAuditReport",
    "GradientAuditor",
    "OnlineCortexLearner",
    "OnlineStepMetrics",
    "OnlineStreamLearner",
    "StageBBenchmarkReport",
    "StreamingVICRegLoss",
    "audit_model_gradients",
]


@dataclass(frozen=True)
class OnlineStepMetrics:
    """Metrics recorded for a single streaming online learning step."""
    tick: int
    pre_update_prediction_loss: float
    total_loss: float | None
    latent_variance: float
    grad_norm: float | None
    updated: bool
    gradient_audit: GradientAuditReport | None = None
    step_latency_ms: float = 0.0


@dataclass(frozen=True)
class StageBBenchmarkReport:
    """Complete report comparing an online learning model against an identical frozen twin."""
    total_ticks: int
    online_pre_update_loss_mean: float
    frozen_reference_loss_mean: float
    prediction_improvement_loss_delta: float  # frozen - online (backward compat)
    relative_improvement_pct: float
    final_latent_variance: float
    variance_guard_passed: bool
    mean_gradient_norm: float
    gradient_flow_healthy: bool
    frozen_model_parameters_drift: float
    status: str
    summary: str
    delta_pred: float = 0.0  # online - frozen (< 0 indicates online outpredicted frozen)
    step_latency_ms_p50: float = 0.0
    step_latency_ms_p95: float = 0.0
    throughput_fps: float = 0.0


class OnlineCortexLearner:
    """Manages online in-memory self-supervised parameter updates for a streaming visual cortex.

    Pre-update evaluation contract:
      On incoming frame x_t, evaluate forecast prediction error L_pre = SmoothL1(hat_z_t, z_t)
      BEFORE applying any optimizer parameter updates.
    """

    def __init__(
        self,
        model: StreamingVisualCortexV1,
        *,
        permissions: StreamingPermissions | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        loss_fn: StreamingVICRegLoss | None = None,
        gradient_auditor: GradientAuditor | None = None,
    ) -> None:
        self.model = model
        self.permissions = permissions or StreamingPermissions(
            enable_observation=False,
            enable_online_learning=False,
            authorize_checkpoint_save=False,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.max_grad_norm = float(max_grad_norm)
        self.loss_fn = loss_fn or StreamingVICRegLoss()
        self.gradient_auditor = gradient_auditor or GradientAuditor()

        self._state: CortexState | None = None
        self._prev_latent: torch.Tensor | None = None
        self._tick = 0
        self.history: list[OnlineStepMetrics] = []

    def reset(self) -> None:
        """Reset internal recurrent state, history, and loss queue."""
        self._state = None
        self._prev_latent = None
        self._tick = 0
        self.history.clear()
        self.loss_fn.reset()

    def assert_learning_allowed(self) -> None:
        """Fail-closed assertion ensuring online learning is explicitly permitted."""
        self.permissions.assert_learning_allowed()

    def process_frame(
        self,
        frame_tensor: torch.Tensor,
        timestamp_delta: float = 0.033,
    ) -> OnlineStepMetrics:
        """Process one incoming frame: evaluate pre-update prediction, then update if allowed.

        Order of operations:
        1. assert_observation_allowed()
        2. forward_step -> extract latent z_t and next recurrent state
        3. Pre-Update Evaluation: L_pre = SmoothL1(predict_future(z_{t-1}), z_t) BEFORE optimizer step
        4. Measure latent variance Var(z_t)
        5. If enable_online_learning:
           assert_learning_allowed() -> backward() -> clip_grad_norm_() -> optimizer.step()
        6. Advance recurrent state (detached)
        """
        t0 = time.perf_counter()
        self.permissions.assert_observation_allowed()

        # 1. Forward step through model
        latent_z, next_state, _outputs = self.model.forward(
            frame_tensor,
            self._state,
            timestamp_delta,
        )

        # 2. Pre-Update Evaluation (BEFORE ANY OPTIMIZER UPDATE)
        pre_update_loss: float = 0.0
        if self._prev_latent is not None:
            with torch.no_grad():
                predicted_z = self.model.predict_future(self._prev_latent)
                pre_update_loss = float(F.smooth_l1_loss(predicted_z, latent_z).item())

        # 3. Compute current latent variance
        with torch.no_grad():
            _, cur_var = self.loss_fn.compute_variance(latent_z)

        # 4. Online Parameter Learning (fail-closed check)
        total_loss_val: float | None = None
        grad_norm_val: float | None = None
        grad_audit_report: GradientAuditReport | None = None
        updated = False

        if self.permissions.enable_online_learning:
            # Enforce fail-closed permission check
            self.permissions.assert_learning_allowed()

            if self._prev_latent is not None:
                self.model.train()
                self.optimizer.zero_grad()

                predicted_z = self.model.predict_future(self._prev_latent)
                loss, _loss_details = self.loss_fn(predicted_z, latent_z)

                loss.backward()

                if self.gradient_auditor is not None:
                    grad_audit_report = self.gradient_auditor.audit(self.model)

                # Gradient clipping with finite norm enforcement
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                grad_norm_val = float(
                    total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
                )

                self.optimizer.step()
                total_loss_val = float(loss.item())
                updated = True
        else:
            self.model.eval()

        # 5. Advance internal state (detach recurrent state to bound BPTT to 1 step in RAM)
        self._state = CortexState(
            hidden=next_state.hidden.detach(),
            previous_features=next_state.previous_features.detach(),
        )
        self._prev_latent = latent_z.detach()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metric = OnlineStepMetrics(
            tick=self._tick,
            pre_update_prediction_loss=pre_update_loss,
            total_loss=total_loss_val,
            latent_variance=cur_var,
            grad_norm=grad_norm_val,
            updated=updated,
            gradient_audit=grad_audit_report,
            step_latency_ms=elapsed_ms,
        )
        self.history.append(metric)
        self._tick += 1
        return metric


# Backwards compatibility alias
OnlineStreamLearner = OnlineCortexLearner


class DualModelStreamingBenchmark:
    """Benchmark comparing an online-learning model against an identical frozen reference twin.

    Contract:
    - Both models initialized with identical random seed and initial weights.
    - Reference twin has requires_grad=False and zero parameter drift across entire stream.
    - Evaluates future prediction error delta: delta_pred = L_pre_online - L_pre_frozen.
    - Zero overclaiming: honest reporting of whether learning improved future prediction on stream.
    """

    def __init__(
        self,
        base_model: StreamingVisualCortexV1 | None = None,
        *,
        seed: int = 42,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        loss_fn: StreamingVICRegLoss | None = None,
        gradient_auditor: GradientAuditor | None = None,
    ) -> None:
        if base_model is None:
            self.base_model = create_clean_random_cortex(seed=seed)
        else:
            self.base_model = base_model

        # Create two twin models from identical initial weights
        self.model_online = copy.deepcopy(self.base_model)
        self.model_frozen = copy.deepcopy(self.base_model)

        # Freeze the reference twin completely
        for param in self.model_frozen.parameters():
            param.requires_grad = False
        self.model_frozen.eval()

        # Verify initial parameter equality (zero initial drift)
        initial_drift = 0.0
        with torch.no_grad():
            for p_base, p_froz in zip(self.model_online.parameters(), self.model_frozen.parameters()):
                initial_drift += float(torch.norm(p_base - p_froz).item())
        if initial_drift > 1e-7:
            raise RuntimeError(f"Twin model initialization failure: initial drift {initial_drift} > 0")

        # Online model learner with observation and online learning authorized
        self.online_perms = StreamingPermissions(
            enable_observation=True,
            enable_online_learning=True,
            authorize_checkpoint_save=False,
        )
        self.loss_fn = loss_fn or StreamingVICRegLoss()
        self.gradient_auditor = gradient_auditor or GradientAuditor()

        self.learner = OnlineCortexLearner(
            self.model_online,
            permissions=self.online_perms,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            loss_fn=self.loss_fn,
            gradient_auditor=self.gradient_auditor,
        )

        # Frozen reference tracking state
        self._frozen_state: CortexState | None = None
        self._frozen_prev_latent: torch.Tensor | None = None
        self.frozen_losses: list[float] = []

    def evaluate_stream(
        self,
        frames: Sequence[torch.Tensor],
        timestamp_delta: float = 0.033,
    ) -> StageBBenchmarkReport:
        """Run identical frame stream synchronously through both models.

        Args:
            frames: Sequence of (1, 3, H, W) float32 frame tensors in [0, 1].
            timestamp_delta: Elapsed time between successive frames.

        Returns:
            StageBBenchmarkReport with pre-update comparison and honest metric analysis.
        """
        self.learner.reset()
        self._frozen_state = None
        self._frozen_prev_latent = None
        self.frozen_losses.clear()

        latencies_ms: list[float] = []

        for frame in frames:
            t_start = time.perf_counter()

            # 1. Step Frozen Reference Model (Pre-update future prediction)
            with torch.no_grad():
                f_latent, next_f_state, _ = self.model_frozen.forward(
                    frame,
                    self._frozen_state,
                    timestamp_delta,
                )
                if self._frozen_prev_latent is not None:
                    pred_f = self.model_frozen.predict_future(self._frozen_prev_latent)
                    f_loss = float(F.smooth_l1_loss(pred_f, f_latent).item())
                    self.frozen_losses.append(f_loss)

                self._frozen_state = next_f_state
                self._frozen_prev_latent = f_latent

            # 2. Step Online Model (Evaluates pre-update loss on frame, then updates parameters)
            self.learner.process_frame(frame, timestamp_delta=timestamp_delta)

            step_elapsed = (time.perf_counter() - t_start) * 1000.0
            latencies_ms.append(step_elapsed)

        # Ignore tick 0 since no prior frame context existed for forecasting
        online_losses = [m.pre_update_prediction_loss for m in self.learner.history[1:]]
        frozen_losses = self.frozen_losses

        if not online_losses or not frozen_losses:
            raise ValueError(
                "Stream was too short to evaluate prediction improvement (need >= 3 frames)."
            )

        mean_online = float(np.mean(online_losses))
        mean_frozen = float(np.mean(frozen_losses))

        # delta_pred: online - frozen (< 0 means online had lower forecast error)
        delta_pred = mean_online - mean_frozen
        # prediction_improvement_loss_delta: frozen - online (> 0 means online improved)
        improvement_delta = mean_frozen - mean_online
        rel_improvement = (improvement_delta / max(1e-6, mean_frozen)) * 100.0

        variances = [m.latent_variance for m in self.learner.history]
        final_var = float(variances[-1]) if variances else 0.0
        var_passed = final_var >= 0.05

        grad_norms = [m.grad_norm for m in self.learner.history if m.grad_norm is not None]
        mean_grad = float(np.mean(grad_norms)) if grad_norms else 0.0
        grad_healthy = mean_grad > 1e-5 and math.isfinite(mean_grad)

        # Verify frozen reference model parameters have strictly ZERO drift
        drift = 0.0
        with torch.no_grad():
            for p_base, p_froz in zip(self.base_model.parameters(), self.model_frozen.parameters()):
                drift += float(torch.norm(p_base - p_froz).item())

        p50_latency = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
        p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
        throughput_fps = (1000.0 / p50_latency) if p50_latency > 0 else 0.0

        is_passed = (var_passed and grad_healthy and drift < 1e-6)
        status = "PASSED" if is_passed else "EXPERIMENTAL"

        summary = (
            f"Stage B Benchmark {status}: Online pre-update loss: {mean_online:.5f} vs "
            f"Frozen twin: {mean_frozen:.5f} (delta_pred={delta_pred:+.5f}, "
            f"improvement={improvement_delta:+.5f}, {rel_improvement:+.2f}%). "
            f"Latent variance: {final_var:.4f} (guard >=0.05: {'PASS' if var_passed else 'FAIL'}). "
            f"Gradient flow: mean norm={mean_grad:.4f} ({'HEALTHY' if grad_healthy else 'ZERO/NAN'}). "
            f"Frozen twin weight drift: {drift:.8f} (immutability guaranteed). "
            f"Step latency: P50={p50_latency:.2f}ms, P95={p95_latency:.2f}ms (~{throughput_fps:.1f} FPS)."
        )

        return StageBBenchmarkReport(
            total_ticks=len(frames),
            online_pre_update_loss_mean=mean_online,
            frozen_reference_loss_mean=mean_frozen,
            prediction_improvement_loss_delta=improvement_delta,
            relative_improvement_pct=rel_improvement,
            final_latent_variance=final_var,
            variance_guard_passed=var_passed,
            mean_gradient_norm=mean_grad,
            gradient_flow_healthy=grad_healthy,
            frozen_model_parameters_drift=drift,
            status=status,
            summary=summary,
            delta_pred=delta_pred,
            step_latency_ms_p50=p50_latency,
            step_latency_ms_p95=p95_latency,
            throughput_fps=throughput_fps,
        )
