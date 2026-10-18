"""Anti-collapse variance regularization and gradient flow auditing for live streaming cortex."""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GradientAuditReport:
    """Detailed audit of gradient flow across functional cortex modules."""
    total_norm: float
    module_norms: dict[str, float]
    is_healthy: bool
    zero_grad_modules: tuple[str, ...]
    nan_or_inf_modules: tuple[str, ...]
    summary: str


class GradientAuditor:
    """Verifies non-zero, finite gradient flow across learnable modules.

    Audits:
    - encoder: scale_half, scale_quarter, scale_eighth, lateral_half, lateral_quarter, lateral_eighth, fuse
    - convgru: recurrent
    - latent_predictor: latent_projection, future_predictor
    """

    CORE_MODULES: tuple[str, ...] = ("encoder", "convgru", "latent_predictor")

    def __init__(
        self,
        *,
        min_grad_norm: float = 1e-6,
        required_modules: Sequence[str] | None = None,
    ) -> None:
        self.min_grad_norm = float(min_grad_norm)
        self.required_modules = tuple(required_modules or self.CORE_MODULES)

    def classify_parameter(self, param_name: str) -> str:
        """Classify a named parameter into its functional module group."""
        if any(prefix in param_name for prefix in ("scale_", "lateral_", "fuse")):
            return "encoder"
        if "recurrent" in param_name:
            return "convgru"
        if any(prefix in param_name for prefix in ("latent_projection", "future_predictor")):
            return "latent_predictor"
        if "change_head" in param_name:
            return "change_head"
        if "motion_head" in param_name:
            return "motion_head"
        if "state_decay_rate" in param_name:
            return "convgru"
        return param_name.split(".")[0]

    def audit(self, model: nn.Module) -> GradientAuditReport:
        """Audit all learnable parameters and compute per-module gradient norms."""
        module_sq_norms: dict[str, float] = {mod: 0.0 for mod in self.required_modules}
        param_counts: dict[str, int] = {mod: 0 for mod in self.required_modules}
        total_sq_norm = 0.0
        nan_or_inf_mods: list[str] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            mod_group = self.classify_parameter(name)
            if mod_group not in module_sq_norms:
                module_sq_norms[mod_group] = 0.0
                param_counts[mod_group] = 0

            param_counts[mod_group] += 1

            if param.grad is None:
                continue

            grad = param.grad
            if not torch.isfinite(grad).all():
                if mod_group not in nan_or_inf_mods:
                    nan_or_inf_mods.append(mod_group)
                continue

            norm_val = float(grad.norm(2).item())
            sq = norm_val * norm_val
            module_sq_norms[mod_group] += sq
            total_sq_norm += sq

        total_norm = math.sqrt(total_sq_norm)
        module_norms = {mod: math.sqrt(sq) for mod, sq in module_sq_norms.items()}

        zero_grad_mods: list[str] = []
        for mod in self.required_modules:
            if module_norms.get(mod, 0.0) <= self.min_grad_norm:
                zero_grad_mods.append(mod)

        is_healthy = (len(zero_grad_mods) == 0) and (len(nan_or_inf_mods) == 0) and (total_norm > self.min_grad_norm)

        summary_parts = [f"{mod}={module_norms.get(mod, 0.0):.6f}" for mod in self.required_modules]
        summary_details = ", ".join(summary_parts)
        if is_healthy:
            summary = f"Gradient flow healthy: total_norm={total_norm:.6f} ({summary_details})"
        else:
            issues = []
            if zero_grad_mods:
                issues.append(f"zero_grad in {zero_grad_mods}")
            if nan_or_inf_mods:
                issues.append(f"non_finite in {nan_or_inf_mods}")
            summary = f"Gradient flow failed: total_norm={total_norm:.6f} ({', '.join(issues)}; {summary_details})"

        return GradientAuditReport(
            total_norm=total_norm,
            module_norms=module_norms,
            is_healthy=is_healthy,
            zero_grad_modules=tuple(zero_grad_mods),
            nan_or_inf_modules=tuple(nan_or_inf_mods),
            summary=summary,
        )

    def assert_healthy(self, model: nn.Module) -> GradientAuditReport:
        """Audit gradients and raise RuntimeError if any required module has zero or non-finite flow."""
        report = self.audit(model)
        if not report.is_healthy:
            raise RuntimeError(f"Gradient audit failed: {report.summary}")
        return report


def audit_model_gradients(model: nn.Module, min_grad_norm: float = 1e-6) -> GradientAuditReport:
    """Convenience functional helper to audit model gradients."""
    return GradientAuditor(min_grad_norm=min_grad_norm).audit(model)


class StreamingVICRegLoss(nn.Module):
    """Sliding-window variance regularizer and predictive coding loss for streaming perception.

    Enforces Var(z_j) > 0.05 across all latent dimensions with margin gamma = 0.25 (or configurable target_std).
    For live stream single-frame feeds (B=1), maintains a volatile sliding queue of detached past latents in RAM,
    concatenating with the active current latent to compute variance and covariance without leaking computation graphs.

    Loss Formulation:
      L = lambda_pred * L_pred + lambda_var * L_var + lambda_cov * L_cov
    where:
      L_pred = SmoothL1(predicted_latent, target_latent)
      L_var  = mean_j(max(0, margin_gamma - sqrt(Var(z_j) + eps)))
      L_cov  = (1 / D) * sum_{i != j} C_ij^2
    """

    def __init__(
        self,
        *,
        lambda_pred: float = 1.0,
        lambda_var: float = 0.5,
        lambda_cov: float = 0.1,
        margin_gamma: float = 0.25,
        target_std: float | None = None,
        min_variance_threshold: float = 0.05,
        window_size: int = 16,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lambda_pred = float(lambda_pred)
        self.lambda_var = float(lambda_var)
        self.lambda_cov = float(lambda_cov)
        # Support both margin_gamma (dispatch default 0.25) and target_std (legacy 0.50)
        self.margin_gamma = float(target_std if target_std is not None else margin_gamma)
        self.target_std = self.margin_gamma
        self.min_variance_threshold = float(min_variance_threshold)
        self.window_size = int(max(2, window_size))
        self.eps = float(eps)

        self._queue: deque[torch.Tensor] = deque(maxlen=self.window_size - 1)

    def reset(self) -> None:
        """Clear internal volatile sliding window buffer."""
        self._queue.clear()

    def compute_variance(self, z: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Compute variance per dimension and mean variance scalar."""
        z_2d = z.view(z.shape[0], -1)
        if z_2d.shape[0] > 1:
            var_d = torch.var(z_2d, dim=0, unbiased=True)
            return var_d, float(torch.mean(var_d).item())

        var_channels = torch.var(z_2d, dim=-1, unbiased=True)
        return var_channels, float(torch.mean(var_channels).item())

    def forward(
        self,
        predicted_latent: torch.Tensor,
        target_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute predictive coding loss, sliding-window variance loss, and covariance loss.

        Args:
            predicted_latent: Predicted future latent tensor (batch, latent_dim).
            target_latent: Observed actual latent tensor (batch, latent_dim).

        Returns:
            Tuple of (total_loss_tensor, metrics_dict).
        """
        # 1. Predictive Coding Loss (Smooth L1)
        pred_loss = F.smooth_l1_loss(predicted_latent, target_latent)

        # 2. Sliding-Window Latent Representations
        z_2d = target_latent.view(target_latent.shape[0], -1)
        if z_2d.shape[0] > 1:
            z_window = z_2d
        else:
            if len(self._queue) > 0:
                history = list(self._queue)
                # Keep active z_2d at end with autograd graph attached; history is detached
                z_window = torch.cat(history + [z_2d], dim=0)
            else:
                z_window = z_2d

        num_samples = z_window.shape[0]
        dim = z_window.shape[1]

        # 3. Variance Regularization Loss
        if num_samples > 1:
            var_per_dim = torch.var(z_window, dim=0, unbiased=True)
            std_per_dim = torch.sqrt(var_per_dim + self.eps)
            var_mean = float(torch.mean(var_per_dim).item())
            min_var = float(torch.min(var_per_dim).item())
            var_loss = torch.mean(F.relu(self.margin_gamma - std_per_dim))

            # 4. Covariance Decorrelation Loss
            centered = z_window - torch.mean(z_window, dim=0, keepdim=True)
            cov_matrix = (centered.T @ centered) / (num_samples - 1)
            off_diag = cov_matrix - torch.diag(torch.diag(cov_matrix))
            cov_loss = (off_diag.pow(2)).sum() / dim
        else:
            # Single sample without history: channel-wise dispersion
            var_channels = torch.var(z_2d, dim=-1, unbiased=True)
            std_channels = torch.sqrt(var_channels + self.eps)
            var_mean = float(torch.mean(var_channels).item())
            min_var = var_mean
            var_loss = torch.mean(F.relu(self.margin_gamma - std_channels))
            cov_loss = torch.tensor(0.0, device=target_latent.device)

        total_loss = (
            self.lambda_pred * pred_loss
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
        )

        # Enqueue detached latent for future temporal window steps (RAM-only, zero leak)
        self._queue.append(z_2d.detach())

        metrics = {
            "pred_loss": float(pred_loss.item()),
            "var_mean": var_mean,
            "min_var": min_var,
            "var_loss": float(var_loss.item() if isinstance(var_loss, torch.Tensor) else var_loss),
            "cov_loss": float(cov_loss.item() if isinstance(cov_loss, torch.Tensor) else cov_loss),
            "total_loss": float(total_loss.item()),
            "variance_healthy": bool(var_mean >= self.min_variance_threshold),
        }
        return total_loss, metrics
