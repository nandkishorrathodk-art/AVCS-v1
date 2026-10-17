"""Experimental self-supervised temporal visual representation model.

Motion is a learned, bounded displacement field trained with photometric
warping. It is not calibrated optical flow or evidence of desktop understanding.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class _VisualBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class TemporalVisualCortex(nn.Module):
    """Encode adjacent RGB frames into a temporal latent, motion field and surprise map.

    Inputs are ``(B, 3, H, W)`` float tensors in ``[0, 1]``. ``motion`` is a
    bounded, dimensionless sampling displacement field in normalized grid
    coordinates; it is not a pixel-per-second velocity estimate.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 96,
        latent_dim: int = 128,
        max_displacement: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or latent_dim <= 1:
            raise ValueError("feature_dim must be positive and latent_dim must exceed one.")
        if not 0.0 < max_displacement <= 1.0:
            raise ValueError("max_displacement must be in (0, 1].")
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.max_displacement = max_displacement
        self.encoder = nn.Sequential(
            _VisualBlock(3, 32, stride=2),
            _VisualBlock(32, 64, stride=2),
            _VisualBlock(64, feature_dim, stride=2),
        )
        self.latent_projection = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.motion_head = nn.Sequential(
            _VisualBlock(feature_dim * 3, feature_dim, stride=1),
            nn.Conv2d(feature_dim, 2, kernel_size=3, padding=1),
        )

    def forward(self, previous: torch.Tensor, current: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_frames(previous, current)
        batch_size = previous.shape[0]
        features = self.encoder(torch.cat((previous, current), dim=0))
        previous_features, current_features = features.split(batch_size, dim=0)

        previous_latent = self.latent_projection(
            F.adaptive_avg_pool2d(previous_features, 1).flatten(1)
        )
        latent = self.latent_projection(
            F.adaptive_avg_pool2d(current_features, 1).flatten(1)
        )
        predicted_latent = self.predictor(previous_latent)

        pair_features = torch.cat(
            (previous_features, current_features, current_features - previous_features),
            dim=1,
        )
        motion = self.motion_head(pair_features).tanh() * self.max_displacement
        previous_small, current_small = self._photometric_frames(previous, current)
        sampling_grid = self._sampling_grid(motion, previous_small.shape[-2:])
        warped_previous = F.grid_sample(
            previous_small,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        surprise = (current_small - warped_previous).abs().mean(dim=1, keepdim=True)

        return {
            "latent": latent,
            "predicted_latent": predicted_latent,
            "motion": motion,
            "surprise": surprise,
            "photometric_source": previous_small,
            "warped_previous": warped_previous,
            "photometric_target": current_small,
        }

    @staticmethod
    def _validate_frames(previous: torch.Tensor, current: torch.Tensor) -> None:
        if previous.ndim != 4 or current.ndim != 4:
            raise ValueError("Frames must have shape (batch, 3, height, width).")
        if previous.shape != current.shape or previous.shape[1] != 3:
            raise ValueError("Adjacent frames must have identical RGB batch shapes.")
        if previous.shape[-2] < 32 or previous.shape[-1] < 32:
            raise ValueError("Frame height and width must each be at least 32 pixels.")
        if not previous.is_floating_point() or not current.is_floating_point():
            raise ValueError("Frames must be floating-point RGB tensors in [0, 1].")
        if previous.device != current.device:
            raise ValueError("Adjacent frames must be on the same device.")

    @staticmethod
    def _photometric_frames(
        previous: torch.Tensor,
        current: torch.Tensor,
        max_side: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = previous.shape[-2:]
        scale = min(1.0, max_side / max(height, width))
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        if size == (height, width):
            return previous, current
        return (
            F.interpolate(previous, size=size, mode="bilinear", align_corners=False),
            F.interpolate(current, size=size, mode="bilinear", align_corners=False),
        )

    @staticmethod
    def _sampling_grid(motion: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        height, width = size
        base_y, base_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=motion.device, dtype=motion.dtype),
            torch.linspace(-1.0, 1.0, width, device=motion.device, dtype=motion.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((base_x, base_y), dim=-1).unsqueeze(0)
        base_grid = base_grid.expand(motion.shape[0], -1, -1, -1)
        resized_motion = F.interpolate(
            motion,
            size=size,
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        return base_grid + resized_motion


def temporal_ssl_loss(
    outputs: dict[str, torch.Tensor],
    *,
    change_threshold: float = 0.03,
    change_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute predictive, anti-collapse, photometric and flow-smoothness losses.

    Use batches of at least two distinct frame pairs: the variance/covariance
    terms are batch statistics and cannot diagnose collapse from one example.
    Change weighting emphasizes changed pixels while preserving nonzero weight
    for static regions. A zero change_weight retains the uniform baseline.
    """
    if not math.isfinite(change_threshold) or not 0.0 <= change_threshold <= 1.0:
        raise ValueError("change_threshold must be in [0, 1].")
    if not math.isfinite(change_weight) or change_weight < 0.0:
        raise ValueError("change_weight must be non-negative.")
    latent = outputs["latent"]
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("Self-supervised loss requires a batch of at least two pairs.")
    predicted = outputs["predicted_latent"]
    prediction = F.smooth_l1_loss(predicted, latent.detach())

    centered = latent - latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).square().mean()
    covariance = centered.T @ centered / max(1, latent.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    decorrelation = off_diagonal.square().sum() / latent.shape[1]

    photometric_error = torch.sqrt(
        (outputs["warped_previous"] - outputs["photometric_target"]).square() + 1e-6
    )
    changed_pixels = (
        (outputs["photometric_target"] - outputs["photometric_source"])
        .abs()
        .mean(dim=1, keepdim=True)
        > change_threshold
    ).to(dtype=photometric_error.dtype)
    changed_fraction = changed_pixels.mean()
    if change_weight > 0.0:
        pixel_weights = 1.0 + change_weight * changed_pixels
        per_pair_error = (photometric_error * pixel_weights).mean(dim=(1, 2, 3))
        per_pair_weight = pixel_weights.mean(dim=(1, 2, 3))
        photometric = (per_pair_error / per_pair_weight).mean()
    else:
        photometric = photometric_error.mean()
    motion = outputs["motion"]
    smooth_x = (motion[..., 1:] - motion[..., :-1]).abs().mean()
    smooth_y = (motion[..., 1:, :] - motion[..., :-1, :]).abs().mean()
    smoothness = smooth_x + smooth_y

    total = prediction + variance + 0.04 * decorrelation + photometric + 0.05 * smoothness
    return {
        "total": total,
        "prediction": prediction,
        "variance": variance,
        "decorrelation": decorrelation,
        "photometric": photometric,
        "changed_pixel_fraction": changed_fraction,
        "motion_smoothness": smoothness,
    }


def temporal_loss_for_frames(
    model: TemporalVisualCortex,
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    change_threshold: float = 0.03,
    change_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Forward a frame pair and calculate the configurable self-supervised loss."""
    outputs = model(previous, current)
    return temporal_ssl_loss(
        outputs,
        change_threshold=change_threshold,
        change_weight=change_weight,
    )
