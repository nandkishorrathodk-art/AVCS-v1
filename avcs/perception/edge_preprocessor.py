"""4-Channel Edge & Theme Preprocessor for AVCS System 1.

Converts raw RGB desktop frames into a 4-channel representation:
  - Channel 0: Luminance (ITU-R BT.601: 0.2989 R + 0.5870 G + 0.1140 B)
  - Channel 1: Sobel X (Gx) with reflection padding and normalized kernel / 8.0
  - Channel 2: Sobel Y (Gy) with reflection padding and normalized kernel / 8.0
  - Channel 3: Local Gradient Magnitude M = sqrt(Gx^2 + Gy^2 + eps)

Provides mathematical edge invariance under photometric/color inversion:
  MaxDiff(M(X), M(1 - X)) < 1e-5.

Also provides vectorized photometric and theme augmentations for training:
  - Photometric inversion (p=0.5)
  - Brightness jitter (+/- 20%)
  - Contrast jitter (0.8x - 1.2x)
  - Gamma jitter (0.8x - 1.2x)
  - ThemeAugmentation composite transform

Purely vectorized PyTorch. Zero learned weights. Zero mutable state.
Sub-1.0ms forward pass latency on CPU at 128x128 resolution.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLuminancePreprocessor(nn.Module):
    """Vectorized 4-Channel Preprocessor (Luminance, Sobel X, Sobel Y, Gradient Magnitude).

    Args:
        eps: Small epsilon added inside sqrt for numerical stability (default: 1e-6).
        normalize_to_unit: If True, shifts Gx and Gy from [-0.5, 0.5] to [0.0, 1.0] via +0.5.
            Gradient magnitude M is always computed from unshifted gradients so invariance is preserved.
    """

    def __init__(self, eps: float = 1e-6, normalize_to_unit: bool = False) -> None:
        super().__init__()
        self.eps = float(eps)
        self.normalize_to_unit = bool(normalize_to_unit)

        # Standard 3x3 Sobel kernels normalized by 8.0 (sum of positive weights = 0.5)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0

        # Stack into [2, 1, 3, 3] for a single conv2d call
        weights = torch.cat([sobel_x, sobel_y], dim=0)
        self.register_buffer("sobel_weights", weights, persistent=False)

    def forward(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Forward pass converting [B, 3, H, W] RGB into [B, 4, H, W].

        Input:
            rgb_images: Float32 tensor of shape [B, 3, H, W] or [3, H, W] in [0.0, 1.0].
        Output:
            Float32 tensor of shape [B, 4, H, W] or [4, H, W]:
              - Channel 0: Luminance in [0.0, 1.0]
              - Channel 1: Sobel X (Gx) in [-0.5, 0.5] (or [0.0, 1.0] if normalize_to_unit=True)
              - Channel 2: Sobel Y (Gy) in [-0.5, 0.5] (or [0.0, 1.0] if normalize_to_unit=True)
              - Channel 3: Gradient Magnitude M in [0.0, 1.0]
        """
        squeeze_batch = False
        if rgb_images.ndim == 3:
            rgb_images = rgb_images.unsqueeze(0)
            squeeze_batch = True
        elif rgb_images.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor [B, 3, H, W] or 3D tensor [3, H, W], got shape {rgb_images.shape}"
            )

        if rgb_images.shape[1] != 3:
            raise ValueError(
                f"Expected 3 channels (RGB) at dim 1, got {rgb_images.shape[1]} channels"
            )

        # 1. Luminance Channel (ITU-R BT.601)
        r = rgb_images[:, 0:1]
        g = rgb_images[:, 1:2]
        b = rgb_images[:, 2:3]
        lum = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # 2 & 3. Directional Sobel Gradients with Reflection Padding
        # Reflection padding mirrors boundary pixels: (pad_left, pad_right, pad_top, pad_bottom)
        padded_lum = F.pad(lum, (1, 1, 1, 1), mode="reflect")

        # Single conv2d computing Gx and Gy simultaneously: [B, 2, H, W]
        grads = F.conv2d(padded_lum, self.sobel_weights)
        gx = grads[:, 0:1]
        gy = grads[:, 1:2]

        # 4. Local Gradient Magnitude: M = sqrt(Gx^2 + Gy^2 + eps)
        # Note: Computed strictly from raw zero-centered gradients
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)

        if self.normalize_to_unit:
            gx_out = torch.clamp(gx + 0.5, 0.0, 1.0)
            gy_out = torch.clamp(gy + 0.5, 0.0, 1.0)
        else:
            gx_out = gx
            gy_out = gy

        out = torch.cat([lum, gx_out, gy_out, mag], dim=1)

        if squeeze_batch:
            out = out.squeeze(0)

        return out

    def compute_gradient_magnitude(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Compute only the gradient magnitude M [B, 1, H, W] directly."""
        four_ch = self.forward(rgb_images)
        if four_ch.ndim == 3:
            return four_ch[3:4]
        return four_ch[:, 3:4]


def compute_edge_map(
    images: torch.Tensor,
    threshold: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary edge map E(X) = (M(X) >= threshold).

    Args:
        images: RGB images tensor [B, 3, H, W] or [3, H, W] in [0, 1].
        threshold: Gradient magnitude binarization threshold (default: 0.05).
        eps: Epsilon for gradient magnitude computation.
    Returns:
        Binary float tensor of same spatial dimensions with 1 channel.
    """
    preprocessor = EdgeLuminancePreprocessor(eps=eps, normalize_to_unit=False)
    # Ensure preprocessor weights match device/dtype
    if images.device != preprocessor.sobel_weights.device:
        preprocessor = preprocessor.to(device=images.device)
    mag = preprocessor.compute_gradient_magnitude(images)
    return (mag >= threshold).to(dtype=images.dtype)


def compute_edge_iou(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute Intersection-over-Union (IoU) between two binary edge masks.

    If both masks are completely empty (e.g. solid blank screens), IoU is 1.0.
    """
    bin_a = mask_a > 0.5
    bin_b = mask_b > 0.5
    intersection = (bin_a & bin_b).sum().item()
    union = (bin_a | bin_b).sum().item()
    if union == 0:
        return 1.0
    return float(intersection / (union + eps))


# ==============================================================================
# Vectorized Theme & Photometric Augmentations
# ==============================================================================


def photometric_inversion(
    images: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    """Photometric inversion augmentation: I_inv = 1.0 - I.

    Args:
        images: Float tensor [B, C, H, W] or [C, H, W] in [0.0, 1.0].
        p: Probability of inversion (per batch item).
    Returns:
        Augmented tensor with inverted images according to probability p.
    """
    if p <= 0.0:
        return images
    if p >= 1.0:
        return 1.0 - images

    if images.ndim == 4:
        batch_size = images.shape[0]
        mask = torch.rand(batch_size, 1, 1, 1, device=images.device) < p
        return torch.where(mask, 1.0 - images, images)
    elif images.ndim == 3:
        if torch.rand(1).item() < p:
            return 1.0 - images
        return images
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {images.shape}")


def brightness_jitter(
    images: torch.Tensor,
    factor_range: Tuple[float, float] = (-0.2, 0.2),
) -> torch.Tensor:
    """Brightness jitter augmentation: I' = clamp(I + delta, 0, 1).

    Args:
        images: Float tensor [B, C, H, W] or [C, H, W] in [0.0, 1.0].
        factor_range: (min_delta, max_delta) additive shift range.
    Returns:
        Clamped augmented tensor in [0.0, 1.0].
    """
    low, high = factor_range
    if low == 0.0 and high == 0.0:
        return images

    if images.ndim == 4:
        b = images.shape[0]
        delta = (low + (high - low) * torch.rand(b, 1, 1, 1, device=images.device, dtype=images.dtype))
        return torch.clamp(images + delta, 0.0, 1.0)
    elif images.ndim == 3:
        delta = low + (high - low) * torch.rand(1, 1, 1, device=images.device, dtype=images.dtype)
        return torch.clamp(images + delta, 0.0, 1.0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {images.shape}")


def contrast_jitter(
    images: torch.Tensor,
    factor_range: Tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """Contrast jitter augmentation: I' = clamp(mean + alpha * (I - mean), 0, 1).

    Args:
        images: Float tensor [B, C, H, W] or [C, H, W] in [0.0, 1.0].
        factor_range: (min_factor, max_factor) multiplicative scale around image mean.
    Returns:
        Clamped augmented tensor in [0.0, 1.0].
    """
    low, high = factor_range
    if low == 1.0 and high == 1.0:
        return images

    if images.ndim == 4:
        b = images.shape[0]
        # Image spatial mean per item
        mean = images.mean(dim=(-3, -2, -1), keepdim=True)
        alpha = low + (high - low) * torch.rand(b, 1, 1, 1, device=images.device, dtype=images.dtype)
        return torch.clamp((images - mean) * alpha + mean, 0.0, 1.0)
    elif images.ndim == 3:
        mean = images.mean(dim=(-3, -2, -1), keepdim=True)
        alpha = low + (high - low) * torch.rand(1, 1, 1, device=images.device, dtype=images.dtype)
        return torch.clamp((images - mean) * alpha + mean, 0.0, 1.0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {images.shape}")


def gamma_jitter(
    images: torch.Tensor,
    gamma_range: Tuple[float, float] = (0.8, 1.2),
    eps: float = 1e-7,
) -> torch.Tensor:
    """Gamma jitter augmentation: I' = I^gamma.

    Args:
        images: Float tensor [B, C, H, W] or [C, H, W] in [0.0, 1.0].
        gamma_range: (min_gamma, max_gamma) power-law exponent range.
        eps: Small epsilon to prevent zero underflow during exponentiation.
    Returns:
        Clamped augmented tensor in [0.0, 1.0].
    """
    low, high = gamma_range
    if low == 1.0 and high == 1.0:
        return images

    if images.ndim == 4:
        b = images.shape[0]
        gamma = low + (high - low) * torch.rand(b, 1, 1, 1, device=images.device, dtype=images.dtype)
        clamped = torch.clamp(images, min=eps, max=1.0)
        return torch.clamp(clamped ** gamma, 0.0, 1.0)
    elif images.ndim == 3:
        gamma = low + (high - low) * torch.rand(1, 1, 1, device=images.device, dtype=images.dtype)
        clamped = torch.clamp(images, min=eps, max=1.0)
        return torch.clamp(clamped ** gamma, 0.0, 1.0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {images.shape}")


class ThemeAugmentation(nn.Module):
    """Composite theme and photometric augmentation pipeline for UI images.

    Applies:
      1. Photometric inversion (with probability p_invert)
      2. Brightness jitter (within brightness_range)
      3. Contrast jitter (within contrast_range)
      4. Gamma jitter (within gamma_range)
      (Steps 2-4 gated by probability p_jitter per item)
    """

    def __init__(
        self,
        p_invert: float = 0.5,
        p_jitter: float = 0.5,
        brightness_range: Tuple[float, float] = (-0.2, 0.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        gamma_range: Tuple[float, float] = (0.8, 1.2),
    ) -> None:
        super().__init__()
        self.p_invert = float(p_invert)
        self.p_jitter = float(p_jitter)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply composite theme augmentation to input images [B, C, H, W] or [C, H, W]."""
        # 1. Photometric Inversion
        out = photometric_inversion(images, p=self.p_invert)

        # 2. Photometric Jitter (brightness, contrast, gamma)
        if self.p_jitter > 0.0:
            if out.ndim == 4:
                b = out.shape[0]
                jitter_mask = torch.rand(b, 1, 1, 1, device=out.device) < self.p_jitter
                jittered = brightness_jitter(out, factor_range=self.brightness_range)
                jittered = contrast_jitter(jittered, factor_range=self.contrast_range)
                jittered = gamma_jitter(jittered, gamma_range=self.gamma_range)
                out = torch.where(jitter_mask, jittered, out)
            elif out.ndim == 3:
                if torch.rand(1).item() < self.p_jitter:
                    out = brightness_jitter(out, factor_range=self.brightness_range)
                    out = contrast_jitter(out, factor_range=self.contrast_range)
                    out = gamma_jitter(out, gamma_range=self.gamma_range)

        return out
