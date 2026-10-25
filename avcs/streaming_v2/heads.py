"""Specialized Output Heads for StreamingVisualCortexV2.

Provides three specialized neural heads decoupling perception from closed text vocabularies:
  - Head 1 (TemporalChangeSpinnerHead): Spatial change heatmap (128x128) and cyclical spinner activity (~55.6k params).
  - Head 2 (GenericUIRegionDetectorHead): Anchor-free 5-class UI morphology detector (16x16 grid) (~28.2k params).
  - Head 3 (PostActionVisualResponseHead): Pre/post action latent comparison for visual feedback (~18.6k params).

Architectural Invariants & Encapsulation:
  1. Pure-pixel: Zero OCR, zero word recognizers, zero closed text vocabularies.
  2. Bounded parameters: All heads strictly stay within lightweight CPU budgets.
  3. System 2 Privacy: Heads produce internal feature maps; raw bounding boxes or coordinates
     must not leak across the System 2 reasoning boundary.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TemporalChangeSpinnerHead(nn.Module):
    """Head 1: Spatial Temporal Change Heatmap & Cyclical Spinner Activity Head.

    Combines temporal hidden representation h_t with optical feature delta |f_t - f_{t-1}|
    to predict:
      1. Spatial change heatmap (B, 1, 128, 128) in [0.0, 1.0] indicating visual change locations.
      2. Spinner activity scalar logit (B, 1) indicating cyclical loading animations.

    Parameter Target: ~55.6k parameters.
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 48,
        heatmap_resolution: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.heatmap_resolution = heatmap_resolution

        # Shared temporal-difference representation: 128 -> 48 (Conv 3x3)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.SiLU(inplace=True),
        )

        # Spatial change heatmap projection: 48 -> 1 (Conv 1x1)
        self.heatmap_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

        # Cyclical spinner / loading activity head (Global Pool + MLP): 48 -> 16 -> 1
        self.spinner_mlp = nn.Sequential(
            nn.Linear(mid_channels, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with Kaiming Normal for convs and Xavier Uniform for linear."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.GroupNorm):
                    if m.weight is not None:
                        m.weight.fill_(1.0)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(
        self,
        hidden: torch.Tensor,
        feature_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting spatial change heatmap and spinner logit.

        Args:
            hidden: Temporal hidden state tensor (B, 64, 16, 16) or concatenated (B, 128, 16, 16).
            feature_delta: Optional feature difference tensor (B, 64, 16, 16). If None, hidden
                must have shape (B, 128, 16, 16) or feature_delta is treated as zeros.

        Returns:
            Tuple of:
              - change_heatmap: (B, 1, 128, 128) float tensor in [0.0, 1.0].
              - spinner_activity_logit: (B, 1) float tensor of raw logits.
        """
        if feature_delta is not None:
            x = torch.cat([hidden, feature_delta], dim=1)
        elif hidden.shape[1] == self.in_channels:
            x = hidden
        else:
            # Fallback for single tensor input: pad zeros for delta
            zeros_delta = torch.zeros_like(hidden)
            x = torch.cat([hidden, zeros_delta], dim=1)

        feat = self.shared_conv(x)  # (B, 48, 16, 16)

        # 1. Spatial change heatmap upsampled to 128x128 with sigmoid activation
        heat_logits = self.heatmap_conv(feat)  # (B, 1, 16, 16)
        change_heatmap = F.interpolate(
            heat_logits,
            size=self.heatmap_resolution,
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        # 2. Spinner / cyclical motion classification logit
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # (B, 48)
        spinner_activity_logit = self.spinner_mlp(pooled)  # (B, 1)

        return change_heatmap, spinner_activity_logit


class GenericUIRegionDetectorHead(nn.Module):
    """Head 2: Generic 5-Class UI-Region Detector.

    Anchor-free grid-based predictor running on 16x16 feature representations:
      - Objectness logits: (B, 1, 16, 16)
      - 5-class logits: (B, 5, 16, 16)
          Class 0: Interactive Controls (clickable elements, action buttons, link controls)
          Class 1: Text Input Fields (text entry boxes, search fields, input lines)
          Class 2: Navigation Toggles (switches, icons, tabs, checkboxes, radio toggles)
          Class 3: Modal Dialogs (dialog windows, popups, alert cards, overlay modals)
          Class 4: Content Regions (structured text areas, code blocks, document paragraphs)
      - Bounding box regression offsets: (B, 4, 16, 16) (dx, dy, dw, dh)

    Universal application-agnostic visual UI shape segmenter operating strictly
    on geometric morphology across arbitrary desktop OS windows. Strictly decoupled
    from application-specific heuristics, OCR, or text vocabularies.
    Parameter Target: ~28.2k parameters.
    """

    NUM_CLASSES: int = 5

    # Universal Geometric Morphology Classes
    CLASS_INTERACTIVE_CONTROL: int = 0
    CLASS_TEXT_INPUT_FIELD: int = 1
    CLASS_NAVIGATION_TOGGLE: int = 2
    CLASS_MODAL_DIALOG: int = 3
    CLASS_CONTENT_REGION: int = 4

    # Backward-compatible aliases
    CLASS_BUTTON: int = 0
    CLASS_INPUT_BOX: int = 1
    CLASS_ICON_TOGGLE: int = 2
    CLASS_DIALOG_POPUP: int = 3
    CLASS_TEXT_AREA: int = 4

    GEOMETRIC_CLASS_NAMES: tuple[str, ...] = (
        "Interactive_Control",
        "Text_Input_Field",
        "Navigation_Toggle",
        "Modal_Dialog",
        "Content_Region",
    )

    def __init__(
        self,
        in_channels: int = 64,
        mid_channels: int = 48,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = num_classes

        # Shared feature processing: 64 -> 48 (Conv 3x3)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, mid_channels),
            nn.SiLU(inplace=True),
        )

        # 1. Objectness logit per grid cell: 48 -> 1 (Conv 1x1)
        self.conv_objectness = nn.Conv2d(mid_channels, 1, kernel_size=1)

        # 2. 5-class generic classification logits: 48 -> 5 (Conv 1x1)
        self.conv_classes = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

        # 3. Bounding box regression offsets (dx, dy, dw, dh): 48 -> 4 (Conv 1x1)
        self.conv_boxes = nn.Conv2d(mid_channels, 4, kernel_size=1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with Kaiming Normal."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.GroupNorm):
                    if m.weight is not None:
                        m.weight.fill_(1.0)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass predicting objectness, 5-class classification, and box offsets.

        Args:
            features: Feature map tensor of shape (B, 64, 16, 16).

        Returns:
            Tuple of:
              - ui_objectness_logits: (B, 1, 16, 16) float tensor.
              - ui_class_logits: (B, 5, 16, 16) float tensor for 5 generic UI classes.
              - ui_box_regression: (B, 4, 16, 16) float tensor of (dx, dy, dw, dh).
        """
        feat = self.shared_conv(features)  # (B, 48, 16, 16)

        ui_objectness_logits = self.conv_objectness(feat)  # (B, 1, 16, 16)
        ui_class_logits = self.conv_classes(feat)          # (B, 5, 16, 16)
        ui_box_regression = self.conv_boxes(feat)          # (B, 4, 16, 16)

        return ui_objectness_logits, ui_class_logits, ui_box_regression


class PostActionVisualResponseHead(nn.Module):
    """Head 3: Post-Action Visual Response Head.

    Evaluates whether an executed action produced a visible optical response on screen.
    Ingests pre-action latent z_pre (96d), post-action latent z_post (96d), and
    their absolute difference |z_post - z_pre| (96d), concatenated into a 288-dimensional
    vector, processed by a 2-layer MLP.

    Output:
      - Scalar response logit (B, 1) indicating visible visual reaction.

    Parameter Target: ~18.6k parameters.
    """

    def __init__(
        self,
        latent_dim: int = 96,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.in_dim = latent_dim * 3  # [z_pre, z_post, |z_post - z_pre|] = 288

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear layers with Xavier Uniform."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
    ) -> torch.Tensor:
        """Compute post-action visual response logit from pre and post action latents.

        Args:
            z_pre: Pre-action latent vector (B, 96).
            z_post: Post-action latent vector (B, 96).

        Returns:
            Scalar response logit (B, 1).
        """
        if z_pre.ndim != 2 or z_post.ndim != 2:
            raise ValueError(f"Expected 2D tensors (B, {self.latent_dim}), got {z_pre.shape} and {z_post.shape}")

        delta = torch.abs(z_post - z_pre)
        combined = torch.cat([z_pre, z_post, delta], dim=-1)  # (B, 288)
        return self.mlp(combined)  # (B, 1)
