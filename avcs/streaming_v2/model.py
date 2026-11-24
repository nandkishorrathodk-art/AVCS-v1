"""StreamingVisualCortexV2: Stateful Multi-Scale ConvGRU Visual Model for Timestamped UI Streams.

Milestone M3 Implementation:
  - 4-Channel Multi-Scale Spatial Pyramid Stem (~51.3k params)
  - Recurrent ConvGRU Temporal Cell with Memory Retention Bias (~221.4k params)
  - Temporal Gap Detection & Fail-Closed State Reset (delta_t >= 2.0s / dropped_frame)
  - Smooth Exponential Decay (lambda = 0.5) for Sub-2.0s Intervals
  - Optical Feature Delta Pathway (|f_t - f_{t-1}|)
  - Latent Projection & Next-Latent Future Predictor (~25.1k params)
  - Three Specialized Output Heads (~103.0k params):
      * Head 1: Temporal Change & Spinner Activity
      * Head 2: Generic 5-Class UI-Region Detector
      * Head 3: Post-Action Visual Response
  - Strict Parameter Budget Target: 400,000 <= total_params <= 500,000 (Calculated: ~400,765)
  - Pure-Pixel Boundary: Zero OCR, zero word tokens, zero DOM/UIA dependencies.
  - System 2 Encapsulation: Strict prohibition of raw coordinate leakage across System 2 boundary.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from avcs.streaming_v2.heads import (
    GenericUIRegionDetectorHead,
    PostActionVisualResponseHead,
    TemporalChangeSpinnerHead,
)

# ==============================================================================
# Encapsulated Temporal State & Output Data Contracts
# ==============================================================================


@dataclass
class CortexTemporalState:
    """Encapsulated hidden recurrent state and feature history for StreamingVisualCortexV2.

    Maintains internal hidden representations across sequential observations.
    Never exposed directly to System 2 reasoning boundaries.
    """

    hidden: torch.Tensor                     # Recurrent hidden state (B, 64, 16, 16)
    previous_features: torch.Tensor          # Previous spatial feature f_{t-1} (B, 64, 16, 16)
    timestamp: float = 0.0                   # Monotonic observation timestamp in seconds
    step_count: int = 0                      # Incremental sequence step counter
    latent: torch.Tensor | None = None    # Projected latent z_t (B, 96)
    predicted_next_latent: torch.Tensor | None = None  # Predicted next latent \hat{z}_{t+1} (B, 96)

    def detach(self) -> CortexTemporalState:
        """Return a detached copy of the state for truncated BPTT."""
        return CortexTemporalState(
            hidden=self.hidden.detach(),
            previous_features=self.previous_features.detach(),
            timestamp=self.timestamp,
            step_count=self.step_count,
            latent=self.latent.detach() if self.latent is not None else None,
            predicted_next_latent=(
                self.predicted_next_latent.detach()
                if self.predicted_next_latent is not None
                else None
            ),
        )


@dataclass
class System2SymbolicSummary:
    """Symbolic perception summary for System 2 reasoning.

    Adheres strictly to the System 2 Privacy Boundary Contract:
    PROHIBITS transmission of raw numeric coordinates (x, y, w, h, bbox tuples, floats).
    Provides categorical presence, status flags, and scalar confidence metrics across
    universal geometric morphology classes (Interactive Controls, Text Input Fields,
    Navigation Toggles, Modal Dialogs, Content Regions).
    """

    has_button: bool
    has_input_box: bool
    has_icon_toggle: bool
    has_dialog_popup: bool
    has_text_area: bool
    visual_change_detected: bool
    spinner_active: bool
    post_action_response_detected: bool
    uncertainty_level: str               # "LOW", "MEDIUM", "HIGH"
    primary_ui_class: str | None      # e.g., "button", "input_box", etc.
    change_coverage_pct: float           # Scalar percentage [0, 100] of changed area

    # Universal Geometric Morphology Accessors
    @property
    def has_interactive_control(self) -> bool:
        """Universal alias for interactive controls / buttons."""
        return self.has_button

    @property
    def has_text_input_field(self) -> bool:
        """Universal alias for text input fields / entry boxes."""
        return self.has_input_box

    @property
    def has_navigation_toggle(self) -> bool:
        """Universal alias for navigation toggles / icons / switches."""
        return self.has_icon_toggle

    @property
    def has_modal_dialog(self) -> bool:
        """Universal alias for modal dialogs / popup containers."""
        return self.has_dialog_popup

    @property
    def has_content_region(self) -> bool:
        """Universal alias for content regions / structured text areas."""
        return self.has_text_area

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary and verify zero coordinate or array leakage."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                raise TypeError(
                    f"System 2 privacy violation: raw coordinate/array {k}={v} leaked into System 2 summary"
                )
        return d


@dataclass
class CortexV2Output:
    """Exact output data contract for StreamingVisualCortexV2 matching PROJECT.md Section 5.2.

    Contains temporal representations, latent projections, and outputs from all 3 heads.
    """

    # Temporal Features & Latents
    latent: torch.Tensor                     # (B, 96)
    predicted_next_latent: torch.Tensor      # (B, 96)
    feature_uncertainty: torch.Tensor        # (B,) normalized MSE error signal
    temporal_state: CortexTemporalState      # Encapsulated hidden state (64, 16, 16)

    # Head 1: Temporal Change & Spinner
    change_heatmap: torch.Tensor             # (B, 1, 128, 128) in [0, 1]
    spinner_activity_logit: torch.Tensor     # (B, 1) float

    # Head 2: Generic 5-Class UI Detector
    ui_objectness_logits: torch.Tensor       # (B, 1, 16, 16)
    ui_class_logits: torch.Tensor            # (B, 5, 16, 16) for 5 generic classes
    ui_box_regression: torch.Tensor          # (B, 4, 16, 16) dx, dy, dw, dh

    # Head 3: Post-Action Visual Response
    post_action_response_logit: torch.Tensor # (B, 1) float

    def to_symbolic_summary(
        self,
        objectness_threshold: float = 0.5,
        change_threshold: float = 0.1,
        spinner_threshold: float = 0.5,
        response_threshold: float = 0.5,
        batch_idx: int = 0,
    ) -> System2SymbolicSummary:
        """Distill tensor predictions into a symbolic summary adhering to System 2 encapsulation.

        Strictly avoids leaking any bounding box floats, offsets, or coordinate arrays.
        """
        # 1. UI class presence from grid predictions
        obj_probs = torch.sigmoid(self.ui_objectness_logits[batch_idx, 0])  # (16, 16)
        active_cells = obj_probs >= objectness_threshold

        has_classes = [False] * 5
        primary_cls_name = None
        if active_cells.any():
            cls_probs = torch.softmax(self.ui_class_logits[batch_idx], dim=0)  # (5, 16, 16)
            for c in range(5):
                if (active_cells & (cls_probs[c] >= 0.3)).any():
                    has_classes[c] = True

            # Determine dominant class
            active_cls_scores = cls_probs[:, active_cells].mean(dim=-1)  # (5,)
            best_cls = int(torch.argmax(active_cls_scores).item())
            class_names = ("button", "input_box", "icon_toggle", "dialog_popup", "text_area")
            primary_cls_name = class_names[best_cls]

        # 2. Change & Spinner status
        heat = self.change_heatmap[batch_idx, 0]  # (128, 128)
        change_mask = heat >= change_threshold
        pct_changed = float(change_mask.float().mean().item()) * 100.0
        change_detected = pct_changed > 0.5

        spinner_prob = float(torch.sigmoid(self.spinner_activity_logit[batch_idx, 0]).item())
        spinner_active = spinner_prob >= spinner_threshold

        # 3. Post-action response
        resp_prob = float(torch.sigmoid(self.post_action_response_logit[batch_idx, 0]).item())
        resp_detected = resp_prob >= response_threshold

        # 4. Uncertainty level
        uncert = float(self.feature_uncertainty[batch_idx].item())
        if uncert < 0.10:
            uncert_level = "LOW"
        elif uncert < 0.35:
            uncert_level = "MEDIUM"
        else:
            uncert_level = "HIGH"

        return System2SymbolicSummary(
            has_button=has_classes[0],
            has_input_box=has_classes[1],
            has_icon_toggle=has_classes[2],
            has_dialog_popup=has_classes[3],
            has_text_area=has_classes[4],
            visual_change_detected=change_detected,
            spinner_active=spinner_active,
            post_action_response_detected=resp_detected,
            uncertainty_level=uncert_level,
            primary_ui_class=primary_cls_name,
            change_coverage_pct=round(pct_changed, 2),
        )


# ==============================================================================
# Multi-Scale 4-Channel Spatial Pyramid Stem
# ==============================================================================


class SpatialPyramidStem(nn.Module):
    """Multi-Scale 4-Channel Spatial Pyramid Stem.

    Accepts 4-channel tensor (B, 4, 128, 128) from EdgeLuminancePreprocessor (I_lum, Gx, Gy, M)
    and downsamples through a multi-scale hierarchy:
      - scale_half:    Conv 4 -> 24 (stride 2, 64x64)
      - scale_quarter: Conv 24 -> 32 (stride 2, 32x32)
      - scale_eighth:  Conv 32 -> 48 (stride 2, 16x16)
    Laterals:
      - lateral_half:    Conv 24 -> 16 (1x1), pooled to 16x16
      - lateral_quarter: Conv 32 -> 16 (1x1), pooled to 16x16
      - lateral_eighth:  Conv 48 -> 16 (1x1)
    Fusion:
      - Concatenate laterals (16 + 16 + 16 = 48 channels) -> Conv 48 -> 64 (3x3)
    Output:
      - Spatial feature map f_t in R^(B x 64 x 16 x 16).

    Parameter Target: ~51.3k parameters (Calculated: 51,296).
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Scale hierarchy
        self.scale_half = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(inplace=True),
        )
        self.scale_quarter = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.SiLU(inplace=True),
        )
        self.scale_eighth = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 48),
            nn.SiLU(inplace=True),
        )

        # Lateral 1x1 projections (16 channels each)
        self.lateral_half = nn.Conv2d(24, 16, kernel_size=1)
        self.lateral_quarter = nn.Conv2d(32, 16, kernel_size=1)
        self.lateral_eighth = nn.Conv2d(48, 16, kernel_size=1)

        # Multi-scale feature fusion: 48 -> 64
        self.fuse = nn.Sequential(
            nn.Conv2d(48, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolution weights with Kaiming Normal."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale fused spatial features.

        Args:
            x: 4-channel tensor of shape (B, 4, 128, 128).

        Returns:
            Fused spatial feature map of shape (B, 64, 16, 16).
        """
        p1 = self.scale_half(x)         # (B, 24, 64, 64)
        p2 = self.scale_quarter(p1)     # (B, 32, 32, 32)
        p3 = self.scale_eighth(p2)      # (B, 48, 16, 16)

        target_size = p3.shape[-2:]     # (16, 16)

        l1 = self.lateral_half(F.adaptive_avg_pool2d(p1, target_size))     # (B, 16, 16, 16)
        l2 = self.lateral_quarter(F.adaptive_avg_pool2d(p2, target_size))  # (B, 16, 16, 16)
        l3 = self.lateral_eighth(p3)                                       # (B, 16, 16, 16)

        pyramid = torch.cat([l1, l2, l3], dim=1)  # (B, 48, 16, 16)
        return self.fuse(pyramid)                 # (B, 64, 16, 16)


# ==============================================================================
# Temporal Recurrent Cell (ConvGRU)
# ==============================================================================


class ConvGRUCell(nn.Module):
    """Convolutional Gated Recurrent Unit (ConvGRU) Cell.

    Maintains a 64-channel spatial hidden state h_t in R^(B x 64 x 16 x 16).
    Recurrent Equations:
      - Reset gate:    r_t = sigmoid(W_xr * x_t + W_hr * h_{t-1} + b_r)
      - Update gate:   z_t = sigmoid(W_xz * x_t + W_hz * h_{t-1} + b_z), b_z initialized to +1.0
      - Candidate:     h_tilde = tanh(W_xh * x_t + W_hh * (r_t * h_{t-1}) + b_h)
      - Output state:  h_t = z_t * h_{t-1} + (1 - z_t) * h_tilde

    Parameter Target: ~221.4k parameters (Calculated: 221,376).
    """

    def __init__(self, in_channels: int = 64, hidden_channels: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        joined = in_channels + hidden_channels  # 128

        # Gates: predicts reset gate r_t and update gate z_t (2 * hidden_channels = 128)
        self.gates = nn.Conv2d(joined, hidden_channels * 2, kernel_size=3, padding=1)

        # Candidate state: predicts h_tilde (hidden_channels = 64)
        self.candidate = nn.Conv2d(joined, hidden_channels, kernel_size=3, padding=1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters: Kaiming for input, Orthogonal for recurrent, +1.0 for update bias."""
        with torch.no_grad():
            # Gates convolution: [:in_channels] input-to-hidden, [in_channels:] recurrent
            nn.init.kaiming_normal_(self.gates.weight[:, :self.in_channels], nonlinearity="sigmoid")
            for i in range(2):
                chunk = self.gates.weight[
                    i * self.hidden_channels : (i + 1) * self.hidden_channels,
                    self.in_channels :,
                ]
                nn.init.orthogonal_(chunk)

            if self.gates.bias is not None:
                self.gates.bias.zero_()
                # Update gate bias set to +1.0 (retaining memory bias)
                self.gates.bias[self.hidden_channels :].fill_(1.0)

            # Candidate convolution
            nn.init.kaiming_normal_(self.candidate.weight[:, :self.in_channels], nonlinearity="tanh")
            nn.init.orthogonal_(self.candidate.weight[:, self.in_channels :])
            if self.candidate.bias is not None:
                self.candidate.bias.zero_()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Compute one recurrent state transition.

        Args:
            x_t: Input spatial features (B, 64, 16, 16).
            h_prev: Previous recurrent hidden state (B, 64, 16, 16).

        Returns:
            Updated recurrent hidden state h_t (B, 64, 16, 16).
        """
        combined = torch.cat([x_t, h_prev], dim=1)  # (B, 128, 16, 16)
        gate_outputs = self.gates(combined)         # (B, 128, 16, 16)

        r_logits, z_logits = gate_outputs.chunk(2, dim=1)  # (B, 64, 16, 16) each
        r_t = torch.sigmoid(r_logits)                      # Reset gate
        z_t = torch.sigmoid(z_logits)                      # Update gate (with +1.0 bias)

        candidate_input = torch.cat([x_t, r_t * h_prev], dim=1)  # (B, 128, 16, 16)
        h_tilde = torch.tanh(self.candidate(candidate_input))    # Candidate state

        # Final state update: h_t = z_t * h_prev + (1.0 - z_t) * h_tilde
        h_t = z_t * h_prev + (1.0 - z_t) * h_tilde
        return h_t


# ==============================================================================
# StreamingVisualCortexV2 Main Model
# ==============================================================================


class StreamingVisualCortexV2(nn.Module):
    """StreamingVisualCortexV2: End-to-end Pure-Pixel Spatiotemporal UI Perception Model.

    Combines:
      - 4-Channel Multi-Scale Spatial Stem (51,296 params)
      - ConvGRU Recurrent Temporal Backbone (221,376 params)
      - Latent Projection & Next-Latent Future Predictor (25,056 params)
      - Head 1: Temporal Change & Spinner Activity (56,242 params)
      - Head 2: Generic 5-Class UI-Region Detector (28,234 params)
      - Head 3: Post-Action Visual Response (18,561 params)

    Total Parameters: 400,765 (strictly in [400,000, 500,000]).
    """

    model_name: str = "StreamingVisualCortexV2"
    version: str = "2.0.0"

    def __init__(
        self,
        in_channels: int = 4,
        spatial_channels: int = 64,
        hidden_channels: int = 64,
        latent_dim: int = 96,
        max_state_gap_seconds: float = 2.0,
        decay_lambda: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.spatial_channels = spatial_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.max_state_gap_seconds = float(max_state_gap_seconds)
        self.decay_lambda = float(decay_lambda)

        # 1. Spatial Stem
        self.stem = SpatialPyramidStem(in_channels=in_channels, out_channels=spatial_channels)

        # 2. Recurrent ConvGRU Cell
        self.conv_gru = ConvGRUCell(in_channels=spatial_channels, hidden_channels=hidden_channels)

        # 3. Latent Projection & Future Predictor: 64 -> 96 -> 96
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_channels, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.future_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # 4. Specialized Heads
        self.change_spinner_head = TemporalChangeSpinnerHead(
            in_channels=hidden_channels + spatial_channels,  # 64 + 64 = 128
            mid_channels=48,
            heatmap_resolution=(128, 128),
        )
        self.ui_detector_head = GenericUIRegionDetectorHead(
            in_channels=hidden_channels,  # 64
            mid_channels=48,
            num_classes=5,
        )
        self.post_action_head = PostActionVisualResponseHead(
            latent_dim=latent_dim,  # 96
            hidden_dim=64,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all model weights."""
        with torch.no_grad():
            self.stem.reset_parameters()
            self.conv_gru.reset_parameters()
            for m in self.latent_projection.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.LayerNorm):
                    if m.weight is not None:
                        m.weight.fill_(1.0)
                    if m.bias is not None:
                        m.bias.zero_()
            for m in self.future_predictor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.zero_()
            self.change_spinner_head.reset_parameters()
            self.ui_detector_head.reset_parameters()
            self.post_action_head.reset_parameters()

    def reset_hidden_state(self) -> None:
        """Interface compatibility hook for resetting temporal recurrent state."""
        pass

    @property
    def total_parameters(self) -> int:
        """Return total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_parameter_breakdown(self) -> dict[str, int]:
        """Return dictionary mapping component names to exact parameter counts."""
        stem_params = sum(p.numel() for p in self.stem.parameters())
        gru_params = sum(p.numel() for p in self.conv_gru.parameters())
        latent_params = sum(p.numel() for p in self.latent_projection.parameters())
        predictor_params = sum(p.numel() for p in self.future_predictor.parameters())
        head1_params = sum(p.numel() for p in self.change_spinner_head.parameters())
        head2_params = sum(p.numel() for p in self.ui_detector_head.parameters())
        head3_params = sum(p.numel() for p in self.post_action_head.parameters())

        return {
            "spatial_stem": stem_params,
            "conv_gru": gru_params,
            "latent_projection": latent_params,
            "future_predictor": predictor_params,
            "head1_change_spinner": head1_params,
            "head2_ui_detector": head2_params,
            "head3_post_action": head3_params,
            "total": sum(p.numel() for p in self.parameters()),
        }

    def forward(
        self,
        x_4ch: torch.Tensor,
        prev_state: CortexTemporalState | None = None,
        timestamp: float = 0.0,
        dropped_frame: bool = False,
    ) -> CortexV2Output:
        """Process a 4-channel spatial tensor with temporal recurrence and gap handling.

        Args:
            x_4ch: 4-channel tensor of shape (B, 4, 128, 128) or (4, 128, 128) in float32.
            prev_state: Optional temporal state from previous observation.
            timestamp: Monotonic observation timestamp in seconds.
            dropped_frame: If True, indicates dropped observations and forces hard state reset.

        Returns:
            CortexV2Output containing latents, uncertainties, new temporal state, and head predictions.
        """
        # 1. Input shape validation
        if x_4ch.ndim == 3:
            x_4ch = x_4ch.unsqueeze(0)
        elif x_4ch.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, 4, H, W) or 3D tensor (4, H, W), got {tuple(x_4ch.shape)}")

        if x_4ch.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels at dim 1, got {x_4ch.shape[1]} (shape: {tuple(x_4ch.shape)})"
            )

        batch_size = x_4ch.shape[0]

        # 2. Multi-Scale Spatial Feature Extraction: f_t in R^(B x 64 x 16 x 16)
        f_t = self.stem(x_4ch)

        # 3. Timestamp & Gap Handling with Fail-Closed Hard Reset
        do_hard_reset = False
        delta_t = 0.0

        if prev_state is None:
            do_hard_reset = True
        else:
            raw_delta = float(timestamp - prev_state.timestamp)
            if dropped_frame or raw_delta < 0.0 or raw_delta >= self.max_state_gap_seconds:
                do_hard_reset = True
            delta_t = max(0.0, raw_delta)

        # 4. Hidden State Decay & Feature Delta
        if do_hard_reset or prev_state is None:
            # Zero out hidden state and set delta to zero
            h_prev = torch.zeros(
                batch_size,
                self.hidden_channels,
                16,
                16,
                dtype=f_t.dtype,
                device=f_t.device,
            )
            prev_f = f_t
            f_delta = torch.zeros_like(f_t)
            z_pre_latent = None
            prev_predicted_latent = None
            step_count = 1
        else:
            # Smooth exponential decay: h_decayed = h_{t-1} * exp(-lambda * delta_t) with lambda = 0.5
            decay_factor = math.exp(-self.decay_lambda * delta_t)
            h_prev = prev_state.hidden.to(device=f_t.device, dtype=f_t.dtype) * decay_factor
            prev_f = prev_state.previous_features.to(device=f_t.device, dtype=f_t.dtype)
            f_delta = torch.abs(f_t - prev_f)
            z_pre_latent = (
                prev_state.latent.to(device=f_t.device, dtype=f_t.dtype)
                if prev_state.latent is not None
                else None
            )
            prev_predicted_latent = (
                prev_state.predicted_next_latent.to(device=f_t.device, dtype=f_t.dtype)
                if prev_state.predicted_next_latent is not None
                else None
            )
            step_count = prev_state.step_count + 1

        # 5. Temporal Recurrent Ingestion: Ingest optical feature difference alongside f_t
        # Feature Delta Pathway: x_t combines spatial features with optical difference
        x_recurrent = f_t + f_delta
        h_t = self.conv_gru(x_recurrent, h_prev)  # (B, 64, 16, 16)

        # 6. Latent Projection & Next-Latent Future Predictor
        pooled_h = F.adaptive_avg_pool2d(h_t, (1, 1)).flatten(1)  # (B, 64)
        latent = self.latent_projection(pooled_h)                   # (B, 96)
        predicted_next_latent = self.future_predictor(latent)       # (B, 96)

        # 7. Feature Uncertainty / Prediction Error
        if prev_predicted_latent is not None and not do_hard_reset:
            # Normalized MSE error signal between actual next latent z_{t+1} and predicted z_hat_{t+1}
            feature_uncertainty = F.mse_loss(latent, prev_predicted_latent, reduction="none").mean(dim=-1)
        else:
            feature_uncertainty = torch.zeros(batch_size, dtype=latent.dtype, device=latent.device)

        # 8. Head 1: Temporal Change Heatmap & Cyclical Spinner Activity
        change_heatmap, spinner_activity_logit = self.change_spinner_head(h_t, f_delta)

        # 9. Head 2: Generic 5-Class UI-Region Detector
        ui_obj_logits, ui_cls_logits, ui_box_reg = self.ui_detector_head(h_t)

        # 10. Head 3: Post-Action Visual Response Logit
        z_pre = z_pre_latent if (z_pre_latent is not None and not do_hard_reset) else latent
        z_post = latent
        post_action_response_logit = self.post_action_head(z_pre, z_post)

        # 11. Encapsulate New Temporal State
        new_temporal_state = CortexTemporalState(
            hidden=h_t,
            previous_features=f_t,
            timestamp=float(timestamp),
            step_count=step_count,
            latent=latent,
            predicted_next_latent=predicted_next_latent,
        )

        return CortexV2Output(
            latent=latent,
            predicted_next_latent=predicted_next_latent,
            feature_uncertainty=feature_uncertainty,
            temporal_state=new_temporal_state,
            change_heatmap=change_heatmap,
            spinner_activity_logit=spinner_activity_logit,
            ui_objectness_logits=ui_obj_logits,
            ui_class_logits=ui_cls_logits,
            ui_box_regression=ui_box_reg,
            post_action_response_logit=post_action_response_logit,
        )

    def forward_step(
        self,
        x_4ch: torch.Tensor,
        prev_state: CortexTemporalState | None = None,
        timestamp: float = 0.0,
        dropped_frame: bool = False,
    ) -> CortexV2Output:
        """Alias for forward() processing a single observation frame."""
        return self.forward(
            x_4ch=x_4ch,
            prev_state=prev_state,
            timestamp=timestamp,
            dropped_frame=dropped_frame,
        )

    def forward_sequence(
        self,
        x_4ch_seq: torch.Tensor,
        timestamps: list[float] | torch.Tensor | None = None,
        dropped_frames: list[bool] | torch.Tensor | None = None,
        initial_state: CortexTemporalState | None = None,
    ) -> list[CortexV2Output]:
        """Process a sequence of frames (B, T, 4, 128, 128) across sequential time steps.

        Args:
            x_4ch_seq: Sequence tensor of shape (B, T, 4, 128, 128).
            timestamps: Optional list or tensor of T timestamps.
            dropped_frames: Optional list or tensor of T dropped frame flags.
            initial_state: Optional initial CortexTemporalState.

        Returns:
            List of T CortexV2Output instances.
        """
        if x_4ch_seq.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, T, 4, H, W), got {tuple(x_4ch_seq.shape)}")

        seq_len = x_4ch_seq.shape[1]
        outputs: list[CortexV2Output] = []
        state = initial_state

        for t in range(seq_len):
            frame = x_4ch_seq[:, t]
            ts = float(timestamps[t]) if timestamps is not None else float(t * (1.0 / 30.0))
            drop = bool(dropped_frames[t]) if dropped_frames is not None else False

            out = self.forward(frame, prev_state=state, timestamp=ts, dropped_frame=drop)
            outputs.append(out)
            state = out.temporal_state

        return outputs
