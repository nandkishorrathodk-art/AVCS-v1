"""Stateful multi-scale ConvGRU visual model for timestamped RGB streams."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for count in range(min(8, channels), 0, -1):
        if channels % count == 0:
            return count
    return 1


class CortexState(NamedTuple):
    hidden: torch.Tensor
    previous_features: torch.Tensor


class ConvGRUCell(nn.Module):
    """Convolutional gated state update at the encoder's coarsest spatial scale."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        joined = input_channels + hidden_channels
        self.gates = nn.Conv2d(joined, hidden_channels * 2, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(joined, hidden_channels, kernel_size=3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Clean initialization: Kaiming normal for input, orthogonal for recurrent."""
        with torch.no_grad():
            # Gates convolution: [:input_channels] is input-to-hidden, [input_channels:] is recurrent
            nn.init.kaiming_normal_(self.gates.weight[:, :self.input_channels], nonlinearity="sigmoid")
            for i in range(2):
                chunk = self.gates.weight[
                    i * self.hidden_channels : (i + 1) * self.hidden_channels,
                    self.input_channels :,
                ]
                nn.init.orthogonal_(chunk)
            if self.gates.bias is not None:
                self.gates.bias.zero_()
                # Update gate bias set to 1.0 (encourages initial state retention)
                self.gates.bias[self.hidden_channels :].fill_(1.0)

            # Candidate convolution
            nn.init.kaiming_normal_(self.candidate.weight[:, :self.input_channels], nonlinearity="tanh")
            nn.init.orthogonal_(self.candidate.weight[:, self.input_channels :])
            if self.candidate.bias is not None:
                self.candidate.bias.zero_()

    def forward(self, value: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((value, hidden), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((value, reset * hidden), dim=1))
        )
        return update * hidden + (1.0 - update) * candidate


class StreamingVisualCortexV1(nn.Module):
    """Learned spatial, recurrent, change, motion and next-feature prediction.

    Input frames are RGB tensors in [0, 1]. ``timestamp_delta`` is elapsed
    seconds since the previous observation; long intervals decay the recurrent
    state rather than being treated as adjacent high-rate frames.
    """

    format_version = 1
    model_name = "StreamingVisualCortexV1"

    def __init__(
        self,
        *,
        feature_dim: int = 48,
        hidden_dim: int = 64,
        latent_dim: int = 96,
        max_state_gap_seconds: float = 2.0,
        temporal_mode: str = "convgru",
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim, latent_dim) <= 0:
            raise ValueError("feature and latent dimensions must be positive.")
        if not math.isfinite(max_state_gap_seconds) or max_state_gap_seconds <= 0:
            raise ValueError("max_state_gap_seconds must be positive and finite.")
        if temporal_mode not in {"spatial", "convgru", "convgru_delta"}:
            raise ValueError("temporal_mode must be spatial, convgru, or convgru_delta.")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_state_gap_seconds = max_state_gap_seconds
        self.temporal_mode = temporal_mode

        self.scale_half = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(24), 24),
            nn.SiLU(inplace=True),
        )
        self.scale_quarter = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(inplace=True),
        )
        self.scale_eighth = nn.Sequential(
            nn.Conv2d(32, feature_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(feature_dim), feature_dim),
            nn.SiLU(inplace=True),
        )
        self.lateral_half = nn.Conv2d(24, 16, 1)
        self.lateral_quarter = nn.Conv2d(32, 16, 1)
        self.lateral_eighth = nn.Conv2d(feature_dim, 16, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(48, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_dim), hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.recurrent = ConvGRUCell(hidden_dim, hidden_dim)
        self.state_decay_rate = nn.Parameter(torch.tensor(-2.0))
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.future_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        change_channels = hidden_dim * (3 if temporal_mode == "convgru_delta" else 2) + feature_dim
        self.change_head = nn.Sequential(
            nn.Conv2d(change_channels, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.motion_head = nn.Sequential(
            nn.Conv2d(change_channels, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize clean random weights without loading any legacy checkpoints.

        - 2D Convolutions: Kaiming Normal with appropriate gain
        - ConvGRU recurrent weights: Orthogonal initialization (via ConvGRUCell.reset_parameters)
        - Linear projections: Xavier Uniform
        - Normalization layers (GroupNorm, LayerNorm): weight=1.0, bias=0.0
        - Biases: 0.0, with ConvGRU update gate bias initialized to 1.0
        """
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, ConvGRUCell):
                    module.reset_parameters()
                elif isinstance(module, nn.Conv2d):
                    if "recurrent" not in name:
                        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                        if module.bias is not None:
                            module.bias.zero_()
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                    if module.weight is not None:
                        module.weight.fill_(1.0)
                    if module.bias is not None:
                        module.bias.zero_()

    def encode_spatial(self, frame: torch.Tensor) -> torch.Tensor:
        half = self.scale_half(frame)
        quarter = self.scale_quarter(half)
        eighth = self.scale_eighth(quarter)
        target_size = eighth.shape[-2:]
        pyramid = torch.cat(
            (
                F.adaptive_avg_pool2d(self.lateral_half(half), target_size),
                F.adaptive_avg_pool2d(self.lateral_quarter(quarter), target_size),
                self.lateral_eighth(eighth),
            ),
            dim=1,
        )
        return self.fuse(pyramid), eighth

    def forward_step(
        self,
        frame: torch.Tensor,
        state: CortexState | None = None,
        *,
        timestamp_delta: float | torch.Tensor = 0.0,
    ) -> tuple[dict[str, torch.Tensor], CortexState]:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError("frame must have shape (batch, 3, height, width).")
        if min(frame.shape[-2:]) < 32:
            raise ValueError("frame height and width must each be at least 32.")
        if not frame.is_floating_point() or not torch.isfinite(frame).all().item():
            raise ValueError("frame must contain finite floating-point RGB values.")
        if frame.min().item() < 0.0 or frame.max().item() > 1.0:
            raise ValueError("frame RGB values must lie in [0, 1].")
        delta = torch.as_tensor(timestamp_delta, dtype=frame.dtype, device=frame.device)
        if delta.numel() not in {1, frame.shape[0]}:
            raise ValueError("timestamp_delta must be scalar or have one value per batch item.")
        delta = delta.reshape(-1)
        if not torch.isfinite(delta).all().item() or (delta < 0).any().item():
            raise ValueError("timestamp_delta values must be finite and non-negative.")
        fused, spatial = self.encode_spatial(frame)
        if state is None:
            hidden = fused.new_zeros((frame.shape[0], self.hidden_dim, *fused.shape[-2:]))
            previous = fused
            has_previous = False
        else:
            if state.hidden.shape != (frame.shape[0], self.hidden_dim, *fused.shape[-2:]):
                raise ValueError("recurrent state shape does not match this frame/model.")
            if state.previous_features.shape != fused.shape:
                raise ValueError("previous feature state shape does not match this frame.")
            hidden, previous = state
            has_previous = True
            decay_rate = F.softplus(self.state_decay_rate)
            decay = torch.exp(
                -decay_rate * delta.clamp(max=self.max_state_gap_seconds)
            )
            if decay.numel() == 1:
                decay = decay.expand(frame.shape[0])
            decay = decay.view(frame.shape[0], 1, 1, 1)
            hidden = hidden * decay

        if self.temporal_mode == "spatial":
            hidden = fused
            previous = fused
        else:
            hidden = self.recurrent(fused, hidden)
        latent = self.latent_projection(F.adaptive_avg_pool2d(hidden, 1).flatten(1))
        predicted_next = self.future_predictor(latent)
        comparison_parts = [hidden, previous, spatial]
        if self.temporal_mode == "convgru_delta":
            comparison_parts.append((fused - previous).abs())
        comparison = torch.cat(comparison_parts, dim=1)
        change_logits = F.interpolate(
            self.change_head(comparison),
            size=frame.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        motion = torch.tanh(self.motion_head(comparison)) * 0.25
        change_score = torch.sigmoid(change_logits).mean(dim=(1, 2, 3))
        outputs = {
            "latent": latent,
            "predicted_next_latent": predicted_next,
            "change_logits": change_logits,
            "change_score": change_score,
            "motion_grid_displacement": motion,
            "has_previous": torch.full(
                (frame.shape[0],), has_previous, dtype=torch.bool, device=frame.device
            ),
        }
        return outputs, CortexState(hidden, fused)

    def forward_sequence(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run (B,T,3,H,W) clips through the same incremental state path."""
        if frames.ndim != 5 or frames.shape[1] < 2:
            raise ValueError("frames must have shape (batch, time>=2, 3, height, width).")
        if timestamps is not None:
            if timestamps.shape != frames.shape[:2] or not torch.isfinite(timestamps).all().item():
                raise ValueError("timestamps must be finite with shape (batch, time).")
            if (timestamps[:, 1:] <= timestamps[:, :-1]).any().item():
                raise ValueError("timestamps must increase strictly within each sequence.")
        state: CortexState | None = None
        collected: dict[str, list[torch.Tensor]] = {}
        for index in range(frames.shape[1]):
            delta: float | torch.Tensor = 0.0 if index == 0 else 1.0
            if timestamps is not None and index:
                differences = timestamps[:, index] - timestamps[:, index - 1]
                delta = differences
            output, state = self.forward_step(
                frames[:, index],
                state,
                timestamp_delta=delta,
            )
            for name, value in output.items():
                collected.setdefault(name, []).append(value)
        return {name: torch.stack(values, dim=1) for name, values in collected.items()}

    def forward(
        self,
        x: torch.Tensor,
        prev_state: CortexState | None = None,
        delta_t: float | torch.Tensor = 0.0,
    ) -> tuple[torch.Tensor, CortexState, dict[str, torch.Tensor]]:
        """Run single observation step matching the SCOPE.md interface contract.

        Args:
            x: Raw input frame tensor (batch, 3, height, width) in [0, 1].
            prev_state: Prior CortexState or None for initial state.
            delta_t: Elapsed seconds since prior observation.

        Returns:
            Tuple of (latent_z, next_state, full_outputs_dict).
        """
        output_dict, next_state = self.forward_step(x, prev_state, timestamp_delta=delta_t)
        return output_dict["latent"], next_state, output_dict

    def predict_future(self, latent_z: torch.Tensor) -> torch.Tensor:
        """Predict future latent representation from recent temporal latent context."""
        return self.future_predictor(latent_z)

    def reset_state(self) -> None:
        """Clean interface for resetting any model-held transient state."""
        pass

    def initial_state(
        self,
        batch_size: int = 1,
        height: int = 16,
        width: int = 16,
        device: torch.device | None = None,
    ) -> CortexState:
        """Construct a clean zero-initialized persistent recurrent hidden state."""
        dev = device if device is not None else self.state_decay_rate.device
        hidden = torch.zeros((batch_size, self.hidden_dim, height, width), device=dev)
        previous = torch.zeros((batch_size, self.hidden_dim, height, width), device=dev)
        return CortexState(hidden=hidden, previous_features=previous)


def create_clean_random_cortex(
    *,
    seed: int | None = None,
    feature_dim: int = 48,
    hidden_dim: int = 64,
    latent_dim: int = 96,
    max_state_gap_seconds: float = 2.0,
    temporal_mode: str = "convgru",
) -> StreamingVisualCortexV1:
    """Instantiate a StreamingVisualCortexV1 with verified clean random initialization.

    Guarantees:
    - Absolutely ZERO weights loaded from legacy checkpoints.
    - Kaiming normal for 2D convolutions.
    - Orthogonal recurrent kernel W_hh for ConvGRUCell.
    - Xavier uniform for linear projections.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = StreamingVisualCortexV1(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_state_gap_seconds=max_state_gap_seconds,
        temporal_mode=temporal_mode,
    )
    model.init_weights()
    return model
