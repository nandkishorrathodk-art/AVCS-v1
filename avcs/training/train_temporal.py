"""Bounded in-memory frame-pair buffer and self-supervised temporal trainer."""
from __future__ import annotations

import random
import tempfile
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from avcs.temporal_cortex import (
    TemporalVisualCortex,
    temporal_loss_for_frames,
    temporal_ssl_loss,
)


class FramePairBuffer:
    """Keep a bounded recent window of letterboxed RGB frame pairs in RAM only."""

    def __init__(
        self,
        *,
        capacity: int = 256,
        frame_size: tuple[int, int] = (128, 128),
    ) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least two frame pairs.")
        width, height = frame_size
        if width < 32 or height < 32:
            raise ValueError("frame_size dimensions must each be at least 32 pixels.")
        self.capacity = capacity
        self.frame_size = frame_size
        self._pairs: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._pairs)

    def append(self, previous_rgb: np.ndarray, current_rgb: np.ndarray) -> None:
        """Add one adjacent uint8 RGB pair; no image or metadata is written to disk."""
        previous = self._prepare(previous_rgb)
        current = self._prepare(current_rgb)
        self._pairs.append((previous, current))

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not indices:
            raise ValueError("At least one frame-pair index is required.")
        pairs = list(self._pairs)
        if any(index < 0 or index >= len(pairs) for index in indices):
            raise IndexError("Frame-pair index is outside the current buffer.")
        previous = torch.stack([pairs[index][0] for index in indices]).to(device=device)
        current = torch.stack([pairs[index][1] for index in indices]).to(device=device)
        return previous.float().div_(255.0), current.float().div_(255.0)

    def _prepare(self, frame: np.ndarray) -> torch.Tensor:
        array = letterbox_rgb_frame(frame, self.frame_size)
        return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def letterbox_rgb_frame(
    frame: np.ndarray,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Resize a uint8 RGB frame proportionally onto a centered RGB canvas."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frames must be numpy uint8 RGB arrays.")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Frames must be uint8 RGB arrays with shape (height, width, 3).")
    if frame.shape[0] < 32 or frame.shape[1] < 32:
        raise ValueError("Source frames must be at least 32x32 pixels.")
    target_width, target_height = frame_size
    if target_width < 32 or target_height < 32:
        raise ValueError("frame_size dimensions must each be at least 32 pixels.")
    image = Image.fromarray(frame)
    if image.size != frame_size:
        source_width, source_height = image.size
        scale = min(
            target_width / source_width,
            target_height / source_height,
        )
        resized_size = (
            min(target_width, max(1, round(source_width * scale))),
            min(target_height, max(1, round(source_height * scale))),
        )
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize(resized_size, resampling.BILINEAR)
        canvas = Image.new("RGB", frame_size, color=(0, 0, 0))
        offset = (
            (target_width - resized_size[0]) // 2,
            (target_height - resized_size[1]) // 2,
        )
        canvas.paste(image, offset)
        image = canvas
    return np.asarray(image, dtype=np.uint8)


def train_temporal_cortex(
    model: TemporalVisualCortex,
    buffer: FramePairBuffer,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    change_threshold: float = 0.03,
    change_weight: float = 0.0,
    model_init_seed: Optional[int] = None,  # noqa: UP045
    checkpoint_path: Optional[Path] = None,  # noqa: UP045
    data_provenance: Optional[dict[str, float | int | str]] = None,  # noqa: UP045
) -> list[dict[str, float]]:
    """Fit on a bounded in-memory buffer and return measured per-epoch losses."""
    if len(buffer) < 2:
        raise ValueError("At least two frame pairs are required for anti-collapse losses.")
    if (
        epochs < 1
        or batch_size < 2
        or learning_rate <= 0
        or not 0.0 <= change_threshold <= 1.0
        or change_weight < 0.0
    ):
        raise ValueError("epochs and batch_size must be >= 1 and >= 2; learning_rate must be positive.")
    device_obj = torch.device(device)
    model.to(device_obj)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        indices = list(range(len(buffer)))
        rng.shuffle(indices)
        totals: dict[str, float] = {}
        batches = 0
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            if len(batch_indices) < 2:
                continue
            previous, current = buffer.batch(batch_indices, device=device_obj)
            optimizer.zero_grad(set_to_none=True)
            losses = temporal_loss_for_frames(
                model,
                previous,
                current,
                change_threshold=change_threshold,
                change_weight=change_weight,
            )
            losses["total"].backward()
            optimizer.step()
            batches += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        if not batches:
            raise RuntimeError("No valid training batches were produced.")
        history.append({
            "epoch": float(epoch + 1),
            **{name: value / batches for name, value in totals.items()},
            "batches": float(batches),
            "pairs": float(len(buffer)),
        })
    if checkpoint_path is not None:
        _save_checkpoint(
            model,
            Path(checkpoint_path),
            history,
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "change_threshold": change_threshold,
                "change_weight": change_weight,
                "model_init_seed": model_init_seed,
                "training_pairs": len(buffer),
            },
            data_provenance or {},
        )
    return history


def evaluate_temporal_cortex(
    model: TemporalVisualCortex,
    buffer: FramePairBuffer,
    *,
    batch_size: int = 8,
    device: str = "cpu",
    change_threshold: float = 0.03,
    change_weight: float = 0.0,
) -> dict[str, float]:
    """Measure losses and latent health over the buffer without updates."""
    if len(buffer) < 2:
        raise ValueError("At least two frame pairs are required for anti-collapse losses.")
    if batch_size < 2:
        raise ValueError("batch_size must be at least two.")
    device_obj = torch.device(device)
    model.to(device_obj)
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    evaluated_pairs = 0
    latents: list[torch.Tensor] = []
    motion_magnitudes: list[torch.Tensor] = []
    surprise_means: list[torch.Tensor] = []
    batches = _evaluation_batches(len(buffer), batch_size)
    try:
        with torch.inference_mode():
            for indices in batches:
                previous, current = buffer.batch(indices, device=device_obj)
                outputs = model(previous, current)
                losses = temporal_ssl_loss(
                    outputs,
                    change_threshold=change_threshold,
                    change_weight=change_weight,
                )
                for name, value in losses.items():
                    totals[name] = (
                        totals.get(name, 0.0) + float(value.cpu()) * len(indices)
                    )
                evaluated_pairs += len(indices)
                latents.append(outputs["latent"].cpu())
                motion_magnitudes.append(
                    outputs["motion"].square().sum(dim=1).sqrt().flatten(1).mean(dim=1).cpu()
                )
                surprise_means.append(
                    outputs["surprise"].flatten(1).mean(dim=1).cpu()
                )
    finally:
        model.train(was_training)
    if not evaluated_pairs:
        raise RuntimeError("No valid evaluation batches were produced.")

    latent_values = torch.cat(latents, dim=0)
    centered = latent_values - latent_values.mean(dim=0, keepdim=True)
    std = centered.var(dim=0, unbiased=False).sqrt()
    covariance = centered.T @ centered / max(1, latent_values.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    off_diagonal_count = max(1, latent_values.shape[1] * (latent_values.shape[1] - 1))
    off_diagonal_rms = off_diagonal.square().sum().div(off_diagonal_count).sqrt()
    motion_values = torch.cat(motion_magnitudes)
    surprise_values = torch.cat(surprise_means)
    return {
        **{name: value / evaluated_pairs for name, value in totals.items()},
        "latent_std_mean": float(std.mean()),
        "latent_std_min": float(std.min()),
        "latent_covariance_offdiagonal_rms": float(off_diagonal_rms),
        "latent_covariance_offdiagonal_frobenius": float(torch.linalg.vector_norm(off_diagonal)),
        "predicted_motion_magnitude_norm_mean": float(motion_values.mean()),
        "predicted_motion_magnitude_norm_p95": float(torch.quantile(motion_values, 0.95)),
        "surprise_mean": float(surprise_values.mean()),
        "surprise_per_pair_mean_p95": float(torch.quantile(surprise_values, 0.95)),
        "evaluated_pairs": float(evaluated_pairs),
    }


def _evaluation_batches(length: int, batch_size: int) -> list[list[int]]:
    batches = [
        list(range(start, min(start + batch_size, length)))
        for start in range(0, length, batch_size)
    ]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def load_temporal_cortex_checkpoint(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[TemporalVisualCortex, dict[str, object]]:
    """Load a versioned temporal checkpoint with strict state validation."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Temporal checkpoint must contain a mapping.")
    if checkpoint.get("format_version") != 1 or checkpoint.get("model") != "TemporalVisualCortex":
        raise ValueError("Unsupported temporal checkpoint format or model.")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    training_config = checkpoint.get("training_config")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise TypeError("Temporal checkpoint is missing model configuration or weights.")
    if not isinstance(training_config, dict):
        raise TypeError("Temporal checkpoint is missing training configuration.")
    try:
        model = TemporalVisualCortex(
            feature_dim=int(model_config["feature_dim"]),
            latent_dim=int(model_config["latent_dim"]),
            max_displacement=float(model_config["max_displacement"]),
        )
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid temporal checkpoint model state: {exc}") from exc
    model.to(torch.device(device))
    return model, checkpoint


def compare_temporal_cortex_on_held_out_stream(
    checkpoint_path: Path,
    held_out_buffer: FramePairBuffer,
    *,
    batch_size: int = 8,
    device: str = "cpu",
    change_threshold: Optional[float] = None,  # noqa: UP045
    change_weight: Optional[float] = None,  # noqa: UP045
) -> dict[str, object]:
    """Compare the checkpoint against its reproducible random-init baseline."""
    trained_model, checkpoint = load_temporal_cortex_checkpoint(
        checkpoint_path,
        device=device,
    )
    training_config = checkpoint["training_config"]
    init_seed = training_config.get("model_init_seed")
    if not isinstance(init_seed, int):
        raise TypeError(
            "Checkpoint has no reproducible model_init_seed; cannot construct "
            "a matched random-initialization baseline."
        )
    evaluation_threshold = (
        float(training_config.get("change_threshold", 0.03))
        if change_threshold is None
        else change_threshold
    )
    evaluation_weight = (
        float(training_config.get("change_weight", 0.0))
        if change_weight is None
        else change_weight
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        random_model = TemporalVisualCortex(**checkpoint["model_config"])
    random_metrics = evaluate_temporal_cortex(
        random_model,
        held_out_buffer,
        batch_size=batch_size,
        device=device,
        change_threshold=evaluation_threshold,
        change_weight=evaluation_weight,
    )
    trained_metrics = evaluate_temporal_cortex(
        trained_model,
        held_out_buffer,
        batch_size=batch_size,
        device=device,
        change_threshold=evaluation_threshold,
        change_weight=evaluation_weight,
    )
    comparable_metrics = (
        "total",
        "prediction",
        "variance",
        "decorrelation",
        "photometric",
        "changed_pixel_fraction",
        "motion_smoothness",
        "latent_std_mean",
        "latent_std_min",
        "latent_covariance_offdiagonal_rms",
        "predicted_motion_magnitude_norm_mean",
        "predicted_motion_magnitude_norm_p95",
        "surprise_mean",
        "surprise_per_pair_mean_p95",
    )
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "held_out_pairs": len(held_out_buffer),
        "evaluation_loss_config": {
            "change_threshold": evaluation_threshold,
            "change_weight": evaluation_weight,
        },
        "random_initialization": random_metrics,
        "trained_checkpoint": trained_metrics,
        "trained_minus_random": {
            name: trained_metrics[name] - random_metrics[name]
            for name in comparable_metrics
        },
        "interpretation": (
            "Held-out self-supervised objective comparison only; not evidence "
            "of general visual understanding or task success."
        ),
        "raw_frames_saved": False,
    }


def _save_checkpoint(
    model: TemporalVisualCortex,
    path: Path,
    history: list[dict[str, float]],
    training_config: dict[str, float | int | str | None],
    data_provenance: dict[str, float | int | str],
) -> None:
    """Atomically save weights and provenance without retaining input frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(
            {
                "format_version": 1,
                "model": "TemporalVisualCortex",
                "model_config": {
                    "feature_dim": model.feature_dim,
                    "latent_dim": model.latent_dim,
                    "max_displacement": model.max_displacement,
                },
                "training_config": training_config,
                "data_provenance": data_provenance,
                "loss_history": history,
                "evaluation_status": "not_evaluated",
                "model_state_dict": model.state_dict(),
            },
            temporary_path,
        )
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise
