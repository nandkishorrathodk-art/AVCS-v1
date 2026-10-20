"""Goal-conditioned, pixel-only target grounding for AVCS."""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from avcs.perception.cortex_2m import CorticalBackbone2M
from avcs.perception.human_eye import ConvBNAct
from avcs.query_limits import MAX_QUERY_BYTES

GROUNDING_MODEL_TYPE = "avcs_goal_conditioned_grounder"
GROUNDING_CHECKPOINT_SCHEMA = 2
LEGACY_GROUNDING_CHECKPOINT_SCHEMA = 1
SUPPORTED_GROUNDING_CHECKPOINT_SCHEMAS = (1, 2)
GROUNDING_DATASET_PROVENANCE = "recorded_ui_click_and_absence_demonstrations"
LEGACY_GROUNDING_DATASET_PROVENANCE = "recorded_ui_click_demonstrations"
MULTI_GENERATOR_GROUNDING_DATASET_PROVENANCE = "multi_generator_grounding_corpus_v1"
GROUNDING_CHECKPOINT_DATASET_PROVENANCES = (
    GROUNDING_DATASET_PROVENANCE,
    MULTI_GENERATOR_GROUNDING_DATASET_PROVENANCE,
)
DEFAULT_INPUT_SIZE = 256


def encode_queries(queries: Sequence[str], max_length: int = MAX_QUERY_BYTES) -> torch.Tensor:
    """Encode query UTF-8 bytes as token IDs; zero is reserved for padding."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    tokens = torch.zeros((len(queries), max_length), dtype=torch.long)
    for row, query in enumerate(queries):
        encoded = query.strip().encode("utf-8")
        if not encoded:
            raise ValueError("Grounding queries must not be empty")
        if len(encoded) > max_length:
            raise ValueError(f"Grounding queries must be at most {max_length} UTF-8 bytes")
        tokens[row, : len(encoded)] = torch.tensor(
            [byte + 1 for byte in encoded], dtype=torch.long
        )
    return tokens


def extract_subpixel_peak(
    logits: torch.Tensor,
    epsilon: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts sub-pixel peak coordinates and peak confidence without center-collapse.

    Applies discrete 2D spatial peak argmax + local 1D quadratic interpolation along x and y.
    Under multimodal distributions or negative screens, it evaluates strictly at the peak cell,
    completely eliminating DSNT center-collapse.

    Args:
        logits: Spatial logits tensor [B, 1, H, W]
        epsilon: Small constant preventing division by zero on flat maps

    Returns:
        coords: Normalized sub-pixel coordinates [B, 2] in [0, 1] range (cx, cy)
        confidence: Peak sigmoid confidence values [B] in [0, 1] range
    """
    B, _, H, W = logits.shape
    padded = F.pad(logits, (1, 1, 1, 1), mode="replicate")
    flat_idx = logits.flatten(1).argmax(dim=-1)
    r_star = flat_idx // W
    c_star = flat_idx % W

    rp = r_star + 1
    cp = c_star + 1

    b_idx = torch.arange(B, device=logits.device)
    l_c = padded[b_idx, 0, rp, cp]
    l_left = padded[b_idx, 0, rp, cp - 1]
    l_right = padded[b_idx, 0, rp, cp + 1]
    l_top = padded[b_idx, 0, rp - 1, cp]
    l_bottom = padded[b_idx, 0, rp + 1, cp]

    denom_x = 2.0 * (2.0 * l_c - l_right - l_left).clamp_min(epsilon)
    delta_x = torch.clamp((l_right - l_left) / denom_x, -0.5, 0.5)

    denom_y = 2.0 * (2.0 * l_c - l_bottom - l_top).clamp_min(epsilon)
    delta_y = torch.clamp((l_bottom - l_top) / denom_y, -0.5, 0.5)

    pred_x = (c_star.float() + 0.5 + delta_x) / float(W)
    pred_y = (r_star.float() + 0.5 + delta_y) / float(H)
    coords = torch.stack([pred_x, pred_y], dim=-1)
    confidence = torch.sigmoid(l_c)
    return coords, confidence


class ByteQueryEncoder(nn.Module):
    """Learned query encoder, independent of OCR/accessibility text extraction."""

    def __init__(self, output_dim: int = 192, embedding_dim: int = 48) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
        self.convolutions = nn.Sequential(
            nn.Conv1d(embedding_dim, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("query tokens must have shape (batch, non-empty sequence)")
        if torch.any(tokens < 0) or torch.any(tokens > 256):
            raise ValueError("query token IDs must be in [0, 256]")
        mask = tokens.ne(0).unsqueeze(1)
        if torch.any(~mask.any(dim=2)):
            raise ValueError("each query must contain at least one non-padding token")
        encoded = self.embedding(tokens).transpose(1, 2)
        encoded = self.convolutions(encoded)
        mask_float = mask.to(encoded.dtype)
        mean = (encoded * mask_float).sum(dim=2) / mask_float.sum(dim=2).clamp_min(1)
        maximum = encoded.masked_fill(~mask, torch.finfo(encoded.dtype).min).amax(dim=2)
        return self.projection(torch.cat((mean, maximum), dim=1))


class GoalConditionedGrounder(nn.Module):
    """Predict a dense click-target heatmap from RGB pixels and a textual subgoal.

    The model learns a location-conditioned compatibility score. It does not
    select high-level task steps, type input, or establish that a click is safe.
    """

    def __init__(
        self, feature_dim: int = 192, input_size: int = DEFAULT_INPUT_SIZE
    ) -> None:
        super().__init__()
        if input_size < 64 or input_size % 16:
            raise ValueError("input_size must be at least 64 and divisible by 16")
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.visual_encoder = CorticalBackbone2M(feature_dim=feature_dim)
        self.query_encoder = ByteQueryEncoder(output_dim=feature_dim)
        self.query_modulation = nn.Linear(feature_dim, feature_dim * 2)
        self.lateral_8 = ConvBNAct(128, 64, kernel_size=1, padding=0)
        self.lateral_4 = ConvBNAct(64, 64, kernel_size=1, padding=0)
        self.lateral_2 = ConvBNAct(32, 32, kernel_size=1, padding=0)
        self.decode_8 = ConvBNAct(feature_dim + 64, 96, kernel_size=3, padding=1)
        self.decode_4 = ConvBNAct(96 + 64, 64, kernel_size=3, padding=1)
        self.decode_2 = ConvBNAct(64 + 32, 64, kernel_size=3, padding=1)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        self.target_presence = nn.Linear(64, 1)
        self._checkpoint_metadata: dict[str, object] = {}

    @property
    def is_trained(self) -> bool:
        schema = self._checkpoint_metadata.get("schema_version")
        legacy_schema = self._checkpoint_metadata.get("legacy_schema_version")
        provenance = self._checkpoint_metadata.get("dataset_provenance")
        valid_schema = (
            schema in SUPPORTED_GROUNDING_CHECKPOINT_SCHEMAS
            or legacy_schema in SUPPORTED_GROUNDING_CHECKPOINT_SCHEMAS
        )
        valid_provenance = provenance in (
            GROUNDING_DATASET_PROVENANCE,
            LEGACY_GROUNDING_DATASET_PROVENANCE,
            MULTI_GENERATOR_GROUNDING_DATASET_PROVENANCE,
        )
        training_samples = self._checkpoint_metadata.get("training_samples")
        validation_samples = self._checkpoint_metadata.get("validation_samples")
        return (
            valid_schema
            and self._checkpoint_metadata.get("model_type") == GROUNDING_MODEL_TYPE
            and valid_provenance
            and isinstance(training_samples, int)
            and not isinstance(training_samples, bool)
            and training_samples > 0
            and isinstance(validation_samples, int)
            and not isinstance(validation_samples, bool)
            and validation_samples > 0
        )

    @property
    def checkpoint_metadata(self) -> Mapping[str, object]:
        return dict(self._checkpoint_metadata)

    def _mark_loaded_checkpoint(self, metadata: dict[str, object]) -> None:
        """Mark a model usable only after the checkpoint loader validates it."""
        self._checkpoint_metadata = dict(metadata)

    def _decode(self, image: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (batch, 3, height, width)")
        if image.shape[0] != query_tokens.shape[0]:
            raise ValueError("image and query batch sizes must match")
        stem, stage1, stage2, _stage3, visual = self.visual_encoder.forward_pyramid(image)
        query = self.query_encoder(query_tokens)
        scale, bias = self.query_modulation(query).chunk(2, dim=1)
        visual = visual * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        decoded = F.interpolate(visual, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_8(torch.cat((decoded, self.lateral_8(stage2)), dim=1))
        decoded = F.interpolate(decoded, size=stage1.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_4(torch.cat((decoded, self.lateral_4(stage1)), dim=1))
        decoded = F.interpolate(decoded, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_2(torch.cat((decoded, self.lateral_2(stem)), dim=1))
        return decoded

    def forward(self, image: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        """Return spatial click logits; retained for localization-only callers."""
        return self.output(self._decode(image, query_tokens))

    def forward_with_presence(
        self, image: torch.Tensor, query_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return click logits and a separate target-present logit per image."""
        decoded = self._decode(image, query_tokens)
        pooled = F.adaptive_avg_pool2d(decoded, output_size=1).flatten(1)
        return self.output(decoded), self.target_presence(pooled).squeeze(1)

    @torch.inference_mode()
    def predict(self, frame_rgb: np.ndarray, query: str) -> GroundingPrediction:
        """Return a candidate point only for a checkpoint trained on real demos."""
        if not self.is_trained:
            raise RuntimeError(
                "GoalConditionedGrounder has no demonstration-trained checkpoint; "
                "train it on recorded UI demonstrations before inference."
            )
        image, transform = letterbox_rgb(frame_rgb, self.input_size)
        device = next(self.parameters()).device
        logits, presence_logits = self.forward_with_presence(
            image.unsqueeze(0).to(device), encode_queries([query]).to(device)
        )
        target_present_probability = float(presence_logits.sigmoid().item())
        threshold = self._checkpoint_metadata.get("target_presence_threshold")
        if not isinstance(threshold, (float, int)) or isinstance(threshold, bool):
            raise TypeError("Grounding checkpoint has no calibrated target-presence threshold")
        target_present = target_present_probability >= float(threshold)
        probabilities = logits.flatten(1).softmax(dim=1).view_as(logits)
        flat_index = int(probabilities.flatten().argmax())
        _, _, heat_h, heat_w = probabilities.shape
        row, column = divmod(flat_index, heat_w)
        canvas_point = ((column + 0.5) / heat_w, (row + 0.5) / heat_h)
        point = transform.canvas_to_original(canvas_point)
        if not target_present or not (0.0 <= point[0] < 1.0 and 0.0 <= point[1] < 1.0):
            point_result = None
        else:
            point_result = point
        return GroundingPrediction(
            candidate_point_norm=point_result,
            peak_probability=float(probabilities.flatten()[flat_index]),
            heatmap=probabilities[0, 0].detach().cpu(),
            target_present_probability=target_present_probability,
            target_present=target_present,
            action_authorized=False,
        )

    @torch.inference_mode()
    def predict_point(self, image_tensor: torch.Tensor, query: str) -> dict[str, object]:
        """Infers candidate point, presence-calibrated confidence, and bounding box with sub-pixel peak extraction."""
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        device = next(self.parameters()).device
        img = image_tensor.to(device)

        clean_q = query.strip() if isinstance(query, str) else ""
        if not clean_q or not any(c.isalnum() for c in clean_q) or len(clean_q.encode("utf-8")) > MAX_QUERY_BYTES:
            return {
                "point": (0.5, 0.5),
                "confidence": 0.0,
                "bbox": (0.46, 0.46, 0.54, 0.54),
                "target_present": False,
            }

        q_tokens = encode_queries([clean_q]).to(device)
        logits, presence_logits = self.forward_with_presence(img, q_tokens)
        coords, peak_conf = extract_subpixel_peak(logits)
        pred_x = float(coords[0, 0].item())
        pred_y = float(coords[0, 1].item())

        exist_prob = float(presence_logits.sigmoid().item())
        peak_prob = float(peak_conf[0].item())
        combined_conf = peak_prob * exist_prob

        threshold = float(self._checkpoint_metadata.get("target_presence_threshold", 0.5))
        target_present = exist_prob >= threshold
        if not target_present:
            combined_conf = min(combined_conf, 0.05)

        pred_box = (
            max(0.0, pred_y - 0.04),
            max(0.0, pred_x - 0.04),
            min(1.0, pred_y + 0.04),
            min(1.0, pred_x + 0.04),
        )
        return {
            "point": (pred_x, pred_y),
            "confidence": combined_conf,
            "bbox": pred_box,
            "target_present": target_present,
            "heatmap": torch.sigmoid(logits[0, 0]).detach().cpu(),
        }



@dataclass(frozen=True)
class LetterboxTransform:
    left_norm: float
    top_norm: float
    width_norm: float
    height_norm: float

    def original_to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            self.left_norm + point[0] * self.width_norm,
            self.top_norm + point[1] * self.height_norm,
        )

    def canvas_to_original(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            (point[0] - self.left_norm) / self.width_norm,
            (point[1] - self.top_norm) / self.height_norm,
        )


@dataclass(frozen=True)
class GroundingPrediction:
    candidate_point_norm: Optional[tuple[float, float]]  # noqa: UP045
    peak_probability: float
    heatmap: torch.Tensor
    target_present_probability: float = 1.0
    target_present: bool = True
    action_authorized: bool = False


@dataclass(frozen=True)
class GroundingStreamFrame:
    sequence: int
    captured_at: float
    inference_latency_ms: float
    query: str
    prediction: GroundingPrediction


class GroundingStream:
    """Process each submitted fresh frame; does not capture or control a desktop.

    This is a stateless per-frame inference stream. It does not provide temporal
    memory, drop frames, or authorize motor actions.
    """

    def __init__(self, model: GoalConditionedGrounder, metrics_window: int = 128) -> None:
        if not model.is_trained:
            raise RuntimeError("GroundingStream requires a demonstration-trained model")
        if metrics_window <= 1:
            raise ValueError("metrics_window must be greater than one")
        self.model = model
        self.metrics_window = metrics_window
        self._sequence = 0
        self._timestamps: list[float] = []
        self._latencies_ms: list[float] = []

    def process_frame(
        self, frame_rgb: np.ndarray, query: str, captured_at: Optional[float] = None  # noqa: UP045
    ) -> GroundingStreamFrame:
        timestamp = time.monotonic() if captured_at is None else float(captured_at)
        if not math.isfinite(timestamp) or timestamp > time.monotonic():
            raise ValueError("captured_at must be a finite monotonic timestamp, not in the future")
        if self._timestamps and timestamp <= self._timestamps[-1]:
            raise ValueError("captured_at must increase for each stream frame")
        started = time.perf_counter()
        prediction = self.model.predict(frame_rgb, query)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._sequence += 1
        self._timestamps.append(timestamp)
        self._latencies_ms.append(latency_ms)
        self._timestamps = self._timestamps[-self.metrics_window :]
        self._latencies_ms = self._latencies_ms[-self.metrics_window :]
        return GroundingStreamFrame(
            sequence=self._sequence,
            captured_at=timestamp,
            inference_latency_ms=latency_ms,
            query=query,
            prediction=prediction,
        )

    def metrics(self) -> dict[str, Optional[float]]:  # noqa: UP045
        """Return recent measured FPS and inference latency; not a performance promise."""
        if not self._latencies_ms:
            return {"observed_input_fps": None, "p50_inference_latency_ms": None}
        if len(self._timestamps) < 2 or self._timestamps[-1] <= self._timestamps[0]:
            observed_fps = None
        else:
            observed_fps = (len(self._timestamps) - 1) / (
                self._timestamps[-1] - self._timestamps[0]
            )
        ordered = sorted(self._latencies_ms)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            median_latency = ordered[middle]
        else:
            median_latency = (ordered[middle - 1] + ordered[middle]) / 2.0
        return {
            "observed_input_fps": observed_fps,
            "p50_inference_latency_ms": median_latency,
        }

    def reset(self) -> None:
        self._sequence = 0
        self._timestamps.clear()
        self._latencies_ms.clear()


def letterbox_rgb(
    frame_rgb: np.ndarray, input_size: int = DEFAULT_INPUT_SIZE
) -> tuple[torch.Tensor, LetterboxTransform]:
    """Resize an RGB frame without distorting its aspect ratio."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must have shape (height, width, 3)")
    if frame_rgb.dtype != np.uint8:
        raise ValueError("frame_rgb must contain uint8 RGB pixels")
    height, width = frame_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("frame_rgb must not be empty")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, min(input_size, round(width * scale)))
    resized_height = max(1, min(input_size, round(height * scale)))
    left = (input_size - resized_width) // 2
    top = (input_size - resized_height) // 2
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(frame_rgb, mode="RGB").resize(
        (resized_width, resized_height), resampling.BILINEAR
    )
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = np.asarray(resized)
    tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1).float().div_(255.0)
    transform = LetterboxTransform(
        left_norm=left / input_size,
        top_norm=top / input_size,
        width_norm=resized_width / input_size,
        height_norm=resized_height / input_size,
    )
    return tensor, transform


def click_heatmap(
    point_norm: tuple[float, float],
    output_size: tuple[int, int],
    sigma_cells: float = 1.5,
) -> torch.Tensor:
    """Build a normalized Gaussian target centered at a normalized click label."""
    x, y = point_norm
    height, width = output_size
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < 1 and 0 <= y < 1):
        raise ValueError("point_norm values must be finite and in [0, 1)")
    if height <= 0 or width <= 0 or sigma_cells <= 0:
        raise ValueError("output dimensions and sigma_cells must be positive")
    yy, xx = torch.meshgrid(
        (torch.arange(height, dtype=torch.float32) + 0.5) / height,
        (torch.arange(width, dtype=torch.float32) + 0.5) / width,
        indexing="ij",
    )
    dist = (xx - x).square() * (width**2) + (yy - y).square() * (height**2)
    target = torch.exp(-dist / (2.0 * sigma_cells**2))
    return target / target.sum().clamp_min(torch.finfo(target.dtype).eps)


def migrate_grounding_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    """Migrate legacy schema 1 checkpoint to schema 2 format.

    Ensures compatibility with both schema 1 legacy checkpoints and schema 2
    (with full evaluation provenance and target-presence support).
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError("grounding checkpoint must be a metadata dictionary")
    schema_version = checkpoint.get("schema_version")
    if schema_version == GROUNDING_CHECKPOINT_SCHEMA:
        return dict(checkpoint)
    if schema_version != LEGACY_GROUNDING_CHECKPOINT_SCHEMA:
        raise ValueError(f"Cannot migrate unsupported schema_version: {schema_version}")

    migrated = dict(checkpoint)
    migrated["legacy_schema_version"] = LEGACY_GROUNDING_CHECKPOINT_SCHEMA
    migrated["schema_version"] = GROUNDING_CHECKPOINT_SCHEMA
    if "target_presence_threshold" not in migrated:
        migrated["target_presence_threshold"] = 0.5
    if "target_presence_calibrated" not in migrated:
        migrated["target_presence_calibrated"] = False

    raw_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(raw_state_dict, dict):
        raise TypeError("grounding checkpoint is missing its model state dictionary")

    state_dict = dict(raw_state_dict)
    if "target_presence.weight" not in state_dict:
        state_dict["target_presence.weight"] = torch.zeros((1, 64), dtype=torch.float32)
    if "target_presence.bias" not in state_dict:
        # Schema 1 only contained positive demonstrations; positive bias declares presence
        state_dict["target_presence.bias"] = torch.tensor([2.0], dtype=torch.float32)
    migrated["model_state_dict"] = state_dict
    return migrated


def _select_grounder_architecture(state: dict[str, Any]) -> type:
    """Return the class matching a checkpoint's architecture, or fail closed.

    Two classes named GoalConditionedGrounder exist: this module's 195-key
    FPN variant and the 81-key variant in ``avcs/goal_grounder.py``. They are
    different networks, so the loader must pick by key names rather than
    assuming one and crashing with a shape error on the other.
    """
    keys = set(state)
    if not keys:
        raise TypeError("grounding checkpoint is missing its model state dictionary")
    is_fpn = any(key.startswith(("lateral_", "decode_")) for key in keys)
    is_flat = any(
        key.startswith(("head_conv1.", "heatmap_head.", "existence_head.")) for key in keys
    )
    if is_fpn and not is_flat:
        return GoalConditionedGrounder
    if is_flat and not is_fpn:
        from avcs.goal_grounder import GoalConditionedGrounder as _FlatGrounder

        return _FlatGrounder
    raise ValueError(
        "grounding checkpoint architecture is ambiguous: expected either "
        "FPN keys (lateral_*/decode_*) or flat-head keys "
        "(head_conv1./heatmap_head./existence_head.), found "
        f"{'both' if is_fpn and is_flat else 'neither'}"
    )


def load_grounding_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"  # noqa: UP007
) -> GoalConditionedGrounder:
    """Load checkpoints carrying real-demonstration training provenance.

    Supports schema 1 legacy checkpoints and schema 2 (with full evaluation
    provenance), and dispatches to whichever GoalConditionedGrounder
    architecture the checkpoint was trained with.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("grounding checkpoint must be a metadata dictionary")

    if checkpoint.get("model_type") != GROUNDING_MODEL_TYPE:
        raise ValueError("grounding checkpoint has invalid 'model_type' provenance")

    schema_version = checkpoint.get("schema_version")
    if schema_version not in SUPPORTED_GROUNDING_CHECKPOINT_SCHEMAS:
        raise ValueError("grounding checkpoint has invalid 'schema_version' provenance")

    dataset_provenance = checkpoint.get("dataset_provenance")
    if schema_version == LEGACY_GROUNDING_CHECKPOINT_SCHEMA:
        if dataset_provenance != LEGACY_GROUNDING_DATASET_PROVENANCE:
            raise ValueError("grounding checkpoint has invalid 'schema_version' provenance")
    elif schema_version == GROUNDING_CHECKPOINT_SCHEMA:
        if dataset_provenance not in GROUNDING_CHECKPOINT_DATASET_PROVENANCES:
            raise ValueError("grounding checkpoint has invalid 'dataset_provenance' provenance")

    training_samples = checkpoint.get("training_samples")
    if (
        isinstance(training_samples, bool)
        or not isinstance(training_samples, int)
        or training_samples <= 0
    ):
        raise ValueError("grounding checkpoint reports no training demonstrations")
    validation_samples = checkpoint.get("validation_samples")
    if (
        isinstance(validation_samples, bool)
        or not isinstance(validation_samples, int)
        or validation_samples <= 0
    ):
        raise ValueError("grounding checkpoint reports no validation demonstrations")

    if schema_version == LEGACY_GROUNDING_CHECKPOINT_SCHEMA:
        checkpoint = migrate_grounding_checkpoint(checkpoint)

    presence_threshold = checkpoint.get("target_presence_threshold")
    if presence_threshold is None:
        presence_threshold = 0.5
    if (
        isinstance(presence_threshold, bool)
        or not isinstance(presence_threshold, (int, float))
        or not math.isfinite(float(presence_threshold))
        or not 0.0 < float(presence_threshold) <= 1.0
    ):
        raise ValueError("grounding checkpoint has no valid calibrated target-presence threshold")
    checkpoint["target_presence_threshold"] = float(presence_threshold)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("grounding checkpoint is missing its model state dictionary")
    input_size = checkpoint.get("input_size", DEFAULT_INPUT_SIZE)
    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size <= 0:
        raise TypeError("grounding checkpoint has invalid input_size")

    grounder_cls = _select_grounder_architecture(state_dict)
    if grounder_cls is GoalConditionedGrounder:
        model = grounder_cls(input_size=input_size).to(device)
    else:
        model = grounder_cls(
            feature_dim=checkpoint.get("feature_dim", 128), input_size=input_size
        ).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"grounding checkpoint does not match {grounder_cls.__module__}."
            f"{grounder_cls.__name__} (strict load failed): {exc}"
        ) from exc
    model.eval()
    metadata = {
        "schema_version": checkpoint["schema_version"],
        "model_type": checkpoint["model_type"],
        **{
            key: checkpoint[key]
            for key in (
                "dataset_provenance",
                "training_samples",
                "validation_samples",
                "held_out_test_samples",
                "target_presence_threshold",
                "target_presence_calibrated",
                "best_val_accuracy_pct",
                "best_val_median_error_px",
                "split_seed",
                "legacy_schema_version",
            )
            if key in checkpoint
        },
    }
    model._mark_loaded_checkpoint(metadata)
    return model
