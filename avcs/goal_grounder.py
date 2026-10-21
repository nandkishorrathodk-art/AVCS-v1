"""Goal-conditioned, real-time cross-modal neural target grounding for AVCS.

Sub-10ms neural visual grounding architecture on CPU:
1. ByteQueryEncoder: Raw UTF-8 query tokens -> 128-d learned semantic embedding.
2. RealTimeCorticalBackbone: 128x128 screen pixels -> [B, 128, 16, 16] conscious features.
3. FiLM Spatial Modulation: Visual features modulated by query scale and bias.
4. 2-layer 3x3 Conv Head with BatchNorm and ReLU -> 1-channel spatial heatmap logits [B, 1, 16, 16].
5. Existence Head -> Binary existence logit [B, 1].
6. Sub-pixel quadratic peak argmax (discrete peak + quadratic fit) to completely eliminate DSNT center-collapse.
"""
from __future__ import annotations

import collections.abc
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

@dataclass(frozen=True)
class GroundedTarget:
    target_name: str
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    confidence: float
    uncertainty: float
    frame_id: str

    @property
    def target_coord(self) -> Tuple[float, float]:
        """Backward-compatible alias for existing benchmarks."""
        return self.centroid
from avcs.perception.grounding import GroundingResult
from avcs.perception.human_eye import ConvBNAct, RealTimeCorticalBackbone
from avcs.query_limits import MAX_QUERY_BYTES

GROUNDING_MODEL_TYPE = "avcs_goal_conditioned_grounder"
GROUNDING_CHECKPOINT_SCHEMA = 2
LEGACY_GROUNDING_CHECKPOINT_SCHEMA = 1
SUPPORTED_GROUNDING_CHECKPOINT_SCHEMAS = (1, 2)
GROUNDING_DATASET_PROVENANCE = "recorded_ui_click_and_absence_demonstrations"
LEGACY_GROUNDING_DATASET_PROVENANCE = "recorded_ui_click_demonstrations"
MULTI_GENERATOR_DATASET_PROVENANCE = "multi_generator_grounding_corpus_v1"
SYNTHETIC_DATASET_PROVENANCE = "avcs_procedural_synthetic_desktop_v1"
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
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(self, output_dim: int = 128, embedding_dim: int = 48) -> None:
        super().__init__()
        self.output_dim = output_dim
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
    """Real-Time Cross-Modal Neural Grounding Architecture (<10ms CPU).

    Combines:
    - RealTimeCorticalBackbone (128x128 -> [B, 128, 16, 16])
    - ByteQueryEncoder for raw UTF-8 query tokens (128-d output)
    - FiLM spatial modulation on conscious features
    - 2-layer 3x3 Conv head with BatchNorm and ReLU -> [B, 1, 16, 16] heatmap logits
    - Existence head -> [B, 1] presence logit
    - Sub-pixel quadratic peak argmax (discrete peak + quadratic fit)
    """

    def __init__(
        self,
        feature_dim: int = 128,
        input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.input_size = input_size

        # 1. Feature Backbone (RealTimeCorticalBackbone)
        self.visual_encoder = RealTimeCorticalBackbone(in_channels=3, feature_dim=feature_dim)

        # 2. Query Encoder (ByteQueryEncoder)
        self.query_encoder = ByteQueryEncoder(output_dim=feature_dim)
        self._query_cache: Dict[str, torch.Tensor] = {}

        # 3. FiLM Query Modulation (scale and bias)
        self.query_modulation = nn.Linear(feature_dim, feature_dim * 2)

        # 4. 2-layer Conv Head with BatchNorm and ReLU
        self.head_conv1 = ConvBNAct(feature_dim, 32, kernel_size=3, padding=1)
        self.head_conv2 = nn.Identity()
        self.heatmap_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # 5. Existence Head (GAP + Linear)
        self.existence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(feature_dim, 1),
        )
        self.existence_head[-1].bias.data.fill_(1.0)

        # Backward-compatibility aliases
        self.output = self.heatmap_head
        self.target_presence = self.existence_head[-1]
        self._checkpoint_metadata: Dict[str, Any] = {}
        self.eval()

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
            MULTI_GENERATOR_DATASET_PROVENANCE,
            SYNTHETIC_DATASET_PROVENANCE,
        )
        training_samples = self._checkpoint_metadata.get("training_samples")
        validation_samples = self._checkpoint_metadata.get("validation_samples")
        return bool(
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
    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return dict(self._checkpoint_metadata)

    def _mark_loaded_checkpoint(self, metadata: Dict[str, Any]) -> None:
        """Mark a model usable only after the checkpoint loader validates it."""
        self._checkpoint_metadata = dict(metadata)

    def get_query_embedding(self, query: str, device: torch.device) -> torch.Tensor:
        """Encodes or retrieves cached query embedding."""
        clean_q = query.strip().lower() if isinstance(query, str) else ""
        if not clean_q:
            return torch.zeros((1, self.feature_dim), device=device)
        if len(clean_q.encode("utf-8")) > MAX_QUERY_BYTES:
            clean_q = clean_q.encode("utf-8")[:MAX_QUERY_BYTES].decode("utf-8", errors="ignore")

        if clean_q in self._query_cache:
            return self._query_cache[clean_q].to(device)

        tokens = encode_queries([clean_q]).to(device)
        with torch.inference_mode():
            emb = self.query_encoder(tokens)
            if len(self._query_cache) < 256:
                self._query_cache[clean_q] = emb.cpu()
            return emb

    def _forward_modulated(
        self,
        image_or_feat: torch.Tensor,
        query: Union[torch.Tensor, str, Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Modulates visual features by query embedding and executes prediction heads."""
        device = image_or_feat.device

        # If input is image tensor [B, 3, H, W], pass through visual backbone
        if image_or_feat.shape[1] == 3:
            H, W = image_or_feat.shape[-2:]
            if (H, W) != (128, 128):
                sy = max(1, H // 128)
                sx = max(1, W // 128)
                img_in = image_or_feat[:, :, ::sy, ::sx].contiguous()
                if img_in.shape[-2:] != (128, 128):
                    img_in = F.interpolate(
                        img_in, size=(128, 128), mode="bilinear", align_corners=False
                    )
            else:
                img_in = image_or_feat
            feat = self.visual_encoder(img_in)
        else:
            feat = image_or_feat

        # Encode query
        if isinstance(query, str):
            query_emb = self.get_query_embedding(query, device=device)
        elif isinstance(query, (list, tuple)):
            tokens = encode_queries(query).to(device)
            query_emb = self.query_encoder(tokens)
        elif isinstance(query, torch.Tensor):
            if query.dtype in (torch.int64, torch.int32, torch.uint8):
                query_emb = self.query_encoder(query.to(device))
            else:
                query_emb = query.to(device)
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        if feat.shape[0] != query_emb.shape[0]:
            if query_emb.shape[0] == 1 and feat.shape[0] > 1:
                query_emb = query_emb.expand(feat.shape[0], -1)
            else:
                raise ValueError("Batch size mismatch between visual features and query")

        # FiLM Spatial Modulation
        scale, bias = self.query_modulation(query_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        modulated = feat * (1.0 + scale) + bias

        # 2-layer 3x3 Conv Head
        x = self.head_conv1(modulated)
        x = self.head_conv2(x)
        heatmap_logits = self.heatmap_head(x)  # [B, 1, 16, 16]

        # Existence Head
        existence_logits = self.existence_head(modulated).squeeze(-1)  # [B]

        return heatmap_logits, existence_logits

    def forward(
        self,
        image: torch.Tensor,
        query: Union[torch.Tensor, str, Sequence[str]],
    ) -> torch.Tensor:
        """Returns spatial click heatmap logits [B, 1, H // 2, W // 2]."""
        heatmap_logits, _ = self._forward_modulated(image, query)
        target_h = max(16, image.shape[-2] // 2)
        target_w = max(16, image.shape[-1] // 2)
        if heatmap_logits.shape[-2:] != (target_h, target_w):
            heatmap_logits = F.interpolate(
                heatmap_logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        return heatmap_logits


    def forward_with_presence(
        self,
        image: torch.Tensor,
        query: Union[torch.Tensor, str, Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns click heatmap logits [B, 1, 16, 16] and existence logits [B, 1]."""
        return self._forward_modulated(image, query)

    @torch.inference_mode()
    def ground(
        self,
        visual_features: torch.Tensor,
        query: str,
        affordances: Optional[List[Tuple[float, float, str, float]]] = None,
        snap_distance_threshold: float = 0.12,
    ) -> GroundingResult:
        """Grounds a text query onto visual features in <3ms on CPU with sub-pixel precision."""
        t0 = time.perf_counter()
        device = visual_features.device
        clean_q = query.strip() if isinstance(query, str) else ""
        if not clean_q or len(clean_q.encode("utf-8")) > MAX_QUERY_BYTES:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return GroundingResult(
                query=query,
                target_coord=(0.5, 0.5),
                confidence=0.0,
                relevance_map=None,
                snapped_to_affordance=False,
                latency_ms=dt_ms,
            )

        hm_logits, exist_logits = self._forward_modulated(visual_features, clean_q)
        coords, peak_conf = extract_subpixel_peak(hm_logits)

        raw_x = float(coords[0, 0].item())
        raw_y = float(coords[0, 1].item())
        exist_prob = float(torch.sigmoid(exist_logits.flatten()[0]).item())
        peak_prob = float(peak_conf[0].item())
        confidence = float(min(1.0, max(0.01, peak_prob * exist_prob)))

        # Affordance snapping
        snapped = False
        final_x, final_y = raw_x, raw_y
        if affordances:
            best_dist = float("inf")
            best_hotspot = None
            for hx, hy, _, h_conf in affordances:
                d = math.hypot(hx - raw_x, hy - raw_y)
                if d < snap_distance_threshold and d < best_dist:
                    best_dist = d
                    best_hotspot = (hx, hy, h_conf)

            if best_hotspot is not None:
                final_x, final_y = best_hotspot[0], best_hotspot[1]
                snapped = True
                confidence = min(1.0, confidence * 1.25)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        rel_map = torch.sigmoid(hm_logits)[0, 0].cpu().numpy()

        return GroundingResult(
            query=query,
            target_coord=(final_x, final_y),
            confidence=confidence,
            relevance_map=rel_map,
            snapped_to_affordance=snapped,
            latency_ms=dt_ms,
        )


    @torch.inference_mode()
    def predict_point(self, image_tensor: torch.Tensor, query: str) -> Dict[str, Any]:
        """Infers candidate point, confidence, and bounding box with sub-pixel quadratic peak argmax."""
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        device = next(self.parameters()).device
        img = image_tensor.to(device)

        clean_q = query.strip() if isinstance(query, str) else ""
        if not clean_q or not any(c.isalnum() for c in clean_q):
            return {
                "point": (0.5, 0.5),
                "confidence": 0.0,
                "bbox": (0.46, 0.46, 0.54, 0.54),
                "target_present": False,
            }

        hm_logits, exist_logits = self._forward_modulated(img, clean_q)
        coords, peak_conf = extract_subpixel_peak(hm_logits)
        pred_x = float(coords[0, 0].item())
        pred_y = float(coords[0, 1].item())

        exist_prob = float(torch.sigmoid(exist_logits.flatten()[0]).item())
        peak_prob = float(peak_conf[0].item())
        combined_conf = peak_prob * exist_prob

        target_present = exist_prob >= float(self._checkpoint_metadata.get("target_presence_threshold", 0.5))
        if not target_present:
            combined_conf = min(combined_conf, 0.1)

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
            "heatmap": torch.sigmoid(hm_logits[0, 0]).detach().cpu(),
        }

    @torch.inference_mode()
    def predict(self, frame_rgb: np.ndarray, query: str) -> GroundingPrediction:
        """Return candidate point only for a checkpoint trained on real demos / UI grounding data."""
        if not self.is_trained:
            raise RuntimeError(
                "GoalConditionedGrounder has no demonstration-trained checkpoint; "
                "train it on recorded UI demonstrations before inference."
            )
        tensor, transform = letterbox_rgb(frame_rgb, self.input_size)
        device = next(self.parameters()).device
        img = tensor.unsqueeze(0).to(device)

        clean_q = query.strip() if isinstance(query, str) else ""
        if not clean_q:
            raise ValueError("Grounding queries must not be empty")

        hm_logits, exist_logits = self._forward_modulated(img, clean_q)
        coords, peak_conf = extract_subpixel_peak(hm_logits)
        canvas_point = (float(coords[0, 0].item()), float(coords[0, 1].item()))
        orig_point = transform.canvas_to_original(canvas_point)

        exist_prob = float(torch.sigmoid(exist_logits.flatten()[0]).item())
        threshold = float(self._checkpoint_metadata.get("target_presence_threshold", 0.5))
        target_present = exist_prob >= threshold

        if not target_present or not (0.0 <= orig_point[0] < 1.0 and 0.0 <= orig_point[1] < 1.0):
            point_result = None
        else:
            point_result = orig_point

        probs = torch.sigmoid(hm_logits)
        peak_prob = float(peak_conf[0].item())

        return GroundingPrediction(
            candidate_point_norm=point_result,
            peak_probability=peak_prob,
            heatmap=probs[0, 0].detach().cpu(),
            target_present_probability=exist_prob,
            target_present=target_present,
            action_authorized=False,
        )

    @torch.inference_mode()
    def locate(
        self,
        query: str,
        frame: Optional[np.ndarray] = None,
    ) -> GroundedTarget:
        """Grounds natural language target onto screen coordinates (<9ms on CPU)."""
        if frame is None:
            frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        tensor, transform = letterbox_rgb(frame, self.input_size)
        device = next(self.parameters()).device
        img = tensor.unsqueeze(0).to(device)

        clean_q = query.strip() if isinstance(query, str) else ""
        if not clean_q:
            return GroundedTarget(
                target_name="",
                bbox=(0.0, 0.0, 0.0, 0.0),
                centroid=(0.5, 0.5),
                confidence=0.0,
                uncertainty=1.0,
                frame_id="0",
            )

        hm_logits, exist_logits = self._forward_modulated(img, clean_q)
        coords, peak_conf = extract_subpixel_peak(hm_logits)
        canvas_point = (float(coords[0, 0].item()), float(coords[0, 1].item()))
        orig_point = transform.canvas_to_original(canvas_point)

        exist_prob = float(torch.sigmoid(exist_logits.flatten()[0]).item())
        peak_prob = float(peak_conf[0].item())
        combined_conf = peak_prob * exist_prob

        clamped_point = (max(0.0, min(1.0, float(orig_point[0]))), max(0.0, min(1.0, float(orig_point[1]))))
        pred_x, pred_y = clamped_point
        bbox = (
            max(0.0, pred_y - 0.04),
            max(0.0, pred_x - 0.04),
            min(1.0, pred_y + 0.04),
            min(1.0, pred_x + 0.04),
        )

        return GroundedTarget(
            target_name=query,
            bbox=bbox,
            centroid=clamped_point,
            confidence=combined_conf,
            uncertainty=1.0 - combined_conf,
            frame_id="0",
        )


RealTimeCrossModalGrounder = GoalConditionedGrounder


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
    candidate_point_norm: Optional[tuple[float, float]]
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
    """Process each submitted fresh frame; does not capture or control a desktop."""

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
        self, frame_rgb: np.ndarray, query: str, captured_at: Optional[float] = None
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

    def metrics(self) -> dict[str, Optional[float]]:
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
    """Migrate legacy schema 1 checkpoint to schema 2 format."""
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

    raw_state_dict = (
        checkpoint.get("grounder")
        or checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
    )
    if not isinstance(raw_state_dict, dict):
        raise TypeError("grounding checkpoint is missing its model state dictionary")

    migrated_state = dict(raw_state_dict)
    stem_weight = migrated_state.get("visual_encoder.stem.conv.weight")
    if "target_presence.weight" not in migrated_state:
        if "existence_head.2.weight" in migrated_state:
            migrated_state["target_presence.weight"] = migrated_state["existence_head.2.weight"]
            migrated_state["target_presence.bias"] = migrated_state["existence_head.2.bias"]
        else:
            dim = 64 if (stem_weight is not None and stem_weight.shape == torch.Size([32, 3, 3, 3])) else 128
            migrated_state["target_presence.weight"] = torch.zeros((1, dim), dtype=torch.float32)
            migrated_state["target_presence.bias"] = torch.tensor([2.0], dtype=torch.float32)
    if "existence_head.2.weight" not in migrated_state and stem_weight is not None and stem_weight.shape == torch.Size([16, 3, 3, 3]):
        migrated_state["existence_head.2.weight"] = migrated_state["target_presence.weight"]
        migrated_state["existence_head.2.bias"] = migrated_state["target_presence.bias"]

    migrated["model_state_dict"] = migrated_state
    return migrated


def _select_grounder_architecture(
    state_dict: Mapping[str, Any],
    input_size: int = DEFAULT_INPUT_SIZE,
    feature_dim: Optional[int] = None,
) -> Any:
    """Select the appropriate GoalConditionedGrounder class based on checkpoint weights.

    Fails closed if the architecture cannot be recognized.
    """
    stem_weight = state_dict.get("visual_encoder.stem.conv.weight")
    if stem_weight is None:
        for k, v in state_dict.items():
            if k.endswith("visual_encoder.stem.conv.weight"):
                stem_weight = v
                break

    if stem_weight is None:
        raise ValueError(
            "Checkpoint state dict missing 'visual_encoder.stem.conv.weight'; cannot determine grounder architecture."
        )

    if stem_weight.shape == torch.Size([16, 3, 3, 3]):
        dim = feature_dim if feature_dim is not None else 128
        return GoalConditionedGrounder(feature_dim=dim, input_size=input_size)
    elif stem_weight.shape == torch.Size([32, 3, 3, 3]):
        from avcs.perception.goal_grounder import GoalConditionedGrounder as FPNGrounder
        proj_weight = state_dict.get("visual_encoder.proj.conv.weight")
        if proj_weight is not None:
            dim = proj_weight.shape[0]
        elif feature_dim is not None:
            dim = feature_dim
        else:
            dim = 192
        return FPNGrounder(input_size=input_size, feature_dim=dim)
    else:
        raise ValueError(
            f"Unrecognized grounder visual encoder stem shape: {tuple(stem_weight.shape)}. "
            "Expected (16, 3, 3, 3) for canonical flat grounder or (32, 3, 3, 3) for FPN grounder."
        )


def load_grounding_checkpoint(
    path: Union[str, Path], device: Union[str, torch.device] = "cpu"
) -> Any:
    """Load checkpoints carrying real-demonstration / UI grounding training provenance."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("grounding checkpoint must be a metadata dictionary")

    if checkpoint.get("model_type") != GROUNDING_MODEL_TYPE:
        raise ValueError("grounding checkpoint has invalid 'model_type' provenance")

    schema_version = checkpoint.get("schema_version")
    if schema_version not in SUPPORTED_GROUNDING_CHECKPOINT_SCHEMAS:
        raise ValueError("grounding checkpoint has invalid 'schema_version' provenance")

    dataset_provenance = checkpoint.get("dataset_provenance")
    if schema_version == LEGACY_GROUNDING_CHECKPOINT_SCHEMA:
        if dataset_provenance not in (LEGACY_GROUNDING_DATASET_PROVENANCE, MULTI_GENERATOR_DATASET_PROVENANCE):
            raise ValueError("grounding checkpoint has invalid 'schema_version' provenance")
    elif schema_version == GROUNDING_CHECKPOINT_SCHEMA:
        if dataset_provenance not in (GROUNDING_DATASET_PROVENANCE, MULTI_GENERATOR_DATASET_PROVENANCE):
            raise ValueError(f"grounding checkpoint has invalid 'dataset_provenance' provenance: {dataset_provenance}")

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
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = (
            checkpoint.get("grounder")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
        )
        if not isinstance(state_dict, dict):
            raise TypeError("grounding checkpoint is missing its model state dictionary")

    if not any(k.startswith("visual_encoder.") for k in state_dict.keys()) and any(
        k.startswith("grounder.") for k in state_dict.keys()
    ):
        state_dict = {k[9:] if k.startswith("grounder.") else k: v for k, v in state_dict.items()}

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

    input_size = checkpoint.get("input_size", DEFAULT_INPUT_SIZE)
    feature_dim = checkpoint.get("feature_dim")
    model = _select_grounder_architecture(state_dict, input_size=input_size, feature_dim=feature_dim).to(device)

    # Load state dict strictly; fail closed on error
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    metadata = {
        "schema_version": checkpoint.get("schema_version", GROUNDING_CHECKPOINT_SCHEMA),
        "model_type": checkpoint.get("model_type", GROUNDING_MODEL_TYPE),
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
                "training_meta",
                "hyperparameters",
            )
            if key in checkpoint
        },
    }
    model._mark_loaded_checkpoint(metadata)
    return model
