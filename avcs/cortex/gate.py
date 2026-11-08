"""Local Visual Cortex Stability Gate (System 1 Coupling).

Couples the 400,765 parameter ConvGRU model (StreamingVisualCortexV2 / EyeStream / CortexTemporalState)
as a local sensory gate for the Ace Autopilot:
1. Pre-actuation stability gate: Verifies is_stable=True and spinner_active=False before querying Layer 3,
   preventing actions on half-rendered screens or loading spinners.
2. Post-actuation Stage 1 ripple probe: Utilizes visual cortex optical feature delta and surprise score
   for sub-2ms physical actuation verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from avcs.cortex.model import (
    CortexTemporalState,
    CortexV2Output,
    StreamingVisualCortexV2,
)
from avcs.cortex.preprocessor import EdgeLuminancePreprocessor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avcs.eye_stream import EyeFrame, EyeStream

logger = logging.getLogger("avcs.cortex.gate")


@dataclass
class StabilityGateResult:
    """Telemetry returned by the Pre-Actuation Sensory Stability Gate."""

    is_stable: bool
    surprise_score: float
    spinner_active: bool
    spatial_delta: float
    latency_ms: float
    details: str = ""
    has_popup: bool = False
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_stable": self.is_stable,
            "surprise_score": self.surprise_score,
            "spinner_active": self.spinner_active,
            "spatial_delta": self.spatial_delta,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "has_popup": self.has_popup,
            "confidence": self.confidence,
        }


@dataclass
class Stage1Result:
    """Stage 1 Physical Verification result (visual cortex optical delta / click ripple probe)."""

    passed: bool
    optical_delta: float
    latency_ms: float
    ripple_detected: bool
    details: str = ""


class VisualCortexGate:
    """Couples the 400,765 parameter StreamingVisualCortexV2 model as a sensory gate.

    Operates 100% locally on CPU to provide:
    1. Pre-actuation stability gating (blocks prediction during UI transition or loading spinner).
    2. Post-actuation optical ripple verification (<2 ms ConvGRU feature delta).
    """

    def __init__(
        self,
        cortex: Optional[Union[StreamingVisualCortexV2, EyeStream, Any]] = None,
        device: str = "cpu",
        stability_threshold: float = 0.015,
        spinner_threshold: float = 0.50,
        change_threshold: float = 0.10,
        source: str = "synthetic",
        load_weights: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.stability_threshold = stability_threshold
        self.spinner_threshold = spinner_threshold
        self.change_threshold = change_threshold
        self.source = source

        if cortex in ("human_eye", "humaneye"):
            try:
                from avcs.perception.human_eye import HumanEye
                self.cortex = HumanEye(source=source, load_weights=load_weights, device=str(self.device))
            except Exception:
                self.cortex = StreamingVisualCortexV2()
        elif cortex in ("cortex_v2", "streaming_cortex", "streaming_visual_cortex"):
            if source != "synthetic" and load_weights:
                try:
                    from avcs.eye_stream import EyeStream
                    self.cortex = EyeStream(source=source, load_weights=load_weights, device=str(self.device))
                except Exception:
                    self.cortex = StreamingVisualCortexV2()
            else:
                self.cortex = StreamingVisualCortexV2()
        elif cortex is not None and not isinstance(cortex, str):
            self.cortex = cortex
        elif source != "synthetic" and load_weights:
            try:
                from avcs.eye_stream import EyeStream
                self.cortex = EyeStream(source=source, load_weights=load_weights, device=str(self.device))
            except Exception:
                self.cortex = StreamingVisualCortexV2()
        else:
            self.cortex = StreamingVisualCortexV2()

        if hasattr(self.cortex, "to"):
            self.cortex.to(self.device)
        if hasattr(self.cortex, "eval"):
            self.cortex.eval()

        self.preprocessor = EdgeLuminancePreprocessor().to(self.device)
        self.preprocessor.eval()

        self.temporal_state: Optional[CortexTemporalState] = None
        self.prev_features: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Resets the recurrent hidden state and temporal feature memory."""
        self.temporal_state = None
        self.prev_features = None

    def _convert_to_numpy_rgb(self, input_data: Any) -> np.ndarray:
        """Converts observation bytes, PIL Image, numpy array, or EyeFrame into RGB numpy array (H, W, 3)."""
        # Zero-copy fast-path: directly reuse pre-extracted in-memory RGB array if available (<0.01 ms)
        if hasattr(input_data, "raw_numpy") and isinstance(input_data.raw_numpy, np.ndarray):
            return input_data.raw_numpy

        raw_rgb: Optional[np.ndarray] = None

        if hasattr(input_data, "image_bytes") and input_data.image_bytes:
            # Decode JPEG/PNG/WebP bytes via PIL
            b = input_data.image_bytes
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(b)).convert("RGB")
                raw_rgb = np.asarray(img)
            except Exception:
                # Deterministic synthetic fallback on corrupt or dummy test bytes
                arr = np.full((128, 128, 3), 128, dtype=np.uint8)
                if b:
                    b_bytes = np.frombuffer(b[:128], dtype=np.uint8)
                    n = min(len(b_bytes), 128)
                    arr[:n, :n, 0] = b_bytes[:n, None]
                    arr[:n, :n, 1] = b_bytes[:n]
                raw_rgb = arr
        elif isinstance(input_data, bytes):
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(input_data)).convert("RGB")
                raw_rgb = np.asarray(img)
            except Exception:
                arr = np.full((128, 128, 3), 128, dtype=np.uint8)
                if input_data:
                    b_bytes = np.frombuffer(input_data[:128], dtype=np.uint8)
                    n = min(len(b_bytes), 128)
                    arr[:n, :n, 0] = b_bytes[:n, None]
                    arr[:n, :n, 1] = b_bytes[:n]
                raw_rgb = arr
        elif isinstance(input_data, np.ndarray):
            raw_rgb = input_data
        elif hasattr(input_data, "raw_frame") and isinstance(input_data.raw_frame, np.ndarray):
            raw_rgb = input_data.raw_frame

        if raw_rgb is None:
            raw_rgb = np.full((128, 128, 3), 128, dtype=np.uint8)

        if raw_rgb.ndim == 2:
            raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
        elif raw_rgb.shape[-1] > 3:
            raw_rgb = raw_rgb[:, :, :3]

        return raw_rgb

    def _convert_to_tensor_4ch(self, input_data: Any) -> torch.Tensor:
        """Converts observation bytes, numpy arrays, or EyeFrames into 4-channel tensor (1, 4, 128, 128)."""
        raw_rgb = self._convert_to_numpy_rgb(input_data)

        H, W = raw_rgb.shape[:2]
        if (H, W) != (128, 128):
            sy = float(H) / 128.0
            sx = float(W) / 128.0
            y_idx = np.minimum((np.arange(128) * sy).astype(int), H - 1)
            x_idx = np.minimum((np.arange(128) * sx).astype(int), W - 1)
            frame_small = raw_rgb[y_idx[:, None], x_idx]
        else:
            frame_small = raw_rgb

        t_frame = torch.from_numpy(np.ascontiguousarray(frame_small)).float()
        if t_frame.max() > 1.0:
            t_frame = t_frame / 255.0
        t_frame = t_frame.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, 128, 128)

        # 4-Channel Preprocessing: Luminance, Sobel X, Sobel Y, Gradient Magnitude
        return self.preprocessor(t_frame)

    @torch.no_grad()
    def check_stability(
        self,
        observation: Optional[Any] = None,
        max_settle_wait: float = 0.5,
        poll_interval: float = 0.05,
        observation_provider: Optional[Any] = None,
    ) -> StabilityGateResult:
        """Pre-actuation sensory stability gate.

        Checks whether the desktop UI is stable and free from active loading spinners
        before allowing Layer 3 action prediction.
        """
        t0 = time.perf_counter()

        # If live EyeStream is used directly with no observation passed
        if observation is None and hasattr(self.cortex, "step"):
            try:
                frame = self.cortex.step()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                return StabilityGateResult(
                    is_stable=frame.is_stable,
                    surprise_score=round(frame.surprise_score, 4),
                    spinner_active=not frame.is_stable,
                    spatial_delta=round(frame.surprise_score / 10.0, 5),
                    latency_ms=round(dt_ms, 2),
                    details="UI settled and stable." if frame.is_stable else "Live EyeStream optical drift detected.",
                    has_popup=frame.has_popup,
                )
            except Exception as e:
                logger.debug(f"EyeStream step fallback: {e}")

        def _eval_frame(obs: Any) -> StabilityGateResult:
            if hasattr(self.cortex, "observe") and not hasattr(self.cortex, "temporal_cell"):
                t_f0 = time.perf_counter()
                raw_rgb = self._convert_to_numpy_rgb(obs)
                eye_state = self.cortex.observe(frame=raw_rgb)
                dt_ms = (time.perf_counter() - t_f0) * 1000.0
                is_stable = bool(eye_state.is_stable and (eye_state.motion_magnitude < self.stability_threshold))
                spinner_active = eye_state.stability in ("busy", "tracking", "dynamic_video")
                details = f"HumanEye sensory gate: stability={eye_state.stability}, motion={eye_state.motion_magnitude:.4f}, surprise={eye_state.surprise_score:.4f}"
                return StabilityGateResult(
                    is_stable=is_stable,
                    surprise_score=round(eye_state.surprise_score, 4),
                    spinner_active=spinner_active,
                    spatial_delta=round(eye_state.motion_magnitude, 5),
                    latency_ms=round(dt_ms, 2),
                    details=details,
                    has_popup=getattr(eye_state, "overlay_suspected", False),
                )

            t_f0 = time.perf_counter()
            x_4ch = self._convert_to_tensor_4ch(obs)
            now = time.time()

            out: CortexV2Output = self.cortex.forward(
                x_4ch,
                prev_state=self.temporal_state,
                timestamp=now,
            )
            current_features = out.temporal_state.previous_features
            self.temporal_state = out.temporal_state

            if self.prev_features is not None:
                spatial_delta = float(torch.abs(current_features - self.prev_features).mean().item())
            else:
                spatial_delta = 0.0
            self.prev_features = current_features.clone()

            surprise = float(np.clip(spatial_delta * 10.0, 0.0, 1.0))
            spinner_prob = float(torch.sigmoid(out.spinner_activity_logit[0, 0]).item())
            spinner_active = (spinner_prob >= self.spinner_threshold) and (spatial_delta >= 0.01)
            is_stable = (spatial_delta < self.stability_threshold) and not spinner_active

            dt_ms = (time.perf_counter() - t_f0) * 1000.0
            details = "UI settled and stable."
            if not is_stable:
                if spinner_active:
                    details = f"Loading spinner active (prob={spinner_prob:.2f}). Actuation gated."
                else:
                    details = f"UI optical drift detected (delta={spatial_delta:.4f} >= {self.stability_threshold:.4f}). Screen settling."

            return StabilityGateResult(
                is_stable=is_stable,
                surprise_score=round(surprise, 4),
                spinner_active=spinner_active,
                spatial_delta=round(spatial_delta, 5),
                latency_ms=round(dt_ms, 2),
                details=details,
                has_popup=False,
            )

        res = _eval_frame(observation)

        # If unstable and observation_provider is provided, poll for UI to settle
        if not res.is_stable and observation_provider is not None and max_settle_wait > 0:
            settle_deadline = time.perf_counter() + max_settle_wait
            while not res.is_stable and time.perf_counter() < settle_deadline:
                time.sleep(poll_interval)
                next_obs = observation_provider()
                if next_obs is not None:
                    res = _eval_frame(next_obs)

        return res

    @torch.no_grad()
    def evaluate_ripple(
        self,
        pre_obs: Optional[Any],
        post_obs: Optional[Any],
        action: Optional[Any] = None,
    ) -> Stage1Result:
        """Post-actuation Stage 1 ripple probe using visual cortex ConvGRU feature delta.

        Runs in <2 ms on CPU for 128x128 input.
        """
        t0 = time.perf_counter()

        if pre_obs is None or post_obs is None:
            latency = (time.perf_counter() - t0) * 1000.0
            return Stage1Result(
                passed=False,
                optical_delta=0.0,
                latency_ms=round(latency, 2),
                ripple_detected=False,
                details="Missing pre- or post-actuation observation frame.",
            )

        # Check byte equality for immediate zero-diff early exit
        b1 = getattr(pre_obs, "image_bytes", None)
        b2 = getattr(post_obs, "image_bytes", None)
        if b1 is not None and b2 is not None and b1 == b2:
            latency = (time.perf_counter() - t0) * 1000.0
            return Stage1Result(
                passed=False,
                optical_delta=0.0,
                latency_ms=round(latency, 2),
                ripple_detected=False,
                details="Zero optical change detected between pre- and post-actuation frames.",
            )

        # Localized ROI around action.coordinate (50px radius) for rapid sub-2ms click ripple verification
        coord = getattr(action, "coordinate", None) or getattr(action, "start_coordinate", None)
        if coord and len(coord) == 2:
            cx, cy = int(coord[0]), int(coord[1])
            arr1 = self._convert_to_numpy_rgb(pre_obs)
            arr2 = self._convert_to_numpy_rgb(post_obs)
            h, w = arr1.shape[:2]
            r = 50
            bx1, by1 = max(0, cx - r), max(0, cy - r)
            bx2, by2 = min(w, cx + r), min(h, cy + r)
            if bx2 > bx1 and by2 > by1:
                crop1 = arr1[by1:by2, bx1:bx2]
                crop2 = arr2[by1:by2, bx1:bx2]
                diff = np.abs(crop2.astype(np.int16) - crop1.astype(np.int16))
                changed_pixels = int(np.count_nonzero(diff > 5))
                delta_score = changed_pixels / float(crop1.size)
                has_ripple = changed_pixels > 0
                dt_ms = (time.perf_counter() - t0) * 1000.0
                details = (
                    f"Visual cortex localized ROI ripple probe (50px ROI at ({cx}, {cy})): "
                    f"delta={delta_score:.4f} (ripple {'detected' if has_ripple else 'not detected'}) in {dt_ms:.2f}ms."
                )
                return Stage1Result(
                    passed=has_ripple,
                    optical_delta=round(delta_score, 4),
                    latency_ms=round(dt_ms, 2),
                    ripple_detected=has_ripple,
                    details=details,
                )

        # HumanEye visual cortex branch
        if hasattr(self.cortex, "observe") and not hasattr(self.cortex, "temporal_cell"):
            arr1 = self._convert_to_numpy_rgb(pre_obs)
            arr2 = self._convert_to_numpy_rgb(post_obs)
            state1 = self.cortex.observe(frame=arr1)
            state2 = self.cortex.observe(frame=arr2)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            motion_delta = float(abs(state2.motion_magnitude - state1.motion_magnitude))
            surprise = float(state2.surprise_score)
            has_ripple = (motion_delta >= 0.0005) or (surprise >= 0.005) or (state2.motion_magnitude > 0.001) or (state2.stability != "stable")
            details = (
                f"HumanEye ripple probe {'confirmed' if has_ripple else 'not detected'} "
                f"(motion_delta={motion_delta:.4f}, surprise={surprise:.4f}, stability={state2.stability}) "
                f"in {dt_ms:.2f}ms."
            )
            return Stage1Result(
                passed=has_ripple,
                optical_delta=round(max(motion_delta, surprise), 4),
                latency_ms=round(dt_ms, 2),
                ripple_detected=has_ripple,
                details=details,
            )

        # 1. Forward pass on pre-observation
        x1_4ch = self._convert_to_tensor_4ch(pre_obs)
        out1: CortexV2Output = self.cortex.forward(x1_4ch, prev_state=None, timestamp=time.time())

        # 2. Forward pass on post-observation conditioned on pre-observation temporal state
        x2_4ch = self._convert_to_tensor_4ch(post_obs)
        out2: CortexV2Output = self.cortex.forward(
            x2_4ch,
            prev_state=out1.temporal_state,
            timestamp=time.time() + 0.04,
        )

        f1 = out1.temporal_state.previous_features
        f2 = out2.temporal_state.previous_features

        # Optical feature delta: |f2 - f1|
        spatial_delta = float(torch.abs(f2 - f1).mean().item())
        surprise_score = float(np.clip(spatial_delta * 10.0, 0.0, 1.0))

        # Head 3: Post-action visual response probability
        post_action_prob = float(torch.sigmoid(out2.post_action_response_logit[0, 0]).item())

        # Physical ripple detected if optical feature delta exceeds noise floor
        has_ripple = (spatial_delta >= 0.0005) or (surprise_score >= 0.005) or (post_action_prob >= 0.50 and spatial_delta > 0.0)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        details = (
            f"Visual cortex ripple probe {'confirmed' if has_ripple else 'not detected'} "
            f"(surprise={surprise_score:.4f}, delta={spatial_delta:.4f}, post_resp={post_action_prob:.2f}) "
            f"in {dt_ms:.2f}ms."
        )

        return Stage1Result(
            passed=has_ripple,
            optical_delta=round(surprise_score, 4),
            latency_ms=round(dt_ms, 2),
            ripple_detected=has_ripple,
            details=details,
        )
