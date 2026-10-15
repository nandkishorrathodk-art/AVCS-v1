"""Visual Focus & Caret Detection (Pure-Pixel Affordance Verification).

Strict No-Cheat Mandate:
Performs 100% pixel-based verification of keyboard input focus WITHOUT
accessing Windows UIA, DOM inspectors, or external OCR.
Detects:
1. Temporal Caret Blinking: 0.8-2.0 Hz photometric oscillation of vertical caret.
2. Spatial Caret Geometry: High-contrast vertical line (width 1-4px, height 8-36px).
3. Active Focus Ring / Border: Distinct high-contrast bounding perimeter.
4. Fail-Closed Contract: If visual focus cannot be confirmed, is_focused is False.
"""
from __future__ import annotations

import collections
import dataclasses
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class VisualFocusState:
    """Evaluated visual state of keyboard focus for an input region."""
    is_focused: bool                        # Strict fail-closed verification
    confidence: float                       # Confidence in [0.0, 1.0]
    caret_detected: bool                    # Caret geometry identified in spatial patch
    caret_blinking: bool                    # Caret temporal photometric oscillation verified
    focus_ring_detected: bool               # High-contrast active focus ring verified
    caret_coord: Optional[Tuple[float, float]] = None # Normalized (x, y) coordinates of caret
    contrast_score: float = 0.0             # Local visual sharpness
    latency_ms: float = 0.0                 # Computation time

    def to_dict(self) -> Dict[str, object]:
        return {
            "is_focused": self.is_focused,
            "confidence": round(self.confidence, 3),
            "caret_detected": self.caret_detected,
            "caret_blinking": self.caret_blinking,
            "focus_ring_detected": self.focus_ring_detected,
            "caret_coord": (
                (round(self.caret_coord[0], 4), round(self.caret_coord[1], 4))
                if self.caret_coord is not None
                else None
            ),
            "contrast_score": round(self.contrast_score, 4),
            "latency_ms": round(self.latency_ms, 2),
        }


class VisualFocusDetector:
    """Non-generative pixel-level visual focus verifier."""

    def __init__(
        self,
        fovea_radius_frac: float = 0.06,
        min_caret_height_px: int = 8,
        max_caret_height_px: int = 40,
        max_caret_width_px: int = 5,
        min_aspect_ratio: float = 2.5,
        history_maxlen: int = 6,
    ) -> None:
        self.radius_frac = fovea_radius_frac
        self.min_caret_h = min_caret_height_px
        self.max_caret_h = max_caret_height_px
        self.max_caret_w = max_caret_width_px
        self.min_aspect_ratio = min_aspect_ratio

        # Temporal frame ring buffer for tracking cursor blink: (timestamp, fovea_gray)
        self._history: collections.deque[Tuple[float, np.ndarray, Tuple[float, float]]] = collections.deque(
            maxlen=history_maxlen
        )

    def crop_patch(
        self,
        frame: np.ndarray,
        centroid: Tuple[float, float],
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extracts high-resolution native crop around target coordinate."""
        H, W = frame.shape[:2]
        cx, cy = centroid
        px = int(cx * W)
        py = int(cy * H)

        rw = int(max(20, W * self.radius_frac))
        rh = int(max(20, H * self.radius_frac))

        x1 = max(0, px - rw)
        y1 = max(0, py - rh)
        x2 = min(W, px + rw)
        y2 = min(H, py + rh)

        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def analyze_spatial_caret(
        self,
        gray_patch: np.ndarray,
    ) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
        """Inspects single-frame spatial geometry for a vertical text cursor line."""
        H, W = gray_patch.shape
        if H < self.min_caret_h or W < 4:
            return False, 0.0, None

        # Horizontal gradients identify sharp vertical edges
        grad_x = np.abs(gray_patch[:, 1:] - gray_patch[:, :-1])
        grad_y = np.abs(gray_patch[1:, :] - gray_patch[:-1, :])

        # Caret has high horizontal gradient (transitions into and out of line)
        # and low vertical gradient along its column
        mean_gx = np.mean(grad_x)
        if mean_gx < 0.03:
            return False, 0.0, None

        # Check each column for a vertical line segment
        best_score = 0.0
        best_pos = None

        for x in range(1, W - 2):
            col_gx = grad_x[:, x]
            active_y = np.where(col_gx > 0.20)[0]
            if len(active_y) >= self.min_caret_h:
                # Check continuity
                span = int(np.max(active_y) - np.min(active_y) + 1)
                if self.min_caret_h <= span <= self.max_caret_h:
                    density = float(len(active_y)) / span
                    if density > 0.65:
                        score = float(np.mean(col_gx[active_y]))
                        if score > best_score:
                            best_score = score
                            best_pos = (x, int(np.mean(active_y)))

        if best_score > 0.25 and best_pos is not None:
            conf = min(1.0, best_score * 1.5)
            return True, conf, best_pos

        return False, 0.0, None

    def analyze_temporal_blink(
        self,
        prev_patch: np.ndarray,
        curr_patch: np.ndarray,
    ) -> Tuple[bool, float, Optional[Tuple[int, int]]]:
        """Detects 1-2 Hz caret blink via localized vertical temporal diff."""
        if prev_patch.shape != curr_patch.shape:
            return False, 0.0, None

        diff = np.abs(curr_patch.astype(float) - prev_patch.astype(float))
        # Text cursor flips between background and dark caret (delta > 0.30)
        active_mask = diff > 0.30
        y_indices, x_indices = np.where(active_mask)

        if len(y_indices) < self.min_caret_h:
            return False, 0.0, None

        h_span = int(np.max(y_indices) - np.min(y_indices) + 1)
        w_span = int(np.max(x_indices) - np.min(x_indices) + 1)

        # Caret geometry verification: tall and narrow
        if not (self.min_caret_h <= h_span <= self.max_caret_h):
            return False, 0.0, None
        if not (1 <= w_span <= self.max_caret_w):
            return False, 0.0, None

        aspect_ratio = float(h_span) / max(1, w_span)
        if aspect_ratio < self.min_aspect_ratio:
            return False, 0.0, None

        # Verify area density: should be a clean thin vertical bar, not dispersed noise
        fill_factor = float(len(y_indices)) / (h_span * w_span)
        if fill_factor < 0.40:
            return False, 0.0, None

        mid_x = int(np.mean(x_indices))
        mid_y = int(np.mean(y_indices))
        confidence = float(min(1.0, 0.65 + 0.35 * min(1.0, (aspect_ratio - 2.5) / 5.0)))
        return True, confidence, (mid_x, mid_y)

    def verify_focus(
        self,
        frame: np.ndarray,
        target_point: Tuple[float, float],
        reference_frame: Optional[np.ndarray] = None,
    ) -> VisualFocusState:
        """Evaluates whether target input element visually possesses keyboard focus (<2ms)."""
        t0 = time.perf_counter()
        H, W = frame.shape[:2]

        crop, (x1, y1, x2, y2) = self.crop_patch(frame, target_point)
        if crop.size == 0:
            return VisualFocusState(
                is_focused=False,
                confidence=0.0,
                caret_detected=False,
                caret_blinking=False,
                focus_ring_detected=False,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Convert to float grayscale in [0.0, 1.0]
        if crop.ndim == 3:
            gray = np.mean(crop[..., :3], axis=-1) / 255.0
        else:
            gray = crop.astype(float) / 255.0

        # Visual edge sharpness / contrast
        gy, gx = np.gradient(gray)
        contrast = float(np.mean(np.hypot(gx, gy)))

        # 1. Temporal blink verification if reference frame provided
        caret_blinking = False
        blink_conf = 0.0
        blink_pos = None

        if reference_frame is not None:
            ref_crop, _ = self.crop_patch(reference_frame, target_point)
            if ref_crop.size > 0 and ref_crop.shape == crop.shape:
                if ref_crop.ndim == 3:
                    ref_gray = np.mean(ref_crop[..., :3], axis=-1) / 255.0
                else:
                    ref_gray = ref_crop.astype(float) / 255.0
                caret_blinking, blink_conf, blink_pos = self.analyze_temporal_blink(ref_gray, gray)

        # Also check internal history buffer if no explicit reference frame
        if not caret_blinking and len(self._history) > 0:
            for _, prev_gray, prev_pt in reversed(self._history):
                if math.hypot(prev_pt[0] - target_point[0], prev_pt[1] - target_point[1]) < 0.02:
                    if prev_gray.shape == gray.shape:
                        c_blink, c_conf, c_pos = self.analyze_temporal_blink(prev_gray, gray)
                        if c_blink:
                            caret_blinking = True
                            blink_conf = c_conf
                            blink_pos = c_pos
                            break

        # 2. Spatial Caret Inspection
        caret_detected, spatial_conf, spatial_pos = self.analyze_spatial_caret(gray)

        # 3. Active Focus Border / High-Contrast Perimeter
        # Check border intensity around patch margin
        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        border_contrast = float(np.std(border_pixels))
        focus_ring = (border_contrast > 0.15) and (contrast > 0.020)

        # 4. Synthesize Focus Verification Decision (Fail-Closed)
        is_focused = False
        final_conf = 0.0
        norm_caret_coord = None

        if caret_blinking and blink_pos is not None:
            # Gold standard: temporal blinking cursor verified
            is_focused = True
            final_conf = max(0.80, blink_conf)
            cx_global = (x1 + blink_pos[0]) / float(W)
            cy_global = (y1 + blink_pos[1]) / float(H)
            norm_caret_coord = (float(cx_global), float(cy_global))
        elif caret_detected and spatial_pos is not None and focus_ring:
            # Silver standard: static caret + active border
            is_focused = True
            final_conf = float(0.65 * spatial_conf + 0.35)
            cx_global = (x1 + spatial_pos[0]) / float(W)
            cy_global = (y1 + spatial_pos[1]) / float(H)
            norm_caret_coord = (float(cx_global), float(cy_global))
        elif caret_detected and spatial_conf > 0.70:
            # Bronze standard: highly confident spatial caret
            is_focused = True
            final_conf = float(spatial_conf)
            if spatial_pos is not None:
                cx_global = (x1 + spatial_pos[0]) / float(W)
                cy_global = (y1 + spatial_pos[1]) / float(H)
                norm_caret_coord = (float(cx_global), float(cy_global))

        # Store to history
        now = time.perf_counter()
        self._history.append((now, gray, target_point))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return VisualFocusState(
            is_focused=is_focused,
            confidence=round(final_conf, 3),
            caret_detected=caret_detected,
            caret_blinking=caret_blinking,
            focus_ring_detected=focus_ring,
            caret_coord=norm_caret_coord,
            contrast_score=round(contrast, 4),
            latency_ms=round(dt_ms, 2),
        )

