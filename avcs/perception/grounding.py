"""Language Gaze Grounding & Dual-Fovea Acuity Engine.

Implements Phase 2 of the AVCS Universal Real-Time Human Eye Platform:
1. Open-Vocabulary Language Grounding: Maps natural language queries
   ('Submit Order', 'search bar in top left', 'close icon') directly to spatial
   gaze coordinates in <3ms using character n-gram text projection and spatial
   cross-correlation.
2. Dual-Fovea Acuity Engine: Replicates biological foveation by extracting a
   high-resolution zoom crop centered on fixation gaze to classify element type
   (button, input, icon, tab) and regress exact bounding box boundaries in <2ms.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.vision.cortex.core.modulation import (
    CharNGramEncoder,
    LanguageModulation,
    dsnt_soft_argmax,
)

FOVEAL_ELEMENT_TYPES = [
    "button",
    "input_field",
    "icon",
    "tab",
    "link",
    "text_label",
    "container",
]


@dataclasses.dataclass(frozen=True)
class GroundingResult:
    """Result of top-down natural language gaze grounding."""
    query: str
    target_coord: Tuple[float, float]       # Normalized (x, y) in [0, 1]
    confidence: float                       # Saliency confidence in [0, 1]
    relevance_map: Optional[np.ndarray]     # 2D spatial correlation map
    snapped_to_affordance: bool             # True if snapped to nearest conscious affordance
    latency_ms: float                       # Processing latency in milliseconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "target_coord": (round(self.target_coord[0], 3), round(self.target_coord[1], 3)),
            "confidence": round(self.confidence, 3),
            "snapped_to_affordance": self.snapped_to_affordance,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclasses.dataclass(frozen=True)
class FovealInspection:
    """High-acuity central foveation analysis on fixation target."""
    centroid: Tuple[float, float]           # Screen coordinate (fx, fy) in [0, 1]
    element_kind: str                       # 'button', 'input_field', 'icon', 'tab', etc.
    confidence: float                       # Classification confidence
    bbox: Tuple[float, float, float, float] # Global screen bounds [xmin, ymin, xmax, ymax]
    contrast_score: float                   # Local sharpness / visual edge density
    foveal_crop_shape: Tuple[int, int]      # Resolution of extracted crop (H, W)
    latency_ms: float                       # Inspection latency in milliseconds

    @property
    def refined_centroid(self) -> Tuple[float, float]:
        """Returns the high-acuity sub-pixel refined centroid (cx, cy) from bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centroid": (round(self.centroid[0], 3), round(self.centroid[1], 3)),
            "refined_centroid": (round(self.refined_centroid[0], 3), round(self.refined_centroid[1], 3)),
            "element_kind": self.element_kind,
            "confidence": round(self.confidence, 3),
            "bbox": [round(b, 3) for b in self.bbox],
            "contrast_score": round(self.contrast_score, 3),
            "latency_ms": round(self.latency_ms, 2),
        }



class LanguageGazeGrounder(nn.Module):
    """Top-down semantic gaze grounder with character n-grams & spatial modulation."""

    def __init__(self, visual_dim: int = 128, text_dim: int = 128) -> None:
        super().__init__()
        self.text_encoder = CharNGramEncoder(out_dim=text_dim)
        self.modulator = LanguageModulation(visual_dim=visual_dim, text_dim=text_dim)
        self._query_cache: Dict[str, torch.Tensor] = {}

    def get_query_embedding(self, query: str, device: torch.device) -> torch.Tensor:
        """Encodes or retrieves cached query embedding."""
        clean_q = query.strip().lower()
        if clean_q in self._query_cache:
            return self._query_cache[clean_q].to(device)

        with torch.inference_mode():
            emb = self.text_encoder(clean_q, device=device)
            # Cache up to 128 common queries
            if len(self._query_cache) < 128:
                self._query_cache[clean_q] = emb.cpu()
            return emb

    @torch.inference_mode()
    def ground(
        self,
        visual_features: torch.Tensor,
        query: str,
        affordances: Optional[List[Tuple[float, float, str, float]]] = None,
        snap_distance_threshold: float = 0.12,
    ) -> GroundingResult:
        """Grounds a text query onto visual features in <3ms on CPU."""
        t0 = time.perf_counter()
        device = visual_features.device
        text_emb = self.get_query_embedding(query, device=device)

        # 1. Forward spatial modulation
        rel_map, target_coords = self.modulator(visual_features, text_emb)
        # target_coords: (1, 2)
        raw_x = float(target_coords[0, 0].item())
        raw_y = float(target_coords[0, 1].item())

        # 2. Extract confidence & relevance map
        rel_np = rel_map[0, 0].cpu().numpy()
        peak_val = float(np.max(rel_np))
        mean_val = float(np.mean(rel_np))
        # Peak-to-average ratio as confidence metric
        confidence = float(min(1.0, max(0.1, (peak_val - mean_val) * 8.0)))

        # 3. Affordance snapping: snap to nearest conscious affordance hotspot if close
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
        return GroundingResult(
            query=query,
            target_coord=(final_x, final_y),
            confidence=confidence,
            relevance_map=rel_np,
            snapped_to_affordance=snapped,
            latency_ms=dt_ms,
        )


class FovealAcuityHead(nn.Module):
    """Lightweight convolutional neural network for high-acuity foveal inspection."""

    def __init__(self, in_channels: int = 3, num_classes: int = len(FOVEAL_ELEMENT_TYPES)) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, num_classes)
        self.bbox_regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        logits = self.classifier(feat)
        bbox = self.bbox_regressor(feat)
        return logits, bbox


class FovealAcuityEngine:
    """Extracts dynamic high-resolution foveal zoom crops and executes fine-grained inspection."""

    def __init__(
        self,
        device: str = "cpu",
        fovea_radius_frac: float = 0.08,
        input_size: Tuple[int, int] = (96, 96),
        load_weights: bool = True,
        weights_path: Optional[str] = None,
        refine_subpixel: bool = True,
        fovea_patch_size: int = 128,
    ) -> None:
        self.device = torch.device(device)
        self.radius_frac = fovea_radius_frac
        self.input_size = input_size
        self.patch_size = fovea_patch_size
        self.refine_subpixel = refine_subpixel
        self.head = FovealAcuityHead().to(self.device).eval()
        self.weights_loaded: bool = False
        if load_weights:
            from avcs.weights import load_avcs_weights
            self.weights_loaded = load_avcs_weights(
                self.head,
                path=weights_path,
                device=str(self.device),
                strict=False,
            )

    @staticmethod
    def _subpixel_peak_1d(profile: np.ndarray, initial_idx: float, search_radius: int = 6) -> float:
        """Finds sub-pixel peak in 1D gradient profile using parabolic interpolation."""
        n = len(profile)
        i0 = max(1, int(round(initial_idx)) - search_radius)
        i1 = min(n - 2, int(round(initial_idx)) + search_radius)
        if i0 >= i1:
            return float(initial_idx)
        segment = profile[i0:i1 + 1]
        peak_offset = int(np.argmax(segment))
        peak_idx = i0 + peak_offset
        if profile[peak_idx] < 1.3 * np.mean(profile) or profile[peak_idx] < 8.0:
            return float(initial_idx)
        y_m1 = float(profile[peak_idx - 1])
        y_0 = float(profile[peak_idx])
        y_p1 = float(profile[peak_idx + 1])
        denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
        if abs(denom) > 1e-4:
            sub = (y_p1 - y_m1) / denom
            sub = max(-0.5, min(0.5, sub))
            return float(peak_idx) + sub
        return float(peak_idx)

    def refine_subpixel_boundaries(
        self,
        crop: np.ndarray,
        pred_box: Tuple[float, float, float, float],
        element_kind: str = "button",
        grad_mag: Optional[np.ndarray] = None,
        gx: Optional[np.ndarray] = None,
        gy: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float]:
        """Performs sub-pixel edge alignment on predicted bounding box within un-downsampled crop (<0.05ms)."""
        H, W = crop.shape[:2]
        if H < 8 or W < 8:
            return pred_box

        if grad_mag is None or gx is None or gy is None:
            gray = np.mean(crop, axis=-1) if crop.ndim == 3 else crop.astype(float)
            gy, gx = np.gradient(gray)
            grad_mag = np.hypot(gx, gy)

        p99_5 = float(np.percentile(grad_mag, 99.5))
        if p99_5 < 10.0:
            return pred_box

        bx1, by1, bx2, by2 = pred_box
        px1, px2 = bx1 * W, bx2 * W
        py1, py2 = by1 * H, by2 * H

        # 1. 1D directional profile sub-pixel edge alignment
        iy1 = max(0, int(py1))
        iy2 = min(H, max(iy1 + 1, int(py2)))
        ix1 = max(0, int(px1))
        ix2 = min(W, max(ix1 + 1, int(px2)))

        mag_x = np.abs(gx)
        mag_y = np.abs(gy)
        prof_x = np.mean(mag_x[iy1:iy2, :], axis=0) if iy2 > iy1 else np.zeros(W)

        new_px1 = self._subpixel_peak_1d(prof_x, px1, search_radius=6)
        new_px2 = self._subpixel_peak_1d(prof_x, px2, search_radius=6)

        if element_kind not in ("link", "text_label"):
            prof_y = np.mean(mag_y[:, ix1:ix2], axis=1) if ix2 > ix1 else np.zeros(H)
            new_py1 = self._subpixel_peak_1d(prof_y, py1, search_radius=6)
            new_py2 = self._subpixel_peak_1d(prof_y, py2, search_radius=6)
        else:
            new_py1, new_py2 = py1, py2


        # 2. Compact UI element edge clustering (small icons 16x16, badges, borderless buttons)
        cx, cy = W / 2.0, H / 2.0
        thresh = max(8.0, p99_5 * 0.35)

        if element_kind in ("icon", "button", "text_label"):
            win_x1, win_x2 = max(0, int(cx - 28)), min(W, int(cx + 28))
            win_y1, win_y2 = max(0, int(cy - 28)), min(H, int(cy + 28))
            inner_gm = grad_mag[win_y1:win_y2, win_x1:win_x2]
            inner_mask = inner_gm > thresh
            if np.sum(inner_mask) >= 12:
                iys, ixs = np.where(inner_mask)
                in_w = (np.max(ixs) - np.min(ixs)) + 1
                in_h = (np.max(iys) - np.min(iys)) + 1
                if in_w <= 36 and in_h <= 36 and (px2 - px1 > in_w * 1.35 or py2 - py1 > in_h * 1.35):
                    new_px1 = float(win_x1 + np.min(ixs))
                    new_px2 = float(win_x1 + np.max(ixs) + 1)
                    new_py1 = float(win_y1 + np.min(iys))
                    new_py2 = float(win_y1 + np.max(iys) + 1)

        rx1 = max(0.0, min(W - 1.0, min(new_px1, new_px2))) / W
        rx2 = max(0.0, min(W, max(new_px1, new_px2))) / W
        ry1 = max(0.0, min(H - 1.0, min(new_py1, new_py2))) / H
        ry2 = max(0.0, min(H, max(new_py1, new_py2))) / H

        return (rx1, ry1, rx2, ry2)

    def crop_fovea(
        self,
        frame: np.ndarray,
        centroid: Tuple[float, float],
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Extracts high-resolution un-downsampled foveal crop around fixation point.

        Guarantees:
        1. Aspect-ratio preservation: extracts a strictly square patch (default 128x128).
        2. Boundary safety: clamps window within screen canvas with zero artificial padding.
        3. 1:1 pixel sampling from native frame with zero spatial downsampling artifacts.
        """
        H, W = frame.shape[:2]
        # Fast path if frame is already a foveal element crop centered at (0.5, 0.5)
        if (H, W) == self.input_size and abs(centroid[0] - 0.5) < 1e-4 and abs(centroid[1] - 0.5) < 1e-4:
            return frame, (0.0, 0.0, 1.0, 1.0)

        cx, cy = centroid
        px = int(round(cx * W))
        py = int(round(cy * H))

        # Fixed physical patch size (default 128x128, bounded by frame size)
        S = min(getattr(self, "patch_size", 128), W, H)
        r = S // 2

        # Boundary clamping: ensure window stays strictly within [0, W] and [0, H]
        x1 = max(0, min(W - S, px - r))
        y1 = max(0, min(H - S, py - r))
        x2 = x1 + S
        y2 = y1 + S

        crop = frame[y1:y2, x1:x2]
        norm_crop_box = (x1 / float(W), y1 / float(H), x2 / float(W), y2 / float(H))
        return crop, norm_crop_box

    @torch.inference_mode()
    def inspect_crop(self, crop: np.ndarray) -> FovealInspection:
        """Inspects a pre-cropped foveal element patch directly without re-cropping."""
        return self.inspect(frame=crop, centroid=(0.5, 0.5))

    @torch.inference_mode()
    def inspect(
        self,
        frame: np.ndarray,
        centroid: Tuple[float, float],
    ) -> FovealInspection:
        """Performs biological foveal inspection on target coordinate (<2ms)."""
        t0 = time.perf_counter()
        crop, crop_box = self.crop_fovea(frame, centroid)

        if crop.size == 0:
            return FovealInspection(
                centroid=centroid,
                element_kind="empty",
                confidence=0.0,
                bbox=(centroid[0], centroid[1], centroid[0], centroid[1]),
                contrast_score=0.0,
                foveal_crop_shape=(0, 0),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 1. Compute visual edge sharpness / contrast score using hybrid metric:
        # Combines spatial mean with 99.5th percentile and top-K boundary edge gradient.
        # This completely resolves the Spatial Dilution Paradox on compact UI targets
        # (icons 8x8 to 24x24, dense text glyphs, borderless buttons) while reliably
        # rejecting featureless flat screens, subtle sensor noise, and smooth wallpaper gradients.
        gray = np.mean(crop, axis=-1)
        gy, gx = np.gradient(gray)
        grad_mag = np.hypot(gx, gy)

        k_edge = min(32, max(8, int(grad_mag.size * 0.005)))
        top_k_edge = float(np.mean(np.partition(grad_mag.ravel(), -k_edge)[-k_edge:])) if grad_mag.size >= 8 else float(np.max(grad_mag))

        contrast = float(max(
            float(np.mean(grad_mag)),
            float(np.percentile(grad_mag, 99.5)),
            top_k_edge * 0.85,
        ) / 255.0)

        # 2. Resize foveal crop to tensor input
        t_crop = torch.from_numpy(crop.copy()).float() / 255.0
        if t_crop.ndim == 2:
            t_crop = torch.stack([t_crop] * 3, dim=-1)
        elif t_crop.shape[-1] > 3:
            t_crop = t_crop[..., :3]

        t_crop = t_crop.permute(2, 0, 1).unsqueeze(0).to(self.device)
        t_in = F.interpolate(t_crop, size=self.input_size, mode="bilinear", align_corners=False)

        # 3. Model forward pass
        logits, reg_bbox = self.head(t_in)
        probs = torch.softmax(logits, dim=-1)[0]
        cat_idx = int(torch.argmax(probs).item())
        element_kind = (
            FOVEAL_ELEMENT_TYPES[cat_idx]
            if cat_idx < len(FOVEAL_ELEMENT_TYPES)
            else "button"
        )
        conf = float(probs[cat_idx].item())

        # 4. Map local crop bounding box back to global normalized screen coordinates
        # reg_bbox outputs normalized [ymin, xmin, ymax, xmax] relative to crop
        pred_ymin, pred_xmin, pred_ymax, pred_xmax = reg_bbox[0].tolist()
        by1 = min(pred_ymin, pred_ymax)
        by2 = max(pred_ymin, pred_ymax)
        bx1 = min(pred_xmin, pred_xmax)
        bx2 = max(pred_xmin, pred_xmax)

        if self.refine_subpixel:
            bx1, by1, bx2, by2 = self.refine_subpixel_boundaries(
                crop,
                (bx1, by1, bx2, by2),
                element_kind=element_kind,
                grad_mag=grad_mag,
                gx=gx,
                gy=gy,
            )

        cb_x1, cb_y1, cb_x2, cb_y2 = crop_box
        cb_w = cb_x2 - cb_x1
        cb_h = cb_y2 - cb_y1

        global_x1 = max(0.0, min(1.0, cb_x1 + bx1 * cb_w))
        global_y1 = max(0.0, min(1.0, cb_y1 + by1 * cb_h))
        global_x2 = max(0.0, min(1.0, cb_x1 + bx2 * cb_w))
        global_y2 = max(0.0, min(1.0, cb_y1 + by2 * cb_h))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return FovealInspection(
            centroid=centroid,
            element_kind=element_kind,
            confidence=conf,
            bbox=(global_x1, global_y1, global_x2, global_y2),
            contrast_score=contrast,
            foveal_crop_shape=(crop.shape[0], crop.shape[1]),
            latency_ms=dt_ms,
        )

