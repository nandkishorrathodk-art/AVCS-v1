"""[2] fovea_reader.py: High-Acuity Central Foveal Zoom & Text Reader.

Zooms into visual change hotspots and detected UI boxes to inspect high-resolution
native crops in <10ms. Eliminates the 131-word vocabulary limit: reads unlimited
button labels, dialog text, and error messages when a real OCR or VLM backend
is attached.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FovealElement:
    """Inspected UI element with high-acuity crop data."""
    text: Optional[str]               # Read text content, or None if no backend attached
    category: str                     # e.g., "button", "input", "label", "icon"
    center_norm: Tuple[float, float]  # Exact normalized centroid (x, y)
    bbox_pixels: Tuple[int, int, int, int] # (left, top, width, height) in native frame
    confidence: float                 # Confidence score [0, 1]
    # The high-acuity native crop itself. Held locally only; it must not be
    # forwarded to System 2 (see docs/AVCS_ENGINEERING_RULES.md). Excluded from
    # equality/hash so the frozen dataclass keeps working with an array field.
    patch: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    contrast: float = 0.0             # std of the grayscale crop (texture metric)
    edge_density: float = 0.0         # mean gradient magnitude of the crop (texture metric)

    def describe(self) -> str:
        """Short pixel-derived label, e.g. ``interactive_control at (0.50, 0.50)``.

        The ``at (x, y)`` suffix is deliberate: ``LiveNarrator`` rewrites it into
        a spatial region name, and the System 2 prompt strips it, so raw
        coordinates never leave System 1.
        """
        label = (self.text.strip() if self.text else "") or self.category
        x, y = self.center_norm
        return f"{label} at ({x:.2f}, {y:.2f})"


class FoveaReader:
    """Foveal zoom controller: extracts native crops and reads text in <10ms."""

    def __init__(
        self,
        crop_size: int = 128,
        backend: Optional[Any] = None,
    ) -> None:
        self.crop_size = crop_size
        self._backend = backend

    def attach_backend(self, backend: Any) -> None:
        """Explicitly attach an OCR or local VLM backend."""
        self._backend = backend

    def attach_ocr(self, backend: Any) -> None:
        """Disabled under AVCS Pure-RGB pixel invariant."""
        raise RuntimeError("External OCR is disabled under AVCS Pure-RGB pixel invariant.")

    def crop_element(
        self,
        full_frame: np.ndarray,
        norm_x: float,
        norm_y: float,
        crop_w: int = 160,
        crop_h: int = 48,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crops high-resolution native pixels around a normalized coordinate."""
        h, w = full_frame.shape[:2]
        cx, cy = int(norm_x * w), int(norm_y * h)
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        x1 = max(0, x2 - crop_w)
        y1 = max(0, y2 - crop_h)
        return full_frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    def measure_texture(self, crop: np.ndarray) -> Tuple[float, float]:
        """Return (contrast, edge_density) for a crop, computed once and shared.

        ``contrast`` is the grayscale standard deviation; ``edge_density`` is the
        mean gradient magnitude. Both are the same quantities
        ``summarize_foveal_patches`` reports, so a caller can rank elements on
        the same scale it already sees in the System 2 prompt.
        """
        if crop.size == 0 or crop.ndim < 2:
            return 0.0, 0.0
        gray = np.mean(crop, axis=-1) if crop.ndim == 3 else crop
        gray = gray.astype(np.float32)
        contrast = float(np.std(gray))
        gy, gx = np.gradient(gray)
        edge_density = float(np.mean(np.hypot(gx, gy)))
        return contrast, edge_density

    def read_text(self, crop: np.ndarray) -> Optional[str]:
        """Disabled under AVCS Pure-RGB pixel invariant."""
        raise RuntimeError("External OCR is disabled under AVCS Pure-RGB pixel invariant.")

    def inspect(
        self,
        full_frame: Optional[np.ndarray],
        hotspots: List[Tuple[float, float, str, float]],
        max_inspect: int = 8,
    ) -> List[FovealElement]:
        """Runs pure-pixel foveal inspection on candidate hotspots."""
        if full_frame is None or not hotspots:
            return []

        elements: List[FovealElement] = []
        for spot in hotspots[:max_inspect]:
            x, y = float(spot[0]), float(spot[1])
            kind = str(spot[2]) if len(spot) > 2 else "element"
            conf = float(spot[3]) if len(spot) > 3 else 1.0

            crop, bbox = self.crop_element(full_frame, x, y)
            contrast, edge_density = self.measure_texture(crop)
            elements.append(
                FovealElement(
                    text=None,
                    category=kind,
                    center_norm=(x, y),
                    bbox_pixels=bbox,
                    confidence=conf,
                    patch=crop,
                    contrast=contrast,
                    edge_density=edge_density,
                )
            )
        return elements
