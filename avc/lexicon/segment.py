"""Pixel-based segmentation: group ink into regions, without reading anything.

This is the front end of AVC Lexicon. It answers *where the ink is*, never *what
the ink says* — there is no charset, no glyph classifier and no OCR call here.

The method is purely geometric:

1. Binarise with an Otsu threshold computed from the patch histogram.
2. Pick the ink polarity (dark-on-light vs light-on-dark) that yields a small
   but non-empty ink fraction. Text is a minority of pixels in both cases.
3. Project ink onto the x axis and group contiguous ink columns into boxes,
   splitting where the gap exceeds a multiple of the typical intra-word gap.

Step 3 is the whole idea: a word is a *run of ink columns with small gaps*,
while the space between two words is a *large gap*. That distinction is
geometry, and it survives fonts, sizes and colours that would defeat a fixed
template — which is exactly the shift a matcher has to survive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GlyphBox:
    """A region of ink in a frame, in pixel coordinates ``(x0, y0, x1, y1)``."""

    x0: int
    y0: int
    x1: int
    y1: int
    ink_pixels: int = 0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y0:self.y1, self.x0:self.x1]


def _to_gray(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        # Rec. 601 luma, matching the rest of the vision code.
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's method, computed with a 256-bin histogram. No OpenCV dependency."""
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    prob = hist / total
    omega = np.cumsum(prob)
    centers = (np.arange(256) + 0.5) / 256.0
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.argmax(sigma_b))])


def _normalize(gray: np.ndarray) -> np.ndarray:
    """Accept raw 0-255 or already-normalised input; always return [0, 1]."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size and float(arr.max()) > 1.0:
        arr = arr / 255.0
    return arr


def _ink_mask(gray: np.ndarray, polarity: str, threshold: Optional[float] = None) -> np.ndarray:
    gray = _normalize(gray)
    thr = _otsu_threshold(gray) if threshold is None else threshold
    if polarity == "dark":
        return gray < thr
    if polarity == "light":
        return gray > thr
    raise ValueError(f"polarity must be 'dark' or 'light', got {polarity!r}")


def choose_polarity(gray: np.ndarray, threshold: Optional[float] = None) -> str:
    """Pick the polarity whose ink is the minority, the usual case for text."""
    gray = _normalize(gray)
    dark = float(_ink_mask(gray, "dark", threshold).mean())
    light = float(_ink_mask(gray, "light", threshold).mean())
    # Prefer the polarity with less ink, but never accept an empty mask when the
    # other polarity has ink: a blank patch has to stay blank.
    if dark <= 0.0 and light <= 0.0:
        return "dark"
    if dark <= 0.0:
        return "light"
    if light <= 0.0:
        return "dark"
    return "dark" if dark <= light else "light"


def is_uniform(gray: np.ndarray, tol: float = 0.02) -> bool:
    """True when a patch has no structure to segment.

    A flat patch is the text analogue of the "degenerate frame" the capability
    kernel refuses: Otsu has no valley to find, so it splits the single intensity
    level arbitrarily and every pixel looks like ink. Without this check a blank
    control reads as one giant word.
    """
    return gray.size == 0 or float(_normalize(gray).std()) < tol


class Segmenter:
    """Groups ink columns into word boxes. Reads no characters.

    The split rule is a gap *normalised by text height*, not an absolute pixel
    count, so it does not need re-tuning per font size. Measured across the fonts
    on this box (28 px): the largest intra-word gap is 0.35 of the text height,
    while a real inter-word space is at least 1.38 of it. ``split_ratio=0.75``
    sits in that empty band with roughly 2x margin either way.
    """

    def __init__(
        self,
        min_ink_pixels: int = 6,
        min_ink_columns: int = 2,
        split_ratio: float = 0.75,
        padding: int = 2,
        threshold: Optional[float] = None,
    ) -> None:
        self.min_ink_pixels = int(min_ink_pixels)
        self.min_ink_columns = int(min_ink_columns)
        self.split_ratio = float(split_ratio)
        self.padding = int(padding)
        self.threshold = threshold

    def _blobs(self, frame: np.ndarray, polarity: Optional[str] = None):
        """Binarise once and return the raw ink components. Shared by both paths."""
        from avc.lexicon.blobs import label_components

        gray = _to_gray(frame)
        if gray.size == 0 or is_uniform(gray):
            return gray, None, []
        pol = polarity or choose_polarity(gray, self.threshold)
        mask = _ink_mask(gray, pol, self.threshold)
        if int(mask.sum()) < self.min_ink_pixels:
            return gray, None, []
        return gray, mask, label_components(mask)

    def segment_lines(self, frame: np.ndarray, polarity: Optional[str] = None) -> List[List[GlyphBox]]:
        """Group ink into *lines*, each a list of control/word boxes.

        A single-line patch and a whole screen differ only in how many lines are
        present, so the caller should not have to know which it holds. Lines are
        found by projecting the *grouped* regions onto y and splitting where the
        gap between consecutive bands is large relative to the text height.
        """
        from avc.lexicon.blobs import group_regions

        gray, _mask, raw = self._blobs(frame, polarity)
        if not raw:
            return []
        containers, words = group_regions(raw)
        glyph_height = float(np.median([w.height for w in words])) if words else 0.0
        line_gap = max(2.0, 0.8 * glyph_height)

        bands = sorted(((c.y0, c.y1) for c in containers + words), key=lambda t: t[0])
        lines: List[Tuple[int, int]] = []
        start, end = bands[0]
        for (a, b) in bands[1:]:
            if a - end > line_gap:
                lines.append((start, end))
                start, end = a, b
            else:
                end = max(end, b)
        lines.append((start, end))

        out: List[List[GlyphBox]] = []
        for (ly0, ly1) in lines:
            members = [c for c in raw if c.y0 < ly1 and c.y1 > ly0]
            boxes = self._boxes_from_blobs(gray, members)
            if boxes:
                out.append(boxes)
        return out

    def _boxes_from_blobs(self, gray: np.ndarray, raw) -> List[GlyphBox]:
        """Turn a line's raw components into boxes: a box per control, and per word.

        Containers (bordered/filled controls) are one box each — the clickable
        unit, with its label inside. Glyphs that sit inside a container are that
        container's label and are dropped; glyphs *outside* any container are
        merged into words and kept. A line can hold both, and dropping the whole
        glyph set whenever one container appears would silently lose the free
        labels beside it.
        """
        from avc.lexicon.blobs import boxes_overlap, group_regions

        h, w = gray.shape[:2]
        containers, words = group_regions(raw)
        outside = [wd for wd in words if not any(boxes_overlap(wd, c) for c in containers)]
        chosen = sorted(list(containers) + outside, key=lambda c: c.x0)

        # Drop specks far below the line's text height.
        heights = [c.height for c in chosen]
        text_height = float(np.median(heights)) if heights else 1.0
        chosen = [c for c in chosen if c.height >= 0.3 * max(1.0, text_height)]

        boxes: List[GlyphBox] = []
        pad = self.padding
        for c in chosen:
            if c.width < self.min_ink_columns or c.area < self.min_ink_pixels:
                continue
            boxes.append(GlyphBox(
                x0=max(0, c.x0 - pad), y0=max(0, c.y0 - pad),
                x1=min(w, c.x1 + pad), y1=min(h, c.y1 + pad),
                ink_pixels=c.area,
            ))
        return boxes

    def segment_regions(self, frame: np.ndarray, polarity: Optional[str] = None) -> List[GlyphBox]:
        """All control/word boxes across the whole frame, in reading order."""
        boxes: List[GlyphBox] = []
        for line in self.segment_lines(frame, polarity):
            boxes.extend(line)
        return boxes

    def label_box(self, frame: np.ndarray, box: GlyphBox) -> GlyphBox:
        """The box of the *label* inside a region, for the encoder to read.

        The region box is the clickable unit (a button frame, say), but the
        encoder must see the word, not the frame: training patches were words
        cropped to their ink, so feeding it a bordered button puts a rectangle
        where the model expects a glyph and the label comes back wrong. The frame
        is the location; the label inside it is the thing to recognise.
        """
        from avc.lexicon.blobs import group_regions

        crop = box.crop(frame)
        gray, _mask, raw = self._blobs(crop)
        if not raw:
            return box
        _containers, words = group_regions(raw)
        inner = words or raw
        if not inner:
            return box
        x0 = min(c.x0 for c in inner)
        y0 = min(c.y0 for c in inner)
        x1 = max(c.x1 for c in inner)
        y1 = max(c.y1 for c in inner)
        # The inner boxes are crop-relative; shift back to frame coordinates.
        return GlyphBox(x0=x0 + box.x0, y0=y0 + box.y0, x1=x1 + box.x0, y1=y1 + box.y0)

    def segment(self, frame: np.ndarray, polarity: Optional[str] = None) -> List[GlyphBox]:
        """Return word boxes, left to right. Empty list for a blank patch."""
        gray = _to_gray(frame)
        if gray.size == 0:
            return []
        if is_uniform(gray):
            return []
        pol = polarity or choose_polarity(gray, self.threshold)
        mask = _ink_mask(gray, pol, self.threshold)
        if int(mask.sum()) < self.min_ink_pixels:
            return []

        row_ink = mask.sum(axis=1)
        ink_rows = np.where(row_ink > 0)[0]
        if ink_rows.size == 0:
            return []
        text_height = int(ink_rows[-1] - ink_rows[0] + 1)
        split_gap = max(2.0, self.split_ratio * text_height)

        col_ink = mask.sum(axis=0)
        active = col_ink > 0
        if not active.any():
            return []

        # Contiguous runs of ink columns, then merge runs separated by a small
        # gap (intra-word) and split where the gap is large (inter-word).
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for x, is_active in enumerate(active):
            if is_active and start is None:
                start = x
            elif not is_active and start is not None:
                runs.append((start, x))
                start = None
        if start is not None:
            runs.append((start, len(active)))

        words: List[Tuple[int, int]] = []
        cur_start, cur_end = runs[0]
        for (a, b) in runs[1:]:
            if (a - cur_end) > split_gap:
                words.append((cur_start, cur_end))
                cur_start, cur_end = a, b
            else:
                cur_end = b
        words.append((cur_start, cur_end))

        boxes: List[GlyphBox] = []
        h, w = mask.shape
        for (x0, x1) in words:
            if (x1 - x0) < self.min_ink_columns:
                continue
            band = mask[:, x0:x1]
            rows = np.where(band.sum(axis=1) > 0)[0]
            if rows.size == 0:
                continue
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            ink = int(band.sum())
            if ink < self.min_ink_pixels:
                continue
            pad = self.padding
            boxes.append(GlyphBox(
                x0=max(0, x0 - pad), y0=max(0, y0 - pad),
                x1=min(w, x1 + pad), y1=min(h, y1 + pad),
                ink_pixels=ink,
            ))
        return boxes


def segment_glyphs(frame: np.ndarray, **kwargs) -> List[GlyphBox]:
    """Convenience wrapper around :class:`Segmenter`."""
    return Segmenter(**kwargs).segment(frame)
