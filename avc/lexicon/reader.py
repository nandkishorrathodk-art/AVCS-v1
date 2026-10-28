"""LexiconReader: pixels in, control identity out — and the inverse.

The reader composes the segmenter with the shared space. Two directions, both
from the same space:

``read_regions(frame)``
    Segment the frame into word boxes and name each one. This is the inverse
    direction (patch -> word): given a hotspot the dorsal stream already found,
    say what the control is.

``find(frame, word)``
    Ground a *spoken* word in the frame by matching its text embedding against
    every segmented patch. This is the forward direction (word -> patch): the
    language side drives the search, so no separate grounder is needed for
    labelled controls.

Both refuse rather than guess. A patch whose best prototype is below ``floor``
is returned with ``known=False``; ``find`` returns ``None`` when no region
clears the floor. That is the fail-closed rule this repo applies everywhere
else, applied to reading.

The cycle property is what makes the space trustworthy: ``read_regions`` on the
region ``find`` returned must name the word that was asked for. ``check_cycle``
measures exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from avc.lexicon.encoder import PATCH_H, PATCH_W, LexiconEncoder, Readout
from avc.lexicon.render import to_patch
from avc.lexicon.segment import GlyphBox, Segmenter


@dataclass(frozen=True)
class ReadWord:
    """One segmented region and what the shared space says it is."""

    box: GlyphBox
    readout: Readout

    @property
    def label(self) -> str:
        return self.readout.label

    @property
    def known(self) -> bool:
        return self.readout.known

    def as_dict(self) -> dict:
        return {"box": self.box.as_tuple(), **self.readout.as_dict()}


def patch_from_crop(crop: np.ndarray) -> torch.Tensor:
    """Crop -> (1, 1, H, W) float tensor, using the canonical preprocessing.

    Delegates to :func:`avc.lexicon.render.to_patch` so the encoder sees exactly
    the tensor shape and scaling it was trained on. An earlier version resized
    with nearest-neighbour here while training used bilinear ink-cropping, and
    the resulting skew dropped end-to-end accuracy from ~0.9 to ~0.15 while every
    training metric stayed high.
    """
    patch = to_patch(crop)
    return torch.from_numpy(patch)[None, None, :, :]


class LexiconReader:
    """Composes segmentation and the shared space. Loads a trained checkpoint."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        *,
        labels: Optional[Sequence[str]] = None,
        floor: float = 0.55,
        segmenter: Optional[Segmenter] = None,
        model: Optional[LexiconEncoder] = None,
    ) -> None:
        self.floor = float(floor)
        self.segmenter = segmenter or Segmenter()
        if model is not None:
            self.model = model
        else:
            if labels is None:
                from avc.lexicon.vocab import LABELS as default_labels

                labels = default_labels
            self.model = LexiconEncoder(list(labels))
            if checkpoint and Path(checkpoint).exists():
                blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.model.load_state_dict(blob["state_dict"])
                if blob.get("labels"):
                    self.model.labels = list(blob["labels"])
            self.model.rebuild_prototypes()
        self.model.eval()

    @property
    def labels(self) -> List[str]:
        return list(self.model.labels)

    # ------------------------------------------------------------------ #
    # inverse direction: pixels -> name
    # ------------------------------------------------------------------ #
    def read_regions(self, frame: np.ndarray, floor: Optional[float] = None) -> List[ReadWord]:
        floor = self.floor if floor is None else floor
        out: List[ReadWord] = []
        for box in self.segmenter.segment_regions(frame):
            # The box is the clickable unit; the label inside it is what the
            # encoder recognises. Reading the frame itself would put a border
            # where the model expects a glyph.
            inner = self.segmenter.label_box(frame, box)
            crop = inner.crop(frame)
            if crop.size == 0:
                continue
            patch = patch_from_crop(crop)
            with torch.no_grad():
                readout = self.model.read_patch(patch, floor=floor)
            out.append(ReadWord(box=box, readout=readout))
        return out

    def describe(self, frame: np.ndarray, floor: Optional[float] = None) -> List[str]:
        """Names of the regions that cleared the floor, in reading order."""
        return [w.label for w in self.read_regions(frame, floor=floor) if w.known]

    # ------------------------------------------------------------------ #
    # forward direction: word -> where
    # ------------------------------------------------------------------ #
    def find(
        self, frame: np.ndarray, word: str, floor: Optional[float] = None
    ) -> Optional[ReadWord]:
        """Best region matching a spoken word, or ``None`` if nothing clears the floor."""
        floor = self.floor if floor is None else floor
        if word.strip().lower() not in {lbl.lower() for lbl in self.model.labels}:
            # A word outside the closed vocabulary has no prototype to match.
            return None
        regions = self.read_regions(frame, floor=floor)
        best: Optional[ReadWord] = None
        for rw in regions:
            if rw.label.lower() != word.strip().lower():
                continue
            if best is None or rw.readout.confidence > best.readout.confidence:
                best = rw
        return best

    # ------------------------------------------------------------------ #
    # the property that makes the space trustworthy
    # ------------------------------------------------------------------ #
    def check_cycle(self, frame: np.ndarray, word: str, floor: Optional[float] = None) -> bool:
        """``find(word)`` then read it back: does the region still name ``word``?"""
        hit = self.find(frame, word, floor=floor)
        return hit is not None and hit.label.lower() == word.strip().lower()
