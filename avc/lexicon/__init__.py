"""AVC Lexicon — reading labels without OCR.

The rest of ``avc`` answers *where* things are (dorsal: affordances) and *what
kind of screen* this is (ventral: categories). Neither can answer **what a
control is called**, and the repo forbids OCR on purpose: recognition by glyph
transcription is a different problem with a different failure mode.

AVC Lexicon is the third stream. It treats a word the way the visual system
treats a face: as a *shape* to be recognised, not a string to be transcribed.

    pixels -> segment (glyph-gap grouping) -> shape encoder -> prototype cosine
                    ^ no OCR                ^ no charset          ^ closed vocab

Design constraints, inherited from this repo:

- **Pure pixels.** No OCR library, no charset, no per-glyph classifier. The
  segmenter groups ink by whitespace geometry; the encoder sees bitmaps.
- **Fail closed.** A shape that is not close enough to any known prototype is
  reported as *unknown*, not snapped to the nearest label. Confidence is the
  cosine margin, and the margin has to clear a floor.
- **Honest about font generalisation.** Fonts are the domain shift here, exactly
  as generators are for the grounder. A held-out *font* is the real test, and
  the benchmark reports it separately from the training fonts.

Nothing here claims general reading. The vocabulary is closed and declared in
``vocab.py``; text outside it is refused.
"""
from __future__ import annotations

from avc.lexicon.blobs import (
    Component,
    group_regions,
    label_components,
    merge_close_components,
    split_containers,
)
from avc.lexicon.encoder import LexiconEncoder, Readout
from avc.lexicon.reader import LexiconReader, ReadWord, patch_from_crop
from avc.lexicon.render import render_word, split_fonts, to_patch
from avc.lexicon.segment import GlyphBox, Segmenter, segment_glyphs
from avc.lexicon.vocab import LABELS, VOCAB, label_to_index

__all__ = [
    "LexiconEncoder",
    "Readout",
    "LexiconReader",
    "ReadWord",
    "patch_from_crop",
    "GlyphBox",
    "Segmenter",
    "segment_glyphs",
    "Component",
    "label_components",
    "merge_close_components",
    "split_containers",
    "group_regions",
    "render_word",
    "split_fonts",
    "to_patch",
    "LABELS",
    "VOCAB",
    "label_to_index",
]
