"""Tests for the AVC Lexicon stream — reading control labels without OCR.

The stream's whole claim is that it recognises a control's *identity* from
pixels and refuses when it does not know. These tests pin both halves, plus the
segmentation geometry they rest on.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from avc.lexicon.blobs import (
    Component,
    boxes_overlap,
    group_regions,
    label_components,
    merge_close_components,
    split_containers,
)
from avc.lexicon.encoder import LexiconEncoder, TextEncoder
from avc.lexicon.reader import LexiconReader, patch_from_crop
from avc.lexicon.render import discover_fonts, render_word, split_fonts
from avc.lexicon.segment import Segmenter, choose_polarity, segment_glyphs
from avc.lexicon.vocab import LABELS, VOCAB, index_to_label, label_to_index, label_words

pytestmark = pytest.mark.skipif(
    not discover_fonts(), reason="no TrueType fonts on this box to render with"
)


def _fonts():
    train, _held = split_fonts()
    return train


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def test_vocabulary_is_closed_and_unique():
    assert len(LABELS) == len(set(LABELS))
    assert len(VOCAB) == len(LABELS)
    assert set(VOCAB.values()) == set(range(len(LABELS)))


def test_unknown_label_is_reported_as_unknown_not_index_zero():
    """The classic lookup bug: a missing key silently becoming index 0."""
    assert label_to_index("save") == VOCAB["save"]
    assert label_to_index("not_a_real_label") == -1


def test_index_to_label_raises_out_of_range():
    with pytest.raises(IndexError):
        index_to_label(len(LABELS))


def test_label_words_splits_multiword_labels():
    assert label_words("edit_menu") == ("edit_menu",)
    assert label_words("save as") == ("save", "as")


# --------------------------------------------------------------------------- #
# Segmentation geometry
# --------------------------------------------------------------------------- #
def test_two_words_separate_and_one_word_stays_together():
    font_path = _fonts()[0]
    font = ImageFont.truetype(font_path, 28)

    two = Image.new("RGB", (320, 60), (245, 245, 245))
    ImageDraw.Draw(two).text((10, 10), "Save   Cancel", font=font, fill=(20, 20, 20))
    assert len(Segmenter().segment(np.asarray(two))) == 2

    one = Image.new("RGB", (320, 60), (245, 245, 245))
    ImageDraw.Draw(one).text((10, 10), "Username", font=font, fill=(20, 20, 20))
    assert len(Segmenter().segment(np.asarray(one))) == 1


def test_blank_patch_produces_no_boxes():
    blank = np.full((60, 200, 3), 240, dtype=np.uint8)
    assert segment_glyphs(blank) == []


def test_light_on_dark_is_segmented_too():
    font_path = _fonts()[0]
    font = ImageFont.truetype(font_path, 28)
    dark = Image.new("RGB", (320, 60), (18, 18, 18))
    ImageDraw.Draw(dark).text((10, 10), "Delete  File", font=font, fill=(235, 235, 235))
    assert len(Segmenter().segment(np.asarray(dark))) == 2


def test_polarity_pick_prefers_the_minority_ink():
    dark_bg = np.full((40, 120), 20, dtype=np.uint8)
    dark_bg[15:25, 20:40] = 230          # small light patch on dark
    assert choose_polarity(dark_bg) == "light"
    light_bg = np.full((40, 120), 230, dtype=np.uint8)
    light_bg[15:25, 20:40] = 20          # small dark patch on light
    assert choose_polarity(light_bg) == "dark"


# --------------------------------------------------------------------------- #
# Shared space
# --------------------------------------------------------------------------- #
def test_both_branches_land_in_the_same_normalised_space():
    import torch

    model = LexiconEncoder(LABELS[:8])
    model.rebuild_prototypes()
    patch = np.full((32, 96), 0.5, dtype=np.float32)
    with torch.no_grad():
        p = model.encode_patches(torch.from_numpy(patch)[None, None])
        t = model.encode_words(["save"])
    assert p.shape == t.shape == (1, 128)
    assert abs(float((p ** 2).sum()) - 1.0) < 1e-5
    assert abs(float((t ** 2).sum()) - 1.0) < 1e-5


def test_prototypes_are_one_per_label_and_unit_norm():
    model = LexiconEncoder(LABELS[:8])
    model.rebuild_prototypes()
    assert model.prototypes.shape == (8, 128)
    norms = model.prototypes.norm(dim=1)
    assert bool((norms > 0.99).all() and (norms < 1.01).all())


def test_readout_refuses_when_prototypes_are_not_built():
    """A checkpoint whose prototypes were never rebuilt must not silently answer."""
    import torch

    model = LexiconEncoder(LABELS[:8])
    with pytest.raises(RuntimeError):
        model.read_patch(torch.zeros(1, 1, 32, 96))


def test_text_encoder_ids_are_stable_across_instances():
    """Same class of bug as the salted hash(): the mapping must not vary."""
    a = TextEncoder.encode_word("save")
    b = TextEncoder.encode_word("save")
    assert a.tolist() == b.tolist()
    assert a.tolist() != TextEncoder.encode_word("cancel").tolist()


def test_unknown_characters_map_to_a_reserved_id_not_a_crash():
    ids = TextEncoder.encode_word("save!")
    assert ids.numel() == 5
    assert int(ids[-1]) == 0            # reserved unknown slot


# --------------------------------------------------------------------------- #
# Reader: both directions and the fail-closed rule
# --------------------------------------------------------------------------- #
def _rendered_frame(word: str, font_path: str, size: int = 26) -> np.ndarray:
    patch = render_word(word, font_path, size=size)
    gray = (patch * 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def test_reader_returns_a_region_for_rendered_ink():
    reader = LexiconReader()
    frame = _rendered_frame("save", _fonts()[0])
    regions = reader.read_regions(frame)
    assert len(regions) == 1
    assert regions[0].box.width > 0


def test_reader_refuses_a_word_outside_the_closed_vocabulary():
    """find() must return None, not the nearest label, for an unknown word."""
    reader = LexiconReader()
    frame = _rendered_frame("save", _fonts()[0])
    assert reader.find(frame, "frobnicate") is None


def test_reader_find_agrees_with_read_regions():
    reader = LexiconReader()
    frame = _rendered_frame("cancel", _fonts()[0])
    regions = reader.read_regions(frame)
    hit = reader.find(frame, regions[0].label)
    assert hit is not None
    assert hit.label == regions[0].label


def test_patch_from_crop_shape_and_range():
    crop = np.full((12, 40), 200, dtype=np.uint8)
    patch = patch_from_crop(crop)
    assert tuple(patch.shape) == (1, 1, 32, 96)
    assert 0.0 <= float(patch.min()) and float(patch.max()) <= 1.0


def test_patch_from_crop_survives_a_degenerate_crop():
    patch = patch_from_crop(np.zeros((0, 0), dtype=np.uint8))
    assert tuple(patch.shape) == (1, 1, 32, 96)


# --------------------------------------------------------------------------- #
# 2D grouping: words and whole screens
# --------------------------------------------------------------------------- #
def _mock_screen(buttons, width: int = 560, height: int = 200) -> np.ndarray:
    """A title line plus a row of bordered buttons, like a real toolbar."""
    font = ImageFont.truetype(_fonts()[0], 26)
    title_font = ImageFont.truetype(_fonts()[0], 18)
    image = Image.new("RGB", (width, height), (238, 238, 240))
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "Untitled document", font=title_font, fill=(60, 60, 70))
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    x = 20
    for name in buttons:
        w = probe.textbbox((0, 0), name, font=font)[2] + 24
        draw.rectangle([x, 70, x + w, 116], fill=(250, 250, 252), outline=(150, 150, 160))
        draw.text((x + 12, 80), name, font=font, fill=(30, 30, 40))
        x += w + 14
    return np.asarray(image)


def test_label_components_finds_each_glyph_blob():
    """Two separated squares must label as two components, not one."""
    mask = np.zeros((20, 40), dtype=bool)
    mask[4:14, 4:14] = True
    mask[4:14, 24:34] = True
    comps = label_components(mask)
    assert len(comps) == 2
    assert comps[0].as_tuple() == (4, 4, 14, 14)
    assert comps[1].as_tuple() == (24, 4, 34, 14)


def test_label_components_empty_mask_is_empty_list():
    assert label_components(np.zeros((10, 10), dtype=bool)) == []


def test_merge_joins_the_dot_of_an_i_to_its_stem():
    """A 5px dot above a 32px stem is one word, despite the height ratio."""
    dot = Component(10, 0, 15, 5, 25)
    stem = Component(10, 7, 15, 32, 100)
    merged = merge_close_components([dot, stem])
    assert len(merged) == 1
    assert merged[0].as_tuple() == (10, 0, 15, 32)


def test_merge_does_not_join_two_distant_words():
    left = Component(0, 0, 20, 20, 100)
    right = Component(80, 0, 100, 20, 100)
    assert len(merge_close_components([left, right])) == 2


def test_container_rule_needs_hollow_and_enclosing():
    """A hollow frame that encloses a glyph is a container; the glyph is not."""
    frame = Component(0, 0, 60, 30, 60)      # hollow: 60 / 1800
    glyph = Component(20, 10, 30, 20, 80)    # solid: 80 / 100
    containers, glyphs = split_containers([frame, glyph])
    assert containers == [frame]
    assert glyphs == [glyph]


def test_enclosure_alone_does_not_make_a_glyph_a_container():
    """The 'a' counter must not be mistaken for a control frame.

    An enclosure-only rule called any blob that surrounds another a container,
    which turned the counter of a letter into a frame and split words apart. The
    hollow test is what prevents it: a letter is mostly solid.
    """
    letter = Component(0, 0, 20, 30, 500)    # solid: 500 / 600
    speck = Component(6, 10, 12, 16, 36)
    containers, glyphs = split_containers([letter, speck])
    assert containers == []
    assert len(glyphs) == 2


def test_boxes_overlap_detects_shared_area():
    assert boxes_overlap(Component(0, 0, 10, 10, 1), Component(5, 5, 15, 15, 1))
    assert not boxes_overlap(Component(0, 0, 10, 10, 1), Component(10, 0, 20, 10, 1))


def test_a_rendered_word_is_one_region_not_a_shred():
    """The whole point: 'save' must come back as one box, not five fragments."""
    frame = _rendered_frame("save", _fonts()[0])
    boxes = Segmenter().segment_regions(frame)
    assert len(boxes) == 1


def test_screen_separates_buttons_and_keeps_their_labels():
    """Four bordered buttons must give four boxes, one per control."""
    frame = _mock_screen(["save", "close", "undo", "redo"])
    boxes = Segmenter().segment_regions(frame)
    # The button row has four controls; the title line is a fifth region.
    button_boxes = [b for b in boxes if b.y0 > 40]
    assert len(button_boxes) == 4
    assert all(b.height > 20 for b in button_boxes)


def test_reader_names_buttons_on_a_whole_screen():
    """End-to-end: the reader must name the controls on a real-size screen."""
    reader = LexiconReader(checkpoint="runs/weights/avc_lexicon.pt")
    import pathlib

    if not pathlib.Path("runs/weights/avc_lexicon.pt").exists():
        pytest.skip("no trained lexicon checkpoint on this box")
    frame = _mock_screen(["save", "close", "undo", "redo"])
    labels = {r.label.lower() for r in reader.read_regions(frame, floor=0.0)}
    # At least the first two buttons must be named correctly; the checkpoint is
    # trained, so this is a real accuracy check, not a plumbing check.
    assert "save" in labels
    assert "close" in labels
