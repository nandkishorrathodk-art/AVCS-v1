"""Synthetic label rendering, with fonts as the domain shift.

The grounder's honest weak point is that a corpus generated one way does not
transfer to another generator. Text has the same structure of problem, and it
has a natural, well-defined version of it: **fonts**. A matcher that works only
on the font it trained on has learned a template, not a shape.

So this module splits the available fonts into disjoint groups:

    TRAIN_FONTS  -> jittered across size, colour, weight, spacing
    HELD_OUT_FONTS -> never seen in training; the generalisation test

A second, harder split holds out *labels* rather than fonts, which is what makes
the open-set rejection number meaningful: the reader has to say "unknown" for a
word whose shape it has never been asked to recognise.
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avc.lexicon.encoder import PATCH_H, PATCH_W
from avc.lexicon.vocab import LABELS

FONT_DIRS = ("/usr/share/fonts", "/usr/local/share/fonts")


def discover_fonts() -> List[str]:
    """Every readable TrueType font on the box, sorted for determinism."""
    found: List[str] = []
    for root in FONT_DIRS:
        found.extend(glob.glob(os.path.join(root, "**", "*.ttf"), recursive=True))
    return sorted(set(found))


def _family(path: str) -> str:
    return os.path.basename(path).lower()


def split_fonts(all_fonts: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
    """Disjoint train / held-out font groups, chosen by family.

    ``Sans`` (and its bold/oblique variants) trains; ``Serif`` and ``Mono`` are
    held out, so a held-out score is a genuine cross-font score rather than a
    differently-rendered copy of the training face.
    """
    fonts = list(all_fonts) if all_fonts is not None else discover_fonts()
    if not fonts:
        return [], []
    train, held = [], []
    for f in fonts:
        name = _family(f)
        if "mono" in name or "serif" in name:
            held.append(f)
        else:
            train.append(f)
    # A held-out set that is empty is not a test; fall back to a deterministic
    # slice so the caller gets a real split on any machine.
    if not held and len(fonts) > 1:
        held = train[len(train) // 2:]
        train = train[: len(train) // 2]
    if not train:
        train = held[:1]
    return train, held


@dataclass(frozen=True)
class RenderSpec:
    """One way of drawing a word. The jitter is the augmentation."""

    font_path: str
    size: int
    color: Tuple[int, int, int]
    background: Tuple[int, int, int]


def _to_patch(image: Image.Image) -> np.ndarray:
    """Crop to ink, resize to the fixed patch size, return float32 in [0, 1].

    Ink is detected *relative to the background level*, not against an absolute
    threshold. ``arr.max(axis=1) > 0.15`` looks reasonable but is wrong on any
    light background: the background itself (0.96) clears the threshold, so every
    row counts as ink and nothing is ever cropped. That went unnoticed because
    the training renderer sizes its canvas tightly to the text, where a failed
    crop is nearly harmless; it becomes fatal the moment a caller passes a
    larger region, because the glyph shrinks inside its padding and the encoder
    sees a scale it never trained on. Taking the median as the background and
    measuring deviation from it works for light-on-dark and dark-on-light alike.
    """
    arr = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    if arr.size == 0:
        return np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
    background = float(np.median(arr))
    deviation = np.abs(arr - background)
    ink = deviation > 0.15
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if rows.size and cols.size:
        arr = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    pil = pil.resize((PATCH_W, PATCH_H), Image.BILINEAR)
    return np.asarray(pil).astype(np.float32) / 255.0


def to_patch(image: "np.ndarray | Image.Image") -> np.ndarray:
    """THE canonical preprocessing. Training and inference must both call this.

    A train/serve skew here is invisible in training metrics and fatal at
    inference: the encoder is fed a differently-scaled glyph than it learned on.
    Keeping one function is the only reliable way to prevent it — the same
    reasoning that makes the tokenizer travel with the weights on the text side.
    """
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = np.stack(
                [0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]] * 3,
                axis=-1,
            )
        image = Image.fromarray(arr.astype(np.uint8))
    return _to_patch(image)


def render_word(
    word: str,
    font_path: str,
    *,
    size: int = 24,
    color: Tuple[int, int, int] = (20, 20, 20),
    background: Tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    """Render one word to a normalised patch. No reading, just drawing."""
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    bbox = probe.textbbox((0, 0), word, font=font)
    pad = max(4, size // 4)
    width = max(8, bbox[2] - bbox[0] + 2 * pad)
    height = max(8, bbox[3] - bbox[1] + 2 * pad)
    image = Image.new("RGB", (width, height), background)
    ImageDraw.Draw(image).text((pad - bbox[0], pad - bbox[1]), word, font=font, fill=color)
    return _to_patch(image)


def random_spec(rng: random.Random, font_path: str) -> RenderSpec:
    size = rng.choice([16, 20, 24, 28, 32])
    dark = rng.random() < 0.5
    if dark:
        color = tuple(rng.randint(0, 90) for _ in range(3))
        background = tuple(rng.randint(200, 255) for _ in range(3))
    else:
        color = tuple(rng.randint(190, 255) for _ in range(3))
        background = tuple(rng.randint(0, 70) for _ in range(3))
    return RenderSpec(font_path, size, color, background)


def make_patch(
    label: str, fonts: Sequence[str], rng: random.Random, augment: bool = False
) -> np.ndarray:
    spec = random_spec(rng, rng.choice(list(fonts)))
    patch = render_word(
        label, spec.font_path, size=spec.size, color=spec.color, background=spec.background
    )
    if augment:
        patch = augment_patch(patch, rng)
    return patch


def augment_patch(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    """Perturb a patch the way a different font would.

    Font is the domain shift for this stream, and only a handful of faces exist
    on any one box. Simulating the axes along which fonts actually differ --
    slant, stroke weight, aspect, blur -- is what lets a model trained on two
    sans faces transfer to a serif it has never seen. This is the text analogue
    of the photometric jitter the grounder experiment tried (and which failed
    *there* because its gap was structural); here the gap really is appearance.
    """
    img = Image.fromarray((patch * 255.0).astype(np.uint8))
    w, h = img.size

    # Slant (shear), the strongest single font cue (italic vs roman).
    if rng.random() < 0.7:
        shear = rng.uniform(-0.35, 0.35)
        img = img.transform(
            (w, h), Image.AFFINE, (1, shear, -shear * h / 2, 0, 1, 0),
            resample=Image.BILINEAR, fillcolor=int(rng.uniform(0.8, 1.0) * 255),
        )
    # Stroke weight / contrast: erode-like thinning or dilate-like thickening.
    if rng.random() < 0.6:
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr * rng.uniform(0.75, 1.25) + rng.uniform(-18, 18), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    # Aspect stretch, which separates condensed from wide faces.
    if rng.random() < 0.6:
        img = img.resize((max(4, int(w * rng.uniform(0.8, 1.25))), h), Image.BILINEAR)
    # Optical blur, for rendering differences and screenshot scaling.
    if rng.random() < 0.4:
        from PIL import ImageFilter

        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    return np.asarray(img.resize((PATCH_W, PATCH_H), Image.BILINEAR)).astype(np.float32) / 255.0


def make_corpus(
    labels: Sequence[str] = LABELS,
    fonts: Optional[Sequence[str]] = None,
    per_label: int = 8,
    seed: int = 7,
    augment: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(patches, targets)`` with ``patches`` shaped (N, 1, H, W) float32."""
    font_list = list(fonts) if fonts else split_fonts()[0]
    if not font_list:
        raise RuntimeError("no TrueType fonts found; cannot render a corpus")
    rng = random.Random(seed)
    index_of = {label: i for i, label in enumerate(labels)}
    patches, targets = [], []
    for label in labels:
        for _ in range(per_label):
            patch = make_patch(label, font_list, rng, augment=augment)
            patches.append(patch[None, :, :])
            targets.append(index_of[label])
    return np.asarray(patches, dtype=np.float32), np.asarray(targets, dtype=np.int64)
