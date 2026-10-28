"""Measure the AVC Lexicon stream honestly.

Four numbers, each catching something the others cannot:

1. **closed_vocab_top1** — seen fonts, unseen samples. Does it recognise at all?
2. **held_out_font_top1** — a font family never trained on. This is the real
   test; a number that only holds on the training font is a template, not a
   recogniser. Same trap as the grounder's cross-generator gap.
3. **unknown_rejection** — words outside the closed vocabulary must be refused,
   not snapped to the nearest known label. A recogniser that always answers is
   dangerous: it will confidently name a control that is not there.
4. **cycle_consistency** — ``find(word)`` then read the region back. If the
   shared space is real, the region names the word that was asked for.

Plus latency, quoted with the resolution, per this repo's rules.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avc.lexicon.reader import LexiconReader, patch_from_crop
from avc.lexicon.render import render_word, split_fonts
from avc.lexicon.vocab import LABELS

# Words the reader was never trained on. Rendering them is not "reading new
# text" — the point is that the reader must NOT claim to know them.
UNKNOWN_WORDS = ["frobnicate", "quux", "zibble", "kthxbye", "plzno", "xyzzy"]

DEFAULT_OUT = "benchmarks/results/avc_lexicon.json"


def _phrase_frame(words: Sequence[str], font_path: str, size: int = 28) -> np.ndarray:
    """Render several words on one canvas, widely spaced, as a UI row would be."""
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    widths = [probe.textbbox((0, 0), w, font=font)[2] for w in words]
    gap = size
    total = sum(widths) + gap * (len(words) + 1)
    height = size * 2
    image = Image.new("RGB", (max(32, total), height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    x = gap
    for w in words:
        draw.text((x, size // 2), w, font=font, fill=(25, 25, 25))
        x += probe.textbbox((0, 0), w, font=font)[2] + gap
    return np.asarray(image)


def _patch_accuracy(reader: LexiconReader, labels: Sequence[str], fonts: Sequence[str],
                    per_label: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    correct = 0
    total = 0
    accepted = 0
    for label in labels:
        for _ in range(per_label):
            font_path = str(rng.choice(list(fonts)))
            patch = render_word(label, font_path, size=int(rng.choice([18, 22, 26, 30])))
            frame = (patch * 255).astype(np.uint8)
            frame = np.stack([frame] * 3, axis=-1)
            words = reader.read_regions(frame)
            if not words:
                total += 1
                continue
            best = max(words, key=lambda w: w.readout.confidence)
            total += 1
            if best.label == label:
                correct += 1
                if best.known:
                    accepted += 1
    return {
        "top1": round(correct / max(1, total), 4),
        "accepted": round(accepted / max(1, total), 4),
        "n": total,
    }


def _unknown_rejection(reader: LexiconReader, fonts: Sequence[str], seed: int) -> Dict[str, float]:
    """Fraction of out-of-vocabulary words the reader refuses to name."""
    rng = np.random.default_rng(seed)
    refused = 0
    total = 0
    for word in UNKNOWN_WORDS:
        for _ in range(4):
            font_path = str(rng.choice(list(fonts)))
            patch = render_word(word, font_path, size=int(rng.choice([20, 26])))
            frame = np.stack([(patch * 255).astype(np.uint8)] * 3, axis=-1)
            words = reader.read_regions(frame)
            total += 1
            if not words or not any(w.known for w in words):
                refused += 1
    return {"rejection_rate": round(refused / max(1, total), 4), "n": total}


def _cycle_consistency(reader: LexiconReader, fonts: Sequence[str], seed: int) -> Dict[str, float]:
    """Render a row of words; ask for each by name; read the hit back."""
    rng = np.random.default_rng(seed)
    ok = 0
    total = 0
    for _ in range(6):
        picked = list(rng.choice(LABELS, size=3, replace=False))
        font_path = str(rng.choice(list(fonts)))
        frame = _phrase_frame([str(w) for w in picked], font_path)
        for word in picked:
            total += 1
            if reader.check_cycle(frame, str(word)):
                ok += 1
    return {"cycle_rate": round(ok / max(1, total), 4), "n": total}


def _latency(reader: LexiconReader, font_path: str, reps: int = 20) -> Dict[str, float]:
    frame = _phrase_frame(["save", "cancel", "username"], font_path)
    times: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        reader.read_regions(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    # Minimum, not mean: on a loaded box the mean is scheduler noise (repo rule).
    return {
        "frame": f"{frame.shape[1]}x{frame.shape[0]}",
        "min_ms": round(min(times), 3),
        "median_ms": round(float(np.median(times)), 3),
    }


def _gate_separability(reader: LexiconReader, fonts: Sequence[str], seed: int) -> Dict[str, float]:
    """Correct-label vs best-wrong-label similarity. The honest gate story.

    A recognition number alone hides whether the *confidence* means anything. On
    the training fonts the correct label's cosine sits well above the best
    competing label; on an unseen font the two distributions overlap, so the
    floor that works at home stops separating abroad. This is the same failure
    the goal grounder's presence gate shows across generators, measured on the
    text stream.
    """
    import torch

    correct: List[float] = []
    wrong: List[float] = []
    for font_path in fonts:
        for label in reader.labels:
            patch = render_word(label, font_path, size=28)
            frame = np.stack([(patch * 255).astype(np.uint8)] * 3, axis=-1)
            regions = reader.read_regions(frame)
            if not regions:
                continue
            with torch.no_grad():
                emb = reader.model.encode_patches(patch_from_crop(regions[0].box.crop(frame)))
                sims = (emb @ reader.model.prototypes.t())[0]
            idx = reader.labels.index(label)
            correct.append(float(sims[idx]))
            others = torch.cat([sims[:idx], sims[idx + 1:]])
            wrong.append(float(others.max()))
    if not correct:
        return {}
    c = np.asarray(correct)
    w = np.asarray(wrong)
    return {
        "correct_median": round(float(np.median(c)), 4),
        "correct_p10": round(float(np.percentile(c, 10)), 4),
        "wrong_median": round(float(np.median(w)), 4),
        "wrong_p90": round(float(np.percentile(w, 90)), 4),
        "separable": bool(np.percentile(c, 10) > np.percentile(w, 90)),
        "n": len(correct),
    }


def benchmark(
    checkpoint: Optional[str] = None,
    out: str = DEFAULT_OUT,
    seed: int = 7,
    verbose: bool = True,
) -> Dict[str, object]:
    train_fonts, held_fonts = split_fonts()
    reader = LexiconReader(checkpoint=checkpoint)
    report: Dict[str, object] = {
        "stream": "avc.lexicon",
        "checkpoint": checkpoint,
        "vocab_size": len(reader.labels),
        "train_fonts": [Path(f).name for f in train_fonts],
        "held_out_fonts": [Path(f).name for f in held_fonts],
        "closed_vocab": _patch_accuracy(reader, reader.labels, train_fonts, 3, seed),
        "gate_train_fonts": _gate_separability(reader, train_fonts, seed),
        "unknown_rejection": _unknown_rejection(reader, train_fonts, seed),
        "cycle_consistency": _cycle_consistency(reader, train_fonts, seed),
        "latency": _latency(reader, train_fonts[0] if train_fonts else ""),
    }
    if held_fonts:
        report["held_out_font"] = _patch_accuracy(reader, reader.labels, held_fonts, 3, seed + 1)
        report["gate_held_out_fonts"] = _gate_separability(reader, held_fonts, seed + 1)
        report["held_out_font_unknown_rejection"] = _unknown_rejection(reader, held_fonts, seed + 1)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
    if verbose:
        print(json.dumps(report, indent=2))
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark the AVC Lexicon stream")
    ap.add_argument("--checkpoint", type=str, default="runs/weights/avc_lexicon.pt")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    benchmark(checkpoint=args.checkpoint, out=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
