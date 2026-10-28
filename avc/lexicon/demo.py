"""Read control labels off a screenshot, with no OCR.

    # Name every labelled region in an image
    python3 -m avc.lexicon.demo --image shot.png

    # Ground a spoken word: where is the Save button?
    python3 -m avc.lexicon.demo --image shot.png --find save

    # Draw the boxes and names to a copy, for eyeballing
    python3 -m avc.lexicon.demo --image shot.png --annotate out.png

    # No image on this box? Render a synthetic row and read that.
    python3 -m avc.lexicon.demo --synthetic "save cancel username"

This prints what the reader *actually* resolved. A region whose best match is
below the confidence floor is printed as ``?`` rather than guessed at, which is
the honest outcome when the ink is not a known control.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avc.lexicon.reader import LexiconReader
from avc.lexicon.render import split_fonts

DEFAULT_CHECKPOINT = "runs/weights/avc_lexicon.pt"


def _synthetic(words: Sequence[str], width: int = 0, height: int = 64) -> np.ndarray:
    fonts, _held = split_fonts()
    if not fonts:
        raise RuntimeError("no TrueType fonts available to render a synthetic frame")
    font = ImageFont.truetype(fonts[0], 28)
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    gap = 28
    widths = [probe.textbbox((0, 0), w, font=font)[2] for w in words]
    total = sum(widths) + gap * (len(words) + 1)
    image = Image.new("RGB", (width or max(64, total), height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    x = gap
    for w in words:
        draw.text((x, 14), w, font=font, fill=(25, 25, 25))
        x += probe.textbbox((0, 0), w, font=font)[2] + gap
    return np.asarray(image)


def _annotate(frame: np.ndarray, reader: LexiconReader, out: str) -> None:
    image = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for rw in reader.read_regions(frame):
        x0, y0, x1, y1 = rw.box.as_tuple()
        draw.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=2)
        text = rw.label if rw.known else "?"
        draw.text((x0, max(0, y0 - 12)), f"{text} {rw.readout.confidence:.2f}", fill=(255, 60, 60))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"wrote {out}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="AVC Lexicon: read control labels without OCR")
    ap.add_argument("--image", type=str, default=None, help="screenshot to read")
    ap.add_argument("--synthetic", type=str, default=None,
                    help="render these space-separated words instead of loading an image")
    ap.add_argument("--find", type=str, default=None,
                    help="ground a spoken word: report where that control is")
    ap.add_argument("--annotate", type=str, default=None, help="write an annotated copy here")
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--floor", type=float, default=0.55,
                    help="confidence floor; below it a region is reported as unknown")
    args = ap.parse_args(argv)

    if args.synthetic:
        frame = _synthetic(args.synthetic.split())
    elif args.image:
        frame = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        ap.error("pass --image FILE or --synthetic \"word word\"")

    ckpt = args.checkpoint if Path(args.checkpoint).exists() else None
    if ckpt is None:
        print(f"[note] no checkpoint at {args.checkpoint}; using an untrained model "
              f"(names will be meaningless — run avc.lexicon.train first)")
    reader = LexiconReader(checkpoint=ckpt, floor=args.floor)

    regions = reader.read_regions(frame)
    print(f"frame {frame.shape[1]}x{frame.shape[0]} | {len(regions)} region(s) | "
          f"vocab {len(reader.labels)} | floor {args.floor}")
    for rw in regions:
        mark = rw.label if rw.known else "?"
        print(f"  box={rw.box.as_tuple()} -> {mark:12s} "
              f"conf={rw.readout.confidence:.3f} margin={rw.readout.margin:.3f}")

    if args.find:
        hit = reader.find(frame, args.find)
        if hit is None:
            print(f"find({args.find!r}): not found above the floor (refused)")
        else:
            x0, y0, x1, y1 = hit.box.as_tuple()
            print(f"find({args.find!r}): centre=({(x0 + x1) // 2}, {(y0 + y1) // 2}) "
                  f"conf={hit.readout.confidence:.3f}")
            print(f"  cycle_consistent={reader.check_cycle(frame, args.find)}")

    if args.annotate:
        _annotate(frame, reader, args.annotate)


if __name__ == "__main__":
    main()
