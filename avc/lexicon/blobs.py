"""Two-dimensional ink grouping, numpy-only.

A one-dimensional column projection cannot separate *bordered* controls: a
button's frame makes its ink span the full cell, so the gap between two buttons
looks the same as the gap between two letters. Projection works only when the
things being separated are bare words on a common baseline.

Real screens are not like that. The unit that has to be separated is a 2D blob,
so this module works on blobs:

1. Label connected ink components (union-find over 8-neighbour adjacency).
2. Discard specks below a size floor.
3. Merge components whose bounding boxes are close *relative to their height*,
   which glues the glyphs of one word together without gluing two controls.

There is no scipy here, so the labelling is a small union-find. It is O(pixels)
with path compression, and it runs on a binarised patch where most pixels are
background, so it stays cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

import numpy as np


class GlyphBoxLike(Protocol):
    """Anything with pixel bounds, so `boxes_overlap` is not tied to one class."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class Component:
    """One connected ink blob."""

    x0: int
    y0: int
    x1: int
    y1: int
    area: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression keeps repeated lookups flat on large blobs.
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _row_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """Ink runs ``(start, end)`` in one boolean row, end exclusive."""
    padded = np.concatenate(([False], row, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def label_components(mask: np.ndarray) -> List[Component]:
    """8-connected components of a boolean mask, as bounding boxes.

    Run-length connected components: label the ink runs per row, then union runs
    in adjacent rows that touch. A screen is mostly background, so each row has
    only a handful of runs and the union loop stays tiny — a per-pixel union-find
    over a full frame is ~100 ms in Python, which does not fit a screen clock.
    """
    if mask.size == 0 or not mask.any():
        return []
    height, width = mask.shape
    # Start with an EMPTY parent list and append one entry per run. Seeding it
    # with a placeholder makes run 1 a child of run 0, so every run collapses
    # into a single component and the whole frame labels as one blob.
    uf = _UnionFind(0)
    run_rows: List[List[int]] = []
    run_meta: List[Tuple[int, int, int]] = []  # (row, start, end)

    for y in range(height):
        ids = []
        for start, end in _row_runs(mask[y]):
            run_id = len(run_meta)
            run_meta.append((y, start, end))
            uf.parent.append(run_id)
            ids.append(run_id)
        run_rows.append(ids)

    for y in range(1, height):
        prev, curr = run_rows[y - 1], run_rows[y]
        if not prev or not curr:
            continue
        i = j = 0
        while i < len(prev) and j < len(curr):
            _, ps, pe = run_meta[prev[i]]
            _, cs, ce = run_meta[curr[j]]
            # 8-connectivity: runs touch when they overlap or are adjacent.
            if ps <= ce and cs <= pe:
                uf.union(prev[i], curr[j])
            if pe < ce:
                i += 1
            else:
                j += 1

    groups: dict[int, List[int]] = {}
    for run_id, (_, start, end) in enumerate(run_meta):
        groups.setdefault(uf.find(run_id), []).append(run_id)

    components: List[Component] = []
    for members in groups.values():
        y0 = min(run_meta[r][0] for r in members)
        y1 = max(run_meta[r][0] for r in members) + 1
        x0 = min(run_meta[r][1] for r in members)
        x1 = max(run_meta[r][2] for r in members)
        area = sum(run_meta[r][2] - run_meta[r][1] for r in members)
        components.append(Component(x0, y0, x1, y1, area))
    components.sort(key=lambda c: (c.y0, c.x0))
    return components


def split_containers(
    components: Sequence[Component],
    *,
    max_fill: float = 0.25,
) -> Tuple[List[Component], List[Component]]:
    """Separate control *frames* from the *glyphs* they contain.

    A component is a container when it is **hollow** (its ink fills a small
    fraction of its bounding box) **and** it **encloses** another component. A
    bordered button frame is hollow and encloses its label; a letter is either
    solid, or — if it has disconnected parts, like the two dots of a colon — too
    small to enclose anything meaningful.

    The hollow test is what the enclosure test alone cannot supply. A glyph such
    as 'a' can enclose a small speck in its own counter, and an enclosure-only
    rule then calls the letter a container, which splits a word into pieces. The
    fill ratio is measured, not guessed: on the fonts here a button frame fills
    0.06 of its box and every glyph fills at least 0.34, so 0.25 sits in an empty
    band. A *solid* filled button is not caught by this rule; it arrives as a
    glyph-shaped blob and its label is read directly, which is the right answer
    for a flat control with no visible border.
    """
    containers: List[Component] = []
    glyphs: List[Component] = []
    for i, comp in enumerate(components):
        box_area = max(1, comp.width * comp.height)
        hollow = (comp.area / box_area) < max_fill
        encloses = any(
            j != i
            and comp.x0 <= other.x0 and comp.y0 <= other.y0
            and comp.x1 >= other.x1 and comp.y1 >= other.y1
            and (other.width < comp.width or other.height < comp.height)
            for j, other in enumerate(components)
        )
        (containers if (hollow and encloses) else glyphs).append(comp)
    return containers, glyphs


def boxes_overlap(a: "GlyphBoxLike", b: "GlyphBoxLike") -> bool:
    """True when two boxes share area. Used to attach a word to its frame."""
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)


def merge_close_components(
    components: Sequence[Component],
    *,
    gap_ratio: float = 0.6,
) -> List[Component]:
    """Glue blobs that belong to one word.

    A blob joins a neighbour when it is close along one axis and overlaps along
    the other:

    - side by side — 'v' and 'a' in "save": small x gap, y overlap;
    - stacked — the dot of an 'i' above its stem: small y gap, x overlap.

    Both are needed. Testing only the horizontal case left the dot of an 'i' as a
    separate box and shredded the word.

    There is no height guard. One used to sit here to stop a button frame from
    absorbing the label beside it, but that also rejected a 5-pixel dot against a
    32-pixel word (ratio 6.4), which is the *normal* case inside one word. Frames
    are now separated structurally, before this function runs
    (:func:`split_containers`), so nothing here needs protecting from them and
    the gap ratio alone decides. The gap uses the taller blob because a word
    mixes ascenders with x-height glyphs, and scaling by the shorter one would
    refuse to join 'v' to 'a'.
    """
    items = list(components)
    if not items:
        return []
    merged = True
    while merged:
        merged = False
        out: List[Component] = []
        items.sort(key=lambda c: (c.y0, c.x0))
        for comp in items:
            for i, other in enumerate(out):
                x_gap = max(other.x0, comp.x0) - min(other.x1, comp.x1)
                y_gap = max(other.y0, comp.y0) - min(other.y1, comp.y1)
                y_overlap = min(comp.y1, other.y1) - max(comp.y0, other.y0)
                x_overlap = min(comp.x1, other.x1) - max(comp.x0, other.x0)
                span = max(comp.height, other.height)
                side_by_side = x_gap <= gap_ratio * span and y_overlap > 0
                stacked = y_gap <= gap_ratio * span and x_overlap > 0
                if side_by_side or stacked:
                    out[i] = Component(
                        min(comp.x0, other.x0), min(comp.y0, other.y0),
                        max(comp.x1, other.x1), max(comp.y1, other.y1),
                        comp.area + other.area,
                    )
                    merged = True
                    break
            else:
                out.append(comp)
        items = out
    items.sort(key=lambda c: (c.y0, c.x0))
    return items


def group_regions(
    components: Sequence[Component],
    *,
    gap_ratio: float = 0.6,
) -> Tuple[List[Component], List[Component]]:
    """Split raw components into control frames and word boxes.

    Containers are identified first, on the *raw* components, so a button frame
    can never absorb its own label during merging. Then the remaining glyphs are
    merged into words. This ordering is what lets one rule serve both a word
    patch (no containers, glyphs merge) and a screen (frames stay whole, labels
    inside them are dropped by the caller).
    """
    containers, glyphs = split_containers(components)
    words = merge_close_components(glyphs, gap_ratio=gap_ratio) if glyphs else []
    return containers, words
