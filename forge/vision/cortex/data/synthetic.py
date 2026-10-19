"""Synthetic Procedural Desktop Generator with Ground-Truth Affordances.

Generates realistic desktop screens and produces corresponding:
- UI Category Label (editor, browser, terminal, modal, etc.)
- Dense Ground-Truth Affordance Masks (static, clickable, typeable, scrollable)
- Interactive Elements with (x, y) target coordinates
- Temporal sequences (consecutive frames with animated cursor or modal emergence).
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from forge.vision.cortex.core.dorsal import AFFORDANCE_CLASSES
from forge.vision.cortex.core.ventral import UI_CATEGORIES


@dataclass
class SyntheticScreenData:
    frame: np.ndarray             # (H, W, 3) uint8 RGB
    affordance_mask: np.ndarray   # (H, W) int64 class indices [0..4]
    category: str                 # e.g. "editor_code", "modal_dialog"
    category_idx: int             # index into UI_CATEGORIES
    interactive_points: List[Tuple[float, float, str]] # (norm_x, norm_y, class_name)
    target_label: str = "element" # natural-language name of the first interactive point
    cursor_xy: Tuple[int, int] = (0, 0) # top-left pixel of the rendered cursor

    def interactive_label(self) -> str:
        """Natural-language query for the primary interactive point."""
        if self.target_label:
            return self.target_label
        if self.interactive_points:
            return self.interactive_points[0][2]
        return "element"

    def choose_primary(self) -> None:
        """Picks which interactive point the grounding query refers to.

        Without this the first point was always the same titlebar button, so the
        grounding target was one constant pixel for every frame and the query
        text carried no information. Selecting a random point per seed makes the
        label-to-coordinate mapping learnable.
        """
        if not self.interactive_points:
            return
        idx = random.randrange(len(self.interactive_points))
        self.interactive_points = (
            [self.interactive_points[idx]] + self.interactive_points[:idx] + self.interactive_points[idx + 1:]
        )
        cls = self.interactive_points[0][2]
        self.target_label = _AFFORDANCE_QUERY_NAMES.get(cls, cls)


# Human-readable names for the primary interactive element of each UI category.
_CATEGORY_TARGET_LABELS = {
    "desktop_canvas": "Desktop icon",
    "editor_code": "Code editor",
    "browser_webpage": "Search bar",
    "terminal_shell": "Terminal prompt",
    "modal_dialog": "Confirm button",
    "form_input": "Input field",
    "media_playback": "Play button",
    "settings_panel": "Settings item",
}

# Affordance class -> the natural-language name used as the grounding query.
# These are vocabulary, not decision rules: they name what a class is.
_AFFORDANCE_QUERY_NAMES = {
    "clickable": "button",
    "typeable": "input field",
    "scrollable": "scroll area",
    "draggable": "slider",
}


def connected_components(grid: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """4-connected component labelling without scipy.

    `grid` is a small (H, W) int array of class indices. Returns bounding boxes
    as (class_idx, x0, y0, x1, y1) for every non-zero component, largest first.

    Runs of equal class within a row are labelled, then overlapping runs in
    adjacent rows are unioned. Row-level run encoding keeps this linear in the
    number of runs rather than pixels, which matters when the grid is the full
    resolution frame.
    """
    H, W = grid.shape
    parent: List[int] = []
    run_cls: List[int] = []
    run_box: List[List[int]] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra

    prev_runs: List[Tuple[int, int, int]] = []  # (x0, x1, node)
    for y in range(H):
        row = grid[y]
        # Run boundaries via diff rather than a per-pixel Python scan; the mask
        # is full resolution when the generator derives its points.
        bounds = np.flatnonzero(row[1:] != row[:-1]) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [W]))
        cur_runs: List[Tuple[int, int, int]] = []
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            cls = int(row[x0])
            if cls == 0:
                continue
            node = len(parent)
            parent.append(node)
            run_cls.append(cls)
            run_box.append([x0, y, x1 - 1, y])
            cur_runs.append((x0, x1 - 1, node))
        for a0, a1, an in prev_runs:
            for b0, b1, bn in cur_runs:
                if a0 <= b1 and b0 <= a1 and run_cls[an] == run_cls[bn]:
                    union(an, bn)
        prev_runs = cur_runs

    merged: dict[int, List[int]] = {}
    for node in range(len(parent)):
        root = find(node)
        box = merged.get(root)
        if box is None:
            merged[root] = run_box[node][:]
        else:
            box[0] = min(box[0], run_box[node][0])
            box[1] = min(box[1], run_box[node][1])
            box[2] = max(box[2], run_box[node][2])
            box[3] = max(box[3], run_box[node][3])

    boxes = [(run_cls[r], b[0], b[1], b[2], b[3]) for r, b in merged.items()]
    boxes.sort(key=lambda t: (t[2] - t[0] + 1) * (t[3] - t[1] + 1), reverse=True)
    return boxes


def points_from_mask(
    mask: np.ndarray,
) -> List[Tuple[float, float, str]]:
    """Derives interactive points from the rendered affordance mask.

    The mask is the ground truth the generator already paints; the interactive
    points are its connected components. Deriving them here means a scene can
    never advertise an element the mask does not contain (and vice versa), which
    is what happened when the point list was maintained by hand.
    """
    H, W = mask.shape
    points: List[Tuple[float, float, str]] = []
    for cls_idx, x0, y0, x1, y1 in connected_components(mask):
        name = AFFORDANCE_CLASSES[cls_idx] if cls_idx < len(AFFORDANCE_CLASSES) else "static"
        if name not in _AFFORDANCE_QUERY_NAMES:
            continue
        points.append((((x0 + x1) / 2.0) / W, ((y0 + y1) / 2.0) / H, name))
    return points

# Background color painted behind the cursor in each rendered scene, restored
# when the cursor moves so the successor frame stays otherwise identical.
_CURSOR_BACKDROP = np.array([32, 32, 32], dtype=np.uint8)


class ProceduralScreenGenerator:
    """Renders structured desktop scenes with pixel-level ground truth."""

    def __init__(self, width: int = 640, height: int = 360) -> None:
        self.width = width
        self.height = height

    def generate(self, category: str | None = None, seed: int | None = None) -> SyntheticScreenData:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if category is None or category not in UI_CATEGORIES:
            category = random.choice(UI_CATEGORIES)

        cat_idx = UI_CATEGORIES.index(category)

        # Base canvas
        canvas = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        mask = np.zeros((self.height, self.width), dtype=np.int64) # 0: static
        cursor_xy = (self.width // 2, self.height // 2)

        # 1. Render Window Header (Titlebar)
        h_bar = 28
        canvas[:h_bar, :] = (45, 45, 48)
        # Window control buttons (clickable)
        for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            bx = 16 + i * 18
            by = 14
            canvas[by - 5:by + 5, bx - 5:bx + 5] = col
            mask[by - 5:by + 5, bx - 5:bx + 5] = 1 # clickable

        # 2. Render Category-Specific Elements
        if category == "editor_code":
            # Code area
            canvas[h_bar:, :60] = (30, 30, 30) # Line numbers gutter
            canvas[h_bar:, 60:] = (24, 24, 24) # Editor background
            mask[h_bar:, 60:] = 2              # typeable area
            # Fake code lines
            for y in range(h_bar + 16, self.height - 20, 16):
                lw = random.randint(80, self.width - 120)
                canvas[y:y + 4, 75:75 + lw] = random.choice([(86, 156, 214), (220, 220, 170), (206, 145, 120)])

        elif category == "terminal_shell":
            canvas[h_bar:, :] = (12, 12, 12) # Dark terminal
            mask[h_bar:, :] = 2              # typeable
            # Terminal prompt
            canvas[h_bar + 15:h_bar + 25, 20:120] = (80, 250, 123)
            # Cursor
            cx, cy = 130, h_bar + 15
            canvas[cy:cy + 10, cx:cx + 6] = (255, 255, 255)
            cursor_xy = (cx, cy)

        elif category == "modal_dialog":
            # Dimmed background
            canvas[h_bar:, :] = (20, 20, 20)
            # Modal box
            mw, mh = int(self.width * 0.5), int(self.height * 0.5)
            mx = (self.width - mw) // 2
            my = (self.height - mh) // 2
            canvas[my:my + mh, mx:mx + mw] = (50, 50, 54)

            # Confirm button (clickable)
            btn_w, btn_h = 70, 26
            bx = mx + mw - btn_w - 20
            by = my + mh - btn_h - 16
            canvas[by:by + btn_h, bx:bx + btn_w] = (0, 122, 255)
            mask[by:by + btn_h, bx:bx + btn_w] = 1 # clickable

        elif category in ("browser_webpage", "form_input"):
            # Web address bar or search box (typeable)
            ax, ay, aw, ah = 80, 36, self.width - 160, 24
            canvas[ay:ay + ah, ax:ax + aw] = (255, 255, 255)
            mask[ay:ay + ah, ax:ax + aw] = 2 # typeable

            # Submit / Search Button (clickable)
            bx, by, bw, bh = ax + aw - 60, ay + 2, 56, 20
            canvas[by:by + bh, bx:bx + bw] = (0, 122, 204)
            mask[by:by + bh, bx:bx + bw] = 1 # clickable

            # Scrollable main feed
            canvas[ay + ah + 15:, 50:self.width - 50] = (40, 42, 48)
            mask[ay + ah + 15:, 50:self.width - 50] = 3 # scrollable

        else: # settings_panel, desktop_canvas, media_playback
            # Sidebar navigation (clickable items)
            canvas[h_bar:, :120] = (37, 37, 38)
            for sy in range(h_bar + 16, self.height - 40, 32):
                canvas[sy:sy + 18, 16:104] = (62, 62, 66)
                mask[sy:sy + 18, 16:104] = 1 # clickable

        # Interactive points are the connected components of the mask the scene
        # just painted, not a separately maintained list.
        points = points_from_mask(mask)

        data = SyntheticScreenData(
            frame=canvas,
            affordance_mask=mask,
            category=category,
            category_idx=cat_idx,
            interactive_points=points,
            target_label=_CATEGORY_TARGET_LABELS.get(category, "element"),
            cursor_xy=cursor_xy,
        )
        data.choose_primary()
        return data

    def generate_successor(self, data: SyntheticScreenData) -> SyntheticScreenData:
        """Returns a temporally plausible next frame of the same scene.

        Simulates what actually changes between adjacent desktop frames: the
        cursor moves a few pixels. Structure, category and affordances are
        preserved, so the world model can learn a non-trivial transition
        (predict the cursor motion) instead of a constant identity map.
        """
        frame = data.frame.copy()
        h, w = frame.shape[:2]

        rng = np.random.RandomState(zlib.crc32(data.category.encode("utf-8")) % (2**31))
        dx = int(rng.randint(-12, 13))
        dy = int(rng.randint(-12, 13))

        # Erase only the cursor we drew, using its recorded position. Matching on
        # white pixels would also erase white UI chrome such as address bars.
        ox, oy = data.cursor_xy
        frame[oy : oy + 10, ox : ox + 6] = _CURSOR_BACKDROP

        cy = int(np.clip(oy + dy, 8, h - 12))
        cx = int(np.clip(ox + dx, 8, w - 12))
        frame[cy : cy + 10, cx : cx + 6] = (255, 255, 255)

        return SyntheticScreenData(
            frame=frame,
            affordance_mask=data.affordance_mask,
            category=data.category,
            category_idx=data.category_idx,
            interactive_points=data.interactive_points,
            target_label=data.target_label,
            cursor_xy=(cx, cy),
        )

