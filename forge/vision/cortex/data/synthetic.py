"""Synthetic Procedural Desktop Generator with Ground-Truth Affordances.

Generates realistic desktop screens and produces corresponding:
- UI Category Label (editor, browser, terminal, modal, etc.)
- Dense Ground-Truth Affordance Masks (static, clickable, typeable, scrollable, draggable)
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

        Groups points by affordance class and samples an affordance class uniformly
        from those present in the scene, preventing the ~93% clickable majority-class
        collapse while keeping label-to-coordinate mapping learnable.
        """
        if not self.interactive_points:
            return
        by_class: dict[str, List[int]] = {}
        for idx, (_, _, cls) in enumerate(self.interactive_points):
            by_class.setdefault(cls, []).append(idx)
        chosen_cls = random.choice(list(by_class.keys()))
        idx = random.choice(by_class[chosen_cls])
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
        mask = np.zeros((self.height, self.width), dtype=np.int64)  # 0: static
        cursor_xy = (self.width // 2, self.height // 2)

        # Helper: Render standard Window Titlebar with 3 control buttons
        def render_titlebar(title_bg: Tuple[int, int, int] = (45, 45, 48)) -> int:
            h_bar = 28
            canvas[:h_bar, :] = title_bg
            for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
                bx = 16 + i * 18
                by = 14
                canvas[by - 5:by + 5, bx - 5:bx + 5] = col
                mask[by - 5:by + 5, bx - 5:bx + 5] = 1  # clickable
            return h_bar

        # 2. Render Category-Specific Distinct Elements
        if category == "desktop_canvas":
            # Desktop Wallpaper: distinctive vertical gradient (teal-to-deep-navy)
            for y_i in range(self.height):
                alpha = y_i / max(self.height - 1, 1)
                canvas[y_i, :] = (int(18 + 12 * alpha), int(38 + 22 * alpha), int(68 + 36 * alpha))
            # Desktop icons in a 2-column grid
            for col_idx, gx in enumerate([32, 96]):
                for row_idx, gy in enumerate(range(36, self.height - 60, 56)):
                    icon_color = (245, 195, 65) if (col_idx + row_idx) % 2 == 0 else (50, 165, 245)
                    canvas[gy:gy + 24, gx:gx + 24] = icon_color
                    mask[gy:gy + 24, gx:gx + 24] = 1  # clickable icon
                    canvas[gy + 26:gy + 32, gx - 4:gx + 28] = (210, 215, 225)

            # Bottom Taskbar
            tb_h = 36
            tb_y = self.height - tb_h
            canvas[tb_y:, :] = (16, 16, 20)
            canvas[tb_y + 6:tb_y + 30, 12:36] = (0, 120, 215)
            mask[tb_y + 6:tb_y + 30, 12:36] = 1  # clickable Start button
            canvas[tb_y + 12:tb_y + 24, self.width - 64:self.width - 12] = (130, 135, 145)

        elif category == "editor_code":
            h_bar = render_titlebar()
            # Code area
            canvas[h_bar:, :60] = (30, 30, 30)  # Line numbers gutter
            canvas[h_bar:, 60:self.width - 16] = (24, 24, 24)  # Editor background
            mask[h_bar:, 60:self.width - 16] = 2               # typeable area
            # Code lines
            for y in range(h_bar + 16, self.height - 20, 16):
                lw = random.randint(80, self.width - 140)
                canvas[y:y + 4, 75:75 + lw] = random.choice([(86, 156, 214), (220, 220, 170), (206, 145, 120)])
            # Vertical code scrollbar on right edge
            sb_x = self.width - 14
            canvas[h_bar:, sb_x:sb_x + 10] = (45, 45, 50)
            canvas[h_bar + 20:h_bar + 80, sb_x:sb_x + 10] = (90, 90, 100)
            mask[h_bar:, sb_x:sb_x + 10] = 3  # scrollable

        elif category == "terminal_shell":
            h_bar = render_titlebar((30, 30, 34))
            canvas[h_bar:, :] = (12, 12, 12)  # Dark terminal
            mask[h_bar:, :] = 2               # typeable
            # Terminal prompt
            canvas[h_bar + 15:h_bar + 25, 20:120] = (80, 250, 123)
            # Cursor
            cx, cy = 130, h_bar + 15
            canvas[cy:cy + 10, cx:cx + 6] = (255, 255, 255)
            cursor_xy = (cx, cy)

        elif category == "modal_dialog":
            h_bar = render_titlebar()
            # Dimmed background
            canvas[h_bar:, :] = (20, 20, 20)
            # Modal box
            mw, mh = int(self.width * 0.52), int(self.height * 0.52)
            mx = (self.width - mw) // 2
            my = (self.height - mh) // 2
            canvas[my:my + mh, mx:mx + mw] = (50, 50, 54)
            # Modal message lines
            canvas[my + 24:my + 30, mx + 20:mx + mw - 20] = (180, 180, 185)
            canvas[my + 38:my + 44, mx + 20:mx + mw - 60] = (150, 150, 155)

            # Cancel button (clickable)
            btn_w, btn_h = 70, 26
            cx_btn = mx + mw - (btn_w * 2) - 30
            cy_btn = my + mh - btn_h - 16
            canvas[cy_btn:cy_btn + btn_h, cx_btn:cx_btn + btn_w] = (75, 75, 80)
            mask[cy_btn:cy_btn + btn_h, cx_btn:cx_btn + btn_w] = 1  # clickable

            # Confirm button (clickable)
            bx = mx + mw - btn_w - 18
            by = my + mh - btn_h - 16
            canvas[by:by + btn_h, bx:bx + btn_w] = (0, 122, 255)
            mask[by:by + btn_h, bx:bx + btn_w] = 1  # clickable

        elif category == "browser_webpage":
            h_bar = render_titlebar((52, 53, 58))
            # Browser Tabs
            canvas[h_bar - 24:h_bar, 80:160] = (65, 68, 75)
            mask[h_bar - 24:h_bar, 80:160] = 1  # clickable tab
            canvas[h_bar - 24:h_bar, 164:240] = (42, 44, 48)
            mask[h_bar - 24:h_bar, 164:240] = 1  # clickable tab

            # Navigation bar (Back / Forward / Refresh)
            nav_y = h_bar + 6
            for btn_i in range(2):
                nx = 14 + btn_i * 26
                canvas[nav_y:nav_y + 18, nx:nx + 18] = (75, 78, 85)
                mask[nav_y:nav_y + 18, nx:nx + 18] = 1  # clickable

            # URL Address Bar (typeable)
            ax, ay, aw, ah = 76, nav_y, self.width - 150, 22
            canvas[ay:ay + ah, ax:ax + aw] = (245, 245, 248)
            mask[ay:ay + ah, ax:ax + aw] = 2  # typeable

            # Search/Go button
            bx, by, bw, bh = ax + aw + 8, nav_y, 50, 22
            canvas[by:by + bh, bx:bx + bw] = (0, 122, 204)
            mask[by:by + bh, bx:bx + bw] = 1  # clickable

            # Scrollable web feed area
            feed_top = nav_y + ah + 10
            canvas[feed_top:, 20:self.width - 20] = (40, 42, 48)
            mask[feed_top:, 20:self.width - 20] = 3  # scrollable

        elif category == "form_input":
            h_bar = render_titlebar((48, 50, 54))
            # Clean Form Card in distinct light-neutral theme
            cw, ch = int(self.width * 0.72), int(self.height * 0.74)
            cx = (self.width - cw) // 2
            cy = h_bar + 14
            canvas[cy:cy + ch, cx:cx + cw] = (235, 238, 245)

            # Form Title header strip (dark blue banner)
            canvas[cy:cy + 24, cx:cx + cw] = (0, 102, 204)

            # Form Input Field 1 (Username)
            f1_y = cy + 38
            canvas[f1_y:f1_y + 22, cx + 110:cx + cw - 30] = (255, 255, 255)
            mask[f1_y:f1_y + 22, cx + 110:cx + cw - 30] = 2  # typeable

            # Form Input Field 2 (Email / Key)
            f2_y = f1_y + 34
            canvas[f2_y:f2_y + 22, cx + 110:cx + cw - 30] = (255, 255, 255)
            mask[f2_y:f2_y + 22, cx + 110:cx + cw - 30] = 2  # typeable

            # Checkbox option
            cb_y = f2_y + 34
            canvas[cb_y:cb_y + 16, cx + 110:cx + 126] = (0, 122, 255)
            mask[cb_y:cb_y + 16, cx + 110:cx + 126] = 1  # clickable checkbox

            # Submit button
            sub_y = cb_y + 28
            canvas[sub_y:sub_y + 26, cx + 110:cx + 190] = (40, 167, 69)
            mask[sub_y:sub_y + 26, cx + 110:cx + 190] = 1  # clickable submit button

        elif category == "media_playback":
            h_bar = render_titlebar((24, 24, 28))
            # Main video display viewport
            video_bottom = self.height - 48
            canvas[h_bar:video_bottom, :] = (14, 16, 22)

            # Distinct Video Preview Artwork / Thumbnail in center
            vw, vh = int(self.width * 0.65), int((video_bottom - h_bar) * 0.72)
            vx = (self.width - vw) // 2
            vy = h_bar + ((video_bottom - h_bar) - vh) // 2
            canvas[vy:vy + vh, vx:vx + vw] = (30, 45, 65)
            canvas[vy + 10:vy + 30, vx + 20:vx + vw - 20] = (230, 90, 40)

            # Center Play icon (clickable, bright red circle)
            p_size = 36
            px0 = (self.width - p_size) // 2
            py0 = h_bar + (video_bottom - h_bar - p_size) // 2
            canvas[py0:py0 + p_size, px0:px0 + p_size] = (235, 45, 45)
            mask[py0:py0 + p_size, px0:px0 + p_size] = 1  # clickable

            # Bottom media control deck
            deck_y = video_bottom
            canvas[deck_y:, :] = (26, 26, 30)

            # Play/Pause button on control deck
            canvas[deck_y + 12:deck_y + 34, 16:38] = (200, 200, 210)
            mask[deck_y + 12:deck_y + 34, 16:38] = 1  # clickable

            # Timeline scrubber slider (track + draggable thumb knob)
            scrubber_x0 = 50
            scrubber_w = self.width - 150
            canvas[deck_y + 21:deck_y + 25, scrubber_x0:scrubber_x0 + scrubber_w] = (65, 68, 75)
            # Scrubber knob (draggable)
            knob_x = scrubber_x0 + 80
            canvas[deck_y + 16:deck_y + 30, knob_x:knob_x + 14] = (255, 255, 255)
            mask[deck_y + 16:deck_y + 30, knob_x:knob_x + 14] = 4  # draggable slider

            # Volume slider knob on right side (draggable)
            vol_x = self.width - 60
            canvas[deck_y + 22:deck_y + 25, vol_x:vol_x + 40] = (65, 68, 75)
            canvas[deck_y + 18:deck_y + 28, vol_x + 15:vol_x + 25] = (255, 255, 255)
            mask[deck_y + 18:deck_y + 28, vol_x + 15:vol_x + 25] = 4  # draggable slider

        elif category == "settings_panel":
            h_bar = render_titlebar((42, 43, 46))
            # Search settings input box (top right of window)
            canvas[h_bar + 6:h_bar + 26, self.width - 180:self.width - 20] = (240, 242, 246)
            mask[h_bar + 6:h_bar + 26, self.width - 180:self.width - 20] = 2  # typeable search

            # Left navigation sidebar
            sb_w = 120
            canvas[h_bar + 32:, :sb_w] = (34, 35, 38)
            for sy in range(h_bar + 42, self.height - 30, 36):
                canvas[sy:sy + 20, 14:sb_w - 14] = (75, 78, 85)
                mask[sy:sy + 20, 14:sb_w - 14] = 1  # clickable category

            # Main Settings items on right
            content_x = sb_w + 24
            # Item 1: Vivid Green Toggle Switch (clickable)
            tog_y = h_bar + 50
            canvas[tog_y:tog_y + 18, content_x + 160:content_x + 196] = (48, 209, 88)
            mask[tog_y:tog_y + 18, content_x + 160:content_x + 196] = 1  # clickable toggle

            # Item 2: Brightness Slider (draggable)
            sl_y = tog_y + 40
            canvas[sl_y + 6:sl_y + 10, content_x:content_x + 160] = (80, 82, 88)
            canvas[sl_y + 2:sl_y + 14, content_x + 60:content_x + 72] = (255, 255, 255)
            mask[sl_y + 2:sl_y + 14, content_x + 60:content_x + 72] = 4  # draggable slider

            # Item 3: Grey Toggle Switch (clickable)
            tog2_y = sl_y + 36
            canvas[tog2_y:tog2_y + 18, content_x + 160:content_x + 196] = (120, 122, 128)
            mask[tog2_y:tog2_y + 18, content_x + 160:content_x + 196] = 1  # clickable toggle

            # Item 4: Scrollable settings list container
            list_y = tog2_y + 32
            canvas[list_y:self.height - 20, content_x:self.width - 20] = (42, 44, 50)
            mask[list_y:self.height - 20, content_x:self.width - 20] = 3  # scrollable list

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
