"""Desktop layout composer across 8 application categories and negative controls with full layout jittering."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from avcs.dataset.synthetic.primitives import (
    SyntheticUIElement,
    draw_background,
    draw_button,
    draw_icon,
    draw_input,
    draw_tab,
    draw_window,
    get_font,
)
from avcs.dataset.synthetic.query_synth import sample_negative_queries
from avcs.dataset.synthetic.themes import AVAILABLE_THEMES, ThemePalette, get_theme
from avcs.query_limits import MAX_QUERY_BYTES

UI_CATEGORIES = [
    "desktop_canvas",
    "editor_code",
    "browser_webpage",
    "terminal_shell",
    "modal_dialog",
    "form_input",
    "media_playback",
    "settings_panel",
]


def jittered_draw_background(
    draw: ImageDraw.ImageDraw,
    size: Tuple[int, int],
    theme: ThemePalette,
    bg_type: str = "solid",
    rng: Optional[random.Random] = None,
) -> None:
    """Jittered background drawing using rng for complete visual uniqueness."""
    w, h = size
    r = rng or random.Random(42)

    if bg_type == "solid":
        base_c = theme.bg_color
        r_j = r.randint(-22, 22)
        g_j = r.randint(-22, 22)
        b_j = r.randint(-22, 22)
        solid_color = (
            max(0, min(255, base_c[0] + r_j)),
            max(0, min(255, base_c[1] + g_j)),
            max(0, min(255, base_c[2] + b_j)),
        )
        draw.rectangle([0, 0, w, h], fill=solid_color)
        # Subtle perimeter vignette border
        edge_c = (
            max(0, solid_color[0] - r.randint(6, 14)),
            max(0, solid_color[1] - r.randint(6, 14)),
            max(0, solid_color[2] - r.randint(6, 14)),
        )
        draw.rectangle([0, 0, w - 1, h - 1], outline=edge_c, width=1)

    elif bg_type == "gradient":
        g_style = r.choice(["v", "h", "diag"])
        base1 = np.array(theme.bg_color, dtype=np.float32)
        base2 = np.array(theme.accent_color, dtype=np.float32) * r.uniform(0.35, 0.65)
        c1 = np.clip(base1 + np.array([r.randint(-18, 18), r.randint(-18, 18), r.randint(-18, 18)]), 0, 255)
        c2 = np.clip(base2 + np.array([r.randint(-18, 18), r.randint(-18, 18), r.randint(-18, 18)]), 0, 255)
        gamma = r.uniform(0.75, 1.35)

        if g_style == "v":
            for y in range(h):
                t = (y / max(1, h - 1)) ** gamma
                color = tuple(np.clip(c1 * (1.0 - t) + c2 * t, 0, 255).astype(int))
                draw.line([(0, y), (w, y)], fill=color)
        elif g_style == "h":
            for x in range(w):
                t = (x / max(1, w - 1)) ** gamma
                color = tuple(np.clip(c1 * (1.0 - t) + c2 * t, 0, 255).astype(int))
                draw.line([(x, 0), (x, h)], fill=color)
        else:  # diagonal
            for s in range(w + h):
                t = (s / max(1, w + h - 2)) ** gamma
                color = tuple(np.clip(c1 * (1.0 - t) + c2 * t, 0, 255).astype(int))
                draw.line([(max(0, s - h), min(s, h - 1)), (min(s, w - 1), max(0, s - w))], fill=color)

    elif bg_type == "radial_gradient":
        cx = w // 2 + r.randint(-25, 25)
        cy = h // 2 + r.randint(-25, 25)
        max_dist = math.hypot(max(cx, w - cx), max(cy, h - cy))
        c1 = np.array(theme.accent_color, dtype=np.float32) * r.uniform(0.35, 0.65)
        c2 = np.array(theme.bg_color, dtype=np.float32)
        gamma = r.uniform(0.7, 1.2)
        for y in range(h):
            for x in range(w):
                d = math.hypot(x - cx, y - cy) / max_dist
                t = min(1.0, d) ** gamma
                color = tuple(np.clip(c1 * (1 - t) + c2 * t, 0, 255).astype(int))
                draw.point((x, y), fill=color)

    elif bg_type == "wallpaper":
        palettes = [
            ((18, 24, 48), (78, 42, 85), (196, 92, 54), (45, 25, 55), (20, 15, 30)),
            ((15, 25, 50), (45, 65, 110), (220, 140, 70), (40, 45, 70), (25, 25, 45)),
            ((8, 12, 28), (20, 45, 55), (40, 140, 110), (18, 28, 42), (10, 16, 26)),
            ((25, 10, 40), (80, 20, 70), (180, 50, 110), (45, 15, 45), (22, 10, 28)),
            ((15, 30, 28), (35, 65, 50), (120, 150, 110), (30, 50, 40), (18, 30, 24)),
        ]
        sky_top, sky_mid, sky_low, r1_c, r2_c = r.choice(palettes)
        j_amt = r.randint(-8, 8)
        sky_top = tuple(max(0, min(255, c + j_amt)) for c in sky_top)
        sky_mid = tuple(max(0, min(255, c + j_amt)) for c in sky_mid)
        sky_low = tuple(max(0, min(255, c + j_amt)) for c in sky_low)

        split_y = int(h * r.uniform(0.60, 0.74))
        for y in range(split_y):
            t = y / max(1, split_y)
            if t < 0.5:
                u = t * 2.0
                c = tuple(int(sky_top[i] * (1 - u) + sky_mid[i] * u) for i in range(3))
            else:
                u = (t - 0.5) * 2.0
                c = tuple(int(sky_mid[i] * (1 - u) + sky_low[i] * u) for i in range(3))
            draw.line([(0, y), (w, y)], fill=c)

        # Celestial body (sun / moon)
        if r.random() < 0.85:
            sun_x = r.randint(int(w * 0.18), int(w * 0.82))
            sun_y = r.randint(int(h * 0.12), int(h * 0.42))
            sun_r = r.randint(8, 18)
            sun_color = r.choice([(255, 230, 180), (255, 200, 120), (220, 240, 255), (255, 160, 80)])
            draw.ellipse([sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r], fill=sun_color)

        # Star field
        num_stars = r.randint(6, 26)
        for _ in range(num_stars):
            sx = r.randint(2, w - 3)
            sy = r.randint(2, int(split_y * 0.7))
            s_bright = r.randint(180, 255)
            draw.point((sx, sy), fill=(s_bright, s_bright, s_bright))

        # Distant ridge
        phase1 = r.uniform(0.0, 6.28)
        freq1 = r.uniform(0.03, 0.06)
        amp1 = r.uniform(0.05, 0.10) * h
        base_y1 = h * r.uniform(0.55, 0.65)
        ridge_pts = [(0, int(base_y1))]
        for x in range(0, w + 16, 16):
            ry = int(base_y1 + math.sin(x * freq1 + phase1) * amp1 + math.cos(x * 0.02) * (h * 0.03))
            ridge_pts.append((x, ry))
        ridge_pts.extend([(w, h), (0, h)])
        draw.polygon(ridge_pts, fill=r1_c)

        # Foreground ridge
        phase2 = r.uniform(0.0, 6.28)
        freq2 = r.uniform(0.05, 0.09)
        amp2 = r.uniform(0.04, 0.08) * h
        base_y2 = h * r.uniform(0.68, 0.78)
        fg_pts = [(0, int(base_y2))]
        for x in range(0, w + 16, 16):
            ry = int(base_y2 + math.sin(x * freq2 + phase2) * amp2)
            fg_pts.append((x, ry))
        fg_pts.extend([(w, h), (0, h)])
        draw.polygon(fg_pts, fill=r2_c)
    else:
        draw.rectangle([0, 0, w, h], fill=theme.bg_color)


class DesktopComposer:
    """Desktop layout composer with deep rng jittering across all UI components and categories."""

    def __init__(
        self,
        screen_size: Tuple[int, int] = (256, 256),
        grid_size: Tuple[int, int] = (16, 16),
    ) -> None:
        self.sw, self.sh = screen_size
        self.gw, self.gh = grid_size

    def compose(
        self,
        category: str = "desktop_canvas",
        theme_name: str = "dark_modern",
        is_negative: bool = False,
        negative_type: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[SyntheticUIElement], List[str]]:
        rng = random.Random(seed)
        theme = get_theme(theme_name)

        img = Image.new("RGB", (self.sw, self.sh), theme.bg_color)
        draw = ImageDraw.Draw(img)

        if is_negative:
            elements: List[SyntheticUIElement] = []
            neg_type = negative_type or rng.choice(["solid", "gradient", "wallpaper"])
            jittered_draw_background(draw, (self.sw, self.sh), theme, bg_type=neg_type, rng=rng)
            mask = np.zeros((self.gh, self.gw), dtype=np.int64)
            neg_queries = sample_negative_queries(present_queries=[], count=5, rng=rng)
            return np.array(img, dtype=np.uint8), mask, elements, neg_queries

        # Positive category generation
        jittered_draw_background(draw, (self.sw, self.sh), theme, bg_type="solid", rng=rng)
        elements = []

        if category == "desktop_canvas":
            elements = self._compose_desktop_canvas(draw, theme, rng)
        elif category == "editor_code":
            elements = self._compose_editor_code(draw, theme, rng)
        elif category == "browser_webpage":
            elements = self._compose_browser_webpage(draw, theme, rng)
        elif category == "terminal_shell":
            elements = self._compose_terminal_shell(draw, theme, rng)
        elif category == "modal_dialog":
            elements = self._compose_modal_dialog(draw, theme, rng)
        elif category == "form_input":
            elements = self._compose_form_input(draw, theme, rng)
        elif category == "media_playback":
            elements = self._compose_media_playback(draw, theme, rng)
        elif category == "settings_panel":
            elements = self._compose_settings_panel(draw, theme, rng)
        else:
            elements = self._compose_desktop_canvas(draw, theme, rng)

        # Ensure that every interactive element rendered within a single frame has a unique query label
        seen_queries: Dict[str, int] = {}
        for el in elements:
            if el.action_type in ("static",) or el.element_kind in ("container",):
                continue
            if el.target_query in seen_queries:
                seen_queries[el.target_query] += 1
                el.target_query = f"{el.target_query} ({seen_queries[el.target_query]})"
            else:
                seen_queries[el.target_query] = 1

        mask = self._build_affordance_mask(elements)

        present_queries = [el.target_query for el in elements]
        for el in elements:
            present_queries.extend(el.query_variations)
        neg_queries = sample_negative_queries(present_queries, count=5, rng=rng)

        return np.array(img, dtype=np.uint8), mask, elements, neg_queries

    def _build_affordance_mask(self, elements: List[SyntheticUIElement]) -> np.ndarray:
        mask = np.zeros((self.gh, self.gw), dtype=np.int64)
        for el in elements:
            if el.action_type == "static" or el.element_kind == "container":
                continue
            x1, y1, x2, y2 = el.bbox_px
            gx1 = max(0, min(self.gw - 1, int(math.floor(x1 * self.gw / self.sw))))
            gy1 = max(0, min(self.gh - 1, int(math.floor(y1 * self.gh / self.sh))))
            gx2 = max(0, min(self.gw - 1, int(math.ceil(x2 * self.gw / self.sw))))
            gy2 = max(0, min(self.gh - 1, int(math.ceil(y2 * self.gh / self.sh))))

            val = 2 if el.action_type == "typeable" else 1
            mask[gy1 : max(gy1 + 1, gy2), gx1 : max(gx1 + 1, gx2)] = np.maximum(
                mask[gy1 : max(gy1 + 1, gy2), gx1 : max(gx1 + 1, gx2)], val
            )
        return mask

    def _compose_desktop_canvas(
        self, draw: ImageDraw.ImageDraw, theme: ThemePalette, rng: random.Random
    ) -> List[SyntheticUIElement]:
        elements: List[SyntheticUIElement] = []
        w, h = self.sw, self.sh

        # Procedural wallpaper with seed variation
        jittered_draw_background(draw, (w, h), theme, bg_type="wallpaper", rng=rng)

        # Jittered taskbar
        tb_h = rng.randint(22, 26)
        tb_y1 = h - tb_h
        draw.rectangle([0, tb_y1, w, h], fill=theme.taskbar_bg, outline=theme.border_color)

        # Start button with jittered width, text, and offset
        start_text = rng.choice(["Start", "Menu", "Apps", "Windows", "Launch"])
        start_w = rng.randint(38, 50)
        start_x1 = rng.randint(3, 8)
        start_btn = draw_button(
            draw,
            bbox=(start_x1, tb_y1 + 2, start_x1 + start_w, h - 2),
            text=start_text,
            theme=theme,
            element_id="taskbar_start",
            screen_size=(w, h),
            variant="default",
            context="taskbar",
            font_size=9,
        )
        elements.append(start_btn)

        # Search bar on taskbar with jittered position and placeholder
        search_x1 = start_x1 + start_w + rng.randint(4, 9)
        search_w = rng.randint(60, 85)
        search_placeholder = rng.choice(["Search...", "Type here", "Search apps", "Find files..."])
        search_in = draw_input(
            draw,
            bbox=(search_x1, tb_y1 + 2, search_x1 + search_w, h - 2),
            text="",
            placeholder=search_placeholder,
            theme=theme,
            element_id="taskbar_search",
            screen_size=(w, h),
            is_focused=False,
            context="taskbar",
            font_size=8,
        )
        elements.append(search_in)

        # Desktop shortcut icons with jittered placement, selection, and coordinates
        all_icons = [
            ("terminal", "Terminal"),
            ("browser", "Browser"),
            ("code", "Code"),
            ("settings", "Settings"),
            ("folder", "Files"),
            ("document", "Notes"),
            ("media", "Music"),
            ("folder", "Projects"),
        ]
        num_icons = rng.randint(4, 6)
        selected_icons = rng.sample(all_icons, num_icons)

        col1_x = rng.randint(10, 16)
        col2_x = rng.randint(62, 70)
        icon_w = rng.randint(38, 44)
        icon_h = rng.randint(42, 48)
        row_dy = rng.randint(52, 56)
        base_y = rng.randint(12, 18)

        for idx, (itype, label) in enumerate(selected_icons):
            col_x = col1_x if idx < 3 else col2_x
            row_idx = idx if idx < 3 else idx - 3
            ix1 = col_x + rng.randint(-2, 3)
            iy1 = base_y + row_idx * row_dy + rng.randint(-2, 3)
            ix2 = ix1 + icon_w
            iy2 = iy1 + icon_h

            icon_el = draw_icon(
                draw,
                bbox=(ix1, iy1, ix2, iy2),
                icon_type=itype,
                label=label,
                theme=theme,
                element_id=f"desktop_icon_{idx}",
                screen_size=(w, h),
            )
            elements.append(icon_el)

        return elements

    def _compose_editor_code(
        self, draw: ImageDraw.ImageDraw, theme: ThemePalette, rng: random.Random
    ) -> List[SyntheticUIElement]:
        elements: List[SyntheticUIElement] = []
        w, h = self.sw, self.sh

        # Jitter window bounds
        wx1 = rng.randint(5, 12)
        wy1 = rng.randint(5, 12)
        wx2 = w - rng.randint(5, 12)
        wy2 = h - rng.randint(5, 12)
        win_box = (wx1, wy1, wx2, wy2)

        fname = rng.choice(["main.py", "app.py", "model.py", "train.py", "pipeline.py", "server.py", "index.ts", "utils.py"])
        ed_name = rng.choice(["Code Editor", "VS Code", "PyCharm", "Sublime", "IDE"])
        title = f"{fname} - {ed_name}"

        win_el, controls = draw_window(draw, win_box, title, theme, "win_editor", (w, h))
        elements.extend(controls)

        # Menu bar
        menus = rng.choice([["File", "Edit", "View", "Run"], ["File", "Edit", "Selection", "Run"], ["File", "Tools", "Run"]])
        mx = wx1 + rng.randint(4, 8)
        my = wy1 + 24
        for idx, mtext in enumerate(menus):
            mw = rng.randint(24, 28)
            mbtn = draw_button(
                draw,
                bbox=(mx, my, mx + mw, my + 13),
                text=mtext,
                theme=theme,
                element_id=f"menu_{idx}",
                screen_size=(w, h),
                variant="default",
                font_size=8,
            )
            elements.append(mbtn)
            mx += mw + rng.randint(2, 5)

        # Run action button on top right of editor (disambiguated from menu bar)
        run_text = rng.choice(["Run Code", "Debug Script", "Execute Code", "Start Code"])
        run_w = rng.randint(56, 68)
        run_rx2 = wx2 - rng.randint(8, 16)
        run_btn = draw_button(
            draw,
            bbox=(run_rx2 - run_w, my - 1, run_rx2, my + 14),
            text=run_text,
            theme=theme,
            element_id="btn_run_code",
            screen_size=(w, h),
            variant="success",
            color_hint="green",
            context="toolbar",
            font_size=9,
        )
        elements.append(run_btn)

        # Tab strip
        tab_y = wy1 + 42
        tab1_w = rng.randint(58, 68)
        tab1 = draw_tab(
            draw,
            bbox=(wx1 + 4, tab_y, wx1 + 4 + tab1_w, tab_y + 16),
            text=fname,
            theme=theme,
            element_id="tab_main_file",
            screen_size=(w, h),
            is_active=True,
            window_id="win_editor",
        )
        sec_fname = rng.choice(["test.py", "config.json", "utils.py", "data.py", "README.md"])
        tab2_w = rng.randint(55, 65)
        tab2 = draw_tab(
            draw,
            bbox=(wx1 + 6 + tab1_w, tab_y, wx1 + 6 + tab1_w + tab2_w, tab_y + 16),
            text=sec_fname,
            theme=theme,
            element_id="tab_second_file",
            screen_size=(w, h),
            is_active=False,
            window_id="win_editor",
        )
        elements.extend([tab1, tab2])

        # Editor code lines
        ey1 = tab_y + 20
        font_code = get_font(8, bold=False)
        snippets = [
            ["import os", "import torch", "def train():", "    model = Eye()", "    return model"],
            ["from cortex import Eye", "async def run(data):", "    feat = extract(data)", "    return feat", "print('ready')"],
            ["import numpy as np", "class VisionHead:", "    def forward(self, x):", "        return x * 2", "model = VisionHead()"],
            ["import json, sys", "def parse(path):", "    with open(path) as f:", "        return json.load(f)", "cfg = parse('c.json')"],
        ]
        code_lines = rng.choice(snippets)
        for i, cline in enumerate(code_lines):
            draw.text((wx1 + 10, ey1 + i * 14), cline, fill=theme.text_color, font=font_code)

        # Search / Replace input at bottom of editor
        find_placeholder = rng.choice(["Find in file...", "Search symbol...", "Find...", "Search code..."])
        find_w = rng.randint(110, 135)
        search_el = draw_input(
            draw,
            bbox=(wx1 + 6, wy2 - 24, wx1 + 6 + find_w, wy2 - 6),
            text="",
            placeholder=find_placeholder,
            theme=theme,
            element_id="editor_find_input",
            screen_size=(w, h),
            is_focused=True,
            has_caret=True,
            context="code editor",
            font_size=8,
        )
        elements.append(search_el)

        return elements

    def _compose_browser_webpage(
        self, draw: ImageDraw.ImageDraw, theme: ThemePalette, rng: random.Random
    ) -> List[SyntheticUIElement]:
        elements: List[SyntheticUIElement] = []
        w, h = self.sw, self.sh

        # Window
        wx1 = rng.randint(4, 10)
        wy1 = rng.randint(4, 10)
        wx2 = w - rng.randint(4, 10)
        wy2 = h - rng.randint(4, 10)
        browser_name = rng.choice(["Google Chrome", "Mozilla Firefox", "Microsoft Edge", "Safari", "Brave"])
        win_el, controls = draw_window(
            draw, (wx1, wy1, wx2, wy2), browser_name, theme, "win_browser", (w, h)
        )
        elements.extend(controls)

        # Tab bar
        tab_text = rng.choice(["Search Web", "Documentation", "Dashboard", "GitHub Repo", "Tech News", "Wikipedia"])
        tab_w = rng.randint(75, 95)
        tab_browser = draw_tab(
            draw,
            bbox=(wx1 + 4, wy1 + 22, wx1 + 4 + tab_w, wy1 + 38),
            text=tab_text,
            theme=theme,
            element_id="tab_browser_1",
            screen_size=(w, h),
            is_active=True,
        )
        elements.append(tab_browser)

        # Nav buttons
        btn_y1 = wy1 + 42
        btn_y2 = btn_y1 + 16
        back_w = rng.randint(14, 18)
        back_btn = draw_button(
            draw,
            bbox=(wx1 + 4, btn_y1, wx1 + 4 + back_w, btn_y2),
            text="<",
            theme=theme,
            element_id="btn_browser_back",
            screen_size=(w, h),
            font_size=9,
        )
        fwd_x1 = wx1 + 6 + back_w
        fwd_w = rng.randint(14, 18)
        fwd_btn = draw_button(
            draw,
            bbox=(fwd_x1, btn_y1, fwd_x1 + fwd_w, btn_y2),
            text=">",
            theme=theme,
            element_id="btn_browser_fwd",
            screen_size=(w, h),
            font_size=9,
        )
        ref_x1 = fwd_x1 + fwd_w + 2
        ref_w = rng.randint(14, 18)
        ref_btn = draw_button(
            draw,
            bbox=(ref_x1, btn_y1, ref_x1 + ref_w, btn_y2),
            text="R",
            theme=theme,
            element_id="btn_browser_refresh",
            screen_size=(w, h),
            font_size=9,
        )
        elements.extend([back_btn, fwd_btn, ref_btn])

        # Address / URL bar
        url_text = rng.choice([
            "https://search.engine.com",
            "https://docs.python.org",
            "https://github.com/repo",
            "https://news.ycombinator.com",
            "https://en.wikipedia.org",
            "https://platform.openai.com",
        ])
        url_x1 = ref_x1 + ref_w + rng.randint(4, 8)
        url_input = draw_input(
            draw,
            bbox=(url_x1, btn_y1, wx2 - rng.randint(6, 12), btn_y2),
            text=url_text,
            placeholder="Type URL or search",
            theme=theme,
            element_id="input_browser_url",
            screen_size=(w, h),
            is_focused=False,
            font_size=8,
        )
        elements.append(url_input)

        # Webpage content body
        page_y1 = btn_y2 + 4
        draw.rectangle([wx1 + 2, page_y1, wx2 - 2, wy2 - 2], fill=theme.input_bg)

        # Centered search box on webpage
        page_search_y = page_y1 + rng.randint(30, 48)
        page_search_h = rng.randint(20, 24)
        page_search_w = rng.randint(160, 200)
        page_search_x1 = (w - page_search_w) // 2 + rng.randint(-5, 5)
        page_search_placeholder = rng.choice(["Search the web...", "Search Google or type URL", "Search docs...", "Enter search terms..."])
        page_search = draw_input(
            draw,
            bbox=(page_search_x1, page_search_y, page_search_x1 + page_search_w, page_search_y + page_search_h),
            text="",
            placeholder=page_search_placeholder,
            theme=theme,
            element_id="input_page_search",
            screen_size=(w, h),
            is_focused=True,
            has_caret=True,
            context="webpage",
            font_size=9,
        )
        elements.append(page_search)

        # Webpage action buttons
        act_y = page_search_y + page_search_h + rng.randint(10, 16)
        btn1_text = rng.choice(["Search", "Google Search", "Find", "Go"])
        btn1_w = rng.randint(60, 75)
        btn1_x1 = page_search_x1 + rng.randint(4, 12)
        search_btn = draw_button(
            draw,
            bbox=(btn1_x1, act_y, btn1_x1 + btn1_w, act_y + 20),
            text=btn1_text,
            theme=theme,
            element_id="btn_page_search",
            screen_size=(w, h),
            variant="primary",
            color_hint="blue",
            font_size=9,
        )

        btn2_text = rng.choice(["I'm Feeling Lucky", "Advanced Search", "Browse", "Bookmarks"])
        btn2_w = rng.randint(85, 100)
        btn2_x1 = btn1_x1 + btn1_w + rng.randint(8, 16)
        lucky_btn = draw_button(
            draw,
            bbox=(btn2_x1, act_y, min(wx2 - 6, btn2_x1 + btn2_w), act_y + 20),
            text=btn2_text,
            theme=theme,
            element_id="btn_page_lucky",
            screen_size=(w, h),
            variant="default",
            font_size=8,
        )
        elements.extend([search_btn, lucky_btn])

        return elements

    def _compose_terminal_shell(
        self, draw: ImageDraw.ImageDraw, theme: ThemePalette, rng: random.Random
    ) -> List[SyntheticUIElement]:
        elements: List[SyntheticUIElement] = []
        w, h = self.sw, self.sh

        wx1 = rng.randint(6, 14)
        wy1 = rng.randint(6, 14)
        wx2 = w - rng.randint(6, 14)
        wy2 = h - rng.randint(6, 14)
        shell_name = rng.choice(["Terminal - PowerShell", "Terminal - bash", "Terminal - zsh", "Command Prompt", "Alacritty"])
        win_el, controls = draw_window(
            draw, (wx1, wy1, wx2, wy2), shell_name, theme, "win_terminal", (w, h)
        )
        elements.extend(controls)

        tab_title = rng.choice(["PowerShell", "bash", "zsh", "cmd", "node"])
        tab_w = rng.randint(60, 75)
        t_tab = draw_tab(
            draw,
            bbox=(wx1 + 4, wy1 + 22, wx1 + 4 + tab_w, wy1 + 38),
            text=tab_title,
            theme=theme,
            element_id="tab_terminal_shell",
            screen_size=(w, h),
            is_active=True,
        )
        elements.append(t_tab)

        # Deep black terminal background
        term_y1 = wy1 + 40
        draw.rectangle([wx1 + 2, term_y1, wx2 - 2, wy2 - 2], fill=(10, 10, 14))

        # Static shell banner lines
        font_mono = get_font(8, bold=False)
        banners = [
            ("Windows PowerShell 7.4.0", "PS C:\\workspace> py -3.14 -m pytest"),
            ("Linux ubuntu-24.04 6.8.0-generic", "dev@station:~/cortex$ git pull"),
            ("GNU bash, version 5.2.21(1)-release", "alex@dev-box:~$ docker ps"),
            ("Darwin Kernel Version 23.4.0", "macbook:avcs user$ cargo check"),
        ]
        b1, b2 = rng.choice(banners)
        draw.text((wx1 + 8, term_y1 + 6), b1, fill=(180, 180, 190), font=font_mono)
        draw.text((wx1 + 8, term_y1 + 20), b2, fill=(200, 200, 210), font=font_mono)

        # Active prompt line with typeable input
        cmd_text = rng.choice(["git status", "pytest", "python app.py", "docker ps", "cargo test", "ls -la"])
        prompt_y = term_y1 + rng.randint(38, 48)
        prompt_input = draw_input(
            draw,
            bbox=(wx1 + 6, prompt_y, wx2 - 6, prompt_y + 19),
            text=cmd_text,
            placeholder="Enter command...",
            theme=theme,
            element_id="terminal_cmd_input",
            screen_size=(w, h),
            is_focused=True,
            has_caret=True,
            context="terminal",
            font_size=8,
        )
        elements.append(prompt_input)

        # Bottom clear / kill buttons
        clear_text = rng.choice(["Clear", "Kill", "Stop", "Reset", "Exit"])
        clear_w = rng.randint(42, 54)
        clear_btn = draw_button(
            draw,
            bbox=(wx2 - clear_w - 6, wy2 - 20, wx2 - 6, wy2 - 6),
            text=clear_text,
            theme=theme,
            element_id="btn_terminal_clear",
            screen_size=(w, h),
            variant="default",
            font_size=8,
        )
        elements.append(clear_btn)

        return elements

    def _compose_modal_dialog(
        self, draw: ImageDraw.ImageDraw, theme: ThemePalette, rng: random.Random
    ) -> List[SyntheticUIElement]:
        elements: List[SyntheticUIElement] = []
        w, h = self.sw, self.sh

        # Background dimmed window
        bg_margin = rng.randint(8, 14)
        bg_win = (bg_margin, bg_margin, w - bg_margin, h - bg_margin)
        draw.rectangle(bg_win, fill=theme.window_bg, outline=theme.border_color)

        # Centered modal dialog box with jittered dimensions
        mw = rng.randint(185, 215)
        mh = rng.randint(120, 140)
        mx1 = (w - mw) // 2 + rng.randint(-6, 6)
        my1 = (h - mh) // 2 + rng.randint(-6, 6)
        mx2 = mx1 + mw
        my2 = my1 + mh

        # Drop shadow & frame
        draw.rectangle([mx1 + 3, my1 + 3, mx2 + 3, my2 + 3], fill=(15, 15, 18))
        draw.rectangle([mx1, my1, mx2, my2], fill=theme.window_bg, outline=theme.border_color, width=2)

        # Modal titlebar
        modal_title = rng.choice(["Confirm Action", "Save Changes", "Discard Draft", "Warning", "Alert"])
        draw.rectangle([mx1, my1, mx2, my1 + 22], fill=theme.titlebar_bg)
        draw.text((mx1 + 8, my1 + 4), modal_title, fill=theme.titlebar_text, font=get_font(9, bold=True))

        # Close button on modal
        close_btn = draw_button(
            draw,
            bbox=(mx2 - 18, my1 + 4, mx2 - 4, my1 + 18),
            text="x",
            theme=theme,
            element_id="modal_close_btn",
            screen_size=(w, h),
            variant="danger",
            color_hint="red",
            font_size=8,
        )
        elements.append(close_btn)

        # Prompt text inside modal
        prompt_text = rng.choice([
            "Save changes to file before exiting?",
            "Are you sure you want to proceed?",
            "Do you want to overwrite existing file?",
            "Unsaved modifications will be lost.",
        ])
        draw.text((mx1 + 10, my1 + 34), prompt_text, fill=theme.text_color, font=get_font(8))

        # Action buttons: Save, Don't Save, Cancel
        btn_y1 = my2 - rng.randint(26, 30)
        btn_y2 = my2 - rng.randint(8, 11)

        save_text = rng.choice(["Save", "Confirm", "OK", "Proceed", "Yes"])
        save_w = rng.randint(48, 56)
        save_btn = draw_button(
            draw,
            bbox=(mx1 + 10, btn_y1, mx1 + 10 + save_w, btn_y2),
            text=save_text,
            theme=theme,
            element_id="btn_modal_save",
            screen_size=(w, h),
            variant="primary",
            color_hint="blue",
            context="confirmation dialog",
            font_size=9,
        )

        dont_save_text = rng.choice(["Don't Save", "Discard", "No", "Skip"])
        dont_save_w = rng.randint(62, 72)
        dont_save_x1 = mx1 + 14 + save_w
        dont_save_btn = draw_button(
            draw,
            bbox=(dont_save_x1, btn_y1, dont_save_x1 + dont_save_w, btn_y2),
            text=dont_save_text,
            theme=theme,
            element_id="btn_modal_dont_save",
            screen_size=(w, h),
            variant="danger",
            color_hint="red",
            context="confirmation dialog",
            font_size=8,
        )

        cancel_text = rng.choice(["Cancel", "Abort", "Dismiss", "Close"])
        cancel_x1 = dont_save_x1 + dont_save_w + 4
        cancel_btn = draw_button(
            draw,
            bbox=(cancel_x1, btn_y1, mx2 - 8, btn_y2),
            text=cancel_text,
            theme=theme,
            element_id="btn_modal_cancel",
            screen_size=(w, h),
            variant="default",
            context="confirmation dialog",
            font_size=8,
        )
        elements.extend([save_btn, dont_save_btn, cancel_btn])

        return elements

    def _compose_form_input(
        self, draw: ImageDraw.ImageDraw, theme: ThemePalette, rng: random.Random
    ) -> List[SyntheticUIElement]:
        elements: List[SyntheticUIElement] = []
        w, h = self.sw, self.sh

        wx1 = rng.randint(6, 14)
        wy1 = rng.randint(6, 14)
        wx2 = w - rng.randint(6, 14)
        wy2 = h - rng.randint(6, 14)
        form_title = rng.choice(["User Account Settings", "Profile Information", "Registration Form", "Contact Details"])
        win_el, controls = draw_window(
            draw, (wx1, wy1, wx2, wy2), form_title, theme, "win_form", (w, h)
        )
        elements.extend(controls)

        font_label = get_font(8, bold=True)
        # Field 1: Username
        uname_label = rng.choice(["Username:", "Display Name:", "User Handle:"])
        uname_val = rng.choice(["alex_dev", "jordan_k", "taylor_m", "casey_r", "samuel_p"])
        f1_y = wy1 + 26
        draw.text((wx1 + 10, f1_y), uname_label, fill=theme.text_muted, font=font_label)
        user_input = draw_input(
            draw,
            bbox=(wx1 + 10, f1_y + 12, wx2 - 10, f1_y + 32),
            text=uname_val,
            placeholder="Enter username",
            theme=theme,
            element_id="input_username",
            screen_size=(w, h),
            is_focused=False,
            font_size=8,
        )
        elements.append(user_input)

        # Field 2: Email
        email_label = rng.choice(["Email Address:", "Work Email:", "Contact Email:"])
        email_ph = rng.choice(["user@example.com", "dev@company.org", "team@visualcortex.ai"])
        f2_y = f1_y + 38
        draw.text((wx1 + 10, f2_y), email_label, fill=theme.text_muted, font=font_label)
        email_input = draw_input(
            draw,
            bbox=(wx1 + 10, f2_y + 12, wx2 - 10, f2_y + 32),
            text="",
            placeholder=email_ph,
            theme=theme,
            element_id="input_email",
            screen_size=(w, h),
            is_focused=True,
            has_caret=True,
            font_size=8,
        )
        elements.append(email_input)

        # Field 3: Password
        pwd_label = rng.choice(["Password:", "Access Token:", "Secret Key:"])
        f3_y = f2_y + 38
        draw.text((wx1 + 10, f3_y), pwd_label, fill=theme.text_muted, font=font_label)
        pwd_input = draw_input(
            draw,
            bbox=(wx1 + 10, f3_y + 12, wx2 - 10, f3_y + 32),
            text="",
            placeholder="********",
            theme=theme,
            element_id="input_password",
            screen_size=(w, h),
            is_focused=False,
            font_size=8,
        )
        elements.append(pwd_input)

        # Submit / Cancel buttons
        btn_y1 = f3_y + 38
        btn_y2 = btn_y1 + 20
        submit_text = rng.choice(["Save Profile", "Submit", "Update Account", "Save Changes"])
        sub_w = rng.randint(70, 85)
        submit_btn = draw_button(
            draw,
            bbox=(wx1 + 10, btn_y1, wx1 + 10 + sub_w, btn_y2),
            text=submit_text,
            theme=theme,
            element_id="btn_submit_profile",
            screen_size=(w, h),
            variant="primary",
            color_hint="blue",
            font_size=8,
        )

        cancel_text = rng.choice(["Cancel", "Reset", "Clear", "Back"])
        cancel_w = rng.randint(52, 65)
        cancel_btn = draw_button(
            draw,
            bbox=(wx1 + 16 + sub_w, btn_y1, wx1 + 16 + sub_w + cancel_w, btn_y2),
            text=cancel_text,
            theme=theme,
            element_id="btn_cancel_form",
            screen_size=(w, h),
            variant="default",
            font_size=8,
        )
        elements.extend([submit_btn, cancel_btn])

        return elements

    def _compose_media_playback(
        self, draw: ImageDraw.ImageDraw, theme: ThemePalette, rng: random.Random
    ) -> List[SyntheticUIElement]:
        elements: List[SyntheticUIElement] = []
        w, h = self.sw, self.sh

        wx1 = rng.randint(6, 14)
        wy1 = rng.randint(6, 14)
        wx2 = w - rng.randint(6, 14)
        wy2 = h - rng.randint(6, 14)
        player_name = rng.choice(["VLC Media Player", "Media Player", "Spotify Desktop", "Audio Player", "QuickTime"])
        win_el, controls = draw_window(
            draw, (wx1, wy1, wx2, wy2), player_name, theme, "win_media", (w, h)
        )
        elements.extend(controls)

        # Video viewport
        vy1 = wy1 + rng.randint(22, 28)
        vy2 = wy2 - rng.randint(48, 56)
        draw.rectangle([wx1 + 4, vy1, wx2 - 4, vy2], fill=(12, 12, 16))

        # Central media graphic
        mid_x = (wx1 + wx2) // 2
        mid_y = (vy1 + vy2) // 2
        cr = rng.randint(20, 26)
        draw.ellipse([mid_x - cr, mid_y - cr, mid_x + cr, mid_y + cr], outline=(60, 60, 75), width=2)
        tri_offset = cr // 2
        draw.polygon([
            (mid_x - tri_offset // 2, mid_y - tri_offset),
            (mid_x + tri_offset, mid_y),
            (mid_x - tri_offset // 2, mid_y + tri_offset),
        ], fill=(240, 120, 30))

        # Progress / Seek bar
        pbar_y = vy2 + rng.randint(5, 8)
        bar_x1 = wx1 + 8
        bar_x2 = wx2 - 8
        draw.line([(bar_x1, pbar_y), (bar_x2, pbar_y)], fill=(60, 64, 72), width=3)
        prog_t = rng.uniform(0.2, 0.8)
        prog_x = int(bar_x1 + (bar_x2 - bar_x1) * prog_t)
        draw.line([(bar_x1, pbar_y), (prog_x, pbar_y)], fill=theme.accent_color, width=3)
        draw.ellipse([prog_x - 4, pbar_y - 4, prog_x + 4, pbar_y + 4], fill=(255, 255, 255))

        # Transport controls
        ctrl_y = wy2 - rng.randint(26, 32)
        ctrl_h = 18

        play_text = rng.choice(["Play", "Pause", "Resume"])
        play_w = rng.randint(34, 44)
        play_btn = draw_button(
            draw,
            bbox=(bar_x1, ctrl_y, bar_x1 + play_w, ctrl_y + ctrl_h),
            text=play_text,
            theme=theme,
            element_id="btn_media_play",
            screen_size=(w, h),
            variant="primary",
            color_hint="blue",
            context="media player",
            font_size=8,
        )

        stop_text = rng.choice(["Stop", "Halt", "Prev"])
        stop_w = rng.randint(34, 44)
        stop_x1 = bar_x1 + play_w + 4
        stop_btn = draw_button(
            draw,
            bbox=(stop_x1, ctrl_y, stop_x1 + stop_w, ctrl_y + ctrl_h),
            text=stop_text,
            theme=theme,
            element_id="btn_media_stop",
            screen_size=(w, h),
            variant="danger",
            color_hint="red",
            context="media player",
            font_size=8,
        )

        mute_text = rng.choice(["Mute", "Volume", "Unmute", "Audio"])
        mute_w = rng.randint(38, 50)
        mute_btn = draw_button(
            draw,
            bbox=(bar_x2 - mute_w, ctrl_y, bar_x2, ctrl_y + ctrl_h),
            text=mute_text,
            theme=theme,
            element_id="btn_media_mute",
            screen_size=(w, h),
            variant="default",
            context="media player",
            font_size=8,
        )
        elements.extend([play_btn, stop_btn, mute_btn])

        return elements

    def _compose_settings_panel(
        self, draw: ImageDraw.ImageDraw, theme: ThemePalette, rng: random.Random
    ) -> List[SyntheticUIElement]:
        elements: List[SyntheticUIElement] = []
        w, h = self.sw, self.sh

        wx1 = rng.randint(5, 12)
        wy1 = rng.randint(5, 12)
        wx2 = w - rng.randint(5, 12)
        wy2 = h - rng.randint(5, 12)
        settings_title = rng.choice(["System Settings", "Preferences", "Control Center", "Device Settings"])
        win_el, controls = draw_window(
            draw, (wx1, wy1, wx2, wy2), settings_title, theme, "win_settings", (w, h)
        )
        elements.extend(controls)

        # Top search field
        search_ph = rng.choice(["Find a setting...", "Search preferences", "Type setting name...", "Search..."])
        search_setting = draw_input(
            draw,
            bbox=(wx1 + 6, wy1 + 24, wx2 - 6, wy1 + 42),
            text="",
            placeholder=search_ph,
            theme=theme,
            element_id="input_settings_search",
            screen_size=(w, h),
            is_focused=True,
            has_caret=True,
            context="settings",
            font_size=8,
        )
        elements.append(search_setting)

        # Left navigation tab column
        tabs = rng.choice([
            ["Display", "Sound", "Network", "About"],
            ["General", "Appearance", "Privacy", "Storage"],
            ["System", "Bluetooth", "Apps", "Power"],
        ])
        tab_col_w = rng.randint(58, 68)
        ty = wy1 + 48
        for idx, tname in enumerate(tabs):
            stab = draw_tab(
                draw,
                bbox=(wx1 + 6, ty, wx1 + 6 + tab_col_w, ty + 17),
                text=tname,
                theme=theme,
                element_id=f"tab_settings_{idx}",
                screen_size=(w, h),
                is_active=(idx == 0),
                context="settings category",
                font_size=8,
            )
            elements.append(stab)
            ty += 21

        # Right panel content
        rx1 = wx1 + 10 + tab_col_w
        rx2 = wx2 - 6
        draw.rectangle([rx1, wy1 + 48, rx2, wy2 - 6], fill=theme.input_bg, outline=theme.border_color)

        toggle_label = rng.choice(["Night light:", "Bluetooth:", "Dark mode:", "Auto-sync:", "Location:"])
        draw.text((rx1 + 6, wy1 + 56), toggle_label, fill=theme.text_color, font=get_font(8))
        toggle_text = rng.choice(["Turn on", "Enable", "Toggle", "Switch on"])
        toggle_btn = draw_button(
            draw,
            bbox=(rx1 + 8, wy1 + 72, rx2 - 8, wy1 + 88),
            text=toggle_text,
            theme=theme,
            element_id="btn_toggle_night_light",
            screen_size=(w, h),
            variant="default",
            context="settings panel",
            font_size=8,
        )
        elements.append(toggle_btn)

        apply_text = rng.choice(["Apply Changes", "Save", "Apply Settings", "OK"])
        apply_btn = draw_button(
            draw,
            bbox=(rx1 + 8, wy2 - 28, rx2 - 8, wy2 - 12),
            text=apply_text,
            theme=theme,
            element_id="btn_settings_apply",
            screen_size=(w, h),
            variant="primary",
            color_hint="blue",
            context="settings panel",
            font_size=8,
        )
        elements.append(apply_btn)

        return elements

# Backward-compatibility alias
JitteredDesktopComposer = DesktopComposer
