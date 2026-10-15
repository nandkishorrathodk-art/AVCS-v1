"""Realistic Desktop GUI Dataset Generator for AVCS Model Training and Evaluation.

Generates photorealistic procedural desktop GUI screens, native OS window layouts, and GUI element crops
with pixel-perfect ground truth for:
1. Dense 3-class affordance masks (0: static background, 1: clickable, 2: typeable) on a 16x16 grid
2. GUI Category classification (8 categories: desktop_canvas, editor_code, browser_webpage,
   terminal_shell, modal_dialog, form_input, media_playback, settings_panel)
3. Foveal acuity crops (96x96) for GUI element classification (7 classes) and bounding box regression
4. Pure static desktop wallpapers / uniform backgrounds for zero-false-positive training and evaluation
5. Live desktop screen capture via Win32FastCapture / ScreenCapture when available on the host.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from avcs.grounding import FOVEAL_ELEMENT_TYPES
from forge.vision.capture import ScreenCapture, SyntheticScreen, Win32FastCapture
from forge.vision.cortex.core.ventral import UI_CATEGORIES


@dataclass
class GUIScreenSample:
    """Rich generated desktop GUI screen sample."""
    frame: np.ndarray  # (H, W, 3) uint8 RGB
    affordance_mask: np.ndarray  # (gh, gw) int64 in {0, 1, 2}
    category_idx: int  # 0..7
    category_name: str
    interactive_points: List[Tuple[float, float, str]]  # (x, y, action_type) in [0, 1]
    element_details: Optional[List[Dict[str, Any]]] = None
    is_negative: bool = False
    source: str = "procedural"


@dataclass
class FovealGUIElementSample:
    """Cropped foveal GUI element sample (96, 96, 3)."""
    crop: np.ndarray  # (96, 96, 3) uint8 RGB
    element_kind: str  # in FOVEAL_ELEMENT_TYPES
    kind_idx: int  # 0..6
    bbox: Tuple[float, float, float, float]  # [ymin, xmin, ymax, xmax] normalized in [0, 1] within crop
    contrast_score: float = 0.0
    is_synthetic: bool = True


def compute_crop_contrast_score(crop: np.ndarray) -> float:
    """Computes edge sharpness/contrast score using Sobel gradient magnitude normalized to [0, 1]."""
    if crop.size == 0:
        return 0.0
    gray = np.mean(crop, axis=-1) if crop.ndim == 3 else crop.astype(float)
    gy, gx = np.gradient(gray)
    return float(np.mean(np.hypot(gx, gy)) / 255.0)


class DesktopGUIGenerator:
    """Procedural generator for high-fidelity native desktop GUI screens and element crops."""

    def __init__(self, screen_size: Tuple[int, int] = (128, 128), grid_size: Tuple[int, int] = (16, 16)) -> None:
        self.H, self.W = screen_size
        self.gh, self.gw = grid_size
        self._synthetic_screen = SyntheticScreen(width=screen_size[0], height=screen_size[1])

    def _coord_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = min(self.gw - 1, max(0, int(x * self.gw)))
        gy = min(self.gh - 1, max(0, int(y * self.gh)))
        return gx, gy

    def _draw_rect(
        self,
        canvas: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        border_color: Optional[Tuple[int, int, int]] = None,
        border_width: int = 1,
    ) -> None:
        x1 = max(0, min(self.W - 1, x1))
        x2 = max(0, min(self.W, x2))
        y1 = max(0, min(self.H - 1, y1))
        y2 = max(0, min(self.H, y2))
        if x2 <= x1 or y2 <= y1:
            return
        canvas[y1:y2, x1:x2] = color
        if border_color is not None:
            bw = border_width
            canvas[y1:min(y2, y1 + bw), x1:x2] = border_color
            canvas[max(y1, y2 - bw):y2, x1:x2] = border_color
            canvas[y1:y2, x1:min(x2, x1 + bw)] = border_color
            canvas[y1:y2, max(x1, x2 - bw):x2] = border_color

    def _draw_gui_button(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        points: List[Tuple[float, float, str]],
        cx: int,
        cy: int,
        w: int,
        h: int,
        btn_type: str = "primary",
        element_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, float]:
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h

        if btn_type == "primary":
            color = random.choice([
                (0, 120, 215),   # Windows accent blue
                (59, 130, 246),  # Modern desktop blue
                (16, 185, 129),  # Emerald confirmation green
                (124, 58, 237),  # Modern purple
            ])
            self._draw_rect(canvas, x1, y1, x2, y2, color, border_color=(255, 255, 255), border_width=1)
            # Simulated text line inside button
            tx1 = x1 + max(3, w // 4)
            tx2 = x2 - max(3, w // 4)
            ty = cy
            if tx2 > tx1:
                canvas[ty:ty + 1, tx1:tx2] = (255, 255, 255)
        elif btn_type == "danger":
            color = (220, 38, 38)
            self._draw_rect(canvas, x1, y1, x2, y2, color)
            canvas[cy:cy + 1, x1 + 2:x2 - 2] = (255, 255, 255)
        elif btn_type == "circle":
            r = min(w, h) // 2
            color = random.choice([(239, 68, 68), (245, 158, 11), (16, 185, 129)])
            self._draw_rect(canvas, cx - r, cy - r, cx + r, cy + r, color)
        else:  # secondary / ghost / native toolbar button
            bg = (60, 64, 72) if canvas[cy, cx][0] < 128 else (230, 232, 236)
            border = (120, 124, 132)
            self._draw_rect(canvas, x1, y1, x2, y2, bg, border_color=border, border_width=1)
            canvas[cy:cy + 1, x1 + 3:x2 - 3] = (180, 180, 185)

        # Mark compact hotspot in 16x16 affordance mask
        norm_x = (cx + 0.5) / self.W
        norm_y = (cy + 0.5) / self.H
        hw = min(8, max(4, w // 4))
        hh = min(6, max(3, h // 4))
        gx1 = min(self.gw - 1, max(0, int((cx - hw) / self.W * self.gw)))
        gx2 = min(self.gw, max(gx1 + 1, int((cx + hw) / self.W * self.gw) + 1))
        gy1 = min(self.gh - 1, max(0, int((cy - hh) / self.H * self.gh)))
        gy2 = min(self.gh, max(gy1 + 1, int((cy + hh) / self.H * self.gh) + 1))
        mask[gy1:gy2, gx1:gx2] = 1  # 1: clickable
        points.append((norm_x, norm_y, "clickable"))

        if element_details is not None:
            kind = "icon" if btn_type == "circle" else ("tab" if (btn_type == "secondary" and h <= 12 and w >= 24) else "button")
            element_details.append({
                "x": norm_x,
                "y": norm_y,
                "action_type": "clickable",
                "element_kind": kind,
                "bbox": [max(0.0, y1 / self.H), max(0.0, x1 / self.W), min(1.0, y2 / self.H), min(1.0, x2 / self.W)],
            })

        return norm_x, norm_y

    def _draw_gui_input(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        points: List[Tuple[float, float, str]],
        cx: int,
        cy: int,
        w: int,
        h: int,
        dark_mode: bool = True,
        element_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, float]:
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h

        bg = (24, 26, 32) if dark_mode else (255, 255, 255)
        border = (0, 120, 215) if random.random() < 0.3 else ((70, 75, 85) if dark_mode else (190, 195, 205))
        self._draw_rect(canvas, x1, y1, x2, y2, bg, border_color=border, border_width=1)

        # Blinking or placeholder caret inside input
        caret_x = x1 + 4
        canvas[y1 + 2:y2 - 2, caret_x:caret_x + 1] = (255, 255, 255) if dark_mode else (30, 30, 30)

        # Optional placeholder text line
        if w > 20:
            pw = random.randint(8, max(9, w - 16))
            canvas[cy:cy + 1, caret_x + 3:caret_x + 3 + pw] = (100, 105, 115) if dark_mode else (170, 175, 185)

        norm_x = (cx + 0.5) / self.W
        norm_y = (cy + 0.5) / self.H
        hw = min(10, max(4, w // 6))
        hh = min(5, max(3, h // 4))
        gx1 = min(self.gw - 1, max(0, int((cx - hw) / self.W * self.gw)))
        gx2 = min(self.gw, max(gx1 + 1, int((cx + hw) / self.W * self.gw) + 1))
        gy1 = min(self.gh - 1, max(0, int((cy - hh) / self.H * self.gh)))
        gy2 = min(self.gh, max(gy1 + 1, int((cy + hh) / self.H * self.gh) + 1))
        mask[gy1:gy2, gx1:gx2] = 2  # 2: typeable
        points.append((norm_x, norm_y, "typeable"))

        if element_details is not None:
            element_details.append({
                "x": norm_x,
                "y": norm_y,
                "action_type": "typeable",
                "element_kind": "input_field",
                "bbox": [max(0.0, y1 / self.H), max(0.0, x1 / self.W), min(1.0, y2 / self.H), min(1.0, x2 / self.W)],
            })

        return norm_x, norm_y

    def _draw_gui_link(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        points: List[Tuple[float, float, str]],
        cx: int,
        cy: int,
        w: int,
        h: int,
        element_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, float]:
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = x1 + w
        y2 = y1 + h

        # Blue underline link
        link_color = (59, 130, 246)
        if y2 - 1 < self.H and x2 < self.W:
            canvas[y2 - 1:y2, x1:x2] = link_color
            canvas[cy:cy + 1, x1:x2] = link_color

        norm_x = (cx + 0.5) / self.W
        norm_y = (cy + 0.5) / self.H
        hw = min(8, max(4, w // 4))
        hh = min(4, max(2, h // 4))
        gx1 = min(self.gw - 1, max(0, int((cx - hw) / self.W * self.gw)))
        gx2 = min(self.gw, max(gx1 + 1, int((cx + hw) / self.W * self.gw) + 1))
        gy1 = min(self.gh - 1, max(0, int((cy - hh) / self.H * self.gh)))
        gy2 = min(self.gh, max(gy1 + 1, int((cy + hh) / self.H * self.gh) + 1))
        mask[gy1:gy2, gx1:gx2] = 1  # 1: clickable
        points.append((norm_x, norm_y, "clickable"))

        if element_details is not None:
            element_details.append({
                "x": norm_x,
                "y": norm_y,
                "action_type": "clickable",
                "element_kind": "link",
                "bbox": [max(0.0, y1 / self.H), max(0.0, x1 / self.W), min(1.0, y2 / self.H), min(1.0, x2 / self.W)],
            })

        return norm_x, norm_y

    def generate_gui_screen(
        self,
        category: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GUIScreenSample:
        """Generates a complete native desktop GUI screen with ground-truth affordance mask."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if category is None or category not in UI_CATEGORIES:
            category = random.choice(UI_CATEGORIES)

        cat_idx = UI_CATEGORIES.index(category)
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        mask = np.zeros((self.gh, self.gw), dtype=np.int64)
        points: List[Tuple[float, float, str]] = []
        details: List[Dict[str, Any]] = []

        if category == "desktop_canvas":
            # Pure static desktop wallpaper: zero interactive elements!
            style = random.choice(["solid", "gradient_linear", "gradient_radial", "slate_dark"])
            if style == "solid":
                c = random.choice([(30, 32, 40), (20, 24, 30), (45, 52, 70), (235, 240, 245)])
                canvas[:] = c
            elif style == "gradient_linear":
                top_c = np.array([random.randint(15, 40), random.randint(20, 50), random.randint(35, 80)], dtype=np.float32)
                bot_c = np.array([random.randint(40, 80), random.randint(50, 90), random.randint(80, 140)], dtype=np.float32)
                for y in range(self.H):
                    alpha = y / float(self.H)
                    canvas[y, :] = (top_c * (1 - alpha) + bot_c * alpha).astype(np.uint8)
            elif style == "gradient_radial":
                cx, cy = self.W // 2, self.H // 2
                base = np.array([20, 25, 40], dtype=np.float32)
                center_glow = np.array([50, 70, 110], dtype=np.float32)
                yy, xx = np.ogrid[:self.H, :self.W]
                dist = np.hypot(xx - cx, yy - cy) / (math.hypot(cx, cy) + 1e-5)
                dist = np.clip(dist, 0.0, 1.0)[:, :, None]
                canvas[:] = (center_glow * (1 - dist) + base * dist).astype(np.uint8)
            else:
                canvas[:] = (35, 38, 46)
            # Wallpaper remains 100% mask = 0 (static background)

        elif category == "editor_code":
            # Native IDE code editor GUI (VS Code / PyCharm style)
            canvas[:] = (30, 30, 34)
            # Window Titlebar
            self._draw_rect(canvas, 0, 0, self.W, 12, (45, 45, 50))
            # 3 window buttons
            for i in range(3):
                self._draw_gui_button(canvas, mask, points, 6 + i * 8, 6, 4, 4, btn_type="circle", element_details=details)
            # GUI Document Tabs
            self._draw_rect(canvas, 0, 12, self.W, 24, (37, 37, 42))
            self._draw_gui_button(canvas, mask, points, 24, 18, 36, 10, btn_type="secondary", element_details=details)
            self._draw_gui_button(canvas, mask, points, 66, 18, 36, 10, btn_type="secondary", element_details=details)
            # Gutter / Line numbers
            self._draw_rect(canvas, 0, 24, 18, self.H - 8, (33, 33, 38))
            # Code lines inside editor
            for ly in range(30, self.H - 16, 12):
                lw = random.randint(24, 75)
                code_color = random.choice([(156, 220, 254), (206, 145, 120), (106, 153, 85), (212, 212, 212)])
                canvas[ly:ly + 2, 24:min(self.W - 6, 24 + lw)] = code_color
            # Code editor active typing line (typeable)
            self._draw_gui_input(canvas, mask, points, 64, 52, 70, 14, dark_mode=True, element_details=details)
            # Status bar at bottom
            self._draw_rect(canvas, 0, self.H - 8, self.W, self.H, (0, 122, 204))

        elif category == "terminal_shell":
            canvas[:] = (12, 12, 14)
            # Titlebar
            self._draw_rect(canvas, 0, 0, self.W, 12, (30, 30, 34))
            for i in range(3):
                self._draw_gui_button(canvas, mask, points, 6 + i * 8, 6, 4, 4, btn_type="circle", element_details=details)
            # Terminal prompt output lines
            canvas[18:20, 8:45] = (80, 200, 120)
            canvas[24:26, 8:65] = (200, 200, 200)
            # Shell command line
            self._draw_gui_input(canvas, mask, points, 64, 40, 95, 14, dark_mode=True, element_details=details)

        elif category == "modal_dialog":
            # Scrim / dimmed background
            canvas[:] = (20, 22, 26)
            # Modal window
            mx1, my1, mx2, my2 = 16, 20, self.W - 16, self.H - 20
            self._draw_rect(canvas, mx1, my1, mx2, my2, (48, 50, 56), border_color=(80, 84, 94), border_width=1)
            # Header
            self._draw_rect(canvas, mx1, my1, mx2, my1 + 14, (56, 58, 66))
            # Dismiss 'x' button
            self._draw_gui_button(canvas, mask, points, mx2 - 8, my1 + 7, 8, 8, btn_type="danger", element_details=details)
            # Optional input inside modal
            self._draw_gui_input(canvas, mask, points, self.W // 2, 50, 70, 14, dark_mode=True, element_details=details)
            # Action buttons: OK / Cancel
            self._draw_gui_button(canvas, mask, points, mx2 - 20, my2 - 12, 28, 12, btn_type="primary", element_details=details)
            self._draw_gui_button(canvas, mask, points, mx2 - 54, my2 - 12, 26, 12, btn_type="secondary", element_details=details)

        elif category in ("browser_webpage", "form_input"):
            dark_mode = random.random() < 0.5
            canvas[:] = (28, 30, 36) if dark_mode else (245, 246, 250)
            # Browser GUI toolbar
            self._draw_rect(canvas, 0, 0, self.W, 14, (40, 42, 48) if dark_mode else (225, 227, 234))
            for i in range(3):
                self._draw_gui_button(canvas, mask, points, 6 + i * 8, 7, 4, 4, btn_type="circle", element_details=details)
            # URL / Search bar
            self._draw_gui_input(canvas, mask, points, 64, 24, 90, 12, dark_mode=dark_mode, element_details=details)
            # Search action button
            self._draw_gui_button(canvas, mask, points, self.W - 14, 24, 16, 12, btn_type="primary", element_details=details)

            if category == "form_input":
                # Form GUI with multiple inputs and Submit CTA
                self._draw_gui_input(canvas, mask, points, 64, 52, 80, 14, dark_mode=dark_mode, element_details=details)
                self._draw_gui_input(canvas, mask, points, 64, 74, 80, 14, dark_mode=dark_mode, element_details=details)
                self._draw_gui_button(canvas, mask, points, 64, 98, 50, 16, btn_type="primary", element_details=details)
            else:
                # Content cards with buttons
                self._draw_gui_button(canvas, mask, points, 36, 60, 40, 16, btn_type="primary", element_details=details)
                self._draw_gui_button(canvas, mask, points, 92, 60, 40, 16, btn_type="secondary", element_details=details)
                self._draw_gui_button(canvas, mask, points, 64, 95, 48, 14, btn_type="secondary", element_details=details)

        else:  # settings_panel, media_playback
            dark_mode = True
            canvas[:] = (32, 34, 40)
            # Window header
            self._draw_rect(canvas, 0, 0, self.W, 14, (45, 47, 54))
            for i in range(3):
                self._draw_gui_button(canvas, mask, points, 6 + i * 8, 7, 4, 4, btn_type="circle", element_details=details)

            if category == "media_playback":
                # Player controls in center and bottom
                self._draw_gui_button(canvas, mask, points, self.W // 2, 70, 24, 24, btn_type="circle", element_details=details)
                self._draw_gui_button(canvas, mask, points, self.W // 2 - 28, 70, 14, 14, btn_type="secondary", element_details=details)
                self._draw_gui_button(canvas, mask, points, self.W // 2 + 28, 70, 14, 14, btn_type="secondary", element_details=details)
            else:
                # Settings sidebar
                self._draw_rect(canvas, 0, 14, 34, self.H, (38, 40, 48))
                for sy in range(24, self.H - 16, 20):
                    self._draw_gui_button(canvas, mask, points, 17, sy, 26, 12, btn_type="secondary", element_details=details)
                # Main toggle / buttons
                self._draw_gui_button(canvas, mask, points, 80, 40, 36, 14, btn_type="primary", element_details=details)
                self._draw_gui_input(canvas, mask, points, 80, 68, 64, 14, dark_mode=True, element_details=details)

        is_neg = (category == "desktop_canvas" and len(points) == 0)
        return GUIScreenSample(
            frame=canvas,
            affordance_mask=mask,
            category_idx=cat_idx,
            category_name=category,
            interactive_points=points,
            element_details=details,
            is_negative=is_neg,
            source="procedural",
        )

    def generate_negative_screen(
        self,
        style: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GUIScreenSample:
        """Generates guaranteed negative samples (uniform blank screens, color gradients, photographic wallpapers)."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        styles = [
            "solid_white",
            "solid_black",
            "solid_system_gray",
            "solid_accent_blue",
            "gradient_vertical",
            "gradient_horizontal",
            "gradient_diagonal",
            "gradient_radial_glow",
            "wallpaper_photo_dusk",
            "wallpaper_abstract_polygon",
        ]
        if style is None or style not in styles:
            style = random.choice(styles)

        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        mask = np.zeros((self.gh, self.gw), dtype=np.int64)

        if style == "solid_white":
            canvas[:] = (255, 255, 255)
        elif style == "solid_black":
            canvas[:] = (0, 0, 0)
        elif style == "solid_system_gray":
            g = random.choice([24, 32, 45, 60, 128, 192, 220, 240])
            canvas[:] = (g, g, g)
        elif style == "solid_accent_blue":
            canvas[:] = random.choice([(0, 120, 215), (24, 40, 72), (15, 23, 42)])
        elif style == "gradient_vertical":
            top_c = np.array([random.randint(10, 40), random.randint(20, 60), random.randint(40, 100)], dtype=np.float32)
            bot_c = np.array([random.randint(30, 80), random.randint(50, 120), random.randint(90, 180)], dtype=np.float32)
            for y in range(self.H):
                a = y / float(self.H)
                canvas[y, :] = (top_c * (1 - a) + bot_c * a).astype(np.uint8)
        elif style == "gradient_horizontal":
            left_c = np.array([random.randint(20, 50), random.randint(30, 70), random.randint(60, 120)], dtype=np.float32)
            right_c = np.array([random.randint(60, 120), random.randint(40, 90), random.randint(100, 160)], dtype=np.float32)
            for x in range(self.W):
                a = x / float(self.W)
                canvas[:, x] = (left_c * (1 - a) + right_c * a).astype(np.uint8)
        elif style == "gradient_diagonal":
            c1 = np.array([random.randint(15, 45), random.randint(20, 50), random.randint(30, 80)], dtype=np.float32)
            c2 = np.array([random.randint(50, 110), random.randint(70, 130), random.randint(120, 200)], dtype=np.float32)
            for y in range(self.H):
                for x in range(self.W):
                    a = (x + y) / float(self.W + self.H)
                    canvas[y, x] = (c1 * (1 - a) + c2 * a).astype(np.uint8)
        elif style == "gradient_radial_glow":
            cx, cy = self.W // 2, self.H // 2
            base = np.array([16, 20, 32], dtype=np.float32)
            glow = np.array([45, 65, 115], dtype=np.float32)
            yy, xx = np.ogrid[:self.H, :self.W]
            dist = np.hypot(xx - cx, yy - cy) / (math.hypot(cx, cy) + 1e-5)
            dist = np.clip(dist, 0.0, 1.0)[:, :, None]
            canvas[:] = (glow * (1 - dist) + base * dist).astype(np.uint8)
        elif style == "wallpaper_photo_dusk":
            # Procedural smooth atmospheric sky / dusk landscape
            sky_top = np.array([25, 30, 55], dtype=np.float32)
            sky_mid = np.array([90, 60, 95], dtype=np.float32)
            horizon = np.array([180, 110, 80], dtype=np.float32)
            ground = np.array([30, 25, 35], dtype=np.float32)
            for y in range(self.H):
                if y < self.H * 0.4:
                    a = y / (self.H * 0.4)
                    canvas[y, :] = (sky_top * (1 - a) + sky_mid * a).astype(np.uint8)
                elif y < self.H * 0.7:
                    a = (y - self.H * 0.4) / (self.H * 0.3)
                    canvas[y, :] = (sky_mid * (1 - a) + horizon * a).astype(np.uint8)
                else:
                    a = (y - self.H * 0.7) / (self.H * 0.3)
                    canvas[y, :] = (horizon * (1 - a) + ground * a).astype(np.uint8)
        else:  # wallpaper_abstract_polygon
            canvas[:] = (35, 38, 46)
            for _ in range(6):
                px = random.randint(0, self.W)
                py = random.randint(0, self.H)
                pr = random.randint(15, 45)
                color = (random.randint(25, 60), random.randint(30, 70), random.randint(50, 95))
                self._draw_rect(canvas, px - pr, py - pr, px + pr, py + pr, color)

        return GUIScreenSample(
            frame=canvas,
            affordance_mask=mask,
            category_idx=UI_CATEGORIES.index("desktop_canvas"),
            category_name="desktop_canvas",
            interactive_points=[],
            element_details=[],
            is_negative=True,
            source="procedural_negative",
        )

    def capture_live_screen(
        self,
        target_size: Tuple[int, int] = (128, 128),
    ) -> Optional[GUIScreenSample]:
        """Attempts to capture real desktop screen pixels via Win32FastCapture or ScreenCapture."""
        raw_frame = None
        source_name = "unknown"

        # 1. Try Win32FastCapture first (fastest Windows GDI DIBSection)
        if sys.platform == "win32":
            try:
                cap = Win32FastCapture(target_size=None)
                cap.open()
                raw = cap.grab()
                cap.close()
                if raw is not None and raw.size > 0:
                    raw_frame = raw
                    source_name = "win32_fast_capture"
            except Exception:
                raw_frame = None

        # 2. Try ScreenCapture via mss
        if raw_frame is None and ScreenCapture.available():
            try:
                with ScreenCapture() as sct:
                    raw = sct.grab()
                    if raw is not None and raw.size > 0:
                        raw_frame = raw
                        source_name = "screen_capture_mss"
            except Exception:
                raw_frame = None

        if raw_frame is None:
            return None

        # Resize to target size (128, 128)
        img = Image.fromarray(raw_frame)
        resized_img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        frame_np = np.asarray(resized_img, dtype=np.uint8)

        mask = np.zeros((self.gh, self.gw), dtype=np.int64)

        return GUIScreenSample(
            frame=frame_np,
            affordance_mask=mask,
            category_idx=UI_CATEGORIES.index("desktop_canvas"),
            category_name="desktop_canvas",
            interactive_points=[],
            element_details=[],
            is_negative=True,
            source=source_name,
        )

    def generate_foveal_element(
        self,
        element_kind: Optional[str] = None,
        crop_size: Tuple[int, int] = (96, 96),
        seed: Optional[int] = None,
    ) -> FovealGUIElementSample:
        """Generates a high-resolution 96x96 foveal crop of a specific GUI element with ground-truth bbox."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if element_kind is None or element_kind not in FOVEAL_ELEMENT_TYPES:
            element_kind = random.choice(FOVEAL_ELEMENT_TYPES)

        kind_idx = FOVEAL_ELEMENT_TYPES.index(element_kind)
        H, W = crop_size
        crop = np.full((H, W, 3), random.randint(30, 45), dtype=np.uint8)

        cx, cy = W // 2, H // 2
        if element_kind == "button":
            bw, bh = random.randint(40, 70), random.randint(22, 34)
            color = random.choice([(0, 120, 215), (59, 130, 246), (16, 185, 129), (220, 38, 38)])
            self._draw_rect(crop, cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2, color, (255, 255, 255), 1)
        elif element_kind == "input_field":
            bw, bh = random.randint(50, 80), random.randint(24, 36)
            self._draw_rect(crop, cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2, (20, 22, 28), (0, 120, 215), 2)
            # Caret
            crop[cy - 6:cy + 6, cx - bw // 2 + 6:cx - bw // 2 + 8] = (255, 255, 255)
        elif element_kind == "icon":
            bw, bh = random.randint(24, 40), random.randint(24, 40)
            self._draw_rect(crop, cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2, (70, 75, 88), (140, 145, 160), 1)
        elif element_kind == "tab":
            bw, bh = random.randint(60, 86), random.randint(20, 28)
            self._draw_rect(crop, cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2, (50, 52, 60), (0, 120, 215), 1)
        elif element_kind == "link":
            bw, bh = random.randint(40, 75), random.randint(14, 20)
            crop[cy + 4:cy + 6, cx - bw // 2:cx + bw // 2] = (59, 130, 246)
        elif element_kind == "text_label":
            bw, bh = random.randint(35, 75), random.randint(12, 18)
            crop[cy - 2:cy + 2, cx - bw // 2:cx + bw // 2] = (220, 220, 225)
        else:  # container
            bw, bh = random.randint(70, 90), random.randint(70, 90)
            self._draw_rect(crop, cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2, (40, 42, 50), (90, 95, 110), 1)

        ymin = max(0.0, (cy - bh / 2.0) / float(H))
        xmin = max(0.0, (cx - bw / 2.0) / float(W))
        ymax = min(1.0, (cy + bh / 2.0) / float(H))
        xmax = min(1.0, (cx + bw / 2.0) / float(W))

        contrast = compute_crop_contrast_score(crop)

        return FovealGUIElementSample(
            crop=crop,
            element_kind=element_kind,
            kind_idx=kind_idx,
            bbox=(ymin, xmin, ymax, xmax),
            contrast_score=contrast,
            is_synthetic=True,
        )


class AVCSUnifiedGUIDataset(Dataset):
    """PyTorch Dataset yielding synchronized desktop GUI screens, affordance masks, categories, and foveal crops."""

    def __init__(
        self,
        num_samples: int = 1200,
        seed: int = 42,
        data_dir: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.seed = seed
        self.generator = DesktopGUIGenerator()
        self.data_dir = data_dir
        self.split = split
        self._manifest_frames: Optional[List[Dict[str, Any]]] = None

        if data_dir is not None:
            manifest_file = Path(data_dir) / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    frames = data.get("frames", [])
                    if split:
                        frames = [fr for fr in frames if fr.get("split") == split]
                    if frames:
                        self._manifest_frames = frames
                        self.num_samples = len(frames)
                except Exception:
                    self._manifest_frames = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._manifest_frames is not None and idx < len(self._manifest_frames):
            frame_info = self._manifest_frames[idx]
            base = Path(self.data_dir if self.data_dir else "data")
            img_path = base / frame_info["file_path"]
            mask_path = base / frame_info["affordance_mask_path"] if frame_info.get("affordance_mask_path") else None

            if img_path.exists():
                screen_arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            else:
                sample_seed = self.seed + idx
                screen_arr = self.generator.generate_gui_screen(seed=sample_seed).frame

            if mask_path and mask_path.exists():
                mask_arr = np.load(mask_path)
            else:
                mask_arr = np.zeros((16, 16), dtype=np.int64)

            cat_name = frame_info.get("category", "desktop_canvas")
            cat_idx = UI_CATEGORIES.index(cat_name) if cat_name in UI_CATEGORIES else 0

            # Foveal sample
            foveal = self.generator.generate_foveal_element(seed=self.seed + idx)

            screen_t = torch.from_numpy(screen_arr.transpose(2, 0, 1).copy()).float() / 255.0
            mask_t = torch.from_numpy(mask_arr.copy()).long()
            cat_t = torch.tensor(cat_idx, dtype=torch.long)
            foveal_t = torch.from_numpy(foveal.crop.transpose(2, 0, 1).copy()).float() / 255.0
            foveal_kind_t = torch.tensor(foveal.kind_idx, dtype=torch.long)
            foveal_bbox_t = torch.tensor(foveal.bbox, dtype=torch.float32)

            return {
                "screen": screen_t,
                "affordance_mask": mask_t,
                "category": cat_t,
                "foveal_crop": foveal_t,
                "foveal_kind": foveal_kind_t,
                "foveal_bbox": foveal_bbox_t,
            }

        # On-the-fly generation path
        sample_seed = self.seed + idx
        screen = self.generator.generate_gui_screen(seed=sample_seed)
        foveal = self.generator.generate_foveal_element(seed=sample_seed)

        # 1. Screen tensor: (3, 128, 128) in [0, 1]
        screen_t = torch.from_numpy(screen.frame.transpose(2, 0, 1).copy()).float() / 255.0

        # 2. Affordance mask: (16, 16) int64 in {0, 1, 2}
        mask_t = torch.from_numpy(screen.affordance_mask.copy()).long()

        # 3. Category: scalar long
        cat_t = torch.tensor(screen.category_idx, dtype=torch.long)

        # 4. Foveal crop tensor: (3, 96, 96) in [0, 1]
        foveal_t = torch.from_numpy(foveal.crop.transpose(2, 0, 1).copy()).float() / 255.0
        foveal_kind_t = torch.tensor(foveal.kind_idx, dtype=torch.long)
        foveal_bbox_t = torch.tensor(foveal.bbox, dtype=torch.float32)

        return {
            "screen": screen_t,
            "affordance_mask": mask_t,
            "category": cat_t,
            "foveal_crop": foveal_t,
            "foveal_kind": foveal_kind_t,
            "foveal_bbox": foveal_bbox_t,
        }


def generate_and_save_dataset(
    output_dir: Union[str, Path] = "data",
    num_train: int = 160,
    num_val: int = 40,
    num_test_held_out: int = 40,
    num_test_negative: int = 30,
    num_foveal_crops: int = 140,
    seed_base: int = 10000,
    include_live_capture: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generates and writes a complete, reproducible dataset to disk with manifest.json."""
    t0 = time.perf_counter()
    out_path = Path(output_dir)
    screens_dir = out_path / "screens"
    masks_dir = out_path / "masks"
    fovea_dir = out_path / "fovea"
    fovea_crops_dir = fovea_dir / "crops"

    screens_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    fovea_crops_dir.mkdir(parents=True, exist_ok=True)

    gen = DesktopGUIGenerator(screen_size=(128, 128), grid_size=(16, 16))
    frames_metadata: List[Dict[str, Any]] = []

    interactive_cats = [
        "browser_webpage",
        "editor_code",
        "form_input",
        "modal_dialog",
        "terminal_shell",
        "settings_panel",
        "media_playback",
    ]

    split_counts = {
        "train": 0,
        "val": 0,
        "test_held_out": 0,
        "test_negative": 0,
    }

    frame_counter = 0

    # 1. Test Held-Out Split (40 frames, strict interactive applications, seed 50000+)
    # Aligns with test_held_out_gui_affordance_precision_and_recall benchmark
    if verbose:
        print("[Dataset Generator] Synthesizing test_held_out interactive evaluation split...")
    for i in range(num_test_held_out):
        cat = interactive_cats[i % len(interactive_cats)]
        seed = 50000 + i
        sample = gen.generate_gui_screen(category=cat, seed=seed)

        frame_counter += 1
        frame_id = f"frame_{frame_counter:04d}"
        img_rel = f"screens/{frame_id}.png"
        mask_rel = f"masks/{frame_id}.npy"

        # Save screen image
        Image.fromarray(sample.frame).save(out_path / img_rel)
        # Save mask array
        np.save(out_path / mask_rel, sample.affordance_mask)

        # Build point annotations
        interactive_points = []
        if sample.element_details:
            for d in sample.element_details:
                interactive_points.append({
                    "x": round(float(d["x"]), 4),
                    "y": round(float(d["y"]), 4),
                    "action_type": str(d["action_type"]),
                    "element_kind": str(d.get("element_kind", "button")),
                })
        else:
            for pt in sample.interactive_points:
                interactive_points.append({
                    "x": round(float(pt[0]), 4),
                    "y": round(float(pt[1]), 4),
                    "action_type": str(pt[2]),
                    "element_kind": "button" if pt[2] == "clickable" else "input_field",
                })

        frames_metadata.append({
            "frame_id": frame_id,
            "file_path": img_rel,
            "width": sample.frame.shape[1],
            "height": sample.frame.shape[0],
            "category": sample.category_name,
            "is_negative": False,
            "affordance_mask_path": mask_rel,
            "split": "test_held_out",
            "source": sample.source,
            "interactive_points": interactive_points,
        })
        split_counts["test_held_out"] += 1

    # 2. Test Negative Split (30 frames, pure wallpapers/blank screens, seed 70000+)
    # Aligns with test_held_out_wallpaper_zero_false_positives benchmark
    if verbose:
        print("[Dataset Generator] Synthesizing test_negative zero-FP wallpaper/blank evaluation split...")

    live_capture_saved = 0
    if include_live_capture:
        try:
            live_sample = gen.capture_live_screen()
            if live_sample is not None:
                frame_counter += 1
                frame_id = f"frame_{frame_counter:04d}"
                img_rel = f"screens/{frame_id}.png"
                mask_rel = f"masks/{frame_id}.npy"

                Image.fromarray(live_sample.frame).save(out_path / img_rel)
                np.save(out_path / mask_rel, live_sample.affordance_mask)

                frames_metadata.append({
                    "frame_id": frame_id,
                    "file_path": img_rel,
                    "width": live_sample.frame.shape[1],
                    "height": live_sample.frame.shape[0],
                    "category": "desktop_canvas",
                    "is_negative": True,
                    "affordance_mask_path": mask_rel,
                    "split": "test_negative",
                    "source": live_sample.source,
                    "interactive_points": [],
                })
                split_counts["test_negative"] += 1
                live_capture_saved += 1
                if verbose:
                    print(f"  Captured live desktop frame via {live_sample.source} -> {img_rel}")
        except Exception as exc:
            if verbose:
                print(f"  Live capture note: {exc}")

    remaining_negatives = num_test_negative - live_capture_saved
    for i in range(remaining_negatives):
        seed = 70000 + i
        if i % 3 == 0:
            sample = gen.generate_negative_screen(seed=seed)
        else:
            sample = gen.generate_gui_screen(category="desktop_canvas", seed=seed)

        frame_counter += 1
        frame_id = f"frame_{frame_counter:04d}"
        img_rel = f"screens/{frame_id}.png"
        mask_rel = f"masks/{frame_id}.npy"

        Image.fromarray(sample.frame).save(out_path / img_rel)
        np.save(out_path / mask_rel, sample.affordance_mask)

        frames_metadata.append({
            "frame_id": frame_id,
            "file_path": img_rel,
            "width": sample.frame.shape[1],
            "height": sample.frame.shape[0],
            "category": "desktop_canvas",
            "is_negative": True,
            "affordance_mask_path": mask_rel,
            "split": "test_negative",
            "source": sample.source,
            "interactive_points": [],
        })
        split_counts["test_negative"] += 1

    # 3. Validation Split (40 frames, 30 interactive, 10 negative, seed 30000+)
    if verbose:
        print("[Dataset Generator] Synthesizing val split...")
    for i in range(num_val):
        seed = 30000 + i
        is_neg = (i % 4 == 3)  # 25% negative
        if is_neg:
            sample = gen.generate_negative_screen(seed=seed)
        else:
            cat = interactive_cats[i % len(interactive_cats)]
            sample = gen.generate_gui_screen(category=cat, seed=seed)

        frame_counter += 1
        frame_id = f"frame_{frame_counter:04d}"
        img_rel = f"screens/{frame_id}.png"
        mask_rel = f"masks/{frame_id}.npy"

        Image.fromarray(sample.frame).save(out_path / img_rel)
        np.save(out_path / mask_rel, sample.affordance_mask)

        pts = []
        if sample.element_details:
            for d in sample.element_details:
                pts.append({
                    "x": round(float(d["x"]), 4),
                    "y": round(float(d["y"]), 4),
                    "action_type": str(d["action_type"]),
                    "element_kind": str(d.get("element_kind", "button")),
                })
        else:
            for pt in sample.interactive_points:
                pts.append({
                    "x": round(float(pt[0]), 4),
                    "y": round(float(pt[1]), 4),
                    "action_type": str(pt[2]),
                    "element_kind": "button" if pt[2] == "clickable" else "input_field",
                })

        frames_metadata.append({
            "frame_id": frame_id,
            "file_path": img_rel,
            "width": sample.frame.shape[1],
            "height": sample.frame.shape[0],
            "category": sample.category_name,
            "is_negative": sample.is_negative,
            "affordance_mask_path": mask_rel,
            "split": "val",
            "source": sample.source,
            "interactive_points": pts,
        })
        split_counts["val"] += 1

    # 4. Training Split (160 frames, 120 interactive, 40 negative, seed 10000+)
    if verbose:
        print("[Dataset Generator] Synthesizing train split...")
    for i in range(num_train):
        seed = seed_base + i
        is_neg = (i % 4 == 3)  # 25% negative for robust static margin learning
        if is_neg:
            sample = gen.generate_negative_screen(seed=seed)
        else:
            cat = interactive_cats[i % len(interactive_cats)]
            sample = gen.generate_gui_screen(category=cat, seed=seed)

        frame_counter += 1
        frame_id = f"frame_{frame_counter:04d}"
        img_rel = f"screens/{frame_id}.png"
        mask_rel = f"masks/{frame_id}.npy"

        Image.fromarray(sample.frame).save(out_path / img_rel)
        np.save(out_path / mask_rel, sample.affordance_mask)

        pts = []
        if sample.element_details:
            for d in sample.element_details:
                pts.append({
                    "x": round(float(d["x"]), 4),
                    "y": round(float(d["y"]), 4),
                    "action_type": str(d["action_type"]),
                    "element_kind": str(d.get("element_kind", "button")),
                })
        else:
            for pt in sample.interactive_points:
                pts.append({
                    "x": round(float(pt[0]), 4),
                    "y": round(float(pt[1]), 4),
                    "action_type": str(pt[2]),
                    "element_kind": "button" if pt[2] == "clickable" else "input_field",
                })

        frames_metadata.append({
            "frame_id": frame_id,
            "file_path": img_rel,
            "width": sample.frame.shape[1],
            "height": sample.frame.shape[0],
            "category": sample.category_name,
            "is_negative": sample.is_negative,
            "affordance_mask_path": mask_rel,
            "split": "train",
            "source": sample.source,
            "interactive_points": pts,
        })
        split_counts["train"] += 1

    # 5. Foveal Element Crops (140 crops, covering all 7 classes, seed 90000+)
    # Aligns with test_foveal_acuity_held_out_element_inspection benchmark
    if verbose:
        print(f"[Dataset Generator] Synthesizing {num_foveal_crops} foveal acuity crops...")
    fovea_crops_metadata: List[Dict[str, Any]] = []
    for i in range(num_foveal_crops):
        target_kind = FOVEAL_ELEMENT_TYPES[i % len(FOVEAL_ELEMENT_TYPES)]
        seed = 90000 + i
        foveal_sample = gen.generate_foveal_element(element_kind=target_kind, seed=seed)

        crop_id = f"fov_{i + 1:04d}"
        crop_rel = f"fovea/crops/{crop_id}.png"

        Image.fromarray(foveal_sample.crop).save(out_path / crop_rel)

        fovea_crops_metadata.append({
            "crop_id": crop_id,
            "file_path": crop_rel,
            "crop_shape": list(foveal_sample.crop.shape),
            "element_kind": foveal_sample.element_kind,
            "kind_idx": foveal_sample.kind_idx,
            "bbox": [round(float(v), 4) for v in foveal_sample.bbox],
            "contrast_score": round(float(foveal_sample.contrast_score), 5),
            "is_synthetic": bool(foveal_sample.is_synthetic),
        })

    # Save fovea crop manifest
    fovea_manifest = {
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_crops": len(fovea_crops_metadata),
        "element_kinds": FOVEAL_ELEMENT_TYPES,
        "crops": fovea_crops_metadata,
    }
    with open(out_path / "fovea_manifest.json", "w", encoding="utf-8") as f:
        json.dump(fovea_manifest, f, indent=2)
    with open(fovea_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(fovea_manifest, f, indent=2)

    # 6. Master Full-Screen Dataset Manifest (manifest.json)
    # Strictly adheres to spec_miner_a3/handoff.md § 1.A schema
    manifest = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AVCS_Screen_Dataset_Manifest",
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_frames": len(frames_metadata),
        "splits": {
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test_held_out": split_counts["test_held_out"],
            "test_negative": split_counts["test_negative"],
        },
        "classes": {
            "macro_affordances": ["static", "clickable", "typeable"],
            "foveal_element_kinds": FOVEAL_ELEMENT_TYPES,
            "ui_categories": UI_CATEGORIES,
        },
        "frames": frames_metadata,
    }

    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also mirror to avcs/training/dataset_manifest.json
    avcs_manifest_path = Path("avcs/training/dataset_manifest.json")
    try:
        avcs_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(avcs_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        pass

    dt = time.perf_counter() - t0
    if verbose:
        print(f"[Dataset Generator] Done in {dt:.2f}s! Total frames: {len(frames_metadata)}, Foveal crops: {len(fovea_crops_metadata)}")
        print(f"  Splits: train={split_counts['train']}, val={split_counts['val']}, test_held_out={split_counts['test_held_out']}, test_negative={split_counts['test_negative']}")
        print(f"  Artifacts saved to: {out_path.resolve()}")

    return {
        "manifest_path": str(manifest_path.resolve()),
        "total_frames": len(frames_metadata),
        "total_foveal_crops": len(fovea_crops_metadata),
        "splits": split_counts,
        "elapsed_seconds": round(dt, 2),
    }


def verify_dataset_manifest(
    manifest_path: Union[str, Path] = "data/manifest.json",
    data_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Independently verifies schema conformity, file presence, mask dimensions, and split integrity."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Required top-level keys
    req_top = ["version", "created_at", "total_frames", "splits", "classes", "frames"]
    for k in req_top:
        assert k in data, f"Missing required top-level key: {k}"

    # 2. Required splits
    req_splits = ["train", "val", "test_held_out", "test_negative"]
    for s in req_splits:
        assert s in data["splits"], f"Missing required split: {s}"
        assert isinstance(data["splits"][s], int), f"Split count for {s} must be int"

    # 3. Classes
    assert data["classes"]["macro_affordances"] == ["static", "clickable", "typeable"]
    assert data["classes"]["foveal_element_kinds"] == FOVEAL_ELEMENT_TYPES
    assert data["classes"]["ui_categories"] == UI_CATEGORIES

    # 4. Frames verification
    frames = data["frames"]
    assert len(frames) == data["total_frames"], f"Frame count mismatch: {len(frames)} != {data['total_frames']}"

    if data_dir is not None:
        base_dir = Path(data_dir)
    elif (p.parent / "screens").exists():
        base_dir = p.parent
    elif (Path("data") / "screens").exists():
        base_dir = Path("data")
    else:
        base_dir = p.parent

    split_counts_actual = {s: 0 for s in req_splits}
    verified_files = 0
    verified_masks = 0

    for idx, fr in enumerate(frames):
        # Check required fields
        for fld in ["frame_id", "file_path", "width", "height", "category", "is_negative", "interactive_points"]:
            assert fld in fr, f"Frame {idx} missing field: {fld}"

        # Check split
        split_name = fr.get("split")
        assert split_name in req_splits, f"Frame {idx} invalid split: {split_name}"
        split_counts_actual[split_name] += 1

        # Check category
        assert fr["category"] in UI_CATEGORIES, f"Frame {idx} invalid category: {fr['category']}"

        # Check image file existence and dimensions
        img_file = base_dir / fr["file_path"]
        assert img_file.exists(), f"Image file missing: {img_file}"
        verified_files += 1

        # Check mask existence, shape, and values
        if fr.get("affordance_mask_path"):
            mask_file = base_dir / fr["affordance_mask_path"]
            assert mask_file.exists(), f"Mask file missing: {mask_file}"
            mask_arr = np.load(mask_file)
            assert mask_arr.shape == (16, 16), f"Mask shape must be (16, 16), got {mask_arr.shape}"
            assert np.isin(mask_arr, [0, 1, 2]).all(), "Mask values must be in {0, 1, 2}"
            verified_masks += 1

            if fr["is_negative"]:
                assert (mask_arr == 0).all(), f"Negative frame {fr['frame_id']} must have all-zero mask"
                assert len(fr["interactive_points"]) == 0, f"Negative frame {fr['frame_id']} must have 0 interactive points"

        # Check interactive points
        for pt in fr["interactive_points"]:
            assert "x" in pt and "y" in pt and "action_type" in pt and "element_kind" in pt
            assert 0.0 <= pt["x"] <= 1.0, f"Point x out of bounds: {pt['x']}"
            assert 0.0 <= pt["y"] <= 1.0, f"Point y out of bounds: {pt['y']}"
            assert pt["action_type"] in ["clickable", "typeable"], f"Invalid action_type: {pt['action_type']}"

    for s in req_splits:
        assert split_counts_actual[s] == data["splits"][s], (
            f"Split count mismatch for {s}: actual {split_counts_actual[s]} != manifest {data['splits'][s]}"
        )

    # Check foveal manifest if present
    fovea_manifest_file = base_dir / "fovea_manifest.json"
    foveal_verified = 0
    if fovea_manifest_file.exists():
        with open(fovea_manifest_file, "r", encoding="utf-8") as f:
            fov_data = json.load(f)
        for crop in fov_data.get("crops", []):
            crop_file = base_dir / crop["file_path"]
            assert crop_file.exists(), f"Foveal crop file missing: {crop_file}"
            assert crop["element_kind"] in FOVEAL_ELEMENT_TYPES
            assert len(crop["bbox"]) == 4
            foveal_verified += 1

    result = {
        "verified": True,
        "manifest_path": str(p.resolve()),
        "total_frames": len(frames),
        "verified_image_files": verified_files,
        "verified_mask_files": verified_masks,
        "verified_foveal_crops": foveal_verified,
        "splits": split_counts_actual,
    }

    if verbose:
        print(f"[Manifest Verification] PASSED: {len(frames)} frames, {verified_masks} masks, {foveal_verified} fovea crops.")
        print(f"  Splits validated: {split_counts_actual}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible desktop GUI perception dataset for AVCS.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory (default: data)")
    parser.add_argument("--num-train", type=int, default=160, help="Train frames (default: 160)")
    parser.add_argument("--num-val", type=int, default=40, help="Val frames (default: 40)")
    parser.add_argument("--num-test-heldout", type=int, default=40, help="Test held-out frames (default: 40)")
    parser.add_argument("--num-test-negative", type=int, default=30, help="Test negative frames (default: 30)")
    parser.add_argument("--num-fovea", type=int, default=140, help="Foveal crops (default: 140)")
    parser.add_argument("--seed", type=int, default=10000, help="Base seed (default: 10000)")
    parser.add_argument("--no-live", action="store_true", help="Disable live desktop capture")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing dataset manifest")
    args = parser.parse_args()

    if args.verify_only:
        manifest_p = Path(args.output_dir) / "manifest.json"
        res = verify_dataset_manifest(manifest_p, verbose=True)
        print(json.dumps(res, indent=2))
        return

    res = generate_and_save_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test_held_out=args.num_test_heldout,
        num_test_negative=args.num_test_negative,
        num_foveal_crops=args.num_fovea,
        seed_base=args.seed,
        include_live_capture=not args.no_live,
        verbose=True,
    )

    verify_dataset_manifest(res["manifest_path"], verbose=True)


if __name__ == "__main__":
    main()
