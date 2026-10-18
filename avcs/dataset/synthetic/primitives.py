"""Low-level UI primitive drawing functions and element dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avcs.dataset.synthetic.query_synth import synthesize_queries
from avcs.dataset.synthetic.themes import ThemePalette, get_theme

# Font cache: (size, bold) -> ImageFont
_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.ImageFont] = {}

FONT_PATHS = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def get_font(size: int = 11, bold: bool = False) -> ImageFont.ImageFont:
    """Retrieve cached font or default fallback font."""
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    for p in FONT_PATHS:
        if bold and ("b.ttf" in p.lower() or "bold" in p.lower()) and Path(p).exists():
            try:
                font = ImageFont.truetype(p, size)
                _FONT_CACHE[key] = font
                return font
            except Exception:
                pass
        elif not bold and Path(p).exists():
            try:
                font = ImageFont.truetype(p, size)
                _FONT_CACHE[key] = font
                return font
            except Exception:
                pass

    # Generic fallback
    try:
        font = ImageFont.load_default()
        _FONT_CACHE[key] = font
        return font
    except Exception:
        font = ImageFont.load_default()
        return font


@dataclass
class SyntheticUIElement:
    """Ground-truth metadata for a rendered UI element."""
    element_id: str
    element_kind: str  # in {"button", "input_field", "icon", "tab", "link", "text_label", "container"}
    semantic_role: str
    text_label: str
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    bbox_norm: Tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax) in [0.0, 1.0]
    centroid_px: Tuple[float, float]  # (cx, cy)
    centroid_norm: Tuple[float, float]  # (norm_x, norm_y) in [0.0, 1.0]
    action_type: str  # "clickable", "typeable", "static"
    target_query: str
    query_variations: List[str] = field(default_factory=list)
    is_focused: bool = False
    has_caret: bool = False
    caret_coord_norm: Optional[Tuple[float, float]] = None
    window_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert element to JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "element_id": self.element_id,
            "x": round(self.centroid_norm[0], 4),
            "y": round(self.centroid_norm[1], 4),
            "action_type": self.action_type,
            "element_kind": self.element_kind,
            "bbox": [round(c, 4) for c in self.bbox_norm],
            "target_query": self.target_query,
            "query_variations": self.query_variations,
        }
        if self.is_focused:
            d["is_focused"] = True
        if self.has_caret and self.caret_coord_norm:
            d["has_caret"] = True
            d["caret_coord"] = [round(c, 4) for c in self.caret_coord_norm]
        return d


def _make_coords(
    bbox: Tuple[int, int, int, int],
    screen_size: Tuple[int, int],
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float], Tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    sw, sh = screen_size
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ymin = max(0.0, min(1.0, y1 / sh))
    xmin = max(0.0, min(1.0, x1 / sw))
    ymax = max(0.0, min(1.0, y2 / sh))
    xmax = max(0.0, min(1.0, x2 / sw))
    norm_x = max(0.0, min(1.0, cx / sw))
    norm_y = max(0.0, min(1.0, cy / sh))
    return (ymin, xmin, ymax, xmax), (cx, cy), (norm_x, norm_y)


def draw_button(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    text: str,
    theme: ThemePalette,
    element_id: str,
    screen_size: Tuple[int, int] = (256, 256),
    variant: str = "default",
    color_hint: Optional[str] = None,
    position_hint: Optional[str] = None,
    context: Optional[str] = None,
    font_size: int = 10,
    window_id: Optional[str] = None,
) -> SyntheticUIElement:
    """Draws a styled button with rounded corners, border, and centered text."""
    x1, y1, x2, y2 = bbox
    font = get_font(font_size, bold=(variant in ("primary", "danger")))

    if variant == "primary":
        fill = theme.accent_color
        text_fill = theme.accent_text
        outline = theme.accent_color
        col_hint = color_hint or "blue"
    elif variant == "danger":
        fill = theme.danger_color
        text_fill = (255, 255, 255)
        outline = theme.danger_color
        col_hint = color_hint or "red"
    elif variant == "success":
        fill = theme.success_color
        text_fill = (255, 255, 255)
        outline = theme.success_color
        col_hint = color_hint or "green"
    else:
        fill = theme.button_bg
        text_fill = theme.button_text
        outline = theme.border_color
        col_hint = color_hint

    radius = min(4, max(2, (y2 - y1) // 4))
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=1)

    # Center text
    try:
        tb = font.getbbox(text)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
    except Exception:
        tw = len(text) * (font_size * 0.6)
        th = font_size
    tx = x1 + max(2, (x2 - x1 - tw) // 2)
    ty = y1 + max(1, (y2 - y1 - th) // 2)
    draw.text((tx, ty), text, fill=text_fill, font=font)

    bbox_norm, centroid_px, centroid_norm = _make_coords(bbox, screen_size)
    tq, qvars = synthesize_queries(
        element_kind="button",
        text_label=text,
        action_type="clickable",
        color_hint=col_hint,
        position_hint=position_hint,
        context=context,
    )

    return SyntheticUIElement(
        element_id=element_id,
        element_kind="button",
        semantic_role=f"{variant}_button",
        text_label=text,
        bbox_px=bbox,
        bbox_norm=bbox_norm,
        centroid_px=centroid_px,
        centroid_norm=centroid_norm,
        action_type="clickable",
        target_query=tq,
        query_variations=qvars,
        window_id=window_id,
    )


def draw_input(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    text: str,
    placeholder: str,
    theme: ThemePalette,
    element_id: str,
    screen_size: Tuple[int, int] = (256, 256),
    is_focused: bool = False,
    has_caret: bool = False,
    position_hint: Optional[str] = None,
    context: Optional[str] = None,
    font_size: int = 10,
    window_id: Optional[str] = None,
) -> SyntheticUIElement:
    """Draws a text input box with focus ring and caret indicator if focused."""
    x1, y1, x2, y2 = bbox
    font = get_font(font_size, bold=False)

    fill = theme.input_bg
    outline = theme.accent_color if is_focused else theme.input_border
    radius = min(3, max(2, (y2 - y1) // 5))
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=1)

    display_text = text if text else placeholder
    text_color = theme.text_color if text else theme.text_muted
    tx = x1 + 6
    ty = y1 + max(1, (y2 - y1 - font_size) // 2)
    if display_text:
        # Clip display text to input width
        max_chars = max(4, (x2 - x1 - 12) // (font_size // 2 + 1))
        draw.text((tx, ty), display_text[:int(max_chars)], fill=text_color, font=font)

    caret_coord_norm = None
    if is_focused and has_caret:
        # Render caret line after text or at start
        text_offset = 0
        if text:
            try:
                tb = font.getbbox(text)
                text_offset = tb[2] - tb[0]
            except Exception:
                text_offset = len(text) * (font_size * 0.6)
        cx_caret = min(x2 - 4, tx + int(text_offset) + 2)
        cy1 = y1 + 3
        cy2 = y2 - 3
        draw.line([(cx_caret, cy1), (cx_caret, cy2)], fill=theme.accent_color, width=1)
        sw, sh = screen_size
        caret_coord_norm = (cx_caret / sw, ((cy1 + cy2) / 2.0) / sh)

    bbox_norm, centroid_px, centroid_norm = _make_coords(bbox, screen_size)
    label_for_query = placeholder or text or "search"
    tq, qvars = synthesize_queries(
        element_kind="input_field",
        text_label=label_for_query,
        action_type="typeable",
        position_hint=position_hint,
        context=context,
    )

    return SyntheticUIElement(
        element_id=element_id,
        element_kind="input_field",
        semantic_role="search_or_text_input",
        text_label=label_for_query,
        bbox_px=bbox,
        bbox_norm=bbox_norm,
        centroid_px=centroid_px,
        centroid_norm=centroid_norm,
        action_type="typeable",
        target_query=tq,
        query_variations=qvars,
        is_focused=is_focused,
        has_caret=has_caret,
        caret_coord_norm=caret_coord_norm,
        window_id=window_id,
    )


def draw_tab(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    text: str,
    theme: ThemePalette,
    element_id: str,
    screen_size: Tuple[int, int] = (256, 256),
    is_active: bool = False,
    position_hint: Optional[str] = None,
    context: Optional[str] = None,
    font_size: int = 9,
    window_id: Optional[str] = None,
) -> SyntheticUIElement:
    """Draws a tab strip item."""
    x1, y1, x2, y2 = bbox
    font = get_font(font_size, bold=is_active)

    fill = theme.window_bg if is_active else theme.titlebar_bg
    outline = theme.border_color
    draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=1)

    if is_active:
        # Active top indicator accent bar
        draw.line([(x1 + 1, y1), (x2 - 1, y1)], fill=theme.accent_color, width=2)

    text_color = theme.text_color if is_active else theme.text_muted
    tx = x1 + 5
    ty = y1 + max(1, (y2 - y1 - font_size) // 2)
    max_chars = max(3, (x2 - x1 - 10) // (font_size // 2 + 1))
    draw.text((tx, ty), text[:int(max_chars)], fill=text_color, font=font)

    bbox_norm, centroid_px, centroid_norm = _make_coords(bbox, screen_size)
    tq, qvars = synthesize_queries(
        element_kind="tab",
        text_label=text,
        action_type="clickable",
        position_hint=position_hint,
        context=context,
    )

    return SyntheticUIElement(
        element_id=element_id,
        element_kind="tab",
        semantic_role="navigation_tab",
        text_label=text,
        bbox_px=bbox,
        bbox_norm=bbox_norm,
        centroid_px=centroid_px,
        centroid_norm=centroid_norm,
        action_type="clickable",
        target_query=tq,
        query_variations=qvars,
        window_id=window_id,
    )


def draw_icon(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    icon_type: str,
    label: str,
    theme: ThemePalette,
    element_id: str,
    screen_size: Tuple[int, int] = (256, 256),
    position_hint: Optional[str] = None,
    font_size: int = 8,
) -> SyntheticUIElement:
    """Draws a desktop shortcut or toolbar icon with geometric glyph and label."""
    x1, y1, x2, y2 = bbox
    font = get_font(font_size, bold=False)

    # Icon tile glyph box
    glyph_h = max(14, int((y2 - y1) * 0.55))
    gx1 = x1 + max(2, (x2 - x1 - glyph_h) // 2)
    gy1 = y1 + 2
    gx2 = gx1 + glyph_h
    gy2 = gy1 + glyph_h

    # Distinct color per icon type
    glyph_colors = {
        "terminal": (34, 197, 94),
        "browser": (59, 130, 246),
        "folder": (234, 179, 8),
        "settings": (156, 163, 175),
        "code": (168, 85, 247),
        "document": (244, 63, 94),
        "media": (249, 115, 22),
    }
    gcolor = glyph_colors.get(icon_type, theme.accent_color)
    draw.rounded_rectangle([gx1, gy1, gx2, gy2], radius=3, fill=gcolor)

    # Simple geometric internal emblem
    draw.rectangle([gx1 + 3, gy1 + 3, gx2 - 3, gy2 - 3], fill=(255, 255, 255))
    draw.rectangle([gx1 + 4, gy1 + 4, gx2 - 4, gy2 - 4], fill=gcolor)

    # Label underneath
    if label:
        try:
            tb = font.getbbox(label)
            tw = tb[2] - tb[0]
        except Exception:
            tw = len(label) * 5
        tx = x1 + max(0, (x2 - x1 - tw) // 2)
        ty = gy2 + 2
        draw.text((tx, ty), label, fill=theme.text_color, font=font)

    bbox_norm, centroid_px, centroid_norm = _make_coords(bbox, screen_size)
    tq, qvars = synthesize_queries(
        element_kind="icon",
        text_label=label or icon_type,
        action_type="clickable",
        position_hint=position_hint,
    )

    return SyntheticUIElement(
        element_id=element_id,
        element_kind="icon",
        semantic_role=f"{icon_type}_icon",
        text_label=label or icon_type,
        bbox_px=bbox,
        bbox_norm=bbox_norm,
        centroid_px=centroid_px,
        centroid_norm=centroid_norm,
        action_type="clickable",
        target_query=tq,
        query_variations=qvars,
    )


def draw_window(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    title: str,
    theme: ThemePalette,
    window_id: str,
    screen_size: Tuple[int, int] = (256, 256),
    is_active: bool = True,
    titlebar_height: int = 20,
) -> Tuple[SyntheticUIElement, List[SyntheticUIElement]]:
    """Draws window frame, titlebar, and min/max/close controls."""
    x1, y1, x2, y2 = bbox
    font = get_font(9, bold=is_active)

    # Window body
    draw.rectangle([x1, y1, x2, y2], fill=theme.window_bg, outline=theme.border_color, width=1)

    # Titlebar
    ty2 = y1 + titlebar_height
    tb_fill = theme.titlebar_bg if is_active else theme.window_bg
    draw.rectangle([x1, y1, x2, ty2], fill=tb_fill, outline=theme.border_color, width=1)

    # Window title text
    max_title_chars = max(4, (x2 - x1 - 50) // 6)
    draw.text((x1 + 6, y1 + 4), title[:int(max_title_chars)], fill=theme.titlebar_text, font=font)

    # Window controls: 3 buttons on top right
    controls: List[SyntheticUIElement] = []
    ctrl_w = 12
    ctrl_h = 12
    ctrl_y = y1 + (titlebar_height - ctrl_h) // 2

    # Close button (red)
    close_x2 = x2 - 4
    close_x1 = close_x2 - ctrl_w
    close_bbox = (close_x1, ctrl_y, close_x2, ctrl_y + ctrl_h)
    draw.rectangle(close_bbox, fill=theme.danger_color)
    draw.text((close_x1 + 3, ctrl_y + 1), "x", fill=(255, 255, 255), font=get_font(8, bold=True))
    c_bbox_norm, c_cent_px, c_cent_norm = _make_coords(close_bbox, screen_size)
    tq, qvars = synthesize_queries("button", "close", "clickable", color_hint="red", context=f"{title} window")
    controls.append(
        SyntheticUIElement(
            element_id=f"{window_id}_close",
            element_kind="button",
            semantic_role="window_close_button",
            text_label="Close",
            bbox_px=close_bbox,
            bbox_norm=c_bbox_norm,
            centroid_px=c_cent_px,
            centroid_norm=c_cent_norm,
            action_type="clickable",
            target_query="Close window button",
            query_variations=["Close window button", "close button", "red close button", "close " + title],
            window_id=window_id,
        )
    )

    # Maximize button
    max_x2 = close_x1 - 3
    max_x1 = max_x2 - ctrl_w
    max_bbox = (max_x1, ctrl_y, max_x2, ctrl_y + ctrl_h)
    draw.rectangle(max_bbox, fill=theme.button_bg, outline=theme.border_color)
    draw.rectangle([max_x1 + 3, ctrl_y + 3, max_x2 - 3, ctrl_y + ctrl_h - 3], outline=theme.text_muted)
    m_bbox_norm, m_cent_px, m_cent_norm = _make_coords(max_bbox, screen_size)
    controls.append(
        SyntheticUIElement(
            element_id=f"{window_id}_maximize",
            element_kind="button",
            semantic_role="window_maximize_button",
            text_label="Maximize",
            bbox_px=max_bbox,
            bbox_norm=m_bbox_norm,
            centroid_px=m_cent_px,
            centroid_norm=m_cent_norm,
            action_type="clickable",
            target_query="Maximize window button",
            query_variations=["Maximize window button", "maximize button"],
            window_id=window_id,
        )
    )

    # Minimize button
    min_x2 = max_x1 - 3
    min_x1 = min_x2 - ctrl_w
    min_bbox = (min_x1, ctrl_y, min_x2, ctrl_y + ctrl_h)
    draw.rectangle(min_bbox, fill=theme.button_bg, outline=theme.border_color)
    draw.line([(min_x1 + 3, ctrl_y + ctrl_h - 4), (min_x2 - 3, ctrl_y + ctrl_h - 4)], fill=theme.text_muted)
    n_bbox_norm, n_cent_px, n_cent_norm = _make_coords(min_bbox, screen_size)
    controls.append(
        SyntheticUIElement(
            element_id=f"{window_id}_minimize",
            element_kind="button",
            semantic_role="window_minimize_button",
            text_label="Minimize",
            bbox_px=min_bbox,
            bbox_norm=n_bbox_norm,
            centroid_px=n_cent_px,
            centroid_norm=n_cent_norm,
            action_type="clickable",
            target_query="Minimize window button",
            query_variations=["Minimize window button", "minimize button"],
            window_id=window_id,
        )
    )

    # Window container element itself
    w_bbox_norm, w_cent_px, w_cent_norm = _make_coords(bbox, screen_size)
    window_elem = SyntheticUIElement(
        element_id=window_id,
        element_kind="container",
        semantic_role="window_container",
        text_label=title,
        bbox_px=bbox,
        bbox_norm=w_bbox_norm,
        centroid_px=w_cent_px,
        centroid_norm=w_cent_norm,
        action_type="clickable",
        target_query=f"{title} window",
        query_variations=[f"{title} window", f"active {title} window"],
        window_id=window_id,
    )

    return window_elem, controls


def draw_background(
    draw: ImageDraw.ImageDraw,
    size: Tuple[int, int],
    theme: ThemePalette,
    bg_type: str = "solid",
    rng: Optional[random.Random] = None,
) -> None:
    """Draws desktop background: solid color, linear/radial gradient, or wallpaper."""
    w, h = size
    r = rng or random.Random(42)

    if bg_type == "solid":
        draw.rectangle([0, 0, w, h], fill=theme.bg_color)
    elif bg_type == "gradient":
        # Linear vertical gradient
        c1 = np.array(theme.bg_color, dtype=np.float32)
        c2 = np.array(theme.accent_color, dtype=np.float32) * 0.4
        for y in range(h):
            t = y / max(1, h - 1)
            color = tuple(np.clip(c1 * (1 - t) + c2 * t, 0, 255).astype(int))
            draw.line([(0, y), (w, y)], fill=color)
    elif bg_type == "radial_gradient":
        # Radial gradient from center
        cx, cy = w // 2, h // 2
        max_dist = math.hypot(cx, cy)
        c1 = np.array(theme.accent_color, dtype=np.float32) * 0.5
        c2 = np.array(theme.bg_color, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                d = math.hypot(x - cx, y - cy) / max_dist
                t = min(1.0, d)
                color = tuple(np.clip(c1 * (1 - t) + c2 * t, 0, 255).astype(int))
                draw.point((x, y), fill=color)
    elif bg_type == "wallpaper":
        # Procedural atmospheric wallpaper with mountain ridges and dusk sky
        # Sky gradient
        sky_top = (18, 24, 48)
        sky_mid = (78, 42, 85)
        sky_low = (196, 92, 54)
        for y in range(int(h * 0.7)):
            t = y / (h * 0.7)
            if t < 0.5:
                u = t * 2.0
                c = tuple(int(sky_top[i] * (1 - u) + sky_mid[i] * u) for i in range(3))
            else:
                u = (t - 0.5) * 2.0
                c = tuple(int(sky_mid[i] * (1 - u) + sky_low[i] * u) for i in range(3))
            draw.line([(0, y), (w, y)], fill=c)

        # Distant mountain ridge
        ridge_pts = [(0, int(h * 0.65))]
        for x in range(0, w + 16, 16):
            ry = int(h * 0.60 + math.sin(x * 0.05 + 1.2) * (h * 0.08) + math.cos(x * 0.02) * (h * 0.04))
            ridge_pts.append((x, ry))
        ridge_pts.extend([(w, h), (0, h)])
        draw.polygon(ridge_pts, fill=(45, 25, 55))

        # Foreground mountain ridge
        fg_pts = [(0, int(h * 0.75))]
        for x in range(0, w + 16, 16):
            ry = int(h * 0.72 + math.sin(x * 0.08 + 2.5) * (h * 0.06))
            fg_pts.append((x, ry))
        fg_pts.extend([(w, h), (0, h)])
        draw.polygon(fg_pts, fill=(20, 15, 30))
    else:
        draw.rectangle([0, 0, w, h], fill=theme.bg_color)
