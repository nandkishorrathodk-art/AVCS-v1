"""Color palettes and theme styling for synthetic desktop generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ThemePalette:
    """Color palette and visual styling attributes for desktop themes."""
    name: str
    bg_color: Tuple[int, int, int]
    window_bg: Tuple[int, int, int]
    titlebar_bg: Tuple[int, int, int]
    titlebar_text: Tuple[int, int, int]
    text_color: Tuple[int, int, int]
    text_muted: Tuple[int, int, int]
    accent_color: Tuple[int, int, int]
    accent_text: Tuple[int, int, int]
    border_color: Tuple[int, int, int]
    input_bg: Tuple[int, int, int]
    input_border: Tuple[int, int, int]
    button_bg: Tuple[int, int, int]
    button_text: Tuple[int, int, int]
    danger_color: Tuple[int, int, int]
    success_color: Tuple[int, int, int]
    taskbar_bg: Tuple[int, int, int]


THEMES: Dict[str, ThemePalette] = {
    "dark_modern": ThemePalette(
        name="dark_modern",
        bg_color=(30, 30, 36),
        window_bg=(37, 37, 42),
        titlebar_bg=(45, 45, 52),
        titlebar_text=(240, 240, 245),
        text_color=(245, 245, 250),
        text_muted=(140, 145, 155),
        accent_color=(0, 120, 215),
        accent_text=(255, 255, 255),
        border_color=(60, 62, 70),
        input_bg=(30, 30, 34),
        input_border=(70, 74, 85),
        button_bg=(55, 58, 68),
        button_text=(240, 240, 245),
        danger_color=(220, 38, 38),
        success_color=(16, 185, 129),
        taskbar_bg=(22, 22, 26),
    ),
    "light_clean": ThemePalette(
        name="light_clean",
        bg_color=(240, 242, 245),
        window_bg=(255, 255, 255),
        titlebar_bg=(230, 233, 238),
        titlebar_text=(30, 32, 38),
        text_color=(20, 22, 28),
        text_muted=(110, 115, 125),
        accent_color=(0, 102, 204),
        accent_text=(255, 255, 255),
        border_color=(205, 210, 220),
        input_bg=(250, 250, 252),
        input_border=(180, 185, 195),
        button_bg=(228, 232, 238),
        button_text=(30, 32, 38),
        danger_color=(220, 38, 38),
        success_color=(16, 185, 129),
        taskbar_bg=(220, 224, 230),
    ),
    "windows_11": ThemePalette(
        name="windows_11",
        bg_color=(32, 32, 32),
        window_bg=(31, 31, 31),
        titlebar_bg=(44, 44, 44),
        titlebar_text=(255, 255, 255),
        text_color=(255, 255, 255),
        text_muted=(160, 160, 160),
        accent_color=(96, 205, 255),
        accent_text=(0, 0, 0),
        border_color=(56, 56, 56),
        input_bg=(24, 24, 24),
        input_border=(80, 80, 80),
        button_bg=(50, 50, 50),
        button_text=(255, 255, 255),
        danger_color=(232, 17, 35),
        success_color=(108, 203, 95),
        taskbar_bg=(26, 26, 26),
    ),
}

AVAILABLE_THEMES = list(THEMES.keys())


def get_theme(name: str | None = None) -> ThemePalette:
    """Retrieve theme palette by name; defaults to dark_modern."""
    if not name or name not in THEMES:
        return THEMES["dark_modern"]
    return THEMES[name]
