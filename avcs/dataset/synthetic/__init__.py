"""Modular synthetic desktop UI generator package."""
from avcs.dataset.synthetic.composer import DesktopComposer, UI_CATEGORIES
from avcs.dataset.synthetic.generator import (
    SyntheticDesktopFrame,
    SyntheticDesktopGenerator,
)
from avcs.dataset.synthetic.primitives import (
    SyntheticUIElement,
    draw_background,
    draw_button,
    draw_icon,
    draw_input,
    draw_tab,
    draw_window,
)
from avcs.dataset.synthetic.query_synth import (
    NEGATIVE_QUERY_BANK,
    sample_negative_queries,
    synthesize_queries,
)
from avcs.dataset.synthetic.themes import (
    AVAILABLE_THEMES,
    THEMES,
    ThemePalette,
    get_theme,
)

__all__ = [
    "AVAILABLE_THEMES",
    "THEMES",
    "UI_CATEGORIES",
    "NEGATIVE_QUERY_BANK",
    "DesktopComposer",
    "SyntheticDesktopFrame",
    "SyntheticDesktopGenerator",
    "SyntheticUIElement",
    "ThemePalette",
    "draw_background",
    "draw_button",
    "draw_icon",
    "draw_input",
    "draw_tab",
    "draw_window",
    "get_theme",
    "sample_negative_queries",
    "synthesize_queries",
]
