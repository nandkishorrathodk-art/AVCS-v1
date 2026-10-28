"""The AVCS GUI assistant: a live screen view and a Hinglish conversation."""
from __future__ import annotations

from avcs.gui.demo_source import CATEGORIES, DemoSceneSource
from avcs.gui.server import GuiApp, ScreenModel, build_app

__all__ = [
    "CATEGORIES",
    "DemoSceneSource",
    "GuiApp",
    "ScreenModel",
    "build_app",
]
