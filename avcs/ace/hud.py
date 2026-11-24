"""Legacy shim for avcs.ace.hud.

Re-exports Dynamic Island HUD and click reticle from canonical avcs.os_io package.
"""
from __future__ import annotations

import avcs.os_io.hud as _hud
from avcs.os_io.hud import *  # noqa: F401, F403
from avcs.os_io.hud import (
    AVCSVisualHUD,
    COLOR_HIGHLIGHT,
    COLOR_OBSIDIAN,
    COLOR_RIM,
    STATE_COLORS,
    _blend_bgr,
    _pack_bgr,
    _setup_win32_signatures,
    _unpack_bgr,
    play_acoustic_earcon,
)

__all__ = [
    "AVCSVisualHUD",
    "COLOR_HIGHLIGHT",
    "COLOR_OBSIDIAN",
    "COLOR_RIM",
    "STATE_COLORS",
    "_blend_bgr",
    "_pack_bgr",
    "_setup_win32_signatures",
    "_unpack_bgr",
    "play_acoustic_earcon",
]


def __getattr__(name: str):
    return getattr(_hud, name)

