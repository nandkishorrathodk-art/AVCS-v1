"""Legacy shim for avcs.ace.streamer.

Re-exports ScreenStreamer from canonical avcs.runtime package.
"""
from __future__ import annotations

import avcs.runtime.streamer as _streamer
from avcs.runtime.streamer import *  # noqa: F401, F403
from avcs.runtime.streamer import AceStreamer, ScreenStreamer

__all__ = [
    "AceStreamer",
    "ScreenStreamer",
]


def __getattr__(name: str):
    return getattr(_streamer, name)
