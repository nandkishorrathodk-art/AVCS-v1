"""A scripted set of screens for the assistant to look at, headless.

A machine with no display still needs *something* for the eyes to perceive, or
the page shows an empty rectangle and the assistant has nothing to talk about.
This renders whole scene archetypes -- a desktop, a terminal, a media view, a
dialog, a scrolling list -- using the same generators the vision model is
trained and evaluated on (`forge.vision.scenes`).

It is deliberately **not** a set of labelled controls. The product is an
assistant that looks at a screen and describes it; drawing buttons would turn
the demo back into the control detector this replaced. A scene here has layout,
tone, texture and motion, which is exactly what `ScenePerception` reports.

Honest scope: these are synthetic scenes, not a generalisation claim about real
applications. They exist so the perception path is exercisable and visible on a
box with no display, and the page says so.
"""
from __future__ import annotations

from typing import List

import numpy as np

CATEGORIES: List[str] = ["desktop", "terminal", "media", "dialog", "scroll"]


class DemoSceneSource:
    """Cycles a scripted scene, so motion and change are visible per frame.

    Implements the callable frame-source contract `ScenePerception` accepts
    (``capture()`` and ``__call__``), so it can be swapped for a real capture
    without changing perception.
    """

    def __init__(
        self,
        category: str = "desktop",
        width: int = 640,
        height: int = 400,
        seed: int = 7,
    ) -> None:
        self.width = width
        self.height = height
        self.seed = seed
        self._generator = None
        self.category = category
        self._build(category)

    def _build(self, category: str) -> None:
        from forge.vision.scenes import SceneGenerator

        self.category = category
        self._generator = SceneGenerator(
            category, width=self.width, height=self.height, seed=self.seed
        )

    def set_category(self, category: str) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"unknown scene category {category!r}")
        self._build(category)

    def capture(self) -> np.ndarray:
        return self._generator.grab()

    def __call__(self) -> np.ndarray:
        return self.capture()
