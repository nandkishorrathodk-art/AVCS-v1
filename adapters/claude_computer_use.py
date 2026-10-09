"""Anthropic Claude Computer Use Streaming Pre-Filter.

Acts as a local System-1 reflex layer in front of Claude 3.5 / 3.7 Computer Use.
Monitors the screen continuously at 30 FPS locally, suppressing redundant frames
and only invoking expensive Claude API calls when meaningful visual state shifts occur.
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from avcs import ScreenEye, CortexState


@dataclass
class StreamFilterStats:
    total_screen_ticks: int = 0
    screenshots_sent_to_llm: int = 0
    frames_suppressed: int = 0
    estimated_tokens_saved: int = 0
    estimated_dollars_saved: float = 0.0

    @property
    def token_reduction_percentage(self) -> float:
        total = self.screenshots_sent_to_llm + self.frames_suppressed
        if total == 0:
            return 0.0
        return (self.frames_suppressed / total) * 100.0


class AVCSClaudeStreamFilter:
    """Intelligent rate limiter and visual pre-filter for Claude Computer Use."""

    # Standard Claude 3.5 Sonnet vision token consumption for 1080p images
    AVG_TOKENS_PER_SCREENSHOT: int = 1600
    COST_PER_MILLION_INPUT_TOKENS: float = 3.0  # $3.00 per 1M tokens

    def __init__(
        self,
        eye: Optional[ScreenEye] = None,
        surprise_threshold: float = 0.35,
        min_seconds_between_frames: float = 0.5,
        settle_timeout: float = 3.0,
    ) -> None:
        self.eye = eye or ScreenEye()
        self.surprise_threshold = surprise_threshold
        self.min_seconds_between_frames = min_seconds_between_frames
        self.settle_timeout = settle_timeout

        self.last_sent_timestamp: float = 0.0
        self.last_sent_category: Optional[str] = None
        self.stats = StreamFilterStats()

    def should_emit_screenshot(self, frame: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """Evaluates whether the current screen state warrants sending to Claude.

        Returns:
            (should_emit: bool, reason: str)
        """
        self.stats.total_screen_ticks += 1
        now = time.time()
        state = self.eye.perceive_now(frame=frame)

        # 1. Emit if this is the very first frame (initial agent observation)
        if self.last_sent_category is None:
            self._record_emission(state, now)
            return True, "Initial observation frame"

        # 2. Enforce minimum throttle interval
        if (now - self.last_sent_timestamp) < self.min_seconds_between_frames:
            self._record_suppression()
            return False, "Rate-limited (min interval)"

        # 3. Suppress if UI is actively animating or in layout shift
        if state.stability_state in ["animating", "transitioning"]:
            self._record_suppression()
            return False, f"Screen animating ({state.stability_state})"

        # 4. Emit if high predictive coding surprise occurred (modal, alert, dialog)
        if state.surprise_score >= self.surprise_threshold:
            self._record_emission(state, now)
            return True, f"Visual surprise triggered (score: {state.surprise_score:.2f})"

        # 5. Emit if application UI category changed (e.g. browser -> code editor)
        if state.primary_category != self.last_sent_category and state.category_confidence > 0.4:
            self._record_emission(state, now)
            return True, f"Category transition ({self.last_sent_category} -> {state.primary_category})"

        # 6. Otherwise, screen has not meaningfully changed; suppress frame
        self._record_suppression()
        return False, "Redundant frame (no semantic shift)"

    def format_claude_screenshot(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Formats the current frame into Anthropic's Computer Use API payload."""
        if frame is None:
            frame = self.eye.source.grab()

        pil_img = Image.fromarray(frame)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64_data,
            },
        }

    def wait_and_capture(self) -> Dict[str, Any]:
        """Blocks until screen settles and returns formatted Claude screenshot."""
        self.eye.wait_until_stable(timeout=self.settle_timeout)
        frame = self.eye.source.grab()
        self._record_emission(self.eye.current_state or self.eye.perceive_now(frame), time.time())
        return self.format_claude_screenshot(frame)

    def _record_emission(self, state: CortexState, timestamp: float) -> None:
        self.stats.screenshots_sent_to_llm += 1
        self.last_sent_timestamp = timestamp
        self.last_sent_category = state.primary_category

    def _record_suppression(self) -> None:
        self.stats.frames_suppressed += 1
        self.stats.estimated_tokens_saved += self.AVG_TOKENS_PER_SCREENSHOT
        cost_savings = (self.AVG_TOKENS_PER_SCREENSHOT / 1_000_000.0) * self.COST_PER_MILLION_INPUT_TOKENS
        self.stats.estimated_dollars_saved += cost_savings
