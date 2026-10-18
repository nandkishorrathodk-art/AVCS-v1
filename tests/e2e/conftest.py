"""Pytest configuration and shared fixtures for AVCS E2E tests."""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pytest

from avcs.sdk import HumanEyeClient
from forge.vision.cortex.motor.driver import VirtualCursorDriver

try:
    from avcs.controller import AVCSController
    HAS_AVCS_CONTROLLER = True
except ImportError:
    HAS_AVCS_CONTROLLER = False


def pytest_configure(config):
    """Register custom markers for the 4-tier test architecture."""
    config.addinivalue_line("markers", "tier1: Tier 1 Feature Coverage tests")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary & Corner Case tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Interaction tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World Scenario tests")


@pytest.fixture
def virtual_driver() -> VirtualCursorDriver:
    """Provides an isolated in-memory virtual cursor driver."""
    return VirtualCursorDriver()


@pytest.fixture
def avcs_client() -> Generator[HumanEyeClient, None, None]:
    """Provides a safely isolated HumanEyeClient with synthetic source and virtual motor."""
    client = HumanEyeClient(
        source="synthetic",
        force_virtual_motor=True,
        allow_synthetic_fallback=True,
    )
    try:
        yield client
    finally:
        try:
            client.close()
        except Exception:
            pass


@pytest.fixture
def controller_or_client(avcs_client: HumanEyeClient) -> Any:
    """Returns AVCSController if available, otherwise HumanEyeClient."""
    if HAS_AVCS_CONTROLLER:
        ctrl = AVCSController(source="synthetic", force_virtual_motor=True)
        try:
            yield ctrl
        finally:
            try:
                ctrl.close()
            except Exception:
                pass
    else:
        yield avcs_client


class SyntheticScreenGenerator:
    """Generates synthetic UI frames with exact known ground truth elements."""

    @staticmethod
    def blank_frame(
        width: int = 1280,
        height: int = 720,
        color: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """Generates a uniform color frame."""
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        return frame

    @staticmethod
    def gradient_frame(
        width: int = 1280,
        height: int = 720,
        direction: str = "horizontal",
    ) -> np.ndarray:
        """Generates a smooth luminance gradient frame."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if direction == "horizontal":
            grad = np.linspace(30, 220, width, dtype=np.uint8)
            frame[:, :, 0] = grad
            frame[:, :, 1] = grad
            frame[:, :, 2] = grad
        else:
            grad = np.linspace(30, 220, height, dtype=np.uint8)[:, None]
            frame[:, :, 0] = grad
            frame[:, :, 1] = grad
            frame[:, :, 2] = grad
        return frame

    @staticmethod
    def desktop_ui_frame(
        width: int = 1280,
        height: int = 720,
        seed: int = 42,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generates a desktop UI screen with known elements and exact ground truth."""
        rng = np.random.RandomState(seed)
        # Background: light desktop canvas
        frame = np.full((height, width, 3), (240, 240, 245), dtype=np.uint8)
        
        # Menu bar at top (height 40)
        frame[0:40, :] = (45, 52, 60)
        
        elements: List[Dict[str, Any]] = []

        # Add buttons with distinct colors
        button_specs = [
            ("Play button", (60, 80, 180, 120), (50, 160, 90)),
            ("Submit button", (220, 80, 360, 120), (30, 100, 220)),
            ("Cancel button", (400, 80, 520, 120), (200, 60, 60)),
            ("Text input", (60, 160, 320, 200), (255, 255, 255)),
        ]

        for label, (x1, y1, x2, y2), col in button_specs:
            frame[y1:y2, x1:x2] = col
            # Add a dark border for contrast
            frame[y1:y1+2, x1:x2] = (20, 20, 20)
            frame[y2-2:y2, x1:x2] = (20, 20, 20)
            frame[y1:y2, x1:x1+2] = (20, 20, 20)
            frame[y1:y2, x2-2:x2] = (20, 20, 20)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            norm_bbox = (x1 / width, y1 / height, x2 / width, y2 / height)
            norm_centroid = (cx / width, cy / height)
            elements.append({
                "label": label,
                "pixel_bbox": (x1, y1, x2, y2),
                "pixel_centroid": (cx, cy),
                "norm_bbox": norm_bbox,
                "norm_centroid": norm_centroid,
            })

        return frame, elements

    @staticmethod
    def caret_frames(
        freq_hz: float = 1.0,
        num_frames: int = 90,
        fps: float = 30.0,
        width: int = 200,
        height: int = 100,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Generates a sequence of frames simulating a blinking caret at freq_hz."""
        frames: List[np.ndarray] = []
        timestamps: List[float] = []
        period_s = 1.0 / freq_hz
        cx, cy = width // 2, height // 2

        for i in range(num_frames):
            t = i / fps
            timestamps.append(t)
            # Input field background (white)
            frame = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
            # Border
            frame[0:2, :] = (150, 150, 150)
            frame[-2:, :] = (150, 150, 150)
            frame[:, 0:2] = (150, 150, 150)
            frame[:, -2:] = (150, 150, 150)

            # Square wave blink state: on for first half of period
            phase = (t % period_s) / period_s
            is_visible = phase < 0.5
            if is_visible:
                # Draw vertical caret (1-2 px wide, 20 px high)
                frame[cy-10:cy+10, cx:cx+2] = (0, 0, 0)

            frames.append(frame)

        return frames, timestamps


@pytest.fixture
def screen_gen() -> SyntheticScreenGenerator:
    """Fixture providing the synthetic screen generator."""
    return SyntheticScreenGenerator()
