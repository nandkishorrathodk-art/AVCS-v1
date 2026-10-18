"""Tier 2: Boundary & Corner Cases E2E Test Suite.

Stress-tests extreme dimensions, resolution limits, pure negative screens
(verifying the zero false-positive mandate), contrast limits, caret frequency
boundaries, and adversarial query/coordinate inputs.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pytest

from avcs.grounding import FovealInspection
from avcs.human_eye import HumanEye, HumanEyeState
from avcs.sdk import HumanEyeClient, UnfocusedInputElementError
from forge.vision.cortex.motor.failsafe import SafetyBreakoutException
from forge.vision.cortex.sdk import UnvalidatedGroundingError


@pytest.mark.tier2
class TestTier2ResolutionAndDimensionBoundaries:
    """Stress-tests resolution and aspect ratio boundaries."""

    def test_boundary_minimum_qvga_resolution(self, avcs_client: HumanEyeClient):
        """Verifies perception pipeline handles small QVGA (320x240) frames cleanly."""
        frame = np.full((240, 320, 3), (180, 180, 180), dtype=np.uint8)
        state = avcs_client.perceive(frame=frame)
        assert isinstance(state, HumanEyeState)

    def test_boundary_standard_fhd_resolution(self, avcs_client: HumanEyeClient, screen_gen):
        """Verifies perception pipeline on standard 1080p FHD (1920x1080) screen."""
        frame, _ = screen_gen.desktop_ui_frame(width=1920, height=1080, seed=1080)
        state = avcs_client.perceive(frame=frame)
        assert isinstance(state, HumanEyeState)

    def test_boundary_4k_ultra_high_resolution(self, avcs_client: HumanEyeClient):
        """Verifies perception handles 4K UHD (3840x2160) input without memory failure."""
        frame = np.full((2160, 3840, 3), (220, 220, 225), dtype=np.uint8)
        state = avcs_client.perceive(frame=frame)
        assert isinstance(state, HumanEyeState)

    def test_boundary_non_standard_aspect_ratio(self, avcs_client: HumanEyeClient):
        """Verifies non-standard ultrawide banner resolution (1600x600, 8:3 ratio)."""
        frame = np.full((600, 1600, 3), (200, 200, 200), dtype=np.uint8)
        state = avcs_client.perceive(frame=frame)
        assert isinstance(state, HumanEyeState)


@pytest.mark.tier2
class TestTier2NegativeScreensZeroFalsePositives:
    """Negative controls: Verifies ZERO false-positive hotspot affordances on blank/wallpaper screens."""

    def test_boundary_negative_pure_black_screen_zero_hotspots(self, avcs_client: HumanEyeClient, screen_gen):
        """Zero false positives on completely black screen (R=G=B=0)."""
        black = screen_gen.blank_frame(width=1280, height=720, color=(0, 0, 0))
        state = avcs_client.perceive(frame=black)
        assert len(state.affordance_hotspots) == 0, f"Expected 0 hotspots on pure black screen, got {len(state.affordance_hotspots)}"

    def test_boundary_negative_dark_oled_screen_zero_hotspots(self, avcs_client: HumanEyeClient, screen_gen):
        """Zero false positives on dark OLED background (20, 20, 20)."""
        oled = screen_gen.blank_frame(width=1280, height=720, color=(20, 20, 20))
        state = avcs_client.perceive(frame=oled)
        assert len(state.affordance_hotspots) == 0, f"Expected 0 hotspots on dark OLED screen, got {len(state.affordance_hotspots)}"

    def test_boundary_negative_uniform_gray_screen_zero_hotspots(self, avcs_client: HumanEyeClient, screen_gen):
        """Zero false positives on uniform middle-gray screen (128, 128, 128)."""
        gray = screen_gen.blank_frame(width=1280, height=720, color=(128, 128, 128))
        state = avcs_client.perceive(frame=gray)
        assert len(state.affordance_hotspots) == 0, f"Expected 0 hotspots on middle gray screen, got {len(state.affordance_hotspots)}"

    def test_boundary_negative_horizontal_gradient_zero_hotspots(self, avcs_client: HumanEyeClient, screen_gen):
        """Zero false positives on smooth horizontal luminance gradient."""
        grad = screen_gen.gradient_frame(width=1280, height=720, direction="horizontal")
        state = avcs_client.perceive(frame=grad)
        assert len(state.affordance_hotspots) == 0, f"Expected 0 hotspots on horizontal gradient, got {len(state.affordance_hotspots)}"

    def test_boundary_negative_vertical_gradient_zero_hotspots(self, avcs_client: HumanEyeClient, screen_gen):
        """Zero false positives on smooth vertical luminance gradient."""
        grad = screen_gen.gradient_frame(width=1280, height=720, direction="vertical")
        state = avcs_client.perceive(frame=grad)
        assert len(state.affordance_hotspots) == 0, f"Expected 0 hotspots on vertical gradient, got {len(state.affordance_hotspots)}"

    def test_boundary_negative_subtle_sensor_noise_zero_hotspots(self, avcs_client: HumanEyeClient):
        """Zero false positives on uniform gray screen with subtle Gaussian noise (std=1.5)."""
        rng = np.random.RandomState(999)
        base = np.full((720, 1280, 3), 128.0, dtype=np.float32)
        noise = rng.normal(0, 1.5, size=(720, 1280, 3)).astype(np.float32)
        noisy = np.clip(base + noise, 0, 255).astype(np.uint8)
        state = avcs_client.perceive(frame=noisy)
        assert len(state.affordance_hotspots) == 0, f"Expected 0 hotspots on noisy background, got {len(state.affordance_hotspots)}"


@pytest.mark.tier2
class TestTier2ContrastThresholdBoundaries:
    """Foveal acuity contrast gating boundary conditions (0.010 minimum contrast threshold)."""

    def test_boundary_zero_contrast_crop_fail_closed(self, avcs_client: HumanEyeClient, screen_gen):
        """Contrast = 0.000 (flat crop) strictly returns None."""
        flat = screen_gen.blank_frame(width=200, height=200, color=(100, 100, 100))
        hotspot = (0.5, 0.5, "button", 0.95)
        res = avcs_client.eye.refine_affordance_with_fovea(hotspot=hotspot, frame=flat, min_contrast=0.010)
        assert res is None

    def test_boundary_sub_threshold_contrast_fail_closed(self, avcs_client: HumanEyeClient):
        """Contrast < 0.010 (subtle gradient below threshold) strictly returns None."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # 1-value difference across 200 pixels -> contrast score is ~0.002
        frame[:, :] = np.linspace(100, 101, 200, dtype=np.uint8)[:, None, None]
        hotspot = (0.5, 0.5, "button", 0.90)
        res = avcs_client.eye.refine_affordance_with_fovea(hotspot=hotspot, frame=frame, min_contrast=0.010)
        assert res is None

    def test_boundary_high_contrast_crop_accepted(self, avcs_client: HumanEyeClient):
        """Clear black/white step edge has contrast >> 0.010 and is accepted."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:, 100:] = 255  # High-contrast vertical edge
        hotspot = (0.5, 0.5, "button", 0.90)
        res = avcs_client.eye.refine_affordance_with_fovea(hotspot=hotspot, frame=frame, min_contrast=0.010)
        assert res is not None
        assert isinstance(res, FovealInspection)
        assert res.contrast_score >= 0.010


@pytest.mark.tier2
class TestTier2CaretBlinkingFrequencies:
    """Temporal caret blink frequency boundaries (0.8 - 2.0 Hz band)."""

    def test_boundary_caret_too_slow_0_2_hz_rejected(self, avcs_client: HumanEyeClient, screen_gen):
        """0.2 Hz frequency (5 second period, too slow) produces is_focused=False."""
        frames, ts = screen_gen.caret_frames(freq_hz=0.2, num_frames=30, fps=30.0)
        state = avcs_client.check_focus(frames=frames, timestamps=ts)
        assert not state.is_focused

    def test_boundary_caret_valid_nominal_1_0_hz_accepted(self, avcs_client: HumanEyeClient, screen_gen):
        """1.0 Hz nominal frequency (standard OS caret blink) produces is_focused=True."""
        frames, ts = screen_gen.caret_frames(freq_hz=1.0, num_frames=90, fps=30.0)
        state = avcs_client.check_focus(frames=frames, timestamps=ts)
        assert state.is_focused
        assert state.confidence >= 0.70

    def test_boundary_caret_valid_lower_band_0_85_hz_accepted(self, avcs_client: HumanEyeClient, screen_gen):
        """0.85 Hz (within biological range 0.8-2.0 Hz) produces is_focused=True."""
        frames, ts = screen_gen.caret_frames(freq_hz=0.85, num_frames=60, fps=30.0)
        state = avcs_client.check_focus(frames=frames, timestamps=ts)
        assert state.is_focused

    def test_boundary_caret_valid_upper_band_1_95_hz_accepted(self, avcs_client: HumanEyeClient, screen_gen):
        """1.95 Hz (near upper bound of 0.8-2.0 Hz) produces is_focused=True."""
        frames, ts = screen_gen.caret_frames(freq_hz=1.95, num_frames=60, fps=30.0)
        state = avcs_client.check_focus(frames=frames, timestamps=ts)
        assert state.is_focused

    def test_boundary_caret_high_frequency_flicker_10_hz_rejected(self, avcs_client: HumanEyeClient, screen_gen):
        """10.0 Hz high-frequency flicker (e.g. video/display artifact) is rejected."""
        frames, ts = screen_gen.caret_frames(freq_hz=10.0, num_frames=45, fps=30.0)
        state = avcs_client.check_focus(frames=frames, timestamps=ts)
        assert not state.is_focused


@pytest.mark.tier2
class TestTier2QueryAndCoordinateBoundaries:
    """Stress-tests query text and coordinate edge values."""

    def test_boundary_empty_query_string(self, avcs_client: HumanEyeClient):
        """Empty query string '' does not crash look_at."""
        res = avcs_client.look_at("")
        assert isinstance(res, tuple) and len(res) == 2

    def test_boundary_ultra_long_query_1000_chars(self, avcs_client: HumanEyeClient):
        """1,000-character long query string does not buffer overflow."""
        long_query = "Search button " * 75
        res = avcs_client.look_at(long_query)
        assert isinstance(res, tuple) and len(res) == 2

    def test_boundary_special_characters_and_emojis_query(self, avcs_client: HumanEyeClient):
        """Query with emojis, quotes, and symbols."""
        weird_query = "🔍 Submit 'Button' & <Cancel> 🚀 \x00\t\n"
        res = avcs_client.look_at(weird_query)
        assert isinstance(res, tuple) and len(res) == 2

    def test_boundary_normalized_coordinate_extremes_0_0_and_1_1(self, avcs_client: HumanEyeClient):
        """Corner breakout (0.0, 0.0) trips failsafe; near-boundary coordinates succeed safely."""
        with pytest.raises(SafetyBreakoutException):
            avcs_client.move_to((0.0, 0.0))
        # Reset latched failsafe to re-enable motor operations
        if hasattr(avcs_client.motor_driver, "failsafe"):
            avcs_client.motor_driver.failsafe.reset()
            avcs_client.motor_driver.move_to_normalized(0.5, 0.5)
        # Near-boundary coordinates outside breakout margin succeed
        res_near0 = avcs_client.move_to((0.05, 0.05))
        assert res_near0 is not None
        res_near1 = avcs_client.move_to((0.95, 0.95))
        assert res_near1 is not None

    def test_boundary_rapid_sequential_observations(self, avcs_client: HumanEyeClient, screen_gen):
        """10 rapid sequential perceive calls execute without race condition or memory leak."""
        frame, _ = screen_gen.desktop_ui_frame(seed=500)
        t0 = time.perf_counter()
        for _ in range(10):
            st = avcs_client.perceive(frame=frame)
            assert st is not None
        elapsed = time.perf_counter() - t0
        # 10 calls should complete in under 500 ms on CPU
        assert elapsed < 0.500
