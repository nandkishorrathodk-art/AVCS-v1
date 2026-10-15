"""Unit tests for Affordance Hardening and Visual Focus Verification.

Validates:
1. FastAffordanceHead 2D Spatial NMS and Competitive Background Margin Gating.
2. Zero-false-positive rejection on static wallpapers and uniform screens.
3. Foveal acuity boundary refinement and flat-contrast rejection.
4. VisualFocusDetector pure-pixel temporal caret blink detection and fail-closed security.
5. End-to-end integration via HumanEye.verify_focus and HumanEye.refine_affordance_with_fovea.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from avcs.focus import VisualFocusDetector, VisualFocusState
from avcs.human_eye import (
    FastAffordanceHead,
    HumanEye,
    HumanEyeState,
    RealTimeCorticalBackbone,
)


def test_fast_affordance_head_wallpaper_rejection():
    """Verifies that static wallpaper and flat screens produce ZERO false positive affordances."""
    head = FastAffordanceHead(in_dim=128, num_classes=3)
    # A flat or zero feature map represents an unactivated / background wallpaper region
    flat_feat = torch.zeros(1, 128, 16, 16)

    hotspots = head.find_hotspots(flat_feat, action_type="clickable", threshold=0.35, margin=0.05)
    assert len(hotspots) == 0, f"Expected 0 hotspots on flat wallpaper, got {len(hotspots)}"

    hotspots_typeable = head.find_hotspots(flat_feat, action_type="typeable", threshold=0.35, margin=0.05)
    assert len(hotspots_typeable) == 0


def test_fast_affordance_head_2d_nms_peak_suppression():
    """Verifies that a wide button region collapses to a single peak without duplicate adjacent clusters."""
    head = FastAffordanceHead(in_dim=128, num_classes=3)
    feat = torch.zeros(1, 128, 16, 16)

    # Inject a simulated button hotspot covering a 3x3 region centered at (8, 8)
    # Give the center the highest activation and neighbors lower activation
    with torch.no_grad():
        head.head.weight.zero_()
        head.head.bias.zero_()
        # Set channel 0 to activate class 1 (clickable)
        head.head.weight[1, 0, 0, 0] = 5.0

    feat[0, 0, 7:10, 7:10] = 1.0  # Neighbors
    feat[0, 0, 8, 8] = 2.0        # Peak center

    hotspots = head.find_hotspots(feat, action_type="clickable", top_k=5, threshold=0.35, margin=0.05)
    assert len(hotspots) == 1, f"Expected exactly 1 peak after 2D NMS, got {len(hotspots)}"
    peak_x, peak_y, act_type, conf = hotspots[0]

    # Verify peak is centered at (8, 8) -> norm coord (8.5 / 16) = 0.53125
    expected_norm = (8 + 0.5) / 16.0
    assert abs(peak_x - expected_norm) < 1e-4
    assert abs(peak_y - expected_norm) < 1e-4
    assert act_type == "clickable"
    assert conf > 0.50


def test_human_eye_refine_affordance_with_fovea():
    """Verifies that foveal refinement rejects flat contrast and refines structured buttons."""
    eye = HumanEye(device="cpu")

    # 1. Flat screen frame -> contrast score is 0.0 -> must be rejected as None
    flat_frame = np.full((360, 640, 3), 128, dtype=np.uint8)
    dummy_hotspot = (0.5, 0.5, "clickable", 0.85)

    rejected = eye.refine_affordance_with_fovea(dummy_hotspot, frame=flat_frame, min_contrast=0.010)
    assert rejected is None, "Foveal refinement must reject featureless flat screen"

    # 2. Structured button frame -> high contrast -> returns valid FovealInspection
    button_frame = np.full((360, 640, 3), 240, dtype=np.uint8)
    # Draw button at center
    button_frame[150:210, 270:370] = 30
    button_frame[165:195, 290:350] = 255  # Inner text/icon

    refined = eye.refine_affordance_with_fovea(dummy_hotspot, frame=button_frame, min_contrast=0.010)
    assert refined is not None
    assert refined.contrast_score > 0.010
    assert len(refined.bbox) == 4


def test_visual_focus_detector_blinking_caret():
    """Verifies that a blinking text cursor correctly triggers is_focused=True."""
    detector = VisualFocusDetector()

    # Frame 1: Caret is ON (black line on white background)
    f1 = np.full((360, 640, 3), 255, dtype=np.uint8)
    # Caret at center (320, 180): width 2px, height 20px
    f1[170:190, 319:321] = 0

    # Frame 2: Caret is OFF (all white background)
    f2 = np.full((360, 640, 3), 255, dtype=np.uint8)

    target_point = (320.0 / 640.0, 180.0 / 360.0)

    # Verify focus comparing frame 2 against reference frame 1
    state = detector.verify_focus(f2, target_point, reference_frame=f1)

    assert state.is_focused is True
    assert state.caret_blinking is True
    assert state.confidence >= 0.70
    assert state.caret_coord is not None
    # Caret coordinate should be very close to (0.5, 0.5)
    assert abs(state.caret_coord[0] - 0.5) < 0.05
    assert abs(state.caret_coord[1] - 0.5) < 0.05


def test_visual_focus_detector_fail_closed_on_unfocused():
    """Verifies that an unfocused, static or blank element strictly fails closed (is_focused=False)."""
    detector = VisualFocusDetector()

    # Two identical frames with no cursor
    f1 = np.full((360, 640, 3), 245, dtype=np.uint8)
    f2 = np.full((360, 640, 3), 245, dtype=np.uint8)

    target_point = (0.5, 0.5)
    state = detector.verify_focus(f2, target_point, reference_frame=f1)

    assert state.is_focused is False
    assert state.caret_blinking is False
    assert state.caret_detected is False


def test_human_eye_verify_focus_integration():
    """Verifies that HumanEye exposes verify_focus smoothly."""
    eye = HumanEye(device="cpu")

    # Frame 1: Caret ON
    f1 = np.full((360, 640, 3), 255, dtype=np.uint8)
    f1[170:190, 319:321] = 0

    # Frame 2: Caret OFF
    f2 = np.full((360, 640, 3), 255, dtype=np.uint8)

    target = (0.5, 0.5)
    focus_state = eye.verify_focus(target_point=target, frame=f2, reference_frame=f1)

    assert isinstance(focus_state, VisualFocusState)
    assert focus_state.is_focused is True
    assert focus_state.caret_blinking is True

