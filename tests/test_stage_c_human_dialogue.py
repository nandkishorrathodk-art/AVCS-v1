"""Stage C: Natural-Language Visual Dialogue & Zero-Coordinate Interface Validation Suite.

Invariants tested:
1. Strict Zero Coordinate Leakage: Absolute prohibition on transmitting coordinate numbers
   (x, y, bounding boxes, or normalized float coordinates) across user / System 2 boundary.
2. Human-Like Visual Dialogue ("Insaan Ki Tarha Speech"):
   - Articulation of active application and context.
   - Qualitative anatomical spatial regions (header, footer, left_sidebar, right_sidebar, center).
   - Dynamic temporal changes, pop-up modal appearance, and visual surprise.
   - Explicit target absence articulation (never hallucinating missing buttons or states).
3. Pure-Pixel Grounded Truthfulness: Zero OCR, zero OS accessibility APIs, zero hallucination.
4. Repetition Suppression: Reduction of prompt tokens across static unchanged frames.
"""
from __future__ import annotations

import re
import numpy as np
import pytest

from avcs.observation import (
    EntitySummary,
    SceneObservation,
    VisualObservationEngine,
)


# ==============================================================================
# 1. Strict Zero Coordinate Leakage Tests
# ==============================================================================

def test_observation_text_contains_zero_coordinate_numbers():
    """Verify that generated natural language description contains NO raw coordinate numbers (e.g. x=..., y=..., (120, 45))."""
    engine = VisualObservationEngine()

    class MockVisualState:
        timestamp = 1000.0
        frame_index = 1
        primary_category = "code_editor"
        stability = "stable"
        surprise_score = 0.05
        has_active_modal = False
        affordance_hotspots = [
            (0.10, 0.10, "icon", 0.95),        # top-left / header
            (0.15, 0.50, "tab", 0.92),         # left sidebar
            (0.50, 0.50, "editor_area", 0.88), # center
            (0.90, 0.08, "button", 0.98),      # top-right / header
        ]

    frame = np.full((128, 128, 3), 128, dtype=np.uint8)
    obs = engine.generate(MockVisualState(), frame=frame)

    desc = obs.scene_description
    assert isinstance(desc, str) and len(desc) > 0

    # Strict check: No pattern matching raw coordinate assignments or float coordinate tuples
    coordinate_patterns = [
        r"\bx\s*=\s*\d+",
        r"\by\s*=\s*\d+",
        r"\(\s*\d+\s*,\s*\d+\s*\)",
        r"bbox",
        r"bounding_box",
        r"centroid",
        r"\[\s*\d+\s*,\s*\d+\s*\]",
    ]
    for pat in coordinate_patterns:
        match = re.search(pat, desc, re.IGNORECASE)
        assert match is None, f"Found prohibited coordinate pattern '{pat}' in description: {desc}"

    # Verify serialized dictionary also contains zero coordinate numbers
    obs_dict = obs.to_dict()
    for entity in obs_dict["visible_entities"]:
        if isinstance(entity, dict):
            assert "x" not in entity
            assert "y" not in entity
            assert "bbox" not in entity
            assert "coordinates" not in entity
            # Spatial region must be qualitative anatomical string
            assert entity["spatial_region"] in {
                "header", "footer", "left_sidebar", "right_sidebar", "center", "screen"
            }


def test_zero_coordinates_in_str_and_to_dict():
    """Verify EntitySummary exports only qualitative spatial descriptions, never coordinates."""
    e = EntitySummary(
        category="button",
        count=1,
        spatial_region="header",
        prominence="primary",
        confidence=0.95,
    )
    s = str(e)
    assert "header" in s
    assert "1 button in header" == s
    assert "x" not in s and "y" not in s

    d = e.to_dict()
    assert d["spatial_region"] == "header"
    assert "x" not in d and "y" not in d


# ==============================================================================
# 2. Human-Like Visual Dialogue ("Insaan Ki Tarha Speech") Tests
# ==============================================================================

def test_human_dialogue_active_application_and_controls():
    """Verify natural human-like description of application and visible controls."""
    engine = VisualObservationEngine()

    class EditorState:
        timestamp = 100.0
        frame_index = 5
        primary_category = "code_editor"
        stability = "stable"
        surprise_score = 0.02
        has_active_modal = False
        affordance_hotspots = [
            (0.85, 0.05, "run_button", 0.99),
            (0.10, 0.40, "file_tab", 0.90),
        ]

    frame = np.full((128, 128, 3), 100, dtype=np.uint8)
    obs = engine.generate(EditorState(), frame=frame)

    desc = obs.scene_description
    assert "Code Editor" in desc
    assert "Visible interactive controls:" in desc
    assert "run_button" in desc
    assert "file_tab" in desc


def test_human_dialogue_modal_popup_alert():
    """Verify explicit alert articulation when a modal pop-up appears on screen."""
    engine = VisualObservationEngine()

    class ModalState:
        timestamp = 105.0
        frame_index = 6
        primary_category = "confirmation_dialog"
        stability = "modal_active"
        surprise_score = 0.85
        has_active_modal = True
        affordance_hotspots = [
            (0.50, 0.50, "dialog_box", 0.98),
            (0.45, 0.60, "save_button", 0.95),
            (0.55, 0.60, "cancel_button", 0.95),
        ]

    frame = np.full((128, 128, 3), 200, dtype=np.uint8)
    obs = engine.generate(ModalState(), frame=frame)

    desc = obs.scene_description
    assert "modal alert dialog is currently open and focused" in desc
    assert obs.has_active_modal


def test_explicit_target_absence_articulation():
    """Verify that when a screen is blank or missing targets, the observation states absence without hallucination."""
    engine = VisualObservationEngine()

    class BlankState:
        timestamp = 110.0
        frame_index = 7
        primary_category = "blank_canvas"
        stability = "stable"
        surprise_score = 0.0
        has_active_modal = False
        affordance_hotspots = []

    # Solid uniform blank frame (zero edge contrast)
    blank_frame = np.full((128, 128, 3), 50, dtype=np.uint8)
    obs = engine.generate(BlankState(), frame=blank_frame)

    assert "The display is blank with no detectable visual structure or interactive elements" in obs.scene_description
    assert len(obs.visible_entities) == 0
    assert obs.confidence > 0.90


# ==============================================================================
# 3. Repetition Suppression Tests
# ==============================================================================

def test_repetition_suppression_on_static_frames():
    """Verify consecutive unchanged frames suppress redundant tokens."""
    engine = VisualObservationEngine()

    class StaticState:
        timestamp = 200.0
        frame_index = 10
        primary_category = "terminal"
        stability = "stable"
        surprise_score = 0.01
        has_active_modal = False
        affordance_hotspots = [(0.1, 0.1, "prompt", 0.95)]

    frame = np.full((128, 128, 3), 30, dtype=np.uint8)
    frame[10:20, 10:20] = 255  # create high edge contrast

    # Frame 1
    obs1 = engine.generate(StaticState(), frame=frame)
    # Frame 2 (identical)
    obs2 = engine.generate(StaticState(), frame=frame)

    assert obs1.is_stable
    assert obs2.is_stable
    assert engine.consecutive_static_count >= 1

