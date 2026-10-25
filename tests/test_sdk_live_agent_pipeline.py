"""Regression tests for the avcs_sdk live-agent pipeline.

These cover defects found by running the real entrypoint
(``python -m avcs_sdk.run_live_agent``) rather than importing helpers in
isolation: the crash only appears when the foveal elements produced by
``FoveaReader.inspect`` reach the agent's scoring and narration code.
"""
from __future__ import annotations

import numpy as np
import pytest

from avcs_sdk.fovea_reader import FovealElement, FoveaReader
from avcs_sdk.run_live_agent import resolve_semantic_target


def _element(**overrides) -> FovealElement:
    base = dict(
        text="UI Element",
        category="interactive_control",
        center_norm=(0.5, 0.5),
        bbox_pixels=(0, 0, 8, 8),
        confidence=0.9,
    )
    base.update(overrides)
    return FovealElement(**base)


def test_foveal_element_exposes_every_attribute_the_agent_reads():
    """The agent reads patch/contrast/edge_density/describe; all must exist.

    A regression here means a phantom attribute reappeared: the original code
    read ``e.patch``, ``elem.glyph_likeness``, ``elem.edge_density``,
    ``elem.contrast`` and ``elem.describe()``, none of which existed, so the
    live agent crashed on turn 1.
    """
    elem = _element()
    assert isinstance(elem.patch, (np.ndarray, type(None)))
    assert isinstance(elem.contrast, float)
    assert isinstance(elem.edge_density, float)
    assert isinstance(elem.describe(), str)


def test_element_with_array_patch_stays_hashable_and_comparable():
    """Adding the crop must not break the frozen dataclass contract."""
    a = _element(patch=np.zeros((4, 4, 3), dtype=np.uint8))
    b = _element(patch=np.ones((4, 4, 3), dtype=np.uint8))
    assert hash(a) == hash(b)
    assert a == b  # patch is excluded from equality on purpose
    assert len({a, b}) == 1


def test_describe_carries_strippable_coordinate_suffix():
    """The narrator rewrites 'at (x, y)'; the System 2 prompt strips it."""
    desc = _element(text="", category="toggle_icon", center_norm=(0.25, 0.75)).describe()
    assert "at (" in desc
    assert "toggle_icon" in desc  # falls back to category when no text was read


@pytest.mark.parametrize(
    "query",
    [
        "click the button",
        "find the text line",
        "open the dialog",
        "top right control",
        "center icon",
        None,
    ],
)
def test_resolve_semantic_target_handles_every_query_branch(query):
    """Every scoring branch used to raise AttributeError; each must resolve."""
    chosen = resolve_semantic_target(query, [_element(), _element(center_norm=(0.9, 0.9))])
    assert chosen is not None


def test_resolve_semantic_target_returns_none_without_elements():
    assert resolve_semantic_target("click the button", []) is None


def test_inspect_populates_texture_metrics_and_patch():
    """inspect() must compute the metrics it exposes, from the real crop."""
    reader = FoveaReader()
    # A high-contrast checkerboard gives non-zero gradient and std.
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[::2, ::2] = 255
    frame[1::2, 1::2] = 255
    elements = reader.inspect(frame, [(0.5, 0.5, "interactive_control", 0.8)])
    assert len(elements) == 1
    elem = elements[0]
    assert elem.patch is not None and elem.patch.size > 0
    assert elem.contrast > 0.0
    assert elem.edge_density > 0.0


def test_inspect_returns_empty_on_missing_frame_or_hotspots():
    reader = FoveaReader()
    assert reader.inspect(None, [(0.5, 0.5, "x", 1.0)]) == []
    assert reader.inspect(np.zeros((8, 8, 3), dtype=np.uint8), []) == []


def test_scoring_branches_reference_the_real_emitted_vocabulary():
    """Every category the scorer branches on must be one a producer emits.

    The original code tested ``elem.category == "text_like"`` and
    ``elem.category in ("graphic", "interactive_control")``. The eye stream
    emits ``text_input``/``toggle_icon``/``modal_dialog``/``content_region``
    and never ``text_like``/``graphic``, so those branches were unreachable.
    """
    from avcs_sdk import run_live_agent
    from avcs_sdk.eye_stream import UI_CATEGORIES

    assert run_live_agent.TEXT_CATEGORIES <= UI_CATEGORIES
    assert run_live_agent.INTERACTIVE_CATEGORIES <= UI_CATEGORIES
    assert "text_like" not in UI_CATEGORIES
    assert "graphic" not in UI_CATEGORIES


def test_narrator_maps_every_emitted_category_to_a_generic_term():
    """A real category must not leak as raw snake_case into System 2 text."""
    from avcs_sdk.eye_stream import UI_CATEGORIES
    from avcs_sdk.narrator import LiveNarrator

    narrator = LiveNarrator()
    for category in UI_CATEGORIES:
        rendered = narrator._clean_target_name(category)
        assert rendered != category, f"{category} leaked unmapped into commentary"


def test_narrator_emits_visible_controls_for_real_elements():
    """The narrator used to drop every element because describe() was missing."""
    from avcs_sdk.eye_stream import EyeStream
    from avcs_sdk.narrator import LiveNarrator

    eye = EyeStream(source="synthetic")
    try:
        frame = eye.step()
        reader = FoveaReader()
        elements = reader.inspect(frame.raw_frame, frame.hotspots, max_inspect=4)
        if not elements:
            pytest.skip("synthetic frame produced no hotspots")
        commentary = LiveNarrator().narrate(
            eye_frame=frame, active_window="Test Window", foveal_elements=elements
        )
        assert commentary is not None
        assert commentary.visible_elements, "element descriptions were silently dropped"
    finally:
        eye.close()


def test_llm_bridge_does_not_send_screen_images_by_default():
    """System 1 -> System 2 boundary: no frames/crops unless explicitly opted in.

    docs/AVCS_ENGINEERING_RULES.md: "Do not send screen images to System 2.
    This includes screenshots, crops, video frames, image encodings, or visual
    embeddings." The bridge used to default to vision_enabled=True, so a caller
    that only passed a goal silently shipped foveal crops.
    """
    from avcs_sdk.llm_bridge import LLMBridge, encode_patch_to_base64

    bridge = LLMBridge(api_key="test-key")
    assert bridge.vision_enabled is False

    patch = np.zeros((8, 8, 3), dtype=np.uint8)
    # Simulate the payload builder without a network call.
    if bridge.vision_enabled and [patch]:
        raise AssertionError("default bridge would attach image_url content")
    assert isinstance(encode_patch_to_base64(patch), str)  # helper still works


def test_run_agent_survives_a_full_turn_without_weights():
    """The end-to-end path that crashed must complete on synthetic input."""
    from avcs_sdk.run_live_agent import run_agent

    run_agent("click the settings button", max_turns=1, dry_run=True)
