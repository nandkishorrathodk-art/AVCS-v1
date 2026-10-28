"""Tests for the lexicon perception adapter (the control-grounding path).

This is the *opt-in* path that can read a control label from pixels and ground a
click; the assistant's default eyes are `ScenePerception` (see
`tests/test_assistant_gui.py`). These run the real code paths: a real
`LexiconReader` on rendered frames, the real `CapabilityKernel` and the real
`Assistant`. Nothing here is mocked, because the thing under test is precisely
whether the pieces agree with each other -- a mock would assert that they do.
"""
from __future__ import annotations

import pytest

from avcs.control.capability import CapabilityKernel
from avcs.gui.desktop import DesktopSource, render
from avcs.voice.assistant import Assistant
from avcs.voice.brain import ConversationBrain, RuleBrain
from avcs.voice.lexicon_perception import LexiconPerception


@pytest.fixture(scope="module")
def desktop():
    return DesktopSource(["save", "close", "undo", "redo"])


@pytest.fixture(scope="module")
def perception(desktop):
    return LexiconPerception(frame_source=desktop)


def _kernel(**kw):
    kw.setdefault("granted_ops", ["click"])
    kw.setdefault("min_peak_probability", 0.45)
    return CapabilityKernel(**kw)


# --------------------------------------------------------------------------- #
# desktop renderer
# --------------------------------------------------------------------------- #
def test_desktop_render_shape():
    frame = render(["save", "close"])
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[0] > 100 and frame.shape[1] > 300
    assert frame.dtype.name == "uint8"


def test_desktop_source_is_callable_and_capturable(desktop):
    a = desktop()
    b = desktop.capture()
    assert a.shape == b.shape
    desktop.set_buttons(["ok", "cancel"])
    assert desktop.buttons == ["ok", "cancel"]
    desktop.set_buttons(["save", "close", "undo", "redo"])


# --------------------------------------------------------------------------- #
# adapter: reading
# --------------------------------------------------------------------------- #
def test_adapter_reads_labels_from_pixels(perception):
    perception.grab()
    labels = {r.label for r in perception.regions()}
    # The reader must recover at least the distinctive words it was shown.
    assert {"save", "close"} <= labels


def test_adapter_observe_reports_presence(perception):
    present = perception.observe("save")
    assert present.target_present is True
    assert "save" in present.description
    assert present.hotspot_count == len(perception.regions())


def test_adapter_refuses_out_of_vocabulary_word(perception):
    perception.grab()
    # "print" is not in the lexicon's closed vocabulary, so there is no
    # prototype to match and the honest answer is "not found", not a nearest
    # label.
    assert perception.grounding_evidence("print") is None
    assert perception.observe("print").target_present is False


def test_adapter_refuses_absent_in_vocabulary_word(perception):
    perception.grab()
    # "upload" is a real vocabulary word that is not on this screen.
    assert perception.grounding_evidence("upload") is None


def test_adapter_point_is_inside_the_named_box(perception):
    perception.grab()
    ev = perception.grounding_evidence("save")
    assert ev is not None
    frame = perception.frame()
    h, w = frame.shape[:2]
    cx, cy = ev.candidate_point_norm
    box = next(r.box for r in perception.regions() if r.label == "save")
    assert box.x0 <= cx * w <= box.x1
    assert box.y0 <= cy * h <= box.y1


def test_adapter_caches_one_read_per_frame(perception, desktop):
    """Two queries in one turn must not re-run the encoder."""
    perception.grab()
    first = perception.regions()
    perception.grounding_evidence("save")
    perception.grounding_evidence("close")
    assert perception.regions() == first
    assert desktop.capture() is not None


def test_adapter_accepts_capture_style_source(desktop):
    class Src:
        def capture(self):
            return desktop.capture()

    perc = LexiconPerception(frame_source=Src())
    perc.grab()
    assert any(r.label == "save" for r in perc.regions())


def test_adapter_handles_no_frame():
    perc = LexiconPerception(frame_source=lambda: None)
    assert perc.grab() is None
    assert perc.observe("save").target_present is None
    assert perc.grounding_evidence("save") is None


# --------------------------------------------------------------------------- #
# kernel agreement: the fix that closes the click step
# --------------------------------------------------------------------------- #
def test_kernel_allows_a_present_control(perception):
    perception.grab()
    ev = perception.grounding_evidence("save")
    decision = _kernel().authorize(
        op="click", query="save", evidence=ev, frame=perception.frame()
    )
    assert decision.allowed is True
    assert decision.coordinate is not None


def test_kernel_denies_a_placeholder_with_no_point(perception):
    """The kernel's contract is a `GroundingEvidence`, never `None`.

    When perception has no signal the assistant builds a placeholder with
    `candidate_point_norm=None`; that is what the kernel must refuse, and the
    reason it gives is the one a user sees.
    """
    from avcs.control.capability import GroundingEvidence

    perception.grab()
    placeholder = GroundingEvidence(
        candidate_point_norm=None,
        peak_probability=0.0,
        target_present=False,
        authorized=False,
        active_query="print",
    )
    decision = _kernel().authorize(
        op="click", query="print", evidence=placeholder, frame=perception.frame()
    )
    assert decision.allowed is False
    assert "no candidate point" in decision.reason


def test_kernel_denies_without_grant(perception):
    perception.grab()
    ev = perception.grounding_evidence("save")
    decision = CapabilityKernel(granted_ops=()).authorize(
        op="click", query="save", evidence=ev, frame=perception.frame()
    )
    assert decision.allowed is False
    assert "grant" in decision.reason


# --------------------------------------------------------------------------- #
# full assistant loop
# --------------------------------------------------------------------------- #
def _assistant(perception):
    # Force the offline rule brain: these tests verify the *grounding and gate*
    # mechanics, which must be deterministic. A live LLM would make the target
    # (and therefore the pass/fail) depend on a network round-trip.
    return Assistant(
        brain=ConversationBrain(provider=_OfflineProvider()),
        perception=perception,
        kernel=_kernel(),
        hand=None,
        stream=False,
    )


class _OfflineProvider:
    """A provider that declares no key, so the brain uses the rule fallback."""

    api_key = ""



def test_assistant_hinglish_target_is_clean(perception):
    """'ab close karo' must yield 'close', not 'ab close'.

    The filler word would otherwise become part of the target and no control is
    named that, so the gate would deny a perfectly actionable request.
    """
    asst = _assistant(perception)
    assert RuleBrain._extract_target("ab close karo") == "close"
    assert RuleBrain._extract_target("mujhe save karna hai") == "save"
    assert RuleBrain._extract_target("please undo karo") == "undo"


def test_assistant_acts_on_present_control(perception):
    asst = _assistant(perception)
    rec = asst.step("mujhe save karna hai")
    assert rec.gate_allowed is True
    assert rec.action == "click"
    assert rec.target == "save"


def test_assistant_denies_absent_control(perception):
    asst = _assistant(perception)
    rec = asst.step("upload karo")
    assert rec.gate_allowed is False
    assert "no candidate point" in rec.gate_reason
