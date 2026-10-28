"""The assistant speaks first: proactive awareness.

These tests drive the real loop and the real brain. A mock brain here would
assert that the gate calls a function, not that the assistant says something
useful, and the whole point of this feature is the wording and the restraint.
Where a stub is used it is a real ``AssistantTurn``-producing object, not a
``Mock``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from avcs.gui.awareness import AwarenessConfig, AwarenessLoop
from avcs.voice.brain import (
    PROACTIVE_SYSTEM_PROMPT,
    STREAM_SENTINEL,
    AssistantTurn,
    ConversationBrain,
    RuleBrain,
    TurnKind,
)


# --------------------------------------------------------------------------- #
# Test doubles that are real objects, not mocks
# --------------------------------------------------------------------------- #
@dataclass
class _State:
    """A GuiState-shaped object with only what the watcher reads."""

    description: str = "a mid-tone, detailed screen"
    reading: Dict[str, Any] = field(default_factory=dict)


class _ScriptedBrain:
    """A brain that returns a fixed reply, or stays silent, on demand."""

    def __init__(self, speech: str = "Sir, kuch dikha.", source: str = "llm") -> None:
        self.speech = speech
        self.source = source
        self.calls: List[Dict[str, Any]] = []

    def comment(self, observation, reason="", on_speech=None):
        self.calls.append({"observation": observation, "reason": reason})
        if not self.speech:
            return None
        turn = AssistantTurn(kind=TurnKind.REPLY, speech=self.speech, source=self.source)
        if on_speech is not None:
            on_speech(turn.speech)
        return turn


def _loop(brain, states, **cfg) -> AwarenessLoop:
    """A watcher over a scripted list of states, one per tick."""
    box = {"i": 0}

    def snapshot():
        i = min(box["i"], len(states) - 1)
        box["i"] += 1
        return states[i]

    spoken: List[str] = []
    return AwarenessLoop(
        snapshot=snapshot, brain=brain, speak=spoken.append,
        config=AwarenessConfig(poll_s=0.01, **cfg),
    ), spoken, box


# --------------------------------------------------------------------------- #
# The gate: when the watcher may interrupt
# --------------------------------------------------------------------------- #
def test_quiet_screen_produces_no_remark():
    brain = _ScriptedBrain()
    loop, spoken, _ = _loop(brain, [_State(reading={"change_fraction": 0.0})])
    assert loop.tick() is None
    assert spoken == []
    assert brain.calls == []


def test_overlay_is_worth_interrupting():
    brain = _ScriptedBrain("Sir, ek dialog box khul gaya hai.")
    state = _State(reading={"overlay_suspected": True, "change_fraction": 0.01})
    loop, spoken, _ = _loop(brain, [state])
    remark = loop.tick()
    assert remark is not None and remark.reason == "overlay"
    assert spoken == ["Sir, ek dialog box khul gaya hai."]
    assert brain.calls[0]["reason"] == "a dialog or overlay appeared"


def test_whole_screen_change_is_reported():
    brain = _ScriptedBrain("Sir, naya view.")
    state = _State(reading={"scene_changed": True, "change_fraction": 0.9})
    loop, spoken, _ = _loop(brain, [state])
    assert loop.tick().reason == "scene_change"
    assert spoken == ["Sir, naya view."]


def test_a_repaint_is_not_a_scene_change():
    # scene_changed true but the change fraction is tiny: a repaint, not a view
    # switch. A remark here would be the noisy assistant we do not want.
    brain = _ScriptedBrain()
    state = _State(reading={"scene_changed": True, "change_fraction": 0.05})
    loop, spoken, _ = _loop(brain, [state])
    assert loop.tick() is None
    assert spoken == []


def test_cooldown_suppresses_a_second_remark():
    brain = _ScriptedBrain("Sir.")
    state = _State(reading={"overlay_suspected": True, "change_fraction": 0.01})
    loop, spoken, _ = _loop(brain, [state, state], cooldown_s=999.0)
    assert loop.tick() is not None
    assert loop.tick() is None  # inside the cooldown
    assert len(spoken) == 1
    assert loop.skipped.get("cooldown") == 1


def test_same_reason_is_not_repeated_even_past_cooldown():
    brain = _ScriptedBrain("Sir.")
    state = _State(reading={"overlay_suspected": True, "change_fraction": 0.01})
    loop, spoken, _ = _loop(brain, [state, state], cooldown_s=0.0, repeat_s=999.0)
    assert loop.tick() is not None
    assert loop.tick() is None
    assert loop.skipped.get("repeat:overlay") == 1


def test_busy_suppresses_a_remark_but_is_not_a_permanent_silence():
    brain = _ScriptedBrain("Sir.")
    state = _State(reading={"overlay_suspected": True, "change_fraction": 0.01})
    box = {"i": 0, "busy": True}

    def snapshot():
        box["i"] += 1
        return state

    spoken: List[str] = []
    loop = AwarenessLoop(
        snapshot=snapshot, brain=brain, speak=spoken.append,
        is_busy=lambda: box["busy"], config=AwarenessConfig(poll_s=0.01),
    )
    assert loop.tick() is None
    assert loop.skipped.get("busy") == 1
    box["busy"] = False
    assert loop.tick() is not None
    assert spoken == ["Sir."]


def test_a_silent_brain_is_recorded_as_silence_not_a_remark():
    brain = _ScriptedBrain(speech="")
    state = _State(reading={"overlay_suspected": True, "change_fraction": 0.01})
    loop, spoken, _ = _loop(brain, [state])
    assert loop.tick() is None
    assert spoken == []
    assert loop.skipped.get("silent:overlay") == 1


def test_the_observation_tells_the_brain_it_has_no_hands():
    # The watcher must never tempt the model into announcing an action it cannot
    # take, so the capabilities it is shown are all false.
    brain = _ScriptedBrain()
    state = _State(description="a dialog is open",
                   reading={"overlay_suspected": True, "change_fraction": 0.01})
    loop, _, _ = _loop(brain, [state])
    loop.tick()
    caps = brain.calls[0]["observation"]["capabilities"]
    assert caps["can_click"] is False and caps["can_see_targets"] is False
    assert brain.calls[0]["observation"]["why_now"] == "a dialog or overlay appeared"


def test_the_watcher_has_no_path_to_an_actuation():
    # A structural guard, not a behavioural one: the module must not import a
    # hand. A watcher that could also click is the clicker this product is not.
    import avcs.gui.awareness as aw

    source = open(aw.__file__, encoding="utf-8").read()
    assert "VirtualHand" not in source
    assert "HandMotor" not in source
    assert "XdotoolHand" not in source
    assert "kernel" not in source.lower()


# --------------------------------------------------------------------------- #
# The brain: spontaneous remarks
# --------------------------------------------------------------------------- #
def test_comment_parses_a_json_reply():
    brain = ConversationBrain(provider=None)
    turn = brain._turn_from_comment(
        '{"speech": "Sir, dialog khul gaya.", "thought": "overlay", "worth_saying": true}'
    )
    assert turn is not None and turn.speech == "Sir, dialog khul gaya."


def test_comment_stays_silent_when_the_model_says_so():
    brain = ConversationBrain(provider=None)
    assert brain._turn_from_comment(
        '{"speech": "", "thought": "just a mouse move", "worth_saying": false}'
    ) is None


def test_comment_never_speaks_the_action_sentinel():
    brain = ConversationBrain(provider=None)
    turn = brain._turn_from_comment(
        f'{{"speech": "Sir, kuch dikha {STREAM_SENTINEL} {{}}", "worth_saying": true}}'
    )
    assert turn is not None
    assert STREAM_SENTINEL not in turn.speech
    assert turn.speech == "Sir, kuch dikha"


def test_offline_comment_speaks_for_an_overlay_and_is_otherwise_silent():
    # `_rule_comment` is the deterministic offline path. Tested directly so the
    # result does not depend on whether this box happens to hold an API key: a
    # test that passes only when the network is down is a flaky test.
    brain = ConversationBrain(provider=None)
    turn = brain._rule_comment(
        {"screen": "screen with overlay suspected", "has_modal_dialog": True}, "overlay"
    )
    assert turn is not None and turn.source == "rule_fallback"
    assert "dialog" in turn.speech.lower()
    assert brain._rule_comment({"screen": "a calm desktop, nothing happening"}, "") is None


def test_proactive_prompt_forbids_coordinates_and_actions():
    low = PROACTIVE_SYSTEM_PROMPT.lower()
    assert "never output screen coordinates" in low
    assert "cannot click or type" in low
    assert "stay silent" in low


# --------------------------------------------------------------------------- #
# Wiring: the GUI
# --------------------------------------------------------------------------- #
def test_gui_without_proactive_has_no_watcher_and_says_so():
    from avcs.gui.server import build_app

    app = build_app(source="demo")
    gui = app.state.gui
    assert gui.awareness is None
    assert gui.capabilities()["proactive"] is False
    assert gui.awareness_summary()["enabled"] is False


def test_gui_with_proactive_starts_a_real_watcher():
    from avcs.gui.server import build_app

    app = build_app(source="demo", proactive=True)
    gui = app.state.gui
    try:
        assert gui.awareness is not None and gui.awareness.running
        assert gui.capabilities()["proactive"] is True
        summary = gui.awareness_summary()
        assert summary["enabled"] is True
    finally:
        gui.stop()
    assert not gui.awareness.running


def test_awareness_endpoint_reports_enabled_and_recent():
    from fastapi.testclient import TestClient

    from avcs.gui.server import build_app

    app = build_app(source="demo", proactive=True)
    gui = app.state.gui
    try:
        client = TestClient(app)
        body = client.get("/api/awareness").json()
        assert body["enabled"] is True
        assert isinstance(body["recent"], list)
    finally:
        gui.stop()


def test_a_remark_is_recorded_in_the_transcript_as_proactive():
    from avcs.gui.server import build_app

    app = build_app(source="demo", proactive=True)
    gui = app.state.gui
    try:
        from avcs.gui.awareness import Remark

        gui._record_remark(Remark(at=time.time(), reason="overlay", text="Sir.", source="llm"))
        last = gui.transcript[-1]
        assert last["proactive"] is True
        assert last["mutating"] is False
        assert last["kind"] == "remark"
        assert last["gate_allowed"] is None
    finally:
        gui.stop()


def test_a_user_turn_marks_the_assistant_busy_while_it_runs():
    # The watcher must stay quiet during a conversation. `turn()` sets the flag
    # for the duration of the call, so a remark cannot talk over the reply.
    from avcs.gui.server import build_app

    app = build_app(source="demo")
    gui = app.state.gui
    seen: List[bool] = []
    real_step = gui.assistant.step

    def spy(text, **kw):
        seen.append(gui._busy())
        return real_step(text, **kw)

    gui.assistant.step = spy
    gui.turn("hello")
    assert seen == [True]
    assert gui._busy() is False
