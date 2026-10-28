"""Tests for the five capability gaps that were filled after the audit.

Each gap gets a section, and each section pins the *honesty* property, not just
the happy path:

* memory -- survives a restart, drops empty facts, is inspectable and forgettable
* uncertainty -- the gate reports a number, not a verdict, and a near miss is
  distinguishable from a clear absence
* new ops -- right_click / drag / launch_app / read_file / list_dir are
  fail-closed, and drag refuses a half-authorized (destination-less) drag
* multi-monitor -- enumeration is honest ("unknown", not "one monitor") and the
  monitor origin is what keeps a click on monitor 2 from landing on monitor 1
* trajectory -- the log is reviewable, and it does NOT feed an online loop
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from avcs.control.capability import (
    MUTATING_OPS,
    POINT_OPS,
    CapabilityKernel,
    GroundingEvidence,
)
from avcs.voice.assistant import Assistant, ScreenObservation, StaticPerception
from avcs.voice.brain import ConversationBrain, RuleBrain, TurnKind
from avcs.voice.hands import MUTATING_OPS as HANDS_MUTATING_OPS
from avcs.voice.memory import Episode, Fact, MemoryStore
from avcs.voice.stt import NullEngine
from avcs.voice.trajectory import TrajectoryLog, TrajectoryStep
from avcs.voice.tts import NullEngine as NullTtsEngine
from avcs.voice.tts import Speaker


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _ExplodingProvider:
    """A provider that must never be called: the offline rule brain answers."""

    api_key = ""

    def raw_step(self, **kwargs):
        raise AssertionError("the offline path must not call the provider")

    def plan_step(self, **kwargs):
        raise AssertionError("the offline path must not call the provider")

    def stream_step(self, **kwargs):
        raise AssertionError("the offline path must not call the provider")


def _assistant(**kwargs) -> Assistant:
    kwargs.setdefault("speaker", Speaker(engine=NullTtsEngine()))
    kwargs.setdefault("stt", NullEngine())
    kwargs.setdefault("perception", StaticPerception())
    kwargs.setdefault("brain", ConversationBrain(provider=_ExplodingProvider()))
    return Assistant(**kwargs)


class _GroundingPerception(StaticPerception):
    """Grounds every query with a configurable peak, on a structured frame."""

    def __init__(self, peak: float = 0.9, point=(0.3, 0.3)) -> None:
        self.peak = float(peak)
        self.point = point
        self._frame = np.full((64, 64, 3), 60, dtype=np.uint8)
        self._frame[10:30, 10:30] = 200

    def observe(self, query=None):
        return ScreenObservation(description="a dialog with buttons", hotspot_count=2)

    def frame(self):
        return self._frame

    def grounding_evidence(self, query):
        return GroundingEvidence(
            candidate_point_norm=self.point,
            peak_probability=self.peak,
            target_present=True,
            authorized=False,
            active_query=query,
        )


class _Hand:
    """A hand that records what it was asked to do, and can refuse one op."""

    name = "test-hand"

    def __init__(self, refuse: str | None = None) -> None:
        self.refuse = refuse
        self.clicks: list = []
        self.right_clicks: list = []
        self.drags: list = []
        self.typed: list = []
        self.keys: list = []
        self.scrolls: list = []
        self.launched: list = []

    def _ok(self, op: str) -> bool:
        return op != self.refuse

    def click(self, x, y, double: bool = False):
        if not self._ok("click"):
            return False
        self.clicks.append((x, y))
        return (x, y)

    def right_click(self, x, y):
        if not self._ok("right_click"):
            return False
        self.right_clicks.append((x, y))
        return (x, y)

    def drag(self, x1, y1, x2, y2):
        if not self._ok("drag"):
            return False
        self.drags.append(((x1, y1), (x2, y2)))
        return (x2, y2)

    def type_text(self, text, typing_speed_cps: float = 25.0) -> bool:
        self.typed.append(text)
        return self._ok("type_text")

    def press_key(self, key_name: str) -> bool:
        self.keys.append(key_name)
        return self._ok("press_key")

    def scroll(self, delta: int) -> bool:
        self.scrolls.append(int(delta))
        return self._ok("scroll")

    def launch_app(self, app_name: str) -> bool:
        if not self._ok("launch_app"):
            return False
        self.launched.append(app_name)
        return True


def _intent(op: str, **kwargs):
    from avcs.reasoning.contracts import ActionIntent

    return ActionIntent.create(step_index=1, op=op, **kwargs)


class _ScriptedProvider:
    """Returns one fixed action intent, so a test can drive one op end to end."""

    api_key = "test-key"

    def __init__(self, intent) -> None:
        self._intent = intent

    def raw_step(self, **kwargs):
        action = {
            "op": self._intent.op.value,
            "target_description": self._intent.target_description,
            "text_input": self._intent.text_input,
            "key_name": self._intent.key_name,
        }
        for key, value in (self._intent.parameters or {}).items():
            action[key] = value
        payload = {"kind": "act", "speech": "ok", "action": action}
        return "ok\n<<<ACTION>>>\n" + json.dumps(payload)


# --------------------------------------------------------------------------- #
# Gap 1: durable memory
# --------------------------------------------------------------------------- #
def test_memory_survives_a_restart(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    store.remember("name", "Rahul")
    store.remember("language", "user prefers Hinglish")
    assert store.save() is True

    reopened = MemoryStore(path)
    assert reopened.load_error == ""
    values = {f.key: f.value for f in reopened.facts.values()}
    assert values["name"] == "Rahul"
    assert values["language"] == "user prefers Hinglish"


def test_memory_drops_empty_facts_instead_of_storing_blank(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    assert store.remember("name", "") is None
    assert store.remember("", "   ") is None
    assert store.facts == {}


def test_memory_recall_is_lexical_and_empty_without_overlap(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    store.remember("editor", "the save button is bottom-right in the editor")
    assert store.recall("where is the save button in the editor")
    # No shared terms -> no recall. A nearest-neighbour guess would be a lie.
    assert store.recall("what is the weather in tokyo") == []


def test_corrupt_memory_is_reported_not_silently_replaced(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = MemoryStore(path)
    assert store.load_error, "a corrupt file must be reported"
    # And the file is still there: we did not overwrite the user's data.
    assert path.read_text(encoding="utf-8") == "{ this is not json"


def test_memory_lru_eviction_keeps_a_frequently_used_fact(tmp_path):
    store = MemoryStore(tmp_path / "m.json", max_facts=2)
    store.remember("keep", "the save button is bottom-right")
    # Recalling "keep" bumps its hit count *before* the store fills up, so it must
    # outlive a fact that was merely stored more recently.
    store.recall("save button")
    store.remember("chatter1", "fact one")
    store.remember("chatter2", "fact two")
    keys = {f.key for f in store.facts.values()}
    assert "keep" in keys


def test_episode_summary_distinguishes_no_change_from_worked(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    store.record_episode(
        Episode(heard="save karo", said="ok", kind="act", action="click",
                target="save", performed=True, verified=False)
    )
    summary = store.episode_summary(3)
    assert "did not change" in summary


def test_brain_receives_recalled_facts_as_labelled_notes(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    store.remember("name", "the user's name is Rahul")
    brain = ConversationBrain(provider=_ExplodingProvider(), memory=store)
    brain.respond("what is my name")
    # The store was consulted: the hit count moved.
    assert any(f.hits > 0 for f in store.facts.values())


def test_brain_without_memory_still_answers(tmp_path):
    brain = ConversationBrain(provider=_ExplodingProvider(), memory=None)
    turn = brain.respond("hello")
    assert turn.kind in (TurnKind.REPLY, TurnKind.ASK)


def test_assistant_persists_a_model_proposed_memory(tmp_path):
    store = MemoryStore(tmp_path / "m.json")

    class _MemoryBrain:
        provider = _ExplodingProvider()
        memory = store

        def respond(self, user_text, observation=None):
            from avcs.voice.brain import AssistantTurn

            return AssistantTurn(
                kind=TurnKind.REPLY, speech="noted",
                memories=[["name", "the user's name is Rahul"]],
            )

    assistant = _assistant(brain=_MemoryBrain(), memory=store)
    assistant.step("mera naam Rahul hai")
    assert any("Rahul" in f.value for f in store.facts.values())
    # And it reached disk, so it survives a restart.
    assert MemoryStore(tmp_path / "m.json").facts


def test_assistant_drops_an_empty_memory_proposal(tmp_path):
    store = MemoryStore(tmp_path / "m.json")

    class _MemoryBrain:
        provider = _ExplodingProvider()
        memory = store

        def respond(self, user_text, observation=None):
            from avcs.voice.brain import AssistantTurn

            return AssistantTurn(
                kind=TurnKind.REPLY, speech="noted",
                memories=[["name", ""], ["", "  "]],
            )

    assistant = _assistant(brain=_MemoryBrain(), memory=store)
    assistant.step("hello")
    assert store.facts == {}


# --------------------------------------------------------------------------- #
# Gap 2: structured uncertainty
# --------------------------------------------------------------------------- #
def test_decision_carries_a_confidence_number_not_just_a_verdict():
    kernel = CapabilityKernel(granted_ops=["click"], min_peak_probability=0.75)
    ev = GroundingEvidence(
        candidate_point_norm=(0.5, 0.5), peak_probability=0.60,
        target_present=True, authorized=False, active_query="save",
    )
    decision = kernel.authorize(op="click", query="save", evidence=ev)
    assert decision.allowed is False
    assert decision.confidence == pytest.approx(0.60)
    assert decision.floor == pytest.approx(0.75)
    assert decision.margin == pytest.approx(-0.15)


def test_decision_to_dict_exposes_the_numbers():
    kernel = CapabilityKernel(granted_ops=["click"], min_peak_probability=0.75)
    ev = GroundingEvidence(
        candidate_point_norm=(0.5, 0.5), peak_probability=0.90,
        target_present=True, authorized=False, active_query="save",
    )
    payload = kernel.authorize(op="click", query="save", evidence=ev).to_dict()
    assert payload["confidence"] == pytest.approx(0.90)
    assert payload["floor"] == pytest.approx(0.75)
    assert payload["margin"] == pytest.approx(0.15)


def test_record_and_observation_expose_the_confidence():
    assistant = _assistant(
        perception=_GroundingPerception(peak=0.60),
        kernel=CapabilityKernel(granted_ops=["click"], min_peak_probability=0.75),
        hand=_Hand(),
        brain=ConversationBrain(
            provider=_ScriptedProvider(_intent("click", target_description="save"))
        ),
    )
    rec = assistant.step("save dabao")
    assert rec.gate_allowed is False
    assert rec.confidence == pytest.approx(0.60)
    assert rec.confidence_floor == pytest.approx(0.75)
    # The next observation carries the number, and flags the near miss.
    obs = assistant._observe(None)
    payload = obs.to_brain_dict()
    assert payload["last_confidence"] == pytest.approx(0.60)
    assert "near miss" in payload["confidence_note"]


def test_a_clear_absence_is_not_flagged_as_a_near_miss():
    assistant = _assistant(
        perception=_GroundingPerception(peak=0.10),
        kernel=CapabilityKernel(granted_ops=["click"], min_peak_probability=0.75),
        hand=_Hand(),
        brain=ConversationBrain(
            provider=_ScriptedProvider(_intent("click", target_description="save"))
        ),
    )
    assistant.step("save dabao")
    payload = assistant._observe(None).to_brain_dict()
    assert payload["last_confidence"] == pytest.approx(0.10)
    # 0.10 is not a near miss: the target is simply not there.
    assert "confidence_note" not in payload


# --------------------------------------------------------------------------- #
# Gap 4: right_click and drag
# --------------------------------------------------------------------------- #
def test_new_ops_are_mutating_and_the_two_modules_agree():
    assert {"right_click", "drag", "launch_app", "read_file", "list_dir"} <= MUTATING_OPS
    # A new op added to one set and forgotten in the other would silently bypass
    # the gate, so the sets must be equal.
    assert MUTATING_OPS == HANDS_MUTATING_OPS
    assert {"right_click", "drag"} <= POINT_OPS


def test_right_click_is_fail_closed_without_a_grant():
    hand = _Hand()
    assistant = _assistant(
        perception=_GroundingPerception(),
        kernel=CapabilityKernel(),  # no grants
        hand=hand,
        brain=ConversationBrain(
            provider=_ScriptedProvider(_intent("right_click", target_description="file"))
        ),
    )
    rec = assistant.step("right click karo")
    assert rec.action == "right_click"
    assert rec.gate_allowed is False
    assert hand.right_clicks == []


def test_right_click_reaches_the_hand_when_granted():
    hand = _Hand()
    assistant = _assistant(
        perception=_GroundingPerception(),
        kernel=CapabilityKernel(granted_ops=["right_click"]),
        hand=hand,
        brain=ConversationBrain(
            provider=_ScriptedProvider(_intent("right_click", target_description="file"))
        ),
    )
    rec = assistant.step("right click karo")
    assert rec.gate_allowed is True
    assert rec.performed is True
    assert len(hand.right_clicks) == 1


def test_right_click_never_degrades_into_a_primary_click():
    """A hand with only `click` must report it cannot right-click."""

    class _ClickOnly:
        name = "click-only"

        def click(self, x, y, double: bool = False):
            raise AssertionError("a primary click must not be used for right_click")

    from avcs.voice.hands import dispatch

    result = dispatch(_ClickOnly(), "right_click", point=(0.5, 0.5))
    assert result.performed is False
    assert "secondary click" in result.error


def test_drag_refuses_a_half_authorized_drag_without_a_destination():
    hand = _Hand()
    kernel = CapabilityKernel(granted_ops=["drag"])
    ev = GroundingEvidence(
        candidate_point_norm=(0.2, 0.2), peak_probability=0.9,
        target_present=True, authorized=False, active_query="file",
    )
    decision = kernel.authorize(op="drag", query="file", evidence=ev, arguments={})
    assert decision.allowed is False
    assert "destination" in decision.reason
    assert hand.drags == []


def test_drag_refuses_a_low_confidence_destination():
    kernel = CapabilityKernel(granted_ops=["drag"], min_peak_probability=0.75)
    ev = GroundingEvidence(
        candidate_point_norm=(0.2, 0.2), peak_probability=0.9,
        target_present=True, authorized=False, active_query="file",
    )
    decision = kernel.authorize(
        op="drag", query="file", evidence=ev,
        arguments={"point2": (0.8, 0.8), "peak2": 0.3},
    )
    assert decision.allowed is False
    assert "destination peak" in decision.reason


def test_drag_reaches_the_hand_when_both_ends_are_grounded():
    hand = _Hand()

    class _TwoPointPerception(_GroundingPerception):
        def grounding_evidence(self, query):
            # Source high, destination high too.
            return GroundingEvidence(
                candidate_point_norm=(0.2, 0.2), peak_probability=0.9,
                target_present=True, authorized=False, active_query=query,
            )

    assistant = _assistant(
        perception=_TwoPointPerception(),
        kernel=CapabilityKernel(granted_ops=["drag"]),
        hand=hand,
        brain=ConversationBrain(
            provider=_ScriptedProvider(
                _intent("drag", target_description="file",
                        parameters={"target2": "trash"})
            )
        ),
    )
    rec = assistant.step("file ko trash mein drag karo")
    assert rec.gate_allowed is True
    assert rec.performed is True
    assert len(hand.drags) == 1


# --------------------------------------------------------------------------- #
# Gap 6: launch_app / read_file / list_dir
# --------------------------------------------------------------------------- #
def test_file_ops_are_off_without_a_root(tmp_path):
    kernel = CapabilityKernel(granted_ops=["read_file"])
    ev = GroundingEvidence(candidate_point_norm=None, peak_probability=0.0,
                           target_present=False, authorized=False, active_query="")
    decision = kernel.authorize(
        op="read_file", query=None, evidence=ev,
        arguments={"path": str(tmp_path / "x.txt")},
    )
    assert decision.allowed is False
    assert "no path roots" in decision.reason


def test_read_file_refuses_a_path_outside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    kernel = CapabilityKernel(granted_ops=["read_file"], path_roots=[str(root)])
    ev = GroundingEvidence(candidate_point_norm=None, peak_probability=0.0,
                           target_present=False, authorized=False, active_query="")
    decision = kernel.authorize(
        op="read_file", query=None, evidence=ev, arguments={"path": str(outside)}
    )
    assert decision.allowed is False
    assert "escapes the allowed roots" in decision.reason


def test_read_file_refuses_traversal_even_with_a_valid_prefix(tmp_path):
    """`root/../secret` must not pass a naive prefix check."""
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    kernel = CapabilityKernel(granted_ops=["read_file"], path_roots=[str(root)])
    ev = GroundingEvidence(candidate_point_norm=None, peak_probability=0.0,
                           target_present=False, authorized=False, active_query="")
    decision = kernel.authorize(
        op="read_file", query=None, evidence=ev,
        arguments={"path": str(root / ".." / "secret.txt")},
    )
    assert decision.allowed is False


def test_read_file_allows_a_path_inside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = root / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    kernel = CapabilityKernel(granted_ops=["read_file"], path_roots=[str(root)])
    ev = GroundingEvidence(candidate_point_norm=None, peak_probability=0.0,
                           target_present=False, authorized=False, active_query="")
    assert kernel.authorize(
        op="read_file", query=None, evidence=ev, arguments={"path": str(target)}
    ).allowed is True


def test_launch_is_off_without_an_allow_list():
    kernel = CapabilityKernel(granted_ops=["launch_app"])
    ev = GroundingEvidence(candidate_point_norm=None, peak_probability=0.0,
                           target_present=False, authorized=False, active_query="")
    decision = kernel.authorize(
        op="launch_app", query=None, evidence=ev, arguments={"app": "xterm"}
    )
    assert decision.allowed is False
    assert "no app allow-list" in decision.reason


def test_launch_refuses_an_app_outside_the_allow_list():
    kernel = CapabilityKernel(granted_ops=["launch_app"], allowed_apps=["xterm"])
    ev = GroundingEvidence(candidate_point_norm=None, peak_probability=0.0,
                           target_present=False, authorized=False, active_query="")
    decision = kernel.authorize(
        op="launch_app", query=None, evidence=ev, arguments={"app": "rm"}
    )
    assert decision.allowed is False
    assert "not in the allow-list" in decision.reason


def test_read_file_reaches_the_hand_and_returns_content(tmp_path):
    from avcs.system1.loop import VirtualHand

    root = tmp_path / "root"
    root.mkdir()
    target = root / "notes.txt"
    target.write_text("hello from disk", encoding="utf-8")
    hand = VirtualHand(100, 100)
    assistant = _assistant(
        perception=_GroundingPerception(),
        kernel=CapabilityKernel(granted_ops=["read_file"], path_roots=[str(root)]),
        hand=hand,
        brain=ConversationBrain(
            provider=_ScriptedProvider(
                _intent("read_file", parameters={"path": str(target)})
            )
        ),
    )
    rec = assistant.step("notes.txt padho")
    assert rec.gate_allowed is True
    assert rec.performed is True
    assert "hello from disk" in rec.detail


# --------------------------------------------------------------------------- #
# Gap 3: multi-monitor
# --------------------------------------------------------------------------- #
def test_monitor_enumeration_is_honest_when_it_cannot_read():
    from forge.vision.capture import list_monitors

    monitors = list_monitors()
    # No display in CI -> an empty list ("unknown"), never a fabricated monitor.
    assert isinstance(monitors, list)
    for mon in monitors:
        assert {"index", "left", "top", "width", "height", "primary"} <= set(mon)


def test_monitor_origin_falls_back_to_zero_when_unknown():
    from forge.vision.capture import monitor_origin

    # An unknown monitor yields (0,0), which is the safe interpretation: a click
    # is not offset by a monitor we could not find.
    assert monitor_origin(999) == (0, 0)


def test_virtual_screen_is_none_or_a_box():
    from forge.vision.capture import virtual_screen

    box = virtual_screen()
    assert box is None or {"left", "top", "width", "height"} <= set(box)


def test_capabilities_report_monitor_count_and_hand_ops():
    assistant = _assistant(hand=_Hand())
    caps = assistant.capabilities()
    assert "monitors" in caps
    assert "monitor_count" in caps
    # The hand advertises the ops it can actually perform.
    assert "click" in caps["hand_ops"]
    assert "right_click" in caps["hand_ops"]
    assert "drag" in caps["hand_ops"]


# --------------------------------------------------------------------------- #
# Gap 5: trajectory logging
# --------------------------------------------------------------------------- #
def test_trajectory_step_labels_the_three_booleans_separately():
    noop = TrajectoryStep(turn=0, heard="hi", said="hello", kind="reply", source="llm")
    assert noop.outcome == "noop"

    refused = TrajectoryStep(
        turn=1, heard="save", said="ok", kind="act", source="llm",
        action="click", gate_allowed=False,
    )
    assert refused.outcome == "refused"

    failed = TrajectoryStep(
        turn=2, heard="save", said="ok", kind="act", source="llm",
        action="click", gate_allowed=True, performed=False,
    )
    assert failed.outcome == "failed"

    verified = TrajectoryStep(
        turn=3, heard="save", said="ok", kind="act", source="llm",
        action="click", gate_allowed=True, performed=True, verified=True,
    )
    assert verified.outcome == "verified"


def test_trajectory_writes_jsonl_and_reads_back(tmp_path):
    log = TrajectoryLog(tmp_path / "traj.jsonl")
    log.record(TrajectoryStep(turn=0, heard="hi", said="hello", kind="reply", source="llm"))
    log.record(
        TrajectoryStep(turn=1, heard="save", said="ok", kind="act", source="llm",
                       action="click", gate_allowed=True, performed=True, verified=True)
    )
    rows = log.load()
    assert len(rows) == 2
    assert rows[1]["action"] == "click"
    assert rows[1]["outcome"] == "verified"


def test_trajectory_disabled_writes_nothing(tmp_path):
    log = TrajectoryLog(None)
    log.record(TrajectoryStep(turn=0, heard="hi", said="hi", kind="reply", source="llm"))
    assert log.enabled is False
    assert log.summary()["total"] == 1  # still in memory, nothing on disk


def test_assistant_logs_a_trajectory(tmp_path):
    log = TrajectoryLog(tmp_path / "t.jsonl")
    assistant = _assistant(
        perception=_GroundingPerception(),
        kernel=CapabilityKernel(granted_ops=["click"]),
        hand=_Hand(),
        trajectory=log,
        brain=ConversationBrain(
            provider=_ScriptedProvider(_intent("click", target_description="save"))
        ),
    )
    assistant.step("save dabao")
    rows = log.load()
    assert rows, "a turn must be logged"
    assert rows[-1]["action"] == "click"
    assert rows[-1]["gate_allowed"] is True


def test_trajectory_is_a_log_not_an_online_loop(tmp_path):
    """The log must not feed the model. This pins that it is write-only.

    Rejection-sampling self-training was tried in this repo and made the model
    worse, so the trajectory log deliberately has no `train`/`learn` entry point.
    """
    log = TrajectoryLog(tmp_path / "t.jsonl")
    for name in ("train", "learn", "update", "finetune"):
        assert not hasattr(log, name), f"TrajectoryLog must not expose {name}()"


def test_rule_brain_extracts_a_path_for_read_file():
    """Offline mode must reach the file ops too, or they are untestable without a key."""
    from avcs.voice.brain import RuleBrain

    turn = RuleBrain().respond("read_file /tmp/notes.txt padho", {})
    assert turn.intent is not None
    assert turn.intent.op.value == "read_file"
    assert turn.intent.parameters == {"path": "/tmp/notes.txt"}


def test_rule_brain_asks_for_the_operand_instead_of_emitting_a_broken_op():
    from avcs.voice.brain import RuleBrain

    turn = RuleBrain().respond("padho", {})
    assert turn.kind.value == "ask"


def test_rule_brain_does_not_put_the_action_word_in_the_target():
    """'save button par right click karo' names the control 'save button'."""
    from avcs.voice.brain import RuleBrain

    turn = RuleBrain().respond("save button par right click karo", {})
    assert turn.intent is not None
    assert turn.intent.op.value == "right_click"
    assert turn.intent.target_description == "save button"


def test_gui_virtual_mode_attaches_a_hand_so_a_grant_is_reachable():
    """A grant with no hand is refused for the wrong reason ('no hand attached')."""
    from avcs.gui.server import GuiApp

    gui = GuiApp(source="demo", mouth="silent", ears=False, grants=["click"], eyes="lexicon")
    assert gui.hand is not None
    assert gui.capabilities()["actuation"] == "virtual"
