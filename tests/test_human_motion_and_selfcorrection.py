"""The three human-behaviour additions, each pinned to a measurable fact.

1. ``ServoHand`` -- a pointer that travels a neuromuscular trajectory instead of
   teleporting, behind the same hand contract ``dispatch`` already speaks.
2. hesitation -- a bounded, visible pause before a mutating pointer action.
3. self-correction -- a performed pointer action that changed nothing is
   re-grounded and retried, but only when the target actually moved.

Every test drives real code. The only stand-ins are a hand that records calls
and a perception that returns a scripted grounding point, because the thing under
test is *which calls are made*, and a real cursor would make that unobservable.
"""
from __future__ import annotations

import numpy as np
import pytest

from avcs.control.capability import CapabilityKernel, GroundingEvidence
from avcs.voice.actuators import ServoHand, XdotoolHand, default_servo_hand
from avcs.voice.hands import dispatch
from forge.vision.cortex.motor.driver import VirtualCursorDriver
from forge.vision.cortex.motor.failsafe import SafetyBreakoutException


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _RecordingHand:
    """A pixel-space hand that records every move and click, never actuating."""

    name = "recording"

    def __init__(self, screen=(1000, 500)) -> None:
        self._screen = screen
        self.moves: list = []
        self.clicks: list = []
        self.drags: list = []
        self.refuse_move = False
        self.refuse_click = False

    def get_screen_size(self):
        return self._screen

    def move_to_pixels(self, x, y):
        self.moves.append((x, y))
        return not self.refuse_move

    def click(self, x, y, double: bool = False):
        if self.refuse_click:
            return False
        self.clicks.append((x, y))
        return (x, y)

    def double_click(self, x, y):
        return self.click(x, y, double=True)

    def drag(self, x1, y1, x2, y2):
        self.drags.append(((x1, y1), (x2, y2)))
        return (x2, y2)


class _ScriptedPerception:
    """Grounds every query at a point the test can move between calls."""

    def __init__(self, point=(0.3, 0.3), peak=0.9, frame_value=10) -> None:
        self.point = point
        self.peak = peak
        self.frame_value = frame_value
        self.ground_calls = 0

    def observe(self, query=None):
        from avcs.control.capability import ScreenObservation

        return ScreenObservation(description="a screen", hotspot_count=1)

    def frame(self):
        arr = np.full((64, 64, 3), self.frame_value, dtype=np.uint8)
        arr[5:25, 5:25] = 200
        return arr

    def grounding_evidence(self, query):
        self.ground_calls += 1
        return GroundingEvidence(
            candidate_point_norm=self.point,
            peak_probability=self.peak,
            target_present=True,
            authorized=False,
        )


# --------------------------------------------------------------------------- #
# 1. ServoHand
# --------------------------------------------------------------------------- #
def test_servo_hand_is_driven_by_dispatch_with_no_special_case():
    """`dispatch` must treat ServoHand exactly like any other pixel hand."""
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    result = dispatch(hand, "click", point=(0.75, 0.25))
    assert result.performed is True
    # 0.75 * 999 = 749.25 -> 749 ; 0.25 * 499 = 124.75 -> 125
    assert result.point == (749, 125)
    assert inner.clicks == [(749, 125)]


def test_servo_hand_paints_a_trajectory_but_clicks_once():
    """A trajectory is many moves; the click is exactly one, at the end."""
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    dispatch(hand, "click", point=(0.8, 0.8))
    assert len(inner.moves) > 30, "the path must have many waypoints"
    assert len(inner.clicks) == 1, "one click, not one per waypoint"
    assert inner.moves[-1] == inner.clicks[0], "the click lands at the path's end"


def test_servo_trajectory_is_minimum_jerk_not_constant_speed():
    """Human reaching is slow-fast-slow; a teleport is one jump."""
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    dispatch(hand, "click", point=(0.9, 0.5))

    pts = inner.moves
    speeds = [
        ((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
        for i in range(len(pts) - 1)
    ]
    n = len(speeds)
    start = sum(speeds[: max(1, n // 10)]) / max(1, n // 10)
    mid = sum(speeds[n // 2 - 2 : n // 2 + 2]) / 4
    end = sum(speeds[-max(1, n // 10) :]) / max(1, n // 10)
    assert mid > start, "peak speed must not be at the start"
    assert mid > end, "peak speed must not be at the end"


def test_servo_hand_fitts_duration_scales_with_distance():
    """A farther target takes longer, as Fitts' Law requires."""
    near = ServoHand(_RecordingHand(), realtime=False)
    far = ServoHand(_RecordingHand(), realtime=False)
    dispatch(near, "click", point=(0.52, 0.52))  # from centre: short
    dispatch(far, "click", point=(0.9, 0.7))     # from centre: long
    assert near.last_result is not None
    assert far.last_result is not None
    assert far.last_result.fitts_index_of_difficulty > near.last_result.fitts_index_of_difficulty


def test_servo_hand_refuses_a_corner_target_without_clicking():
    """The panic corner (top-left / bottom-right) aborts; no click is issued.

    Top-left and bottom-right are the emergency breakout zones. A target that
    resolves into one must be refused, not clicked -- the user keeps that corner
    as a panic gesture.
    """
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    result = dispatch(hand, "click", point=(0.995, 0.995))
    assert result.performed is False
    assert inner.clicks == []
    assert hand.overrides >= 1


def test_servo_hand_can_click_a_top_right_window_control():
    """A Close box lives in the top-right; it is a control, not a panic corner.

    Excluding top-right (and bottom-left) from the corner rule is what makes the
    assistant able to close a window. Without it every close-button click would
    trip the failsafe.
    """
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    result = dispatch(hand, "click", point=(0.995, 0.005))
    assert result.performed is True
    assert len(inner.clicks) == 1


def test_servo_hand_failsafe_latches_and_reports_not_raises():
    """A latched failsafe must refuse the click, not raise into the caller."""
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    hand.controller.failsafe._tripped = True
    result = dispatch(hand, "click", point=(0.4, 0.4))
    assert result.performed is False
    assert inner.clicks == [], "a latched servo must not click"
    hand.reset_failsafe()
    assert dispatch(hand, "click", point=(0.4, 0.4)).performed is True


def test_servo_hand_clicks_when_the_cursor_starts_in_a_panic_corner():
    """Where the user parked the pointer is not our emergency.

    The corner rule guards the *destination*. Checking the start position too
    latched the failsafe before a single waypoint when the pointer happened to be
    resting in a corner, and because the next attempt read the same start it
    stayed latched forever -- a permanent lockout with no click ever performed.
    """
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    # Park the cursor in the bottom-right panic corner. The driver's readback
    # still tracks the servo's moves, so this is a corner *start*, not a takeover.
    hand._driver._x_norm, hand._driver._y_norm = 0.999, 0.999
    result = dispatch(hand, "click", point=(0.4, 0.4))
    assert result.performed is True, "a corner start must not block an interior click"
    assert len(inner.clicks) == 1
    assert hand.is_failsafe_tripped is False


def test_servo_hand_refuses_a_corner_destination_after_a_corner_start():
    """The corner start fix must not weaken the corner-target refusal."""
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    hand._driver._x_norm, hand._driver._y_norm = 0.999, 0.999
    result = dispatch(hand, "click", point=(0.999, 0.999))
    assert result.performed is False
    assert inner.clicks == []


def test_mid_flight_takeover_is_detected_by_the_servo_loop():
    """The override detector must actually run, not just exist.

    `override_threshold` was stored on the hand and `check_override` existed on
    the failsafe, but nothing in `servo_to` called it -- so a user grabbing the
    mouse mid-trajectory was never noticed and the servo kept dragging the
    pointer out from under them.
    """
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    # The pointer reports a fixed position far from every planned waypoint, which
    # is what a hand on the mouse looks like to the controller.
    hand._driver.get_position_normalized = lambda: (0.02, 0.02)
    result = dispatch(hand, "click", point=(0.9, 0.45))
    assert result.performed is False
    assert inner.clicks == [], "a taken-over servo must not click"
    assert hand.overrides >= 1
    assert hand.is_failsafe_tripped is True


def test_no_takeover_means_no_override_on_a_normal_click():
    """The detector must not fire on its own; a clean path stays clean."""
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    result = dispatch(hand, "click", point=(0.5, 0.25))
    assert result.performed is True
    assert hand.overrides == 0
    assert hand.is_failsafe_tripped is False


def test_servo_hand_mid_flight_takeover_is_detected():
    """If the cursor is diverted mid-path, the servo gives the user the pointer.

    This drives the real failsafe's override detector: the driver reports a
    position far from the planned waypoint, which is what a human grabbing the
    mouse looks like to the controller.
    """
    inner = _RecordingHand()
    hand = ServoHand(inner, realtime=False)
    hand._driver.get_position_normalized = lambda: (0.95, 0.95)  # yanked away
    with pytest.raises(SafetyBreakoutException):
        hand.controller.failsafe.check_override((0.95, 0.95), (0.1, 0.1))
    assert hand.controller.failsafe.is_tripped


def test_default_servo_hand_over_a_virtual_cursor_is_reported_virtual():
    """With no real hand the plan is real and the side effect is inert."""
    hand = ServoHand(VirtualCursorDriver(screen_size=(800, 600)), virtual=True)
    assert hand.virtual is True
    assert hand.get_screen_size() == (800, 600)
    assert dispatch(hand, "click", point=(0.5, 0.5)).performed is True


def test_xdotool_hand_exposes_a_bare_move_and_position():
    """The servo needs a move that does not click, and a position readback."""
    hand = XdotoolHand()
    assert hasattr(hand, "move_to_pixels")
    assert hasattr(hand, "get_cursor_pos")
    # On a box with no display these fail closed rather than raising.
    x, y = hand.get_cursor_pos()
    assert isinstance(x, int) and isinstance(y, int)


def test_servo_hand_delegates_non_pointer_ops():
    """Typing and key ops are not pointer moves and go straight through."""
    class _Typing:
        name = "typing"

        def get_screen_size(self):
            return (800, 600)

        def type_text(self, text, cps=25.0):
            self.typed = text
            return True

        def press_key(self, key):
            self.key = key
            return True

        def scroll(self, delta):
            self.delta = delta
            return True

    inner = _Typing()
    hand = ServoHand(inner, realtime=False)
    assert dispatch(hand, "type_text", text="hello").performed is True
    assert inner.typed == "hello"
    assert dispatch(hand, "press_key", key_name="ctrl+s").performed is True
    assert inner.key == "ctrl+s"
    assert dispatch(hand, "scroll", scroll_delta=-3).performed is True
    assert inner.delta == -3


def test_default_servo_hand_returns_a_servo_hand():
    hand = default_servo_hand()
    assert isinstance(hand, ServoHand)
    assert hand.available() is True


# --------------------------------------------------------------------------- #
# 2. Hesitation
# --------------------------------------------------------------------------- #
def _assistant(perception, hand, kernel, think_pause_s=0.05, max_corrections=2):
    from avcs.voice.assistant import Assistant
    from avcs.voice.brain import ConversationBrain

    class _Provider:
        api_key = ""
        model = "scripted"

        def __init__(self, turn):
            self._turn = turn

        def plan_step(self, *a, **k):
            return self._turn

        def stream_step(self, *a, **k):
            yield self._turn

    return Assistant(
        brain=ConversationBrain(provider=_Provider(None)),
        perception=perception,
        kernel=kernel,
        hand=hand,
        think_pause_s=think_pause_s,
        max_corrections=max_corrections,
    )


def test_hesitation_pauses_before_a_pointer_action():
    """A mutating pointer op records a visible pause; an observe op does not."""
    from avcs.voice.brain import ActionIntent, AssistantTurn, TurnKind
    from avcs.reasoning.contracts import ActionOp
    from avcs.voice.assistant import Assistant

    class _ScriptedBrain:
        memory = None
        provider = type("P", (), {"api_key": "", "model": "s"})()

        def __init__(self, turn):
            self._turn = turn

        def respond(self, *a, **k):
            return self._turn

    intent = ActionIntent.create(1, ActionOp.CLICK, target_description="save")
    turn = AssistantTurn(kind=TurnKind.ACT, speech="theek hai", intent=intent)
    hand = _RecordingHand()
    a = Assistant(
        brain=_ScriptedBrain(turn),
        perception=_ScriptedPerception(peak=0.9),
        kernel=CapabilityKernel(granted_ops=["click"], min_peak_probability=0.5),
        hand=hand,
        think_pause_s=0.05,
        max_corrections=0,
    )
    rec = a.step("save karo")
    assert rec.hesitated_s == pytest.approx(0.05, abs=0.01)
    assert hand.clicks, "the action still ran after the pause"


def test_hesitation_off_means_no_pause():
    from avcs.voice.brain import ActionIntent, AssistantTurn, TurnKind
    from avcs.reasoning.contracts import ActionOp
    from avcs.voice.assistant import Assistant

    class _ScriptedBrain:
        memory = None
        provider = type("P", (), {"api_key": "", "model": "s"})()

        def respond(self, *a, **k):
            return AssistantTurn(
                kind=TurnKind.ACT, speech="ok",
                intent=ActionIntent.create(1, ActionOp.CLICK, target_description="save"),
            )

    a = Assistant(
        brain=_ScriptedBrain(),
        perception=_ScriptedPerception(peak=0.9),
        kernel=CapabilityKernel(granted_ops=["click"], min_peak_probability=0.5),
        hand=_RecordingHand(),
        think_pause_s=0.0,
        max_corrections=0,
    )
    rec = a.step("save karo")
    assert rec.hesitated_s == 0.0


def test_the_thinking_line_is_spoken_before_the_action_reply():
    """The aside must precede the sentence that announces the click.

    Spoken the other way round, the user hears "click kar raha hoon" and only
    then "ruko, dekh leta hoon" -- a contradiction, and the exact macro
    behaviour the hesitation exists to avoid.
    """
    from avcs.voice.brain import ActionIntent, AssistantTurn, TurnKind
    from avcs.reasoning.contracts import ActionOp
    from avcs.voice.assistant import Assistant

    class _Speaker:
        def __init__(self):
            self.lines = []

        def available(self):
            return True

        def speak(self, text, **k):
            self.lines.append(text)
            return type("R", (), {"spoken": True, "error": None})()

    class _ThoughtBrain:
        memory = None
        provider = type("P", (), {"api_key": "", "model": "s"})()

        def respond(self, *a, **k):
            return AssistantTurn(
                kind=TurnKind.ACT, speech="Theek hai sir, click kar raha hoon.",
                thought="ruko, pehle dekh leta hoon",
                intent=ActionIntent.create(1, ActionOp.CLICK, target_description="save"),
            )

    speaker = _Speaker()
    a = Assistant(
        brain=_ThoughtBrain(),
        perception=_ScriptedPerception(peak=0.9),
        kernel=CapabilityKernel(granted_ops=["click"], min_peak_probability=0.5),
        hand=_RecordingHand(),
        speaker=speaker,
        think_pause_s=0.05,
        max_corrections=0,
    )
    a.step("save karo")
    assert speaker.lines[0] == "ruko, pehle dekh leta hoon"
    assert speaker.lines[1] == "Theek hai sir, click kar raha hoon."


def test_an_observing_op_gets_no_thinking_line():
    """A glance is not worth an aside; only a mutating pointer action is."""
    from avcs.voice.brain import ActionIntent, AssistantTurn, TurnKind
    from avcs.reasoning.contracts import ActionOp
    from avcs.voice.assistant import Assistant

    class _ThoughtBrain:
        memory = None
        provider = type("P", (), {"api_key": "", "model": "s"})()

        def respond(self, *a, **k):
            return AssistantTurn(
                kind=TurnKind.ACT, speech="dekh raha hoon",
                thought="screen ko scan kar raha hoon",
                intent=ActionIntent.create(1, ActionOp.OBSERVE_TARGET,
                                           target_description="save"),
            )

    a = Assistant(
        brain=_ThoughtBrain(),
        perception=_ScriptedPerception(peak=0.9),
        kernel=CapabilityKernel(granted_ops=["click"], min_peak_probability=0.5),
        hand=_RecordingHand(),
        think_pause_s=0.05,
    )
    assert a._thinking_line(_ThoughtBrain().respond("x")) is None


# --------------------------------------------------------------------------- #
# The availability probe must not report a false success
# --------------------------------------------------------------------------- #
def _fake_xdotool(tmp_path, *, pointer_moves: bool, size=(1280, 800)):
    """A stand-in ``xdotool`` whose pointer honours (or ignores) a move.

    ``pointer_moves=False`` reproduces the real Xvfb-without-`-noreset` server:
    ``mousemove`` exits 0 but the pointer stays put. The old probe (exit code
    only) called that hand available, and every click then silently no-opped.
    """
    import os
    import stat
    import textwrap

    w, h = size
    script = textwrap.dedent(f"""
        #!/usr/bin/env python3
        import sys, os
        STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pos.txt")
        MOVES = {str(bool(pointer_moves))}
        def read():
            try:
                return tuple(int(v) for v in open(STATE).read().split(","))
            except Exception:
                return ({w // 2}, {h // 2})
        def write(p):
            open(STATE, "w").write("%d,%d" % p)
        args = sys.argv[1:]
        if not args:
            sys.exit(2)
        cmd = args[0]
        if cmd == "getdisplaygeometry":
            print({w}, {h}); sys.exit(0)
        if cmd == "getmouselocation":
            x, y = read()
            print("X=%d\\nY=%d\\nSCREEN=0\\nWINDOW=1" % (x, y)); sys.exit(0)
        if cmd == "mousemove":
            x, y = int(args[1]), int(args[2])
            if MOVES:
                write((x, y))
            # else: exit 0, pointer unchanged -- the broken server
            sys.exit(0)
        sys.exit(0)
    """).strip() + "\n"
    path = tmp_path / "xdotool"
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def test_probe_rejects_a_display_whose_pointer_does_not_move(tmp_path):
    """Exit code 0 is not proof of actuation; the read-back is.

    This is the exact bug: on Xvfb started without `-noreset` the pointer resets
    on every client disconnect, so `mousemove` returns 0 and moves nothing. The
    hand reported available and every click silently did nothing.
    """
    binary = _fake_xdotool(tmp_path, pointer_moves=False)
    hand = XdotoolHand(display=":0", binary=binary)
    assert hand.available() is False


def test_probe_accepts_a_display_whose_pointer_moves(tmp_path):
    binary = _fake_xdotool(tmp_path, pointer_moves=True)
    hand = XdotoolHand(display=":0", binary=binary)
    assert hand.available() is True
    assert hand.get_screen_size() == (1280, 800)


def test_probe_restores_the_pointer_it_nudged(tmp_path):
    """A probe that left the user's cursor 1 px away would be a real side effect."""
    binary = _fake_xdotool(tmp_path, pointer_moves=True)
    hand = XdotoolHand(display=":0", binary=binary)
    before = hand.get_cursor_pos()
    hand.available()
    assert hand.get_cursor_pos() == before


def test_probe_does_not_recurse_through_run(tmp_path):
    """`_run` calls `available`; the probe must not call `_run`, or it loops."""
    binary = _fake_xdotool(tmp_path, pointer_moves=True)
    hand = XdotoolHand(display=":0", binary=binary)
    # Would raise RecursionError if _probe went through the guarded _run.
    assert hand.available() is True
    assert hand.click(10, 10) is True


def test_unavailable_hand_refuses_every_op(tmp_path):
    """Fail closed: a hand we cannot drive must never report a click happened."""
    binary = _fake_xdotool(tmp_path, pointer_moves=False)
    hand = XdotoolHand(display=":0", binary=binary)
    assert hand.click(5, 5) is False
    assert hand.move_to_pixels(5, 5) is False
    assert hand.get_cursor_pos() == (-1, -1)


# --------------------------------------------------------------------------- #
# 3. Self-correction
# --------------------------------------------------------------------------- #
def _click_assistant(perception, hand, max_corrections=2):
    from avcs.voice.brain import ActionIntent, AssistantTurn, TurnKind
    from avcs.reasoning.contracts import ActionOp
    from avcs.voice.assistant import Assistant

    class _ScriptedBrain:
        memory = None
        provider = type("P", (), {"api_key": "", "model": "s"})()

        def respond(self, *a, **k):
            return AssistantTurn(
                kind=TurnKind.ACT, speech="clicking",
                intent=ActionIntent.create(1, ActionOp.CLICK, target_description="save"),
            )

    return Assistant(
        brain=_ScriptedBrain(),
        perception=perception,
        kernel=CapabilityKernel(granted_ops=["click"], min_peak_probability=0.5),
        hand=hand,
        think_pause_s=0.0,
        max_corrections=max_corrections,
        delta_threshold=0.01,
    )


def test_a_same_point_miss_is_not_retried():
    """If re-grounding gives the same pixel there is nothing to correct."""
    perception = _ScriptedPerception(point=(0.3, 0.3), peak=0.9)
    hand = _RecordingHand()
    a = _click_assistant(perception, hand)
    rec = a.step("save karo")
    assert rec.performed is True
    assert rec.corrections == 0, "the identical point must not be clicked again"
    assert len(hand.clicks) == 1
    assert "nothing to correct" in rec.detail


def test_a_moved_target_is_regrounded_and_retried():
    """When the target moves after a miss, the retry follows the new point.

    The frame only changes once the *second* click lands, so the first attempt
    verifies as a miss (nothing changed) and the correction path runs. The
    grounding then returns a different point, which is what a target that moved
    looks like, and the retry clicks there and verifies.
    """
    perception = _ScriptedPerception(point=(0.3, 0.3), peak=0.9)
    hand = _RecordingHand()
    a = _click_assistant(perception, hand)

    original_frame = perception.frame

    def moving_frame():
        frame = original_frame().copy()
        if len(hand.clicks) >= 2:
            frame[:] = 120
        return frame

    perception.frame = moving_frame

    calls = {"n": 0}
    original_evidence = perception.grounding_evidence

    def moving_evidence(query):
        calls["n"] += 1
        if calls["n"] > 2:
            perception.point = (0.6, 0.6)
        return original_evidence(query)

    perception.grounding_evidence = moving_evidence

    rec = a.step("save karo")
    assert rec.corrections == 1, "a moved target must be retried exactly once"
    assert rec.corrected is True
    assert len(hand.clicks) == 2
    assert hand.clicks[0] != hand.clicks[1], "the retry must follow the new point"


def test_self_correction_is_bounded():
    """Corrections stop at the configured maximum, even if the point keeps moving."""
    perception = _ScriptedPerception(point=(0.2, 0.2), peak=0.9)
    hand = _RecordingHand()
    a = _click_assistant(perception, hand, max_corrections=1)

    n = {"v": 0}
    original = perception.grounding_evidence

    def always_moving(query):
        n["v"] += 1
        perception.point = (0.2 + 0.01 * n["v"], 0.2)
        return original(query)

    perception.grounding_evidence = always_moving
    rec = a.step("save karo")
    assert rec.corrections <= 1


def test_self_correction_off_by_default_knob():
    perception = _ScriptedPerception(point=(0.3, 0.3), peak=0.9)
    hand = _RecordingHand()
    a = _click_assistant(perception, hand, max_corrections=0)
    rec = a.step("save karo")
    assert rec.corrections == 0
    assert len(hand.clicks) == 1


def test_a_refused_action_is_not_retried():
    """A refusal is a different outcome from a miss and must not be re-attempted."""
    perception = _ScriptedPerception(point=(0.3, 0.3), peak=0.1)  # below the floor
    hand = _RecordingHand()
    a = _click_assistant(perception, hand)
    rec = a.step("save karo")
    assert rec.gate_allowed is False
    assert rec.performed is False
    assert rec.corrections == 0
    assert hand.clicks == []


# --------------------------------------------------------------------------- #
# GUI wiring: --servo must never actuate without --live
# --------------------------------------------------------------------------- #
def test_gui_servo_without_live_stays_virtual_and_moves_no_cursor():
    """`--servo` on its own is a human trajectory over a virtual cursor.

    The important half is the negative: without `--live` the hand must not be a
    real one, or the flag would silently grant machine actuation the user never
    asked for.
    """
    from avcs.gui.server import GuiApp, _hand_is_virtual

    gui = GuiApp(source="demo", eyes="lexicon", mouth="silent", ears=False, servo=True)
    assert isinstance(gui.hand, ServoHand)
    assert _hand_is_virtual(gui.hand) is True
    assert gui.capabilities()["actuation"] == "virtual"


def test_gui_default_hand_is_still_the_plain_virtual_hand():
    """With no `--servo` the GUI hand is unchanged, so the default is untouched."""
    from avcs.gui.server import GuiApp, _hand_is_virtual
    from avcs.system1.loop import VirtualHand

    gui = GuiApp(source="demo", mouth="silent", ears=False)
    assert isinstance(gui.hand, VirtualHand)
    assert _hand_is_virtual(gui.hand) is True


# --------------------------------------------------------------------------- #
# The trajectory dataset must carry the human-behaviour fields
# --------------------------------------------------------------------------- #
def test_trajectory_line_records_hesitation_and_corrections(tmp_path):
    """A reviewable dataset is the point of the log; the new fields belong in it.

    Without this, a correction would be invisible in the very file that exists to
    make corrections reviewable, and the recorded turn would read as a first-try
    success.
    """
    import json

    from avcs.voice.trajectory import TrajectoryLog, TrajectoryStep

    path = tmp_path / "trajectories.jsonl"
    log = TrajectoryLog(path)
    log.record(
        TrajectoryStep(
            turn=3, heard="save karo", said="ok", kind="act", source="llm",
            action="click", target="save", gate_allowed=True, performed=True,
            verified=False, hesitated_s=0.45, corrections=1, corrected=False,
        )
    )
    line = json.loads(path.read_text().strip().splitlines()[-1])
    assert line["hesitated_s"] == 0.45
    assert line["corrections"] == 1
    assert line["corrected"] is False
    assert line["outcome"] == "no_change"


def test_assistant_writes_hesitation_into_the_trajectory(tmp_path):
    """End to end: the assistant's real turn reaches the log with the pause in it."""
    import json

    from avcs.voice.brain import ActionIntent, AssistantTurn, TurnKind
    from avcs.reasoning.contracts import ActionOp
    from avcs.voice.assistant import Assistant
    from avcs.voice.trajectory import TrajectoryLog

    class _ScriptedBrain:
        memory = None
        provider = type("P", (), {"api_key": "", "model": "s"})()

        def respond(self, *a, **k):
            return AssistantTurn(
                kind=TurnKind.ACT, speech="ok",
                intent=ActionIntent.create(1, ActionOp.CLICK, target_description="save"),
            )

    log = TrajectoryLog(tmp_path / "t.jsonl")
    a = Assistant(
        brain=_ScriptedBrain(),
        perception=_ScriptedPerception(peak=0.9),
        kernel=CapabilityKernel(granted_ops=["click"], min_peak_probability=0.5),
        hand=_RecordingHand(),
        trajectory=log,
        think_pause_s=0.05,
        max_corrections=0,
    )
    a.step("save karo")
    line = json.loads((tmp_path / "t.jsonl").read_text().strip().splitlines()[-1])
    assert line["hesitated_s"] == pytest.approx(0.05, abs=0.01)
    assert line["performed"] is True


def test_driver_shape_hand_refuses_a_click_after_a_failed_servo():
    """A failed servo must not become a click somewhere else.

    A driver-shape hand clicks wherever the pointer already is, so if the servo
    never reached the target the click lands in the wrong place and still returns
    truthy -- a false success. The hand must refuse instead. A coordinate-shape
    hand (`XdotoolHand`) moves and clicks in one call and is unaffected.
    """
    from avcs.voice.actuators import ServoHand

    class _DriverShape:
        name = "driver-shape"

        def __init__(self):
            self.clicks = []

        def get_screen_size(self):
            return (1000, 500)

        def get_position_normalized(self):
            return (0.5, 0.5)

        def move_to_pixels(self, x, y):
            # A move that raises for a non-safety reason: the servo aborts with
            # success=False and *without* tripping the failsafe.
            raise RuntimeError("cannot move: the pointer is held")

        def click(self, button="left"):
            self.clicks.append(button)

    inner = _DriverShape()
    hand = ServoHand(inner, realtime=False)
    assert hand.click(900, 450) is False, "a failed servo must not report a click"
    assert inner.clicks == [], "nothing may be clicked after a failed servo"
