"""The real-time voice assistant loop.

    you speak  ->  ears (STT)  ->  brain (LLM)  ->  mouth (TTS)  ->  you hear
                                        |
                                        +-> AVCS eyes (pure pixels) feed the brain
                                        +-> capability gate -> hand -> verify

Everything above the gate is conversation. Everything below it is the existing
AVCS machinery, unchanged: perception is pure-pixel, the gate is the production
``CapabilityKernel`` (fail-closed, grants nothing by default), and actuation is a
``VirtualHand`` unless a caller explicitly opts into a real motor.

The loop never raises for the ordinary "this machine has no mic / no speaker / no
API key" cases. Each of those degrades independently and is reported per turn, so
a headless laptop still runs the full reasoning path and says so.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from avcs.control.capability import CapabilityKernel, GroundingEvidence
from avcs.voice.brain import AssistantTurn, ConversationBrain, TurnKind
from avcs.voice.hands import MUTATING_OPS, dispatch
from avcs.voice.stt import AudioClip, SttEngine, Transcript, default_stt_engine
from avcs.voice.tts import Speaker, SpeechQueue, SpeechResult


class FrameSource(Protocol):
    """Anything that can hand back the current screen as an RGB uint8 array."""

    def capture(self) -> np.ndarray:
        ...


@dataclasses.dataclass
class ScreenObservation:
    """The coordinate-free view of the screen the brain is allowed to see."""

    description: str = "no visual detail available"
    target_present: Optional[bool] = None
    stability: str = "stable"
    has_modal: bool = False
    active_window: str = ""
    hotspot_count: int = 0
    last_action: str = ""
    # Structured uncertainty from the last gate decision, if there was one. A
    # number, not a sentence, so the model can reason about how close it was.
    last_confidence: Optional[float] = None
    last_confidence_floor: Optional[float] = None

    def to_brain_dict(self, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"interactive_elements": self.hotspot_count}
        if self.last_action:
            extra["last_action"] = self.last_action
        if self.last_confidence is not None:
            extra["last_confidence"] = self.last_confidence
            if self.last_confidence_floor is not None:
                extra["last_confidence_floor"] = self.last_confidence_floor
                # Spell out the near-miss case, because "0.72 vs 0.75" is easy to
                # read as a success. A near miss is the one case where asking the
                # user beats retrying blindly. The epsilon matters: 0.6 - 0.75 is
                # -0.15000000000000002 in float, which would miss an exact -0.15.
                margin = self.last_confidence - self.last_confidence_floor
                if margin < 0 and abs(margin) <= 0.15 + 1e-9:
                    extra["confidence_note"] = (
                        "the last target was a near miss: it may be present but "
                        "unclear. Ask the user rather than clicking a guess."
                    )
        if capabilities:
            # What the assistant can actually *do* right now. Without this the
            # model does not know whether it holds a click/type grant, so it
            # cannot choose between acting and asking -- it stalls on `wait`.
            # This is not screen data, so it does not cross the System 1 -> 2
            # image boundary; it is the same category as `stability`.
            extra["capabilities"] = capabilities
        return ConversationBrain.build_observation(
            screen_description=self.description,
            target_present=self.target_present,
            stability=self.stability,
            has_modal=self.has_modal,
            active_window=self.active_window,
            extra=extra,
        )


class Perception(Protocol):
    """Turns the current screen into a `ScreenObservation` (pixels only)."""

    def observe(self, query: Optional[str] = None) -> ScreenObservation:
        ...

    def frame(self) -> Optional[np.ndarray]:
        ...


class StaticPerception:
    """A fixed observation for tests and demos with no display attached."""

    def __init__(self, observation: Optional[ScreenObservation] = None) -> None:
        self._obs = observation or ScreenObservation(
            description="a desktop with a few interactive controls",
            target_present=True,
            stability="stable",
            hotspot_count=3,
        )
        self._frame = np.full((64, 64, 3), 96, dtype=np.uint8)
        self._frame[20:40, 20:40] = 200

    def observe(self, query: Optional[str] = None) -> ScreenObservation:
        return self._obs

    def frame(self) -> Optional[np.ndarray]:
        return self._frame


@dataclasses.dataclass
class AssistantTurnRecord:
    """Everything that happened in one turn, with honest provenance."""

    turn: int
    heard: str
    said: str
    kind: str
    source: str
    action: Optional[str] = None
    target: Optional[str] = None
    gate_allowed: Optional[bool] = None
    gate_reason: str = ""
    actuation: str = "none"        # "none" | "virtual" | "live"
    performed: bool = False
    detail: str = ""
    clicked: Optional[tuple[int, int]] = None
    verified: Optional[bool] = None
    visual_delta: Optional[float] = None
    # Structured uncertainty from the gate, so a caller can tell "close call,
    # worth asking" from "target simply is not there" without parsing the reason.
    confidence: Optional[float] = None
    confidence_floor: Optional[float] = None
    spoken: bool = False
    listen_error: Optional[str] = None
    speak_error: Optional[str] = None
    # Per-stage wall time in milliseconds, so "real-time" is a measurement and
    # not a claim. listen_ms is 0 when the turn came from a script, not a mic.
    listen_ms: float = 0.0
    brain_ms: float = 0.0
    act_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ConversationReport:
    """Summary of a whole session."""

    turns: List[AssistantTurnRecord] = dataclasses.field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def actions_taken(self) -> int:
        return sum(1 for t in self.turns if t.actuation != "none")

    @property
    def actions_verified(self) -> int:
        return sum(1 for t in self.turns if t.verified)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "actions_taken": self.actions_taken,
            "actions_verified": self.actions_verified,
            "duration_s": round(self.duration_s, 3),
            "turns": [t.to_dict() for t in self.turns],
        }


class Assistant:
    """Conversational AVCS agent: ears + brain + mouth, gated hands, pixel eyes.

    Parameters
    ----------
    brain:
        The LLM conversation brain. Defaults to ``ConversationBrain()`` (reads its
        endpoint from the environment and falls back to the offline rule brain).
    speaker / stt:
        The mouth and the ears. Both default to the best available engine and
        degrade to a silent mouth / null ears when nothing is installed.
    perception:
        A `Perception`. When omitted the assistant runs without eyes and says so.
    listen_seconds:
        How long to record per turn. The loop is turn-based rather than always-on
        VAD so it stays predictable and testable on CPU.
    """

    def __init__(
        self,
        brain: Optional[ConversationBrain] = None,
        speaker: Optional[Speaker] = None,
        stt: Optional[SttEngine] = None,
        perception: Optional[Perception] = None,
        kernel: Optional[CapabilityKernel] = None,
        hand: Any = None,
        listen_seconds: float = 4.0,
        record_audio: Optional[Callable[[float], AudioClip]] = None,
        actuation: str = "virtual",
        stream: bool = True,
        max_task_steps: int = 8,
        delta_threshold: float = 0.015,
        step_delay_s: float = 0.25,
        memory: Any = None,
        trajectory: Any = None,
    ) -> None:
        self.brain = brain or ConversationBrain()
        self.speaker = speaker or Speaker()
        self.stt = stt or default_stt_engine()
        self.perception = perception
        self.kernel = kernel or CapabilityKernel()
        self.hand = hand
        self.listen_seconds = float(listen_seconds)
        self._record_audio = record_audio
        self.actuation = actuation  # "virtual" | "live"
        self.max_task_steps = int(max_task_steps)
        self.delta_threshold = float(delta_threshold)
        self.step_delay_s = float(step_delay_s)
        self._last_point: Optional[tuple[float, float]] = None
        self._last_outcome: str = ""
        self._last_confidence: Optional[float] = None
        self._last_confidence_floor: Optional[float] = None
        # Durable memory of the user, and a reviewable log of what was done.
        # Both optional; wiring them to the brain here means a caller cannot
        # attach a memory store and then forget to let the model read it.
        self.memory = memory
        self.trajectory = trajectory
        if memory is not None and getattr(self.brain, "memory", None) is None:
            self.brain.memory = memory
        # Streaming makes the assistant talk as the reply is generated. It is on
        # by default because that is the difference between a live conversation
        # and a walkie-talkie; a caller can turn it off for a deterministic demo.
        self.stream = bool(stream)

    # ------------------------------------------------------------------ #
    # Capabilities (reported honestly)
    # ------------------------------------------------------------------ #
    def capabilities(self) -> Dict[str, Any]:
        caps = {
            "brain": type(self.brain.provider).__name__,
            "brain_has_key": bool(getattr(self.brain.provider, "api_key", "")),
            "mouth": self.speaker.engine_name,
            "mouth_available": self.speaker.available(),
            "ears": self.stt.name,
            "ears_available": self.stt.available(),
            "eyes": type(self.perception).__name__ if self.perception else "none",
            "mutating_grants": sorted(self.kernel.granted_ops),
            "actuation": self.actuation,
            "memory": bool(self.memory),
            "trajectory": bool(self.trajectory),
        }
        # What the *hand* can do, so a caller can tell "not granted" from "granted
        # but this hand has no such method". A right-click grant on a hand with
        # no right_click() is a capability the machine does not have.
        hand = self.hand
        if hand is None:
            caps["hand"] = None
            caps["hand_ops"] = []
        else:
            caps["hand"] = getattr(hand, "name", type(hand).__name__)
            caps["hand_ops"] = sorted(
                op for op in MUTATING_OPS if callable(getattr(hand, op, None))
                or (op == "type" and callable(getattr(hand, "type_text", None)))
            )
        # Report the monitor layout only when it can be read; an empty list means
        # "unknown", not "one monitor".
        try:
            from forge.vision.capture import list_monitors

            monitors = list_monitors()
        except Exception:
            monitors = []
        caps["monitors"] = monitors
        caps["monitor_count"] = len(monitors)
        return caps

    # ------------------------------------------------------------------ #
    # One turn
    # ------------------------------------------------------------------ #
    def _listen(self) -> Transcript:
        if self._record_audio is not None:
            try:
                clip = self._record_audio(self.listen_seconds)
            except Exception as exc:
                return Transcript(text="", engine=self.stt.name, error=f"capture failed: {exc}")
        else:
            from avcs.voice.stt import record_microphone, record_microphone_until_silence

            try:
                # End-point on trailing silence so the reply starts as soon as the
                # user stops talking, instead of waiting out a fixed window.
                clip = record_microphone_until_silence(max_seconds=max(self.listen_seconds, 5.0))
            except Exception:
                try:
                    clip = record_microphone(self.listen_seconds)
                except Exception as exc:
                    return Transcript(text="", engine=self.stt.name, error=str(exc))
        clip = clip.trim_silence()
        if clip.samples.size == 0:
            return Transcript(text="", engine=self.stt.name, error="no speech detected")
        return self.stt.transcribe(clip)

    def _speak(self, text: str) -> SpeechResult:
        return self.speaker.speak(text)

    def _observe(self, query: Optional[str]) -> ScreenObservation:
        if self.perception is None:
            return ScreenObservation(description="no eyes attached (perception disabled)")
        try:
            obs = self.perception.observe(query)
        except Exception as exc:
            obs = ScreenObservation(description=f"perception error: {exc}")
        # Tell the brain what the previous step did, so an autonomous run can
        # decide the next move instead of blindly repeating itself.
        obs.last_action = self._last_outcome
        # And the gate's confidence number, so "denied" is not a flat verdict.
        obs.last_confidence = self._last_confidence
        obs.last_confidence_floor = self._last_confidence_floor
        return obs

    def step(
        self,
        user_text: str,
        turn: int = 0,
        stream: Optional[bool] = None,
        on_speech: Optional[Callable[[str], None]] = None,
    ) -> AssistantTurnRecord:
        """Run one conversational turn from already-transcribed user text.

        ``on_speech`` receives each sentence the moment it is spoken. The GUI
        uses it to forward speech to the browser as it is produced, so the page
        hears the first sentence while the model is still writing the rest.
        """
        self.stream = bool(stream) if stream is not None else self.stream
        return self._run_one(user_text, turn, on_speech=on_speech)

    def _run_one(
        self,
        user_text: str,
        turn: int,
        on_speech: Optional[Callable[[str], None]] = None,
    ) -> AssistantTurnRecord:
        """Observe, ask the brain (timed), and act if asked. One loop step."""
        obs = self._observe(None)
        _t0 = time.perf_counter()
        turn_result, spoken, speak_error = self._ask_brain(
            user_text, obs, self.stream, on_speech=on_speech
        )
        brain_ms = (time.perf_counter() - _t0) * 1000.0

        record = AssistantTurnRecord(
            turn=turn,
            heard=user_text,
            said=turn_result.speech,
            kind=turn_result.kind.value,
            source=turn_result.source,
            spoken=spoken,
            speak_error=speak_error,
            brain_ms=round(brain_ms, 1),
        )
        if turn_result.wants_action and turn_result.intent is not None:
            _t1 = time.perf_counter()
            self._execute(turn_result, record)
            record.act_ms = round((time.perf_counter() - _t1) * 1000.0, 1)
        # Persist what the model asked to remember, and log the step for review.
        self._apply_memories(turn_result)
        self._log_episode(record)
        self._log_trajectory(record, obs)
        # Close the loop for whatever the next step decides.
        self._last_outcome = self._outcome_summary(record)
        return record

    def _apply_memories(self, turn_result: AssistantTurn) -> int:
        """Store the facts the model proposed. Returns how many were stored.

        The model proposes; the assistant writes. An empty or malformed proposal
        is dropped rather than stored, so a hallucinated note cannot become a
        permanent false belief about the user.
        """
        proposals = getattr(turn_result, "memories", None) or []
        if not proposals or self.memory is None:
            return 0
        stored = 0
        for item in proposals:
            try:
                key, value = item
            except Exception:
                continue
            if self.memory.remember(str(key), str(value)) is not None:
                stored += 1
        if stored:
            self.memory.save()
        return stored

    def _log_episode(self, record: AssistantTurnRecord) -> None:
        """Add this turn to durable memory as an episode, and persist it.

        Only turns that did something (an action, or a spoken reply the user
        needs continuity from) are kept; a silent turn is not an episode.
        """
        if self.memory is None:
            return
        if not record.heard and not record.action:
            return
        try:
            from avcs.voice.memory import Episode

            self.memory.record_episode(
                Episode(
                    heard=record.heard,
                    said=record.said,
                    kind=record.kind,
                    action=record.action,
                    target=record.target,
                    performed=record.performed,
                    gate_allowed=record.gate_allowed,
                    verified=record.verified,
                    detail=record.detail,
                )
            )
            self.memory.save()
        except Exception:
            pass

    def _log_trajectory(self, record: AssistantTurnRecord, obs: Optional[ScreenObservation]) -> None:
        if self.trajectory is None:
            return
        from avcs.voice.trajectory import TrajectoryStep

        try:
            self.trajectory.record(
                TrajectoryStep(
                    turn=record.turn,
                    heard=record.heard,
                    said=record.said,
                    kind=record.kind,
                    source=record.source,
                    observation=(obs.description if obs is not None else ""),
                    action=record.action,
                    target=record.target,
                    gate_allowed=record.gate_allowed,
                    gate_reason=record.gate_reason,
                    confidence=record.confidence,
                    confidence_floor=record.confidence_floor,
                    performed=record.performed,
                    actuation=record.actuation,
                    detail=record.detail,
                    verified=record.verified,
                    visual_delta=record.visual_delta,
                    capabilities=self.capabilities(),
                )
            )
        except Exception:
            # A logging failure must never break a live turn.
            pass

    def run_task(self, user_text: str, max_steps: Optional[int] = None,
                 on_record: Optional[Callable[[AssistantTurnRecord], None]] = None,
                 on_speech: Optional[Callable[[str], None]] = None) -> List[AssistantTurnRecord]:
        """Autonomously execute a whole task, not just one action.

        A single conversational turn produces at most one action. A real
        assistant keeps going: act, look at the result, decide the next step,
        repeat, until the brain says ``finish``/``stop`` or the budget runs out.

        The loop stops early -- and says so -- when an action is refused by the
        gate or does not reach the hand, because continuing would just repeat the
        same failing step. Speech is streamed on every step, so the assistant
        narrates the work while it does it.
        """
        budget = int(max_steps if max_steps is not None else self.max_task_steps)
        records: List[AssistantTurnRecord] = []
        # Signatures of the last two steps. A person tries a thing, sees it did
        # not work, and changes approach; repeating the identical action at the
        # identical point forever is not progress, and without this guard the
        # model does exactly that when the screen never changes (e.g. a click
        # that has no visible effect). Two repeats is the limit: the third
        # identical step is refused and reported as stuck.
        recent: List[str] = []

        for i in range(1, budget + 1):
            record = self._run_one(user_text, i, on_speech=on_speech)
            records.append(record)
            if on_record:
                on_record(record)

            if record.kind in ("finish", "stop"):
                break
            # A pure conversational turn (a reply or a clarifying question) has
            # no action to carry forward: stop here rather than looping the model
            # over a greeting. Only a proposed action starts autonomous work.
            if not record.action:
                break
            # Stop on a *refused* mutating step, not on an observing one. A
            # perception step ("look at the screen", "observe_target") sets
            # `performed=False` because it actuates nothing -- that is success,
            # not failure. Breaking on it made every task end after one glance,
            # which is why the GUI could never carry a task through.
            if record.action in MUTATING_OPS and not record.performed:
                break

            signature = f"{record.action}:{record.clicked}:{record.verified}"
            recent.append(signature)
            repeats = 1
            for prev in reversed(recent[:-1]):
                if prev == signature:
                    repeats += 1
                else:
                    break
            if repeats >= 3:
                # A stop record, not an action record: it carries no `action`,
                # because no action was attempted on this step. Keeping the
                # invariant "a record with an action means an action was tried"
                # lets a caller count attempts without subtracting the stop.
                stuck = AssistantTurnRecord(
                    turn=record.turn + 1,
                    heard="",
                    said=(
                        f"Sir, main {record.action} baar-baar kar raha hoon par screen "
                        "nahi badal rahi. Main yahin ruk raha hoon -- aap bataiye aage kya karna hai."
                    ),
                    kind="stop",
                    source="loop",
                    target=record.target,
                    detail=(
                        f"stopped: {record.action} repeated {repeats} times with no change"
                    ),
                )
                records.append(stuck)
                if on_record:
                    on_record(stuck)
                break
            # Tell the next step how many times in a row this exact action has now
            # produced no change, so the model can decide to stop rather than
            # repeat a third time. Without the count it cannot tell "first try"
            # from "third identical try".
            if repeats > 1:
                self._last_outcome = (
                    f"{self._last_outcome} (attempt {repeats}, no change from this action yet)"
                )

            # From here the screen is the speaker: the next decision is driven by
            # what changed, not by the user, so the user text is not repeated.
            user_text = ""
            if self.step_delay_s > 0:
                time.sleep(self.step_delay_s)

        return records

    @staticmethod
    def _outcome_summary(record: AssistantTurnRecord) -> str:
        """One line describing what the last step actually did, for the brain.

        Carries the gate's confidence *number*, not just a verdict, so the model
        can tell a near miss (0.72 against a 0.75 floor -- worth asking about)
        from a clear absence (0.10 -- the target is simply not on screen). Without
        the number both read as the same flat "denied".
        """
        if not record.action:
            return ""
        if not record.gate_allowed:
            conf, floor = record.confidence, record.confidence_floor
            if conf is not None and floor:
                return (
                    f"last step {record.action} was REFUSED by the gate "
                    f"(grounding confidence {conf:.2f} vs floor {floor:.2f}; {record.gate_reason})"
                )
            return f"last step {record.action} was REFUSED by the capability gate ({record.gate_reason})"
        if not record.performed:
            return f"last step {record.action} did not reach the hand ({record.gate_reason})"
        change = "screen changed" if record.verified else "no visible change"
        conf = f", confidence {record.confidence:.2f}" if record.confidence is not None else ""
        return f"last step performed {record.action}{conf}, {change}"

    def _ask_brain(
        self,
        user_text: str,
        obs: ScreenObservation,
        use_stream: bool,
        on_speech: Optional[Callable[[str], None]] = None,
    ) -> tuple[AssistantTurn, bool, Optional[str]]:
        """Send one turn to the brain and return (turn, spoken, speak_error)."""
        # Tell the brain what it can actually do, so it can choose between acting
        # and asking instead of stalling. `can_click` reflects the real hand, not
        # just a grant: a grant with no hand still refuses every click.
        capabilities = {
            "can_click": bool(self.hand is not None and self.kernel.is_granted("click")),
            "can_type": bool(self.hand is not None and self.kernel.is_granted("type_text")),
            "can_press_key": bool(self.hand is not None and self.kernel.is_granted("press_key")),
            "can_scroll": bool(self.hand is not None and self.kernel.is_granted("scroll")),
            "can_right_click": bool(
                self.hand is not None and self.kernel.is_granted("right_click")
                and (callable(getattr(self.hand, "right_click", None))
                     or callable(getattr(self.hand, "secondary_click", None)))
            ),
            "can_drag": bool(
                self.hand is not None and self.kernel.is_granted("drag")
                and callable(getattr(self.hand, "drag", None))
            ),
            "can_launch": bool(
                self.hand is not None and self.kernel.is_granted("launch_app")
                and callable(getattr(self.hand, "launch_app", None))
            ),
            "can_read_files": bool(
                self.hand is not None and self.kernel.is_granted("read_file")
                and callable(getattr(self.hand, "read_file", None))
            ),
            "can_list_dirs": bool(
                self.hand is not None and self.kernel.is_granted("list_dir")
                and callable(getattr(self.hand, "list_dir", None))
            ),
            "can_see_targets": callable(getattr(self.perception, "grounding_evidence", None)),
        }
        brain_obs = obs.to_brain_dict(capabilities)
        if use_stream and hasattr(self.brain, "respond_stream"):
            # The mouth runs on a background thread, so the first sentence is
            # already audible while the rest of the reply is still being written.
            queue = SpeechQueue(self.speaker)

            def _say(sentence: str) -> None:
                if on_speech is not None:
                    on_speech(sentence)
                queue.say(sentence)

            turn_result = self.brain.respond_stream(user_text, brain_obs, on_speech=_say)
            results = queue.drain()
            spoken = any(r.spoken for r in results)
            speak_error = next((r.error for r in results if r.error), None)
        else:
            turn_result = self.brain.respond(user_text, brain_obs)
            if on_speech is not None and turn_result.speech:
                on_speech(turn_result.speech)
            result = self._speak(turn_result.speech)
            spoken, speak_error = result.spoken, result.error
        return turn_result, spoken, speak_error

    def _execute(self, turn_result: AssistantTurn, record: AssistantTurnRecord) -> None:
        """Ground the named target, then gate and run the action, then verify.

        Verification uses a fresh post-action capture of the *same* screen. When
        there is no post-frame, ``verified`` stays ``None`` rather than being
        reported as success.
        """
        intent = turn_result.intent
        # Ask perception to ground *that target* (not the raw utterance) before
        # gating the action.
        self._observe(intent.target_description)
        pre_frame = self.perception.frame() if self.perception else None
        self._act(turn_result, record)

        if record.performed and self.perception is not None:
            post_frame = self.perception.frame()
            if pre_frame is not None and post_frame is not None and record.clicked is not None:
                from avcs.control.verifier import VisualRippleVerifier
                from avcs.system1.loop import to_luma

                # The delta helper only reads `ripple_radius`; a full verifier
                # needs a HumanEye we do not have here, so call it on a shim.
                shim = type("_Ripple", (), {"ripple_radius": 0.08})()
                point = self._last_point or (0.5, 0.5)
                delta = float(
                    VisualRippleVerifier.calculate_local_photometric_delta(
                        shim, point, to_luma(pre_frame), to_luma(post_frame)
                    )
                )
                record.visual_delta = round(delta, 4)
                record.verified = delta >= self.delta_threshold

    def _act(self, turn_result: AssistantTurn, record: AssistantTurnRecord) -> None:
        intent = turn_result.intent
        op = intent.op.value
        record.action = op
        record.target = intent.target_description

        frame = self.perception.frame() if self.perception else None
        # Prefer *real* grounding evidence when the perception can produce it;
        # only fall back to a placeholder when it cannot. A placeholder peak of
        # 0.0 fails the kernel's floor, which is the honest outcome when there is
        # no grounding signal at all.
        evidence = None
        producer = getattr(self.perception, "grounding_evidence", None)
        if callable(producer):
            try:
                evidence = producer(intent.target_description)
            except Exception:
                evidence = None
        if evidence is None:
            # No grounding signal at all: report no candidate point so the kernel
            # denies for the honest reason ("grounder produced no candidate point")
            # rather than for a fabricated low peak at a fake centre.
            evidence = GroundingEvidence(
                candidate_point_norm=None,
                peak_probability=0.0,
                target_present=False,
                authorized=False,
                active_query=intent.target_description,
            )

        # Named operands for the non-grounded ops. The kernel contains these, so
        # a path or an app name never reaches a hand unvalidated.
        parameters = intent.parameters or {}
        path_arg = str(parameters.get("path") or intent.text_input or "").strip() or None
        app_arg = str(parameters.get("app") or intent.text_input or "").strip() or None
        arguments: dict = {"app": app_arg, "path": path_arg}

        # A drag has two ends; both must be grounded, so resolve the destination
        # here and hand it to the same kernel decision.
        point2 = None
        if op == "drag":
            dest_query = str(parameters.get("target2") or parameters.get("destination") or "").strip()
            if callable(producer) and dest_query:
                try:
                    dest_ev = producer(dest_query)
                except Exception:
                    dest_ev = None
                if dest_ev is not None:
                    point2 = dest_ev.candidate_point_norm
                    arguments["point2"] = point2
                    arguments["peak2"] = float(dest_ev.peak_probability)
            if "point2" not in arguments:
                arguments["point2"] = point2
                arguments["peak2"] = 0.0

        decision = self.kernel.authorize(
            op=op, query=intent.target_description, evidence=evidence,
            frame=frame, arguments=arguments,
        )
        record.gate_allowed = decision.allowed
        record.gate_reason = decision.reason
        record.confidence = float(decision.confidence)
        record.confidence_floor = float(decision.floor)
        self._last_confidence = record.confidence
        self._last_confidence_floor = record.confidence_floor

        if not decision.allowed:
            record.actuation = "none"
            return
        if op not in MUTATING_OPS:
            record.actuation = "none"
            return
        if self.hand is None:
            record.actuation = "none"
            record.gate_reason = decision.reason + "; no hand attached"
            return

        point = decision.coordinate or (0.5, 0.5)
        self._last_point = point
        result = dispatch(
            self.hand,
            op,
            point=point,
            point2=point2,
            text=intent.text_input,
            key_name=intent.key_name,
            scroll_delta=intent.scroll_delta,
            app_name=app_arg,
            path_arg=path_arg,
        )
        record.performed = result.performed
        record.detail = result.detail
        record.clicked = result.point
        if result.performed:
            record.actuation = self.actuation
        else:
            record.actuation = "none"
            record.gate_reason = f"{decision.reason}; {result.error or 'hand did not perform'}"

    # ------------------------------------------------------------------ #
    # Session
    # ------------------------------------------------------------------ #
    def converse(
        self,
        max_turns: int = 6,
        initial_utterance: Optional[str] = None,
        utterances: Optional[List[str]] = None,
    ) -> ConversationReport:
        """Turn-based voice loop.

        `initial_utterance` skips the first listen. `utterances` scripts a whole
        conversation without a microphone: each entry is fed as the user's turn,
        in order, before falling back to listening. That is what makes the
        clarify-then-act flow testable end to end ("mujhe X kaam karna hai" ->
        the assistant asks -> the user names the task).
        """
        report = ConversationReport(started_at=time.time())
        pending = list(utterances) if utterances else None
        if pending is None and initial_utterance is not None:
            pending = [initial_utterance]
        for i in range(1, max_turns + 1):
            if pending:
                user_text = pending.pop(0)
                listen_error = None
            else:
                transcript = self._listen()
                user_text = transcript.normalized()
                listen_error = transcript.error
                if not user_text:
                    rec = AssistantTurnRecord(
                        turn=i, heard="", said="", kind="silent", source="ears",
                        listen_error=listen_error,
                    )
                    report.turns.append(rec)
                    break

            rec = self.step(user_text, turn=i)
            report.turns.append(rec)
            if rec.kind in ("finish", "stop"):
                break
        report.ended_at = time.time()
        return report

    def loop(
        self,
        max_turns: int = 0,
        stop_phrases: Optional[Sequence[str]] = None,
        on_record: Optional[Callable[[AssistantTurnRecord], None]] = None,
    ) -> ConversationReport:
        """Continuous real-time session: listen -> think -> speak -> act -> repeat.

        This is the always-on assistant. Each turn:
          1. listens on the microphone (silence-terminated),
          2. streams a reply and starts speaking it while the model is still
             writing (barge-in: a new utterance stops the mouth),
          3. grounds, gates and performs any action the brain asked for,
          4. repeats until the user says a stop phrase or ``max_turns`` is hit
             (``max_turns=0`` means run until stopped).

        Ctrl-C ends the session cleanly and the partial report is returned, so an
        interrupted run still reports what actually happened.
        """
        stops = {p.strip().lower() for p in (stop_phrases or ("stop", "band karo", "bas", "exit", "quit")) if p}
        report = ConversationReport(started_at=time.time())
        i = 0
        try:
            while max_turns <= 0 or i < max_turns:
                i += 1
                _t0 = time.perf_counter()
                transcript = self._listen()
                listen_ms = (time.perf_counter() - _t0) * 1000.0
                user_text = transcript.normalized()
                if not user_text:
                    rec = AssistantTurnRecord(
                        turn=i, heard="", said="", kind="silent", source="ears",
                        listen_error=transcript.error, listen_ms=round(listen_ms, 1),
                    )
                    report.turns.append(rec)
                    if on_record:
                        on_record(rec)
                    # A silent turn is normal between utterances; keep waiting.
                    continue

                if user_text.strip().lower() in stops:
                    rec = AssistantTurnRecord(
                        turn=i, heard=user_text, said="Theek hai sir, ruk gaya.",
                        kind="stop", source="ears", listen_ms=round(listen_ms, 1),
                    )
                    report.turns.append(rec)
                    if on_record:
                        on_record(rec)
                    break

                rec = self._run_one(user_text, i)
                rec.listen_ms = round(listen_ms, 1)
                report.turns.append(rec)
                if on_record:
                    on_record(rec)
                if rec.kind in ("finish", "stop"):
                    break

                # The user asked for something that needs work. Keep going on our
                # own — act, look, decide — until the brain says finish or the
                # step budget runs out, then go back to listening.
                if rec.action and rec.performed:
                    for j in range(1, max(0, self.max_task_steps) + 1):
                        i += 1
                        step_rec = self._run_one("", i)
                        report.turns.append(step_rec)
                        if on_record:
                            on_record(step_rec)
                        if step_rec.kind in ("finish", "stop"):
                            break
                        if step_rec.action and not step_rec.performed:
                            break
                        if self.step_delay_s > 0:
                            time.sleep(self.step_delay_s)
        except KeyboardInterrupt:
            pass
        finally:
            self.speaker.stop()
            report.ended_at = time.time()
        return report
