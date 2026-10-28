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

    def to_brain_dict(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"interactive_elements": self.hotspot_count}
        if self.last_action:
            extra["last_action"] = self.last_action
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
        # Streaming makes the assistant talk as the reply is generated. It is on
        # by default because that is the difference between a live conversation
        # and a walkie-talkie; a caller can turn it off for a deterministic demo.
        self.stream = bool(stream)

    # ------------------------------------------------------------------ #
    # Capabilities (reported honestly)
    # ------------------------------------------------------------------ #
    def capabilities(self) -> Dict[str, Any]:
        return {
            "brain": type(self.brain.provider).__name__,
            "brain_has_key": bool(getattr(self.brain.provider, "api_key", "")),
            "mouth": self.speaker.engine_name,
            "mouth_available": self.speaker.available(),
            "ears": self.stt.name,
            "ears_available": self.stt.available(),
            "eyes": type(self.perception).__name__ if self.perception else "none",
            "mutating_grants": sorted(self.kernel.granted_ops),
            "actuation": self.actuation,
        }

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
        # Close the loop for whatever the next step decides.
        self._last_outcome = self._outcome_summary(record)
        return record

    def run_task(self, user_text: str, max_steps: Optional[int] = None,
                 on_record: Optional[Callable[[AssistantTurnRecord], None]] = None) -> List[AssistantTurnRecord]:
        """Autonomously execute a whole task, not just one action.

        A single conversational turn produces at most one action. A real
        assistant keeps going: act, look at the result, decide the next step,
        repeat, until the brain says ``finish``/``stop`` or the budget runs out.

        The loop stops early — and says so — when an action is refused by the
        gate or does not reach the hand, because continuing would just repeat the
        same failing step. Speech is streamed on every step, so the assistant
        narrates the work while it does it.
        """
        budget = int(max_steps if max_steps is not None else self.max_task_steps)
        records: List[AssistantTurnRecord] = []

        for i in range(1, budget + 1):
            record = self._run_one(user_text, i)
            records.append(record)
            if on_record:
                on_record(record)

            if record.kind in ("finish", "stop"):
                break
            if record.action and not record.performed:
                break
            # From here the screen is the speaker: the next decision is driven by
            # what changed, not by the user, so the user text is not repeated.
            user_text = ""
            if self.step_delay_s > 0:
                time.sleep(self.step_delay_s)

        return records

    @staticmethod
    def _outcome_summary(record: AssistantTurnRecord) -> str:
        """One line describing what the last step actually did, for the brain."""
        if not record.action:
            return ""
        if not record.gate_allowed:
            return f"last step {record.action} was REFUSED by the capability gate ({record.gate_reason})"
        if not record.performed:
            return f"last step {record.action} did not reach the hand ({record.gate_reason})"
        change = "screen changed" if record.verified else "no visible change"
        return f"last step performed {record.action}, {change}"

    def _ask_brain(
        self,
        user_text: str,
        obs: ScreenObservation,
        use_stream: bool,
        on_speech: Optional[Callable[[str], None]] = None,
    ) -> tuple[AssistantTurn, bool, Optional[str]]:
        """Send one turn to the brain and return (turn, spoken, speak_error)."""
        if use_stream and hasattr(self.brain, "respond_stream"):
            # The mouth runs on a background thread, so the first sentence is
            # already audible while the rest of the reply is still being written.
            queue = SpeechQueue(self.speaker)

            def _say(sentence: str) -> None:
                if on_speech is not None:
                    on_speech(sentence)
                queue.say(sentence)

            turn_result = self.brain.respond_stream(
                user_text, obs.to_brain_dict(), on_speech=_say
            )
            results = queue.drain()
            spoken = any(r.spoken for r in results)
            speak_error = next((r.error for r in results if r.error), None)
        else:
            turn_result = self.brain.respond(user_text, obs.to_brain_dict())
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
        decision = self.kernel.authorize(
            op=op, query=intent.target_description, evidence=evidence, frame=frame
        )
        record.gate_allowed = decision.allowed
        record.gate_reason = decision.reason

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
            text=intent.text_input,
            key_name=intent.key_name,
            scroll_delta=intent.scroll_delta,
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
