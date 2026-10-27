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
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

from avcs.control.capability import CapabilityKernel, GroundingEvidence
from avcs.voice.brain import AssistantTurn, ConversationBrain, TurnKind
from avcs.voice.stt import AudioClip, SttEngine, Transcript, default_stt_engine
from avcs.voice.tts import Speaker, SpeechQueue, SpeechResult

# Ops that change machine state; everything else is free to observe.
MUTATING_OPS = frozenset({"click", "double_click", "type", "type_text", "press_key", "scroll"})


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

    def to_brain_dict(self) -> Dict[str, Any]:
        return ConversationBrain.build_observation(
            screen_description=self.description,
            target_present=self.target_present,
            stability=self.stability,
            has_modal=self.has_modal,
            active_window=self.active_window,
            extra={"interactive_elements": self.hotspot_count},
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
    clicked: Optional[tuple[int, int]] = None
    verified: Optional[bool] = None
    spoken: bool = False
    listen_error: Optional[str] = None
    speak_error: Optional[str] = None

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
            return self.perception.observe(query)
        except Exception as exc:
            return ScreenObservation(description=f"perception error: {exc}")

    def step(self, user_text: str, turn: int = 0, stream: Optional[bool] = None) -> AssistantTurnRecord:
        """Run one conversational turn from already-transcribed user text."""
        # First observation: morphology only, no query. The brain needs a general
        # "what is on screen" summary before it can decide what to do.
        obs = self._observe(None)
        use_stream = self.stream if stream is None else bool(stream)

        spoken = False
        speak_error: Optional[str] = None
        if use_stream and hasattr(self.brain, "respond_stream"):
            # The mouth runs on a background thread, so the first sentence is
            # already audible while the rest of the reply is still being written.
            queue = SpeechQueue(self.speaker)
            turn_result = self.brain.respond_stream(
                user_text, obs.to_brain_dict(), on_speech=queue.say
            )
            results = queue.drain()
            spoken = any(r.spoken for r in results)
            speak_error = next((r.error for r in results if r.error), None)
        else:
            turn_result = self.brain.respond(user_text, obs.to_brain_dict())
            result = self._speak(turn_result.speech)
            spoken, speak_error = result.spoken, result.error

        record = AssistantTurnRecord(
            turn=turn,
            heard=user_text,
            said=turn_result.speech,
            kind=turn_result.kind.value,
            source=turn_result.source,
            spoken=spoken,
            speak_error=speak_error,
        )

        if turn_result.wants_action and turn_result.intent is not None:
            # Now that the brain has named a target, ask perception to ground
            # *that target* (not the raw utterance) before gating the action.
            self._observe(turn_result.intent.target_description)
            self._act(turn_result, record)
        return record

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
        try:
            clicked = self.hand.click(*point)
            record.clicked = tuple(clicked)
            record.actuation = self.actuation
        except Exception as exc:
            record.actuation = "none"
            record.gate_reason = f"hand raised: {exc}"

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
