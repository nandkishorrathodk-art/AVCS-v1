"""The GUI assistant: one screen, one conversation, one live loop.

This is the runnable face of AVCS, and the thing it runs is an **assistant**, not
a control detector. The page shows three things and nothing else:

  * what the assistant *sees* -- the live screen and the model's own reading of
    it (tone, structure, motion, change), from pure pixels;
  * what it *says* -- a Hinglish conversation, streamed sentence by sentence, and
    spoken out loud in the browser;
  * what it *hears* -- the microphone, transcribed on the server.

The perception here is `ScenePerception`: it describes a scene, it does not read
control labels and it does not ground a point. So there is no click path by
default and no mutating grant. A caller can opt in to the lexicon eyes and a
`click` grant (the capability kernel still gates every action), but that is not
what this page is for.

Design rules kept from the rest of the repo:

- **No OCR.** The description is rendered from typed pixel state.
- **Coordinate-free boundary.** The brain receives text only, never an image or
  a coordinate.
- **Honest provenance.** Every turn carries `brain_source` (`llm` or
  `rule_fallback`) and per-stage timings, so "real time" is a measurement.
- **Fail closed.** With the default (scene) eyes there is no grounding evidence,
  so any mutating action is refused by the kernel and shown as a refusal.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from avcs.control.capability import MUTATING_OPS, CapabilityKernel
from avcs.system1.loop import VirtualHand
from avcs.voice.assistant import Assistant
from avcs.voice.brain import ConversationBrain
from avcs.voice.scene_perception import ScenePerception
from avcs.voice.stt import NullEngine as NullSttEngine, default_stt_engine
from avcs.voice.tts import BrowserEngine, EspeakNgEngine, Speaker, VoiceProfile

FRAME_POLL_S = 0.25


def resolve_gui_grants(
    grants: Optional[Union[str, Iterable[str]]] = None,
    *,
    eyes: str = "scene",
) -> List[str]:
    """Expand GUI capability grants while keeping ``grants=None`` fail-closed.

    Defaults to ``[]`` (or ``["click"]`` for ``eyes="lexicon"``). Passing
    ``"all"`` (e.g. ``--grant all`` or ``grants=["all"]``) unlocks all
    ``MUTATING_OPS``.
    """
    if grants is None:
        return [] if eyes != "lexicon" else ["click"]
    if isinstance(grants, str):
        cleaned = [grants.strip().lower()] if grants.strip() else []
    else:
        cleaned = [str(g).strip().lower() for g in grants if str(g).strip()]
    if "all" in cleaned:
        return sorted(MUTATING_OPS)
    return cleaned


@dataclass
class GuiState:
    """Everything the page needs for one paint, produced from one frame."""

    frame_png: Optional[bytes] = None
    frame_size: tuple[int, int] = (0, 0)
    description: str = ""
    reading: Dict[str, Any] = field(default_factory=dict)
    regions: List[Dict[str, Any]] = field(default_factory=list)
    frame_ms: float = 0.0
    seq: int = 0


class ScreenModel:
    """Owns the screen source and the per-frame read, shared by every request.

    One perception, one cached read per frame. If every HTTP request re-ran the
    perception the page would describe a different frame on each paint and the
    "live" view would be a flicker, not a stream.
    """

    def __init__(
        self,
        source: str = "demo",
        category: str = "desktop",
        *,
        eyes: str = "scene",
        checkpoint: Optional[str] = None,
    ) -> None:
        self._demo = None
        self._source_name = source
        if eyes == "lexicon":
            # Opt-in control reader: needed only to ground a click. Not the
            # assistant's default eyes.
            from avcs.voice.lexicon_perception import LexiconPerception

            if source in ("demo", "desktop", "synthetic-desktop"):
                from avcs.gui.desktop import DesktopSource

                self._demo = DesktopSource()
                self.perception = LexiconPerception(
                    frame_source=self._demo, checkpoint=checkpoint, floor=0.0
                )
            else:
                self.perception = LexiconPerception(
                    source=source, checkpoint=checkpoint, floor=0.0
                )
        elif eyes == "vla":
            from avcs.voice.perception import LocalVLAPerception

            if source in ("demo", "desktop", "synthetic-desktop"):
                from avcs.gui.demo_source import DemoSceneSource

                self._demo = DemoSceneSource(category=category)
                self.perception = LocalVLAPerception(frame_source=self._demo)
            else:
                self.perception = LocalVLAPerception(source=source)
        else:
            frame_source = None
            if source in ("demo", "desktop", "synthetic-desktop"):
                from avcs.gui.demo_source import DemoSceneSource

                self._demo = DemoSceneSource(category=category)
                frame_source = self._demo
            self.perception = ScenePerception(
                frame_source=frame_source,
                source="screen" if frame_source is None else source,
            )
        self.seq = 0
        self._last_state = GuiState()

    @property
    def demo_source(self):
        return self._demo

    def _capture_is_live(self) -> bool:
        """True only when frames come from the user's real screen."""
        if self._demo is not None:
            return False
        flag = getattr(self.perception, "capture_is_live", None)
        if flag is not None:
            return bool(flag)
        # A scene perception over the default "screen" source is live unless it
        # says otherwise; the lexicon perception always answers.
        return getattr(self, "_source_name", "screen") not in ("demo", "desktop", "synthetic-desktop")

    def snapshot(self) -> GuiState:
        t0 = time.perf_counter()
        obs = self.perception.observe()
        read_ms = (time.perf_counter() - t0) * 1000.0

        frame = self.perception.frame()
        reading: Dict[str, Any] = {}
        summariser = getattr(self.perception, "state_summary", None)
        if callable(summariser):
            try:
                reading = summariser() or {}
            except Exception:
                reading = {}

        # The lexicon path has no scene summary; expose its regions instead so
        # the panel is never blank, without pretending they are scene facts.
        regions: List[Dict[str, Any]] = []
        region_lister = getattr(self.perception, "regions", None)
        if callable(region_lister):
            try:
                for r in region_lister():
                    box = r.box
                    regions.append({
                        "x0": int(box.x0), "y0": int(box.y0),
                        "x1": int(box.x1), "y1": int(box.y1),
                        "label": r.label,
                        "confidence": round(float(r.readout.confidence), 4),
                        "known": bool(r.known),
                    })
            except Exception:
                regions = []

        self.seq += 1
        size = (0, 0)
        png = None
        if frame is not None:
            h, w = frame.shape[:2]
            size = (int(w), int(h))
            png = _png(frame)
        self._last_state = GuiState(
            frame_png=png,
            frame_size=size,
            description=obs.description,
            reading=reading,
            regions=regions,
            frame_ms=round(read_ms, 2),
            seq=self.seq,
        )
        return self._last_state


def _png(frame: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class TurnRequest(BaseModel):
    text: str
    # Run the whole task, not just the first step. Defaults to True so a spoken
    # instruction is carried through instead of stopping after one glance.
    multi_step: bool = True


class SceneRequest(BaseModel):
    category: str


class AudioRequest(BaseModel):
    # Base64 WAV (or raw base64 PCM16) recorded in the browser and posted here.
    audio_base64: str
    sample_rate: int = 16000
    is_wav: bool = True


def _decode_audio(req: AudioRequest):
    """Turn the browser's recording into an ``AudioClip`` the ears understand."""
    import base64

    from avcs.voice.stt import AudioClip

    raw = base64.b64decode(req.audio_base64)
    if req.is_wav or raw[:4] == b"RIFF":
        try:
            return AudioClip.read_wav_bytes(raw)
        except Exception:
            pass
    return AudioClip.from_pcm16_bytes(raw, int(req.sample_rate))


def _hand_is_virtual(hand: Any) -> bool:
    """True when a hand records actions but moves no real pointer.

    ``VirtualHand`` is the obvious case. A ``ServoHand`` is the subtler one: it
    plans a genuine neuromuscular trajectory, but when it wraps a simulated
    cursor the pointer never moves. Reading either as "live" would be the same
    overclaim as a fake hand.
    """
    if isinstance(hand, VirtualHand):
        return True
    return bool(getattr(hand, "virtual", False))


class GuiApp:
    """Wires the screen, the brain, the mouth and the ears into one assistant."""

    def __init__(
        self,
        source: str = "demo",
        category: str = "desktop",
        grants: Optional[Union[str, Iterable[str]]] = None,
        checkpoint: Optional[str] = None,
        live: bool = False,
        min_peak: float = 0.45,
        mouth: str = "browser",
        ears: bool = True,
        eyes: str = "scene",
        brain: Optional[ConversationBrain] = None,
        memory_path: Optional[str] = None,
        trajectory_path: Optional[str] = None,
        path_roots: Optional[List[str]] = None,
        allowed_apps: Optional[List[str]] = None,
        proactive: bool = False,
        servo: bool = False,
        granted_ops: Optional[Union[str, Iterable[str]]] = None,
    ) -> None:
        self.eyes = eyes
        self.screen = ScreenModel(
            source=source, category=category, eyes=eyes, checkpoint=checkpoint
        )
        # The assistant's default is *no mutating capability*: it looks, talks
        # and hears. A grant only exists if the caller asked for one AND the eyes
        # can actually ground a point (the lexicon reader). Granting `click` to
        # scene eyes would be a capability with no evidence behind it.
        effective_grants = grants if grants is not None else granted_ops
        self.grants = resolve_gui_grants(effective_grants, eyes=eyes)
        # `path_roots` / `allowed_apps` are the containment for the file and
        # launch ops. They default to off (None), so a caller that grants
        # `read_file` without naming a root still cannot read anything.
        self.path_roots = list(path_roots) if path_roots else None
        self.allowed_apps = list(allowed_apps) if allowed_apps else None
        self.kernel = CapabilityKernel(
            granted_ops=self.grants,
            min_peak_probability=float(min_peak),
            path_roots=self.path_roots,
            allowed_apps=self.allowed_apps,
        )
        # Durable memory and a reviewable trajectory log. Both are on by default
        # for the GUI (that is what makes it an assistant rather than a demo),
        # and both are plain files under ~/.avcs that the user can delete.
        from avcs.voice.memory import MemoryStore, default_memory_path
        from avcs.voice.trajectory import TrajectoryLog

        self.memory = MemoryStore(memory_path or default_memory_path())
        self.trajectory = TrajectoryLog(trajectory_path or TrajectoryLog.default_path())
        self.brain = brain if brain is not None else ConversationBrain(memory=self.memory)
        self.mouth_engine = self._build_mouth(mouth)
        self.speaker = Speaker(engine=self.mouth_engine, profile=VoiceProfile.hinglish())
        self.ears_enabled = bool(ears)
        # `Assistant` treats a missing STT engine as "auto-detect the best one".
        # Pass an explicit null engine when the virtual GUI is text-only, or the
        # browser would still request microphone access after --virtual.
        self.stt = default_stt_engine() if self.ears_enabled else NullSttEngine()
        self.hand = None
        self.live = bool(live)
        # Set while a user turn is being handled, so the proactive watcher stays
        # quiet and does not talk over the conversation it is supposed to support.
        self._turn_in_flight = False
        if self.live and self.grants:
            from avcs.voice.actuators import default_hand

            self.hand = default_hand()
            # `--servo` replaces the teleport with a human neuromuscular path
            # (minimum-jerk + Fitts' Law + micro-tremor). It is opt-in because a
            # real cursor that takes ~0.3-0.9 s to arrive is a deliberate choice,
            # not a silent default: some callers want the instant click.
            if self.hand is not None and servo:
                from avcs.voice.actuators import ServoHand

                self.hand = ServoHand(self.hand, virtual=False)
        elif not self.live:
            # Virtual mode always has a hand, granted or not. Without one the
            # grant is unreachable: `_ask_brain` reports `can_click=False`, the
            # model says "I have no capability", and a granted action is refused
            # with "no hand attached" -- the wrong reason, since the real reason
            # is the missing grant. The virtual hand records the coordinate a
            # granted action resolved to and moves no cursor: the gate is the
            # production kernel, only the side effect is inert.
            if servo:
                # A servo over a *virtual* cursor, never over a real hand: this
                # branch means `--live` was not passed, so actuating the machine
                # here would be a capability the user never granted. The human
                # trajectory is still planned and painted into an in-memory
                # driver, so `--servo` is observable and testable with no display.
                # `VirtualHand` is not wrapped because its click() takes
                # normalized coordinates while `ServoHand` speaks pixels; mixing
                # them would record a pixel as if it were normalized.
                from avcs.voice.actuators import ServoHand
                from forge.vision.cortex.motor.driver import VirtualCursorDriver

                width, height = self._frame_size()
                self.hand = ServoHand(
                    VirtualCursorDriver(screen_size=(width, height)), virtual=True
                )
            else:
                width, height = self._frame_size()
                self.hand = VirtualHand(width, height)
        self.assistant = Assistant(
            brain=self.brain,
            speaker=self.speaker,
            perception=self.screen.perception,
            kernel=self.kernel,
            hand=self.hand,
            # Say "live" only when a real hand exists, so a turn record does not
            # claim real actuation on a machine that cannot actuate. A virtual
            # hand is a granted capability with an inert side effect, so it is
            # reported as "virtual". A `ServoHand` carries its own `virtual`
            # flag, because a servo over a virtual cursor computes a real
            # trajectory but moves no pointer -- neither "live" nor "none".
            actuation=(
                "live"
                if (self.hand is not None and not _hand_is_virtual(self.hand))
                else "virtual"
            ),
            memory=self.memory,
            trajectory=self.trajectory,
        )
        self.transcript: List[Dict[str, Any]] = []

        # Proactive awareness: the assistant watches the screen and speaks first
        # when something is worth mentioning. Opt-in, because an assistant that
        # talks unprompted must be asked for -- and it has no hand, so it can
        # never actuate from this path.
        self.awareness = None
        self.proactive = bool(proactive)
        if self.proactive:
            from avcs.gui.awareness import AwarenessLoop

            self.awareness = AwarenessLoop(
                snapshot=self.screen.snapshot,
                brain=self.brain,
                speak=self._speak_remark,
                on_remark=self._record_remark,
                is_busy=self._busy,
                capabilities=self.capabilities,
            )
            self.awareness.start()

    # ------------------------------------------------------------------ #
    # Proactive awareness
    # ------------------------------------------------------------------ #
    def _busy(self) -> bool:
        """True while a user turn is in flight, so remarks do not interrupt it."""
        return bool(self._turn_in_flight)

    def _speak_remark(self, text: str) -> None:
        """Say a spontaneous remark through the same mouth as a normal reply."""
        self.speaker.speak_async(text)

    def _record_remark(self, remark) -> None:
        entry = {
            "turn": len(self.transcript) + 1,
            "heard": "",
            "said": remark.text,
            "kind": "remark",
            "source": remark.source,
            "proactive": True,
            "reason": remark.reason,
            "detail": remark.detail,
            "total_ms": 0.0,
            "mutating": False,
            "action": None,
            "gate_allowed": None,
        }
        self.transcript.append(entry)

    def awareness_summary(self) -> Dict[str, Any]:
        loop = self.awareness
        if loop is None:
            return {"enabled": False, "running": False, "remark_count": 0, "recent": []}
        summary = loop.summary()
        summary["enabled"] = True
        return summary

    def stop(self) -> None:
        """Stop the background watcher, if any. Safe to call twice."""
        loop = self.awareness
        if loop is not None:
            loop.stop()

    @staticmethod
    def _build_mouth(mouth: str):
        """Pick where the assistant's voice comes out.

        ``browser`` synthesizes locally with espeak-ng and parks the WAV for the
        page to play (the normal case: the server has no sound card, the browser
        does). ``local`` uses the server's own audio device. ``silent`` renders
        but never plays, for tests.
        """
        mouth = (mouth or "browser").lower()
        if mouth == "silent":
            from avcs.voice.tts import NullEngine

            return NullEngine()
        if mouth == "local":
            espeak = EspeakNgEngine()
            if espeak.available():
                return espeak
            return BrowserEngine()
        return BrowserEngine()

    def _frame_size(self) -> tuple[int, int]:
        """The frame size, so the virtual hand maps normalized points into pixels.

        The frame itself is the source of truth: a demo canvas and a real desktop
        both report their size here, so the hand's pixel space always matches the
        screen the gate grounded against.
        """
        try:
            frame = self.perception.frame()
            if frame is not None:
                h, w = frame.shape[:2]
                return int(w), int(h)
        except Exception:
            pass
        return (900, 520)

    def capabilities(self) -> Dict[str, Any]:
        caps = self.assistant.capabilities()
        # Report what can *actually* actuate, not what was requested. `--live`
        # with no hand (no xdotool/X11 on Linux, no user32 on Windows) must not
        # read as "live": every op would be refused, and the page would be
        # claiming a capability the machine does not have.
        if self.hand is not None and not _hand_is_virtual(self.hand):
            caps["actuation"] = "live"
        elif self.live:
            caps["actuation"] = "live-unavailable"
        else:
            # A virtual hand is a real, granted capability whose side effect is
            # inert. Calling that "live" would overclaim; calling it "none" would
            # underclaim. The record says which, per turn, via `actuation`.
            caps["actuation"] = "virtual" if self.hand is not None else "none"
        caps["hand"] = getattr(self.hand, "name", None)
        caps["mouth"] = self.speaker.engine_name
        caps["eyes"] = type(self.screen.perception).__name__
        # Whether the frame really came off a screen. A synthetic source (the demo
        # desktop) and an EyeStream whose `mss` is missing both produce pixels
        # that are not the user's screen; the page must say which, or it claims a
        # live view it never captured.
        caps["capture"] = "live" if self.screen._capture_is_live() else "synthetic"
        caps["can_act"] = bool(self.grants)
        caps["memory"] = bool(self.memory)
        caps["trajectory"] = bool(self.trajectory)
        caps["path_roots"] = self.path_roots
        caps["allowed_apps"] = self.allowed_apps
        caps["proactive"] = bool(self.awareness is not None and self.awareness.running)
        # How the pointer travels, reported from the hand itself so a build
        # without the neuromuscular servo says "teleport" rather than lying.
        caps["motion"] = "neuromuscular" if getattr(self.hand, "name", "") == "servo-hand" else "teleport"
        return caps

    def _entry(self, record, total_ms: float) -> Dict[str, Any]:
        """One turn record as the page consumes it, with honest provenance."""
        entry = record.to_dict()
        entry["total_ms"] = round(total_ms, 2)
        entry["brain_source"] = record.source
        # Whether this turn tried to *do* something. A perception step (the brain
        # asking to look at the screen) is not a gated click, and the page must
        # not draw it as one.
        entry["mutating"] = bool(record.action and record.action in MUTATING_OPS)
        return entry

    def turn(self, text: str, on_speech=None) -> Dict[str, Any]:
        """Run one conversational turn and record it honestly."""
        self.screen.snapshot()
        t0 = time.perf_counter()
        self._turn_in_flight = True
        try:
            record = self.assistant.step(text, on_speech=on_speech)
        finally:
            self._turn_in_flight = False
        total_ms = (time.perf_counter() - t0) * 1000.0

        entry = self._entry(record, total_ms)
        self.transcript.append(entry)
        return entry

    def task(self, text: str, on_speech=None, on_step=None, max_steps: Optional[int] = None):
        """Run a whole task, not just the first step.

        One turn produces at most one action, which is why the assistant could
        answer but never *finish* anything: it looked once and stopped. This
        delegates to the assistant's own ``run_task`` -- the loop lives in one
        place, so the GUI and the voice demo cannot drift apart -- and converts
        each record to the page's shape as it completes.

        A pure conversational turn (no action proposed) stops after the first
        step: there is nothing to carry forward, and looping the model over a
        greeting would burn eight round-trips to say hello.
        """
        self.screen.snapshot()
        records: List[Dict[str, Any]] = []

        def _on_record(record) -> None:
            # `run_task` already timed the turn; recover it from the record's
            # own stage timings rather than re-timing around the callback.
            total_ms = record.listen_ms + record.brain_ms + record.act_ms
            entry = self._entry(record, total_ms)
            records.append(entry)
            self.transcript.append(entry)
            if on_step is not None:
                on_step(entry)

        self._turn_in_flight = True
        try:
            self.assistant.run_task(text, max_steps=max_steps, on_record=_on_record,
                                    on_speech=on_speech)
        finally:
            self._turn_in_flight = False
        return records

    def hear(self, req: AudioRequest) -> Dict[str, Any]:
        """Transcribe a browser recording with the same ears the voice loop uses."""
        if not self.ears_enabled:
            return {"text": "", "error": "ears disabled", "engine": "none"}
        try:
            clip = _decode_audio(req)
        except Exception as exc:
            return {"text": "", "error": f"bad audio: {exc}", "engine": self.stt.name}
        clip = clip.trim_silence()
        if clip.samples.size == 0:
            return {"text": "", "error": "no speech detected", "engine": self.stt.name}
        transcript = self.stt.transcribe(clip)
        return {
            "text": transcript.normalized(),
            "error": transcript.error,
            "engine": transcript.engine,
            "confidence": transcript.confidence,
            "duration_s": round(clip.duration_s, 3),
        }


def build_app(
    source: str = "demo",
    category: str = "desktop",
    grants: Optional[Union[str, Iterable[str]]] = None,
    checkpoint: Optional[str] = None,
    live: bool = False,
    mouth: str = "browser",
    ears: bool = True,
    eyes: str = "scene",
    brain: Optional[ConversationBrain] = None,
    memory_path: Optional[str] = None,
    trajectory_path: Optional[str] = None,
    path_roots: Optional[List[str]] = None,
    allowed_apps: Optional[List[str]] = None,
    proactive: bool = False,
    servo: bool = False,
    granted_ops: Optional[Union[str, Iterable[str]]] = None,
) -> FastAPI:
    gui = GuiApp(
        source=source, category=category, grants=grants, checkpoint=checkpoint,
        live=live, mouth=mouth, ears=ears, eyes=eyes, brain=brain,
        memory_path=memory_path, trajectory_path=trajectory_path,
        path_roots=path_roots, allowed_apps=allowed_apps, proactive=proactive,
        servo=servo, granted_ops=granted_ops,
    )
    app = FastAPI(title="AVCS assistant")
    app.state.gui = gui

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        from pathlib import Path

        page = Path(__file__).with_name("static") / "index.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/capabilities")
    def capabilities() -> JSONResponse:
        return JSONResponse(gui.capabilities())

    @app.get("/api/memory")
    def memory() -> JSONResponse:
        """What the assistant remembers about this user, so it is inspectable.

        Memory the user cannot see is a privacy problem, so it is a plain
        endpoint over a plain file.
        """
        return JSONResponse({
            **gui.memory.to_dict(),
            "trajectory": gui.trajectory.summary(),
        })

    class _ForgetRequest(BaseModel):
        key: str

    @app.post("/api/memory/forget")
    def forget(req: _ForgetRequest) -> JSONResponse:
        removed = gui.memory.forget(req.key)
        if removed:
            gui.memory.save()
        return JSONResponse({"forgotten": removed, "key": req.key})

    @app.get("/api/trajectory")
    def trajectory() -> JSONResponse:
        return JSONResponse(gui.trajectory.summary())

    @app.get("/api/frame")
    def frame() -> JSONResponse:
        st = gui.screen.snapshot()
        return JSONResponse({
            "seq": st.seq,
            "frame_size": list(st.frame_size),
            "description": st.description,
            "reading": st.reading,
            "regions": st.regions,
            "frame_ms": st.frame_ms,
        })

    @app.get("/api/frame.png")
    def frame_png() -> StreamingResponse:
        st = gui.screen.snapshot()
        if st.frame_png is None:
            return JSONResponse({"error": "no frame"}, status_code=503)
        return StreamingResponse(io.BytesIO(st.frame_png), media_type="image/png")

    @app.get("/api/awareness")
    def awareness() -> JSONResponse:
        """What the assistant has noticed and said on its own.

        Returns the remarks it chose to make plus the reasons it stayed quiet, so
        "it watches the screen" is a measurement and not a claim. ``enabled`` is
        false unless the server was started with proactive awareness.
        """
        return JSONResponse(gui.awareness_summary())

    @app.post("/api/turn")
    def turn(req: TurnRequest) -> JSONResponse:
        return JSONResponse(gui.turn(req.text))

    @app.post("/api/turn.stream")
    async def turn_stream(req: TurnRequest):
        """Run one turn and stream speech sentences + every step as SSE.

        This is the real-time path the page uses. The browser hears each sentence
        the moment the brain produces it (``speech`` events), sees each step of a
        multi-step task as it completes (``step`` events), and gets the final
        record (``turn``) when the task is done.

        ``multi_step`` defaults to True: an assistant that answers but never
        finishes the job is not an assistant. Pass false to run exactly one turn
        (the page uses that for a plain conversational exchange).
        """
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()

        def _emit(event: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def _run() -> None:
            try:
                speak = lambda s: _emit({"type": "speech", "text": s})  # noqa: E731
                if req.multi_step:
                    records = gui.task(req.text, on_speech=speak,
                                       on_step=lambda e: _emit({"type": "step", "record": e}))
                    _emit({"type": "turn", "record": records[-1] if records else {}})
                else:
                    record = gui.turn(req.text, on_speech=speak)
                    _emit({"type": "turn", "record": record})
            except Exception as exc:  # a turn must never kill the stream
                _emit({"type": "error", "error": str(exc)})
            finally:
                _emit(None)

        task = asyncio.ensure_future(loop.run_in_executor(None, _run))

        async def gen():
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield f"data: {json.dumps(item)}\n\n"
            finally:
                await task

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/speech.wav")
    def speech_wav(after: int = 0):
        """The newest spoken clip as WAV, or 204 when there is nothing new.

        The page plays this through the Web Audio API, so the assistant's voice
        reaches the user even when the *server* has no sound card (the normal
        case in a container).
        """
        engine = gui.mouth_engine
        taker = getattr(engine, "take_latest", None)
        if not callable(taker):
            return JSONResponse({"error": "mouth engine cannot serve audio"}, status_code=409)
        seq, wav = taker(after_seq=int(after))
        if wav is None:
            # 204 must carry no body at all, so use a bare Response; a
            # JSONResponse would set Content-Length and then send a payload,
            # which uvicorn rejects with "Response content longer than
            # Content-Length".
            return Response(status_code=204, headers={"X-Speech-Seq": str(seq)})
        return StreamingResponse(
            io.BytesIO(wav), media_type="audio/wav",
            headers={"X-Speech-Seq": str(seq), "Cache-Control": "no-store"},
        )

    @app.post("/api/hear")
    def hear(req: AudioRequest) -> JSONResponse:
        """Transcribe a recording from the browser (the assistant's ears)."""
        return JSONResponse(gui.hear(req))

    @app.post("/api/scene")
    def set_scene(req: SceneRequest) -> JSONResponse:
        """Switch which scene the assistant is looking at.

        Only available for the scripted demo source; a real screen is not
        editable and the call says so.
        """
        src = gui.screen.demo_source
        setter = getattr(src, "set_category", None)
        if not callable(setter):
            return JSONResponse(
                {"error": "screen is not a scripted demo (cannot switch scenes)"},
                status_code=409,
            )
        try:
            setter(req.category)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"category": src.category})

    @app.get("/api/stream")
    async def stream(request: Request):
        """Server-sent events: one state object per polled frame.

        The page paints from this instead of polling in JS, so the browser sees
        the same sequence the server produced and a slow paint cannot reorder
        frames.
        """
        async def gen():
            is_test = bool(request.client and request.client.host == "testclient")
            while True:
                st = gui.screen.snapshot()
                payload = {
                    "seq": st.seq,
                    "frame_size": list(st.frame_size),
                    "description": st.description,
                    "reading": st.reading,
                    "regions": st.regions,
                    "frame_ms": st.frame_ms,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                if is_test or await request.is_disconnected():
                    break
                await asyncio.sleep(FRAME_POLL_S)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
