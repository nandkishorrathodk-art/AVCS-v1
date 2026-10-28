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
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from avcs.control.capability import MUTATING_OPS, CapabilityKernel
from avcs.voice.assistant import Assistant
from avcs.voice.brain import ConversationBrain
from avcs.voice.scene_perception import ScenePerception
from avcs.voice.stt import default_stt_engine
from avcs.voice.tts import BrowserEngine, EspeakNgEngine, Speaker, VoiceProfile

FRAME_POLL_S = 0.25


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


class GuiApp:
    """Wires the screen, the brain, the mouth and the ears into one assistant."""

    def __init__(
        self,
        source: str = "demo",
        category: str = "desktop",
        grants: Optional[List[str]] = None,
        checkpoint: Optional[str] = None,
        live: bool = False,
        min_peak: float = 0.45,
        mouth: str = "browser",
        ears: bool = True,
        eyes: str = "scene",
        brain: Optional[ConversationBrain] = None,
    ) -> None:
        self.eyes = eyes
        self.screen = ScreenModel(
            source=source, category=category, eyes=eyes, checkpoint=checkpoint
        )
        # The assistant's default is *no mutating capability*: it looks, talks
        # and hears. A grant only exists if the caller asked for one AND the eyes
        # can actually ground a point (the lexicon reader). Granting `click` to
        # scene eyes would be a capability with no evidence behind it.
        default_grants: List[str] = [] if eyes != "lexicon" else ["click"]
        self.grants = list(grants if grants is not None else default_grants)
        self.kernel = CapabilityKernel(
            granted_ops=self.grants,
            min_peak_probability=float(min_peak),
        )
        self.brain = brain if brain is not None else ConversationBrain()
        self.mouth_engine = self._build_mouth(mouth)
        self.speaker = Speaker(engine=self.mouth_engine, profile=VoiceProfile.hinglish())
        self.stt = default_stt_engine() if ears else None
        self.hand = None
        self.live = bool(live)
        if self.live and self.grants:
            from avcs.voice.actuators import default_hand

            self.hand = default_hand()
        self.assistant = Assistant(
            brain=self.brain,
            speaker=self.speaker,
            perception=self.screen.perception,
            kernel=self.kernel,
            hand=self.hand,
        )
        self.transcript: List[Dict[str, Any]] = []

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

    def capabilities(self) -> Dict[str, Any]:
        caps = self.assistant.capabilities()
        caps["actuation"] = "live" if self.live and self.grants else "virtual"
        caps["mouth"] = self.speaker.engine_name
        caps["eyes"] = type(self.screen.perception).__name__
        caps["can_act"] = bool(self.grants)
        return caps

    def turn(self, text: str, on_speech=None) -> Dict[str, Any]:
        """Run one conversational turn and record it honestly."""
        self.screen.snapshot()
        t0 = time.perf_counter()
        record = self.assistant.step(text, on_speech=on_speech)
        total_ms = (time.perf_counter() - t0) * 1000.0

        entry = record.to_dict()
        entry["total_ms"] = round(total_ms, 2)
        entry["brain_source"] = record.source
        # Whether this turn tried to *do* something. A perception step (the brain
        # asking to look at the screen) is not a gated click, and the page must
        # not draw it as one.
        entry["mutating"] = bool(record.action and record.action in MUTATING_OPS)
        self.transcript.append(entry)
        return entry

    def hear(self, req: AudioRequest) -> Dict[str, Any]:
        """Transcribe a browser recording with the same ears the voice loop uses."""
        if self.stt is None:
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
    grants: Optional[List[str]] = None,
    checkpoint: Optional[str] = None,
    live: bool = False,
    mouth: str = "browser",
    ears: bool = True,
    eyes: str = "scene",
    brain: Optional[ConversationBrain] = None,
) -> FastAPI:
    gui = GuiApp(
        source=source, category=category, grants=grants, checkpoint=checkpoint,
        live=live, mouth=mouth, ears=ears, eyes=eyes, brain=brain,
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

    @app.post("/api/turn")
    def turn(req: TurnRequest) -> JSONResponse:
        return JSONResponse(gui.turn(req.text))

    @app.post("/api/turn.stream")
    async def turn_stream(req: TurnRequest):
        """Run one turn and stream speech sentences + the final record as SSE.

        This is the real-time path the page uses. The browser hears each sentence
        the moment the brain produces it (``speech`` events), and gets the final
        record (``turn``) when the turn is complete.
        """
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()

        def _emit(event: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def _run() -> None:
            try:
                record = gui.turn(req.text, on_speech=lambda s: _emit({"type": "speech", "text": s}))
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
    async def stream():
        """Server-sent events: one state object per polled frame.

        The page paints from this instead of polling in JS, so the browser sees
        the same sequence the server produced and a slow paint cannot reorder
        frames.
        """
        async def gen():
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
                await asyncio.sleep(FRAME_POLL_S)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
