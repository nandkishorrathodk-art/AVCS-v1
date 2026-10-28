"""The GUI model: one screen, one conversation, one live loop.

This is the runnable face of AVCS. It puts the pieces that already exist behind
one browser page and shows what each one is doing, in real time:

  screen  ->  lexicon reader  ->  named regions  ->  brain  ->  kernel  ->  hand
             (pixels, no OCR)      (this is new)     (LLM or      (fail-     (virtual
                                                     fallback)    closed)     by default)

The screen panel is the *model's* view, not a screenshot with boxes drawn on it
by hand: the boxes come from `LexiconReader.read_regions`, the same call the
gate uses. So a box on the page is evidence the reader saw that control.

Design rules this server keeps, matching the rest of the repo:

- **No OCR.** Region labels come from glyph-shape matching against a closed
  vocabulary. Nothing here transcribes characters.
- **Fail closed.** The kernel decides every action. A refusal is shown as a
  refusal, with the kernel's own reason, and is never dressed up as success.
- **Honest provenance.** Every turn carries `brain_source` (`llm` or
  `rule_fallback`), `gate_allowed`, and per-stage timings, so "real time" is a
  measurement rather than a claim.
- **Virtual by default.** The hand records coordinates and moves no cursor
  unless the server was started with live actuation explicitly enabled.
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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from avcs.control.capability import CapabilityKernel
from avcs.gui.desktop import HEIGHT, WIDTH
from avcs.voice.assistant import Assistant
from avcs.voice.brain import ConversationBrain
from avcs.voice.lexicon_perception import LexiconPerception

FRAME_POLL_S = 0.25


@dataclass
class GuiState:
    """Everything the page needs for one paint, produced from one frame."""

    frame_png: Optional[bytes] = None
    frame_size: tuple[int, int] = (0, 0)
    regions: List[Dict[str, Any]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    frame_ms: float = 0.0
    seq: int = 0


class ScreenModel:
    """Owns the screen source and the per-frame read, shared by every request.

    One reader, one cached read per frame. If every HTTP request re-ran the
    encoder the page would show a different region set on each paint and the
    "live" view would be a flicker, not a stream.
    """

    def __init__(self, source: str = "synthetic", checkpoint: Optional[str] = None) -> None:
        if source in ("desktop", "synthetic-desktop"):
            # A synthetic desktop with real controls in it, so the reader has
            # something to read. The flat gray EyeStream buffer has none.
            from avcs.gui.desktop import DesktopSource

            self._desktop = DesktopSource()
            self.perception = LexiconPerception(
                frame_source=self._desktop,
                checkpoint=checkpoint,
                floor=0.0,
            )
        else:
            self._desktop = None
            self.perception = LexiconPerception(
                source=source,
                checkpoint=checkpoint,
                floor=0.0,
            )
        self.seq = 0
        self._last_state = GuiState()

    def snapshot(self) -> GuiState:
        t0 = time.perf_counter()
        frame = self.perception.grab()
        grab_ms = (time.perf_counter() - t0) * 1000.0
        if frame is None:
            self._last_state = GuiState(frame_ms=grab_ms, seq=self.seq)
            return self._last_state

        regions = []
        for r in self.perception.regions():
            box = r.box
            regions.append({
                "x0": int(box.x0), "y0": int(box.y0),
                "x1": int(box.x1), "y1": int(box.y1),
                "label": r.label,
                "confidence": round(float(r.readout.confidence), 4),
                "known": bool(r.known),
            })

        h, w = frame.shape[:2]
        self.seq += 1
        self._last_state = GuiState(
            frame_png=_png(frame),
            frame_size=(int(w), int(h)),
            regions=regions,
            labels=sorted({r["label"] for r in regions}),
            frame_ms=round(grab_ms, 2),
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


class ScreenRequest(BaseModel):
    buttons: List[str]


class GuiApp:
    """Wires the screen, the brain, the kernel and the hand into one object."""

    def __init__(
        self,
        source: str = "synthetic",
        grants: Optional[List[str]] = None,
        checkpoint: Optional[str] = None,
        live: bool = False,
        min_peak: float = 0.45,
    ) -> None:
        self.screen = ScreenModel(source=source, checkpoint=checkpoint)
        self.grants = list(grants or ["click"])
        self.kernel = CapabilityKernel(
            granted_ops=self.grants,
            min_peak_probability=float(min_peak),
        )
        self.brain = ConversationBrain()
        self.hand = None
        self.live = bool(live)
        if self.live:
            from avcs.voice.actuators import default_hand

            self.hand = default_hand()
        else:
            # A virtual hand records the coordinates a granted click resolved to
            # and touches nothing. Attaching it (rather than leaving the hand
            # None) means an authorized click is *shown* to have happened, at a
            # real point, while the machine stays untouched.
            from avcs.system1.loop import VirtualHand

            self.hand = VirtualHand(width=WIDTH, height=HEIGHT)
        self.assistant = Assistant(
            brain=self.brain,
            perception=self.screen.perception,
            kernel=self.kernel,
            hand=self.hand,
        )
        self.transcript: List[Dict[str, Any]] = []

    def capabilities(self) -> Dict[str, Any]:
        caps = self.assistant.capabilities()
        caps["actuation"] = "live" if self.live else "virtual"
        eye = getattr(self.screen.perception, "_eye", None)
        caps["source"] = type(eye).__name__ if eye is not None else "fixture"
        return caps

    def turn(self, text: str) -> Dict[str, Any]:
        """Run one conversational turn and record it honestly."""
        self.screen.snapshot()
        t0 = time.perf_counter()
        record = self.assistant.step(text)
        total_ms = (time.perf_counter() - t0) * 1000.0

        entry = record.to_dict()
        entry["total_ms"] = round(total_ms, 2)
        entry["brain_source"] = record.source
        self.transcript.append(entry)
        return entry


def build_app(
    source: str = "synthetic",
    grants: Optional[List[str]] = None,
    checkpoint: Optional[str] = None,
    live: bool = False,
) -> FastAPI:
    gui = GuiApp(source=source, grants=grants, checkpoint=checkpoint, live=live)
    app = FastAPI(title="AVCS GUI model")
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
            "regions": st.regions,
            "labels": st.labels,
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

    @app.post("/api/screen")
    def set_screen(req: ScreenRequest) -> JSONResponse:
        """Change which controls are on the synthetic desktop.

        This is how the "target absent" path is shown rather than described: put
        a control on screen, ask for it (allowed), take it away, ask again
        (refused). Only available for the synthetic desktop; a real screen is not
        editable and the call says so.
        """
        desktop = getattr(gui.screen, "_desktop", None)
        if desktop is None:
            return JSONResponse(
                {"error": "screen is not editable (not a synthetic desktop)"},
                status_code=409,
            )
        desktop.set_buttons(req.buttons)
        return JSONResponse({"buttons": list(desktop.buttons)})

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
                    "regions": st.regions,
                    "labels": st.labels,
                    "frame_ms": st.frame_ms,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(FRAME_POLL_S)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
