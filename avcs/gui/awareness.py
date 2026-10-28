"""Proactive awareness: the assistant notices the screen and speaks first.

An assistant that only answers when spoken to is a command line with a voice. A
person sitting next to you watches what you are doing and mentions things -- a
dialog that wants an answer, a screen that changed to something else, a moment
where you look stuck. This module is that half of the behaviour.

How it decides to speak
-----------------------
It never runs a model on every frame. The cheap pixel reflex already computes the
signals that matter, so the watcher reads them and applies a *reason* when one
crosses a threshold:

  * ``overlay``      -- something small and high-contrast appeared over the scene
                        (a dialog, a popup). The most worth-interrupting signal.
  * ``scene_change`` -- the whole screen changed at once (new view or app switch).
  * ``appear``       -- something new and prominent appeared in one place.
  * ``stuck``        -- local motion repeating in one spot with no whole-screen
                        change for a long time: the user is probably struggling.

Everything else is ignored on purpose. Narrating a moving mouse is noise, and a
noisy assistant is worse than a silent one. Each accepted reason then faces two
more gates: a **cooldown** (never speak twice within N seconds) and a
**deduplication** window (do not repeat the same reason for the same screen).

What it never does
------------------
It has no hand. There is no code path from this module to an actuation: it can
speak, and that is all. A watcher that could also click would be the clicker the
product is explicitly not.

Honest limits
-------------
The signals are coarse pixel statistics, not understanding. ``overlay`` is a
bright small region, not "an error dialog about your file" -- the brain turns the
observation into words, and if the brain is the offline rule fallback it will
speak a generic sentence. Thresholds are defaults on this box's synthetic demo,
not calibrated across applications; ``AwarenessConfig`` exists so a caller can
retune rather than trusting the constants.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AwarenessConfig:
    """Thresholds for when the watcher is allowed to interrupt.

    Defaults are deliberately conservative: a remark is an interruption, and a
    wrong interruption is worse than none.
    """

    poll_s: float = 0.75
    cooldown_s: float = 20.0
    # The same reason is not repeated for this long, even past the cooldown.
    repeat_s: float = 120.0
    # Overlays are the one signal worth interrupting for at a low bar.
    min_overlay_change: float = 0.004
    # A whole-screen change must be a real change, not a repaint.
    min_scene_change: float = 0.30
    # New prominent region.
    min_appear_novelty: float = 0.45
    min_appear_salience: float = 0.55
    # "Looks stuck": motion in one place, no scene change, for this long.
    stuck_after_s: float = 45.0
    min_stuck_motion: float = 0.12
    # A remark longer than this is not a remark.
    max_speech_chars: int = 240


@dataclass
class Remark:
    """One spontaneous thing the assistant said, with its provenance."""

    at: float
    reason: str
    text: str
    source: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "at": round(self.at, 3),
            "reason": self.reason,
            "text": self.text,
            "source": self.source,
            "detail": self.detail,
        }


class AwarenessLoop:
    """Watches the screen and lets the brain speak unprompted.

    The loop is a daemon thread, so a caller that forgets to ``stop()`` it cannot
    hang the process. It is opt-in: nothing constructs one unless asked, and with
    no brain it does nothing rather than guessing.
    """

    def __init__(
        self,
        *,
        snapshot: Callable[[], Any],
        brain: Any,
        speak: Callable[[str], Any],
        on_remark: Optional[Callable[[Remark], None]] = None,
        is_busy: Optional[Callable[[], bool]] = None,
        config: Optional[AwarenessConfig] = None,
        capabilities: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self._snapshot = snapshot
        self._brain = brain
        self._speak = speak
        self._on_remark = on_remark
        # `is_busy` lets the owner suppress remarks while the user is talking to
        # the assistant. Interrupting a conversation to describe the screen is
        # exactly the wrong moment.
        self._is_busy = is_busy or (lambda: False)
        self._capabilities = capabilities or (lambda: {})
        self.config = config or AwarenessConfig()

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_spoke_at = 0.0
        self._last_reason_at: Dict[str, float] = {}
        self._motion_since = 0.0
        self._last_scene_seq = -1
        self.remarks: List[Remark] = []
        self.skipped: Dict[str, int] = {}
        self.error: str = ""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "AwarenessLoop":
        if self.running:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="avcs-awareness", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        while not self._stop.wait(self.config.poll_s):
            try:
                self.tick()
            except Exception as exc:  # a watcher must never kill the server
                self.error = str(exc)

    def tick(self) -> Optional[Remark]:
        """One observation cycle. Returns the remark if one was spoken."""
        state = self._snapshot()
        reading = getattr(state, "reading", None) or {}
        reason, detail = self._reason(reading)
        if reason is None:
            return None
        if self._is_busy():
            self._note_skip("busy")
            return None
        now = time.time()
        if now - self._last_spoke_at < self.config.cooldown_s:
            self._note_skip("cooldown")
            return None
        if now - self._last_reason_at.get(reason, -1e9) < self.config.repeat_s:
            self._note_skip(f"repeat:{reason}")
            return None
        return self._say(reason, detail, state)

    def _reason(self, reading: Dict[str, Any]) -> tuple[Optional[str], str]:
        """Pick the most worth-interrupting signal, or nothing.

        Order is priority order: an overlay beats a scene change beats a new
        region beats "looks stuck".
        """
        change = float(reading.get("change_fraction") or 0.0)
        motion = float(reading.get("peak_motion") or 0.0)
        novelty = float(reading.get("novelty") or 0.0)
        scene_changed = bool(reading.get("scene_changed"))
        overlay = bool(reading.get("overlay_suspected"))

        if overlay and change >= self.config.min_overlay_change:
            return "overlay", "a dialog or overlay appeared"
        if scene_changed and change >= self.config.min_scene_change:
            return "scene_change", "the whole screen changed"
        if novelty >= self.config.min_appear_novelty and change >= self.config.min_appear_salience:
            return "appear", "something new and prominent appeared"

        # "Looks stuck": local motion repeating with no whole-screen change. The
        # timer resets the moment the screen actually changes.
        now = time.time()
        if scene_changed:
            self._motion_since = 0.0
        elif motion >= self.config.min_stuck_motion:
            if self._motion_since == 0.0:
                self._motion_since = now
            elif now - self._motion_since >= self.config.stuck_after_s:
                self._motion_since = now  # re-arm rather than fire every tick
                return "stuck", "the user has been working in one spot for a while"
        else:
            self._motion_since = 0.0
        return None, ""

    def _say(self, reason: str, detail: str, state: Any) -> Optional[Remark]:
        """Ask the brain for a remark and speak it. Silence is a valid answer."""
        observation = self._observation(state, detail)
        try:
            turn = self._brain.comment(observation, reason=detail)
        except Exception as exc:
            self.error = f"brain.comment failed: {exc}"
            return None
        if turn is None or not getattr(turn, "speech", ""):
            # The model decided this was not worth saying. That is the intended
            # common case, and it must not be recorded as a remark.
            self._note_skip(f"silent:{reason}")
            return None
        text = str(turn.speech).strip()[: self.config.max_speech_chars]
        if not text:
            self._note_skip(f"silent:{reason}")
            return None
        now = time.time()
        self._last_spoke_at = now
        self._last_reason_at[reason] = now
        remark = Remark(
            at=now, reason=reason, text=text,
            source=getattr(turn, "source", "llm"), detail=detail,
        )
        try:
            self._speak(text)
        except Exception as exc:
            self.error = f"speak failed: {exc}"
        with self._lock:
            self.remarks.append(remark)
            # Keep the list bounded; this is a live log, not an archive.
            if len(self.remarks) > 50:
                del self.remarks[:-50]
        if self._on_remark is not None:
            try:
                self._on_remark(remark)
            except Exception:
                pass
        return remark

    def _observation(self, state: Any, detail: str) -> Dict[str, Any]:
        """The coordinate-free description the brain sees, plus why-now.

        Built from the same ``reading`` the page shows, so the remark can only be
        about facts the perception actually reported. No image, no coordinate.
        """
        reading = getattr(state, "reading", None) or {}
        description = getattr(state, "description", "") or ""
        caps = {}
        try:
            caps = self._capabilities() or {}
        except Exception:
            caps = {}
        observation: Dict[str, Any] = {
            "screen": description,
            "stability": "changing" if reading.get("scene_changed") else "stable",
            "has_modal_dialog": bool(reading.get("overlay_suspected")),
            "active_window": "",
            "interactive_elements": int(reading.get("regions") or 0),
            "why_now": detail,
            # The brain is told it has no hand here, so it cannot be tempted to
            # announce an action it will never take.
            "capabilities": {
                "can_click": False, "can_type": False, "can_press_key": False,
                "can_scroll": False, "can_see_targets": False,
            },
            "watching": True,
        }
        return observation

    def _note_skip(self, key: str) -> None:
        with self._lock:
            self.skipped[key] = self.skipped.get(key, 0) + 1

    # ------------------------------------------------------------------ #
    # Introspection for the page
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            remarks = [r.to_dict() for r in self.remarks[-10:]]
            skipped = dict(self.skipped)
        return {
            "running": self.running,
            "remark_count": len(self.remarks),
            "recent": remarks,
            "skipped": skipped,
            "error": self.error,
        }
