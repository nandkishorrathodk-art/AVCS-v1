"""A real, cross-platform hand that actuates the machine.

`avcs_sdk.hand_motor.HandMotor` drives the native cursor but is Windows-only
(ctypes to user32). On Linux the natural equivalent is ``xdotool``. This backend
gives the assistant a hand that actually clicks, types, presses keys and scrolls
on a Linux/X11 desktop, so "complete AI assistant" is not limited to Windows.

It is still **fail-closed**: if ``xdotool`` or ``$DISPLAY`` is missing,
``available()`` is False and every operation returns ``False`` rather than
pretending to have worked. The capability kernel sits in front of it either way.

``ServoHand`` wraps any of these hands and replaces the teleport with a
human neuromuscular trajectory (``avcs.control.servoing``). It exposes the same
hand contract, so ``avcs.voice.hands.dispatch`` needs no special case for it.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from typing import Any, List, Optional, Tuple


class XdotoolHand:
    """Real actuation via the ``xdotool`` command-line tool (X11).

    Coordinates are **absolute pixels**, matching ``HandMotor``'s contract, so
    ``avcs.voice.hands.dispatch`` maps a normalized point the same way for both.
    """

    name = "xdotool"

    def __init__(
        self,
        display: Optional[str] = None,
        binary: Optional[str] = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.binary = binary or shutil.which("xdotool")
        self.display = display if display is not None else os.environ.get("DISPLAY")
        self.timeout_s = float(timeout_s)
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """True only when both the tool and an X display are present.

        The probe result is cached after the first call. A servo trajectory calls
        this once per waypoint, and re-probing would spawn a second ``xdotool``
        process for every move -- doubling the process count of a single click.
        ``refresh_availability()`` re-probes when the display may have changed.
        """
        if self._available is None:
            self._available = self._probe()
        return self._available

    def refresh_availability(self) -> bool:
        """Force a fresh probe (e.g. after a display connect/disconnect)."""
        self._available = self._probe()
        return self._available

    def _probe(self) -> bool:
        """True only when the tool, an X display, *and* a working pointer exist.

        Checking the binary and a `getdisplaygeometry` exit code is not enough.
        On an Xvfb server started without `-noreset`, every client disconnect
        resets the pointer to the screen centre, so `mousemove` returns 0 while
        moving nothing. A probe that stopped at the exit code reported a hand
        that "worked" and then silently no-opped every click -- a false success,
        the one failure a fail-closed hand must never produce.

        The check must therefore move the pointer in *one* process and read it
        back in *another*. A single chained invocation (`mousemove 1 1
        getmouselocation`) reads the position before the client disconnects and
        so reports success even on a broken server; only the second connection
        sees the reset. The nudge is 1 px and is restored, so a real user's
        cursor ends where it started.
        """
        if not self.binary or not self.display:
            return False
        start = self._read_position()
        if start is None:
            return False
        # Nudge *toward* the middle so the target is always on-screen. A blind
        # `+1` at the right/bottom edge would be clamped by the server and the
        # read-back would differ for a reason that has nothing to do with the
        # pointer being broken.
        size = self._read_screen_size()
        if size is None:
            return False
        w, h = size
        dx = 1 if start[0] < (w - 1) / 2 else -1
        dy = 1 if start[1] < (h - 1) / 2 else -1
        target = (start[0] + dx, start[1] + dy)
        try:
            # Process A: move. Process B: read. The gap between them is the
            # disconnect, which is exactly the condition being tested.
            self._run_raw(["mousemove", str(target[0]), str(target[1])])
            after = self._read_position()
            # Always put the pointer back, pass or fail, so the probe is invisible.
            self._run_raw(["mousemove", str(start[0]), str(start[1])])
            if after is None:
                return False
            return after == target
        except Exception:
            return False

    def _read_screen_size(self) -> Optional[Tuple[int, int]]:
        """Display geometry, or None. Raw, so ``_probe`` can use it safely."""
        if not self.binary:
            return None
        try:
            proc = subprocess.run(
                [self.binary, "getdisplaygeometry"],
                capture_output=True, timeout=self.timeout_s, env=self._env(), text=True,
            )
            if proc.returncode != 0:
                return None
            w, h = proc.stdout.split()
            return int(w), int(h)
        except Exception:
            return None

    def _read_position(self) -> Optional[Tuple[int, int]]:
        """Pointer position in pixels, or None when it cannot be read."""
        try:
            proc = subprocess.run(
                [self.binary, "getmouselocation", "--shell"],
                capture_output=True, timeout=self.timeout_s, env=self._env(), text=True,
            )
            if proc.returncode != 0:
                return None
            x = y = None
            for line in proc.stdout.splitlines():
                if line.startswith("X="):
                    x = int(line.split("=", 1)[1])
                elif line.startswith("Y="):
                    y = int(line.split("=", 1)[1])
            if x is None or y is None:
                return None
            return (x, y)
        except Exception:
            return None

    def _env(self) -> dict:
        env = dict(os.environ)
        if self.display:
            env["DISPLAY"] = self.display
        return env

    def _run(self, args: List[str]) -> bool:
        if not self.available():
            return False
        return self._run_raw(args)

    def _run_raw(self, args: List[str]) -> bool:
        """Run xdotool without the availability check.

        ``_probe`` must not go through ``_run``: that would call ``available()``,
        which calls ``_probe``, which calls ``_run`` -- infinite recursion. This
        is the unchecked primitive both paths share.
        """
        if not self.binary:
            return False
        try:
            proc = subprocess.run(
                [self.binary, *args],
                capture_output=True, timeout=self.timeout_s, env=self._env(),
            )
            return proc.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Hand surface (matches HandMotor / VirtualHand where it can)
    # ------------------------------------------------------------------ #
    def get_screen_size(self) -> Tuple[int, int]:
        if not self.available():
            return 0, 0
        return self._read_screen_size() or (0, 0)

    def click(self, x: int, y: int, double: bool = False) -> bool:
        self._run(["mousemove", str(int(x)), str(int(y))])
        return self._run(["click", "--repeat", "2" if double else "1", "1"])

    def move_to_pixels(self, x: int, y: int) -> bool:
        """Move the pointer without clicking.

        ``click`` above moves *and* clicks, which is fine for a single click but
        wrong for a trajectory: a servo path has dozens of waypoints and calling
        ``click`` for each would fire dozens of clicks. This is the bare move the
        neuromuscular servo drives, so a real pointer travels along the planned
        path instead of teleporting at the end.
        """
        return self._run(["mousemove", str(int(x)), str(int(y))])

    def get_cursor_pos(self) -> Tuple[int, int]:
        """Current pointer position in pixels, or (-1, -1) when unreadable."""
        if not self.available():
            return -1, -1
        return self._read_position() or (-1, -1)

    def double_click(self, x: int, y: int) -> bool:
        return self.click(x, y, double=True)

    def right_click(self, x: int, y: int) -> bool:
        # Button 3 is the secondary button in X11; button 1 is primary.
        self._run(["mousemove", str(int(x)), str(int(y))])
        return self._run(["click", "3"])

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Press at (x1,y1), move to (x2,y2), release. Fails closed on any step."""
        if not self._run(["mousemove", str(int(x1)), str(int(y1))]):
            return False
        if not self._run(["mousedown", "1"]):
            return False
        if not self._run(["mousemove", str(int(x2)), str(int(y2))]):
            self._run(["mouseup", "1"])
            return False
        return self._run(["mouseup", "1"])

    def launch_app(self, app_name: str) -> bool:
        """Start an allow-listed application, detached from this process.

        The binary is looked up with ``shutil.which`` and run without a shell, so
        a name containing shell metacharacters cannot become a command. The
        allow-list lives in the kernel; this layer only refuses an empty name.
        """
        name = (app_name or "").strip()
        if not name:
            return False
        binary = shutil.which(name)
        if binary is None:
            return False
        try:
            subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=self._env(),
            )
            return True
        except Exception:
            return False

    def read_file(self, path: str) -> Optional[str]:
        """Read a text file. Returns None (never raises) when it cannot."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:
            return None

    def list_dir(self, path: str) -> Optional[str]:
        """List a directory, one entry per line. Returns None when it cannot."""
        try:
            entries = sorted(os.listdir(path))
        except Exception:
            return None
        return "\n".join(entries)

    def type_text(self, text: str, typing_speed_cps: float = 25.0) -> bool:
        # --delay is per-key milliseconds; 25 chars/s -> 40 ms.
        delay = int(max(0, round(1000.0 / max(1.0, typing_speed_cps))))
        return self._run(["type", "--delay", str(delay), "--", text])

    def press_key(self, key_name: str) -> bool:
        return self._run(["key", "--clearmodifiers", self._xdotool_key(key_name)])

    def scroll(self, delta: int) -> bool:
        # Positive delta scrolls down (button 5), negative scrolls up (button 4).
        button = "5" if delta >= 0 else "4"
        repeats = max(1, abs(int(delta)))
        return self._run(["click", "--repeat", str(repeats), button])

    # ------------------------------------------------------------------ #
    # Key-name translation
    # ------------------------------------------------------------------ #
    # Names the brain may emit -> xdotool key names.
    _ALIASES = {
        "enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape",
        "space": "space", "tab": "Tab", "backspace": "BackSpace", "delete": "Delete",
        "up": "Up", "down": "Down", "left": "Left", "right": "Right",
        "home": "Home", "end": "End", "pageup": "Prior", "pagedown": "Next",
    }

    @classmethod
    def _xdotool_key(cls, key_name: str) -> str:
        """Translate ``ctrl+s`` / ``enter`` into xdotool's ``ctrl+s`` / ``Return``."""
        parts = [p.strip().lower() for p in str(key_name).split("+") if p.strip()]
        if not parts:
            return ""
        mods = {"ctrl": "ctrl", "control": "ctrl", "shift": "shift", "alt": "alt"}
        out = []
        for mod in parts[:-1]:
            out.append(mods.get(mod, mod))
        base = parts[-1]
        out.append(cls._ALIASES.get(base, base))
        return "+".join(out)


class _CursorDriver:
    """An ``OSCursorDriver``-shaped view over a hand, for the servo controller.

    The servo controller talks to a *cursor driver*, not to a hand: it reads the
    cursor position and moves it. A hand has a different surface, so this adapter
    bridges the two and owns the translation between the servo's normalized
    [0,1] space and the hand's pixel space.

    Every method is fail-closed. ``move_to_normalized`` calls the hand's real
    move so the cursor actually travels; if the hand cannot move, it returns
    without pretending, and ``servo_to``'s final position check then reports
    ``success=False`` for the honest reason.
    """

    def __init__(self, hand: Any, screen_size: Tuple[int, int]) -> None:
        self._hand = hand
        self._size = (int(screen_size[0]), int(screen_size[1]))
        self._x_norm = 0.5
        self._y_norm = 0.5
        from forge.vision.cortex.motor.failsafe import CornerBreakoutFailSafe

        # Top-left and bottom-right are the panic corners: entering them aborts
        # immediately. Top-right (a window Close box) and bottom-left (a Start
        # button) are *ordinary controls*, so excluding them from the corner rule
        # is what lets the assistant click a close button at all. This mirrors the
        # repo's own hardware failsafe, whose docstring says exactly that; the
        # panic gesture is the top-left/bottom-right diagonal.
        self.failsafe = CornerBreakoutFailSafe(
            enabled=True, exclude_top_right=True, exclude_bottom_left=True
        )
        self.click_history: List[Tuple[float, float, str]] = []
        self.trajectory_history: List[Tuple[float, float]] = []

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self._size

    # -- position ------------------------------------------------------ #
    def get_position_normalized(self) -> Tuple[float, float]:
        reader = getattr(self._hand, "get_cursor_pos", None)
        if callable(reader):
            try:
                px, py = reader()
                w, h = self._size
                return (px / max(1, w - 1), py / max(1, h - 1))
            except Exception:
                pass
        return (self._x_norm, self._y_norm)

    def get_position_pixels(self) -> Tuple[int, int]:
        nx, ny = self.get_position_normalized()
        w, h = self._size
        return (int(round(nx * (w - 1))), int(round(ny * (h - 1))))

    # -- movement ------------------------------------------------------ #
    def move_to_normalized(self, nx: float, ny: float) -> None:
        self._x_norm = max(0.0, min(1.0, float(nx)))
        self._y_norm = max(0.0, min(1.0, float(ny)))
        self.trajectory_history.append((self._x_norm, self._y_norm))
        w, h = self._size
        px = int(round(self._x_norm * (w - 1)))
        py = int(round(self._y_norm * (h - 1)))
        mover = getattr(self._hand, "move_to_pixels", None)
        if callable(mover):
            mover(px, py)
            return
        # `XdotoolHand`/`HandMotor` have no bare move; their `click` moves and
        # clicks. That would click once per waypoint, so a bare move is done
        # through the underlying tool when one is reachable, and otherwise the
        # trajectory is computed but not painted. Silence here is deliberate:
        # `servo_to` still checks the final position and reports the truth.
        mover = getattr(self._hand, "move_to", None)
        if callable(mover):
            try:
                mover(px, py)
            except TypeError:
                mover(self._x_norm, self._y_norm)

    def move_to_pixels(self, x: int, y: int) -> None:
        w, h = self._size
        self.move_to_normalized(x / max(1, w - 1), y / max(1, h - 1))

    # -- clicks -------------------------------------------------------- #
    def click(self, button: str = "left", hold_duration: float = 0.02) -> None:
        self.click_history.append((self._x_norm, self._y_norm, button))

    def double_click(self, button: str = "left", interval: float = 0.08) -> None:
        self.click_history.append((self._x_norm, self._y_norm, button))
        self.click_history.append((self._x_norm, self._y_norm, button))

    def close(self) -> None:
        pass


class ServoHand:
    """A hand whose pointer travel is a human neuromuscular trajectory.

    ``XdotoolHand`` teleports: ``mousemove x y``. A person does not. This wrapper
    keeps the exact same hand contract -- ``dispatch`` cannot tell the difference
    -- but instead of one jump it plans a minimum-jerk path with Fitts' Law
    duration and 8-12 Hz micro-tremor (``avcs.control.servoing``), then clicks at
    the end.

    Why wrap instead of rewrite: the trajectory model, the physiological tremor,
    and the emergency-breakout failsafe were already built and measured. The only
    thing missing was that nothing on the voice/GUI path used them. This is that
    wire.

    The failsafe is the "user takes the mouse back" behaviour: if the physical
    cursor is diverted from the planned path by more than ``override_threshold``,
    the servo latches and refuses to keep driving, and every later call fails
    closed until ``reset_failsafe()``. The user always wins the pointer.
    """

    name = "servo-hand"

    def __init__(
        self,
        inner: Any,
        *,
        virtual: bool = False,
        realtime: bool = True,
        enable_micro_tremor: bool = True,
        fovea_width: float = 0.04,
        override_threshold: float = 0.08,
        tick_rate_hz: int = 60,
    ) -> None:
        self.inner = inner
        # True when the wrapped cursor is simulated, so a caller can report
        # "virtual" instead of claiming real actuation. A ServoHand over a real
        # hand is live; over a VirtualCursorDriver the plan is real and the
        # side effect is inert.
        self.virtual = bool(virtual)
        self.realtime = bool(realtime)
        self.enable_micro_tremor = bool(enable_micro_tremor)
        self.fovea_width = float(fovea_width)
        self.override_threshold = float(override_threshold)

        size = self._screen_size()
        self._driver = _CursorDriver(inner, size)
        from avcs.control.servoing import NeuromuscularServoingController

        self.controller = NeuromuscularServoingController(
            driver=self._driver,
            tick_rate_hz=tick_rate_hz,
            fovea_width=self.fovea_width,
            enable_micro_tremor=self.enable_micro_tremor,
            override_threshold_norm=self.override_threshold,
        )
        # Remember the last planned path so a mid-flight takeover can be detected
        # against what was actually intended, not against the final target.
        self._last_path: List[Tuple[float, float]] = []
        self.last_result: Any = None
        self.overrides: int = 0

    # ------------------------------------------------------------------ #
    # Hand contract
    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        if self.virtual:
            return True
        fn = getattr(self.inner, "available", None)
        return True if fn is None else bool(fn())

    def _screen_size(self) -> Tuple[int, int]:
        fn = getattr(self.inner, "get_screen_size", None)
        if callable(fn):
            try:
                w, h = fn()
                if w and h:
                    return int(w), int(h)
            except Exception:
                pass
        # A `VirtualCursorDriver` exposes `screen_size` as a property.
        size = getattr(self.inner, "screen_size", None)
        if size and len(size) == 2 and all(size):
            return int(size[0]), int(size[1])
        return (1920, 1080)

    def get_screen_size(self) -> Tuple[int, int]:
        """Pixel dimensions, so ``dispatch`` maps a normalized point for us.

        This is the same contract ``XdotoolHand``/``HandMotor`` follow: a hand
        that reports its size receives absolute pixels and owns the conversion.
        Exposing it is what keeps ``dispatch`` free of any ServoHand special case.
        """
        return self._screen_size()

    def reset_failsafe(self) -> None:
        """Hand the pointer back to the motor after a takeover latched it."""
        self.controller.reset_failsafe()

    @property
    def is_failsafe_tripped(self) -> bool:
        return bool(self.controller.is_failsafe_tripped)

    def _to_norm(self, px: float, py: float) -> Tuple[float, float]:
        w, h = self._screen_size()
        return (
            max(0.0, min(1.0, float(px) / max(1, w - 1))),
            max(0.0, min(1.0, float(py) / max(1, h - 1))),
        )

    # ------------------------------------------------------------------ #
    # Movement + click
    # ------------------------------------------------------------------ #
    def _servo(self, x: float, y: float) -> bool:
        """Plan and (optionally) paint a neuromuscular path to normalized (x, y).

        Returns True when the servo reached the target. A latched failsafe or a
        refused move returns False; it never raises into the caller, because a
        refused pointer must be reported as "did not perform", not as a crash.
        """
        from forge.vision.cortex.motor.failsafe import SafetyBreakoutException

        try:
            result = self.controller.servo_to(
                (float(x), float(y)),
                target_width=self.fovea_width,
                realtime=self.realtime,
                enable_micro_tremor=self.enable_micro_tremor,
            )
        except SafetyBreakoutException:
            self.overrides += 1
            return False
        except Exception:
            return False
        self.last_result = result
        if result.trajectory:
            self._last_path = [(p.x, p.y) for p in result.trajectory]
        return bool(result.success)

    def _click_pixels(self, px: int, py: int, double: bool = False, button: str = "left") -> bool:
        """Click through the wrapped hand, matching its own click contract.

        Two shapes exist: a *hand* clicks at a coordinate (``click(x, y)``), while
        a *cursor driver* clicks wherever the pointer already is (``click(button)``)
        because the servo just moved it there. Passing coordinates to a driver
        would silently record a pixel as a button name, so the driver shape is
        detected and called without them.
        """
        is_driver = callable(getattr(self.inner, "get_position_normalized", None))
        try:
            if is_driver:
                # A driver's click returns None on success and raises on failure,
                # so `is not False` is the honest test -- `bool(None)` would read
                # a successful click as a refusal.
                if button == "right":
                    fn = getattr(self.inner, "right_click", None)
                    return fn() is not False if callable(fn) else False
                if double:
                    fn = getattr(self.inner, "double_click", None)
                    return fn() is not False if callable(fn) else False
                fn = getattr(self.inner, "click", None)
                return fn() is not False if callable(fn) else False

            click = getattr(self.inner, "click", None)
            if click is None:
                return False
            if button == "right":
                right = getattr(self.inner, "right_click", None)
                if callable(right):
                    return right(px, py) is not False
                return False
            if double:
                dbl = getattr(self.inner, "double_click", None)
                if callable(dbl):
                    return dbl(px, py) is not False
            return click(px, py) is not False
        except Exception:
            return False

    def click(self, x: float, y: float, double: bool = False) -> Any:
        """Servo to the pixel point, then click there.

        Coordinates are **absolute pixels**, matching ``XdotoolHand`` and
        ``HandMotor``. The travel is planned as a human neuromuscular path in
        normalized space; only the final click is issued to the wrapped hand.
        """
        nx, ny = self._to_norm(x, y)
        reached = self._servo(nx, ny)
        if self.is_failsafe_tripped:
            return False
        # A *driver-shape* hand clicks wherever the pointer already is, so a servo
        # that never reached the target would click the wrong place and still
        # report success -- a false success. Refuse instead. A *coordinate-shape*
        # hand (`XdotoolHand`, `HandMotor`) moves and clicks in one call, so it
        # self-corrects even after a failed servo and is not gated here.
        if not reached and callable(getattr(self.inner, "get_position_normalized", None)):
            return False
        px = int(round(float(x)))
        py = int(round(float(y)))
        if not self._click_pixels(px, py, double=double):
            return False
        return (px, py)

    def double_click(self, x: float, y: float) -> Any:
        return self.click(x, y, double=True)

    def right_click(self, x: float, y: float) -> Any:
        nx, ny = self._to_norm(x, y)
        reached = self._servo(nx, ny)
        if self.is_failsafe_tripped:
            return False
        if not reached and callable(getattr(self.inner, "get_position_normalized", None)):
            return False
        px = int(round(float(x)))
        py = int(round(float(y)))
        if self._click_pixels(px, py, button="right"):
            return (px, py)
        return False

    def drag(self, x1: float, y1: float, x2: float, y2: float) -> Any:
        """Servo to the source, press, drag to the destination, release.

        The destination is also a trajectory rather than a jump, so a drag looks
        like a hand carrying something, not two teleports with a button held.
        """
        nx1, ny1 = self._to_norm(x1, y1)
        self._servo(nx1, ny1)
        if self.is_failsafe_tripped:
            return False
        sx, sy = int(round(float(x1))), int(round(float(y1)))
        dx, dy = int(round(float(x2))), int(round(float(y2)))
        fn = getattr(self.inner, "drag", None)
        if callable(fn):
            try:
                if fn(sx, sy, dx, dy) is False:
                    return False
                return (dx, dy)
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------ #
    # Non-pointer ops delegate straight through
    # ------------------------------------------------------------------ #
    def type_text(self, text: str, typing_speed_cps: float = 25.0) -> bool:
        fn = getattr(self.inner, "type_text", None)
        if fn is None:
            return False
        try:
            return fn(text, typing_speed_cps) is not False
        except TypeError:
            return fn(text) is not False

    def press_key(self, key_name: str) -> bool:
        fn = getattr(self.inner, "press_key", None)
        if fn is None:
            return False
        return fn(key_name) is not False

    def scroll(self, delta: int) -> bool:
        fn = getattr(self.inner, "scroll", None)
        if fn is None:
            return False
        return fn(int(delta)) is not False

    def launch_app(self, app_name: str) -> bool:
        fn = getattr(self.inner, "launch_app", None)
        return bool(fn(app_name)) if callable(fn) else False

    def read_file(self, path: str) -> Optional[str]:
        fn = getattr(self.inner, "read_file", None)
        return fn(path) if callable(fn) else None

    def list_dir(self, path: str) -> Optional[str]:
        fn = getattr(self.inner, "list_dir", None)
        return fn(path) if callable(fn) else None


def default_hand():
    """Best available real hand for this platform, else None (fail closed)."""
    if sys.platform == "win32":
        try:
            from avcs_sdk.hand_motor import HandMotor

            return HandMotor()
        except Exception:
            return None
    hand = XdotoolHand()
    return hand if hand.available() else None


def default_servo_hand():
    """A ``ServoHand`` over the best available hand, or over a virtual cursor.

    On a machine with a real hand (Windows ``HandMotor`` / Linux ``xdotool``) the
    trajectory drives the actual pointer. With no hand at all this still returns
    a working ``ServoHand`` backed by a virtual cursor, so the human motion model
    runs and can be measured headlessly -- it just moves no real cursor. That is
    the same honest split as ``VirtualHand``: the plan is real, the side effect
    is inert, and ``get_screen_size`` says which.
    """
    from forge.vision.cortex.motor.driver import get_default_driver

    inner = default_hand()
    if inner is not None:
        return ServoHand(inner, virtual=False)
    return ServoHand(get_default_driver(), virtual=True)
