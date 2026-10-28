"""A real, cross-platform hand that actuates the machine.

`avcs_sdk.hand_motor.HandMotor` drives the native cursor but is Windows-only
(ctypes to user32). On Linux the natural equivalent is ``xdotool``. This backend
gives the assistant a hand that actually clicks, types, presses keys and scrolls
on a Linux/X11 desktop, so "complete AI assistant" is not limited to Windows.

It is still **fail-closed**: if ``xdotool`` or ``$DISPLAY`` is missing,
``available()`` is False and every operation returns ``False`` rather than
pretending to have worked. The capability kernel sits in front of it either way.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple


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

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """True only when both the tool and an X display are present."""
        if not self.binary or not self.display:
            return False
        try:
            proc = subprocess.run(
                [self.binary, "getdisplaygeometry"],
                capture_output=True, timeout=self.timeout_s, env=self._env(),
            )
            return proc.returncode == 0
        except Exception:
            return False

    def _env(self) -> dict:
        env = dict(os.environ)
        if self.display:
            env["DISPLAY"] = self.display
        return env

    def _run(self, args: List[str]) -> bool:
        if not self.available():
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
        try:
            proc = subprocess.run(
                [self.binary, "getdisplaygeometry"],
                capture_output=True, timeout=self.timeout_s, env=self._env(), text=True,
            )
            w, h = proc.stdout.split()
            return int(w), int(h)
        except Exception:
            return 0, 0

    def click(self, x: int, y: int, double: bool = False) -> bool:
        self._run(["mousemove", str(int(x)), str(int(y))])
        return self._run(["click", "--repeat", "2" if double else "1", "1"])

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
