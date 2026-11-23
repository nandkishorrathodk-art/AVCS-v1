"""Action Space Schema for the Ace-Control-Medium Architecture.

Defines the structured Action schema emitted directly by the Vision-Action Model (VAM)
without verbose reasoning tokens, following the generalagents-python operational model.
"""
from __future__ import annotations

import dataclasses
from enum import Enum
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple


class ActionKind(str, Enum):
    """Deterministic Action Types."""
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    TRIPLE_CLICK = "triple_click"
    MIDDLE_CLICK = "middle_click"
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK_DRAG = "left_click_drag"
    SCROLL = "scroll"
    TYPE = "type"
    TYPE_TEXT = "type_text"
    KEY = "key"
    WAIT = "wait"
    STOP = "stop"

    @classmethod
    def _missing_(cls, value: object) -> Optional["ActionKind"]:
        if isinstance(value, str):
            v = value.strip().lower()
            aliases = {
                "click": cls.LEFT_CLICK,
                "left_click": cls.LEFT_CLICK,
                "mouse_click": cls.LEFT_CLICK,
                "primary_click": cls.LEFT_CLICK,
                "right_click": cls.RIGHT_CLICK,
                "secondary_click": cls.RIGHT_CLICK,
                "double_click": cls.DOUBLE_CLICK,
                "triple_click": cls.TRIPLE_CLICK,
                "middle_click": cls.MIDDLE_CLICK,
                "mouse_move": cls.MOUSE_MOVE,
                "move": cls.MOUSE_MOVE,
                "hover": cls.MOUSE_MOVE,
                "cursor_move": cls.MOUSE_MOVE,
                "left_click_drag": cls.LEFT_CLICK_DRAG,
                "drag": cls.LEFT_CLICK_DRAG,
                "scroll": cls.SCROLL,
                "mouse_scroll": cls.SCROLL,
                "wheel": cls.SCROLL,
                "type": cls.TYPE,
                "type_text": cls.TYPE_TEXT,
                "input_text": cls.TYPE_TEXT,
                "write": cls.TYPE,
                "key": cls.KEY,
                "press": cls.KEY,
                "press_key": cls.KEY,
                "hotkey": cls.KEY,
                "shortcut": cls.KEY,
                "key_press": cls.KEY,
                "keydown": cls.KEY,
                "wait": cls.WAIT,
                "sleep": cls.WAIT,
                "delay": cls.WAIT,
                "stop": cls.STOP,
                "finish": cls.STOP,
                "done": cls.STOP,
                "complete": cls.STOP,
            }
            if v in aliases:
                return aliases[v]
            for member in cls:
                if member.value == v or member.name.lower() == v:
                    return member
        return None


# USB HID Usage Table - Page 0x07 (Keyboard / Keypad)
# Standardized hardware HID usage IDs for keyboard actuation
HID_KEYBOARD_TABLE: Dict[str, int] = {
    # Letters (0x04 - 0x1D) - lowercase and uppercase
    "a": 0x04, "b": 0x05, "c": 0x06, "d": 0x07, "e": 0x08, "f": 0x09,
    "g": 0x0A, "h": 0x0B, "i": 0x0C, "j": 0x0D, "k": 0x0E, "l": 0x0F,
    "m": 0x10, "n": 0x11, "o": 0x12, "p": 0x13, "q": 0x14, "r": 0x15,
    "s": 0x16, "t": 0x17, "u": 0x18, "v": 0x19, "w": 0x1A, "x": 0x1B,
    "y": 0x1C, "z": 0x1D,
    "A": 0x04, "B": 0x05, "C": 0x06, "D": 0x07, "E": 0x08, "F": 0x09,
    "G": 0x0A, "H": 0x0B, "I": 0x0C, "J": 0x0D, "K": 0x0E, "L": 0x0F,
    "M": 0x10, "N": 0x11, "O": 0x12, "P": 0x13, "Q": 0x14, "R": 0x15,
    "S": 0x16, "T": 0x17, "U": 0x18, "V": 0x19, "W": 0x1A, "X": 0x1B,
    "Y": 0x1C, "Z": 0x1D,
    # Numbers (0x1E - 0x27)
    "1": 0x1E, "2": 0x1F, "3": 0x20, "4": 0x21, "5": 0x22,
    "6": 0x23, "7": 0x24, "8": 0x25, "9": 0x26, "0": 0x27,
    # Common Controls & Modifiers
    "enter": 0x28, "return": 0x28,
    "esc": 0x29, "escape": 0x29,
    "backspace": 0x2A,
    "tab": 0x2B,
    "space": 0x2C, "spacebar": 0x2C,
    "minus": 0x2D, "-": 0x2D,
    "equal": 0x2E, "equals": 0x2E, "=": 0x2E,
    "left_bracket": 0x2F, "[": 0x2F,
    "right_bracket": 0x30, "]": 0x30,
    "backslash": 0x31, "\\": 0x31,
    "semicolon": 0x33, ";": 0x33,
    "quote": 0x34, "apostrophe": 0x34, "'": 0x34,
    "grave": 0x35, "tilde": 0x35, "`": 0x35,
    "comma": 0x36, ",": 0x36,
    "period": 0x37, "dot": 0x37, ".": 0x37,
    "slash": 0x38, "/": 0x38,
    "caps_lock": 0x39, "capslock": 0x39,
    # Function Keys (0x3A - 0x45)
    "f1": 0x3A, "f2": 0x3B, "f3": 0x3C, "f4": 0x3D, "f5": 0x3E, "f6": 0x3F,
    "f7": 0x40, "f8": 0x41, "f9": 0x42, "f10": 0x43, "f11": 0x44, "f12": 0x45,
    # Navigation & Editing
    "print_screen": 0x46, "prtscr": 0x46, "printscreen": 0x46,
    "scroll_lock": 0x47, "scrolllock": 0x47,
    "pause": 0x48, "break": 0x48,
    "insert": 0x49,
    "home": 0x4A,
    "page_up": 0x4B, "pageup": 0x4B, "pgup": 0x4B, "pg_up": 0x4B,
    "delete": 0x4C, "del": 0x4C,
    "end": 0x4D,
    "page_down": 0x4E, "pagedown": 0x4E, "pgdn": 0x4E, "pg_dn": 0x4E, "pgdown": 0x4E,
    "right": 0x4F, "arrowright": 0x4F, "arrow_right": 0x4F, "right_arrow": 0x4F,
    "left": 0x50, "arrowleft": 0x50, "arrow_left": 0x50, "left_arrow": 0x50,
    "down": 0x51, "arrowdown": 0x51, "arrow_down": 0x51, "down_arrow": 0x51,
    "up": 0x52, "arrowup": 0x52, "arrow_up": 0x52, "up_arrow": 0x52,
    "num_lock": 0x53, "numlock": 0x53,
    # Modifier Keys (0xE0 - 0xE7)
    "left_ctrl": 0xE0, "lctrl": 0xE0, "ctrl": 0xE0, "control": 0xE0,
    "left_shift": 0xE1, "lshift": 0xE1, "shift": 0xE1,
    "left_alt": 0xE2, "lalt": 0xE2, "alt": 0xE2, "opt": 0xE2, "option": 0xE2,
    "left_gui": 0xE3, "lgui": 0xE3, "win": 0xE3, "windows": 0xE3, "super": 0xE3, "cmd": 0xE3, "command": 0xE3,
    "right_ctrl": 0xE4, "rctrl": 0xE4,
    "right_shift": 0xE5, "rshift": 0xE5,
    "right_alt": 0xE6, "ralt": 0xE6,
    "right_gui": 0xE7, "rgui": 0xE7, "rwin": 0xE7,
}

HID_CODE_TO_KEY_TABLE: Dict[int, str] = {
    0x04: "a", 0x05: "b", 0x06: "c", 0x07: "d", 0x08: "e", 0x09: "f",
    0x0A: "g", 0x0B: "h", 0x0C: "i", 0x0D: "j", 0x0E: "k", 0x0F: "l",
    0x10: "m", 0x11: "n", 0x12: "o", 0x13: "p", 0x14: "q", 0x15: "r",
    0x16: "s", 0x17: "t", 0x18: "u", 0x19: "v", 0x1A: "w", 0x1B: "x",
    0x1C: "y", 0x1D: "z",
    0x1E: "1", 0x1F: "2", 0x20: "3", 0x21: "4", 0x22: "5",
    0x23: "6", 0x24: "7", 0x25: "8", 0x26: "9", 0x27: "0",
    0x28: "enter", 0x29: "esc", 0x2A: "backspace", 0x2B: "tab",
    0x2C: "space", 0x2D: "minus", 0x2E: "equal",
    0x2F: "left_bracket", 0x30: "right_bracket", 0x31: "backslash",
    0x33: "semicolon", 0x34: "quote", 0x35: "grave",
    0x36: "comma", 0x37: "period", 0x38: "slash", 0x39: "caps_lock",
    0x3A: "f1", 0x3B: "f2", 0x3C: "f3", 0x3D: "f4", 0x3E: "f5", 0x3F: "f6",
    0x40: "f7", 0x41: "f8", 0x42: "f9", 0x43: "f10", 0x44: "f11", 0x45: "f12",
    0x46: "print_screen", 0x47: "scroll_lock", 0x48: "pause", 0x49: "insert",
    0x4A: "home", 0x4B: "page_up", 0x4C: "delete", 0x4D: "end", 0x4E: "page_down",
    0x4F: "right", 0x50: "left", 0x51: "down", 0x52: "up", 0x53: "num_lock",
    0xE0: "ctrl", 0xE1: "shift", 0xE2: "alt", 0xE3: "win",
    0xE4: "rctrl", 0xE5: "rshift", 0xE6: "ralt", 0xE7: "rwin",
}

HID_MOUSE_BUTTON_TABLE: Dict[str, int] = {
    "left": 1,
    "primary": 1,
    "right": 2,
    "secondary": 2,
    "middle": 3,
    "wheel": 3,
    "button4": 4,
    "button5": 5,
}


def key_to_hid(key: str) -> Optional[int]:
    """Converts key name to USB HID Usage Page 0x07 code."""
    if not key or not isinstance(key, str):
        return None
    k = key.strip()
    if k in HID_KEYBOARD_TABLE:
        return HID_KEYBOARD_TABLE[k]
    k_low = k.lower()
    arrow_map = {
        "arrow_up": "up",
        "arrowup": "up",
        "up_arrow": "up",
        "arrow_down": "down",
        "arrowdown": "down",
        "down_arrow": "down",
        "arrow_left": "left",
        "arrowleft": "left",
        "left_arrow": "left",
        "arrow_right": "right",
        "arrowright": "right",
        "right_arrow": "right",
    }
    k_norm = arrow_map.get(k_low, k_low)
    return HID_KEYBOARD_TABLE.get(k_norm)


def hid_to_key(code: int) -> Optional[str]:
    """Converts USB HID Usage code to canonical key name."""
    return HID_CODE_TO_KEY_TABLE.get(code)


def mouse_button_to_hid(button: str) -> Optional[int]:
    """Converts mouse button name to USB HID mouse button code (1=left, 2=right, 3=middle)."""
    if not button or not isinstance(button, str):
        return None
    return HID_MOUSE_BUTTON_TABLE.get(button.strip().lower())


def interpolate_drag_chunks(
    start: Tuple[int, int],
    end: Tuple[int, int],
    num_chunks: int = 5,
    linear: bool = False,
) -> List[Tuple[int, int]]:
    """Generates intermediate interpolated chunk coordinates for drag motions."""
    if start is None or end is None:
        return []
    if not isinstance(start, (list, tuple)) or not isinstance(end, (list, tuple)) or len(start) < 2 or len(end) < 2:
        return []
    try:
        num_chunks = int(num_chunks)
    except (ValueError, TypeError):
        num_chunks = 5
    try:
        fx1, fy1 = float(start[0]), float(start[1])
        fx2, fy2 = float(end[0]), float(end[1])
        if (
            math.isnan(fx1) or math.isnan(fy1) or math.isnan(fx2) or math.isnan(fy2)
            or math.isinf(fx1) or math.isinf(fy1) or math.isinf(fx2) or math.isinf(fy2)
        ):
            return []
        sx, sy = int(round(fx1)), int(round(fy1))
        ex, ey = int(round(fx2)), int(round(fy2))
    except (ValueError, TypeError, OverflowError):
        return []

    if num_chunks <= 1:
        return [(sx, sy), (ex, ey)]
    chunks: List[Tuple[int, int]] = []
    for i in range(num_chunks + 1):
        t = i / float(num_chunks)
        if linear:
            st = t
        else:
            # Cosine/S-curve smoothing for natural biological cursor trajectory
            st = 0.5 * (1.0 - math.cos(math.pi * t))
        cx = int(round(sx + (ex - sx) * st))
        cy = int(round(sy + (ey - sy) * st))
        chunks.append((cx, cy))
    return chunks


def normalize_coordinate(
    coord: Optional[Tuple[int, int]],
    canvas_w: int = 1280,
    canvas_h: int = 800,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[Tuple[float, float]]:
    """Normalizes pixel coordinates (x, y) to canonical [0.0, 1.0] range."""
    if width is not None:
        canvas_w = width
    if height is not None:
        canvas_h = height
    if coord is None or not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    try:
        x, y = float(coord[0]), float(coord[1])
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return None
        norm_x = max(0.0, min(1.0, round(x / max(1, canvas_w), 6)))
        norm_y = max(0.0, min(1.0, round(y / max(1, canvas_h), 6)))
        return norm_x, norm_y
    except (ValueError, TypeError):
        return None


def denormalize_coordinate(
    norm_coord: Optional[Tuple[float, float]],
    canvas_w: int = 1280,
    canvas_h: int = 800,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Denormalizes [0.0, 1.0] coordinates to canonical pixel canvas [0, canvas_w] x [0, canvas_h]."""
    if width is not None:
        canvas_w = width
    if height is not None:
        canvas_h = height
    if norm_coord is None or not isinstance(norm_coord, (list, tuple)) or len(norm_coord) < 2:
        return None
    try:
        nx, ny = float(norm_coord[0]), float(norm_coord[1])
        if math.isnan(nx) or math.isnan(ny) or math.isinf(nx) or math.isinf(ny):
            return None
        # Clamp to [0.0, 1.0]
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        px = int(round(nx * max(1, canvas_w)))
        py = int(round(ny * max(1, canvas_h)))
        px = max(0, min(canvas_w, px))
        py = max(0, min(canvas_h, py))
        return px, py
    except (ValueError, TypeError):
        return None


@dataclasses.dataclass
class Action:
    """A single executable physical OS action on the canonical 1280x800 canvas."""

    kind: ActionKind
    coordinate: Optional[Tuple[int, int]] = None  # (x, y) in [0, 1280] x [0, 800]
    start_coordinate: Optional[Tuple[int, int]] = None
    end_coordinate: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    keys: Optional[List[str]] = None
    scroll_delta: Optional[Tuple[int, int]] = None  # (delta_x, delta_y)
    duration: float = 0.0
    reason: str = ""
    press_enter: bool = False
    confidence: float = 0.0
    clear_first: bool = False
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    drag_chunks: Optional[List[Tuple[int, int]]] = None
    hid_codes: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if isinstance(self.kind, str) and not isinstance(self.kind, ActionKind):
            try:
                self.kind = ActionKind(self.kind)
            except (ValueError, KeyError):
                pass

    def is_type(self) -> bool:
        """Returns True if action represents text input."""
        return self.kind in (ActionKind.TYPE, ActionKind.TYPE_TEXT)

    def is_click(self) -> bool:
        """Returns True if action represents a mouse click."""
        return self.kind in (
            ActionKind.LEFT_CLICK,
            ActionKind.RIGHT_CLICK,
            ActionKind.DOUBLE_CLICK,
            ActionKind.TRIPLE_CLICK,
            ActionKind.MIDDLE_CLICK,
        )

    @property
    def resolved_hid_codes(self) -> List[int]:
        """Resolves USB HID usage codes for key or mouse button payloads."""
        if self.hid_codes is not None:
            return list(self.hid_codes)
        codes: List[int] = []
        if self.kind == ActionKind.KEY and self.keys:
            for k in self.keys:
                c = key_to_hid(k)
                if c is not None:
                    codes.append(c)
                elif isinstance(k, str) and ("+" in k or "-" in k or " " in k):
                    for sub_k in re.split(r'[\+\-\s]+', k):
                        if sub_k:
                            sc = key_to_hid(sub_k)
                            if sc is not None:
                                codes.append(sc)
        elif self.is_type() and self.text:
            for ch in self.text:
                c = key_to_hid(ch)
                if c is not None:
                    codes.append(c)
        elif self.kind in (ActionKind.LEFT_CLICK, ActionKind.DOUBLE_CLICK, ActionKind.TRIPLE_CLICK, ActionKind.LEFT_CLICK_DRAG):
            codes.append(1)
        elif self.kind == ActionKind.RIGHT_CLICK:
            codes.append(2)
        elif self.kind == ActionKind.MIDDLE_CLICK:
            codes.append(3)
        return codes

    def get_drag_chunks(self, num_chunks: int = 5) -> List[Tuple[int, int]]:
        """Returns interpolated path chunks for drag action."""
        if self.drag_chunks:
            return list(self.drag_chunks)
        start = self.start_coordinate
        end = self.end_coordinate or self.coordinate
        if start and end:
            return interpolate_drag_chunks(start, end, num_chunks=num_chunks)
        if start:
            return [start]
        if end:
            return [end]
        return []

    @property
    def normalized_coordinate(self) -> Optional[Tuple[float, float]]:
        return normalize_coordinate(self.coordinate)

    @property
    def normalized_start_coordinate(self) -> Optional[Tuple[float, float]]:
        return normalize_coordinate(self.start_coordinate)

    @property
    def normalized_end_coordinate(self) -> Optional[Tuple[float, float]]:
        return normalize_coordinate(self.end_coordinate)

    @property
    def normalized_drag_chunks(self) -> Optional[List[Tuple[float, float]]]:
        if not self.drag_chunks:
            return None
        res: List[Tuple[float, float]] = []
        for c in self.drag_chunks:
            nc = normalize_coordinate(c)
            if nc:
                res.append(nc)
        return res if res else None

    def to_normalized(
        self,
        canvas_w: int = 1280,
        canvas_h: int = 800,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Serializes action with coordinates normalized to [0.0, 1.0]."""
        if width is not None:
            canvas_w = width
        if height is not None:
            canvas_h = height
        d = self.to_dict()
        if self.coordinate is not None:
            nc = normalize_coordinate(self.coordinate, canvas_w, canvas_h)
            d["coordinate"] = list(nc) if nc else [0.0, 0.0]
        if self.start_coordinate is not None:
            nsc = normalize_coordinate(self.start_coordinate, canvas_w, canvas_h)
            d["start_coordinate"] = list(nsc) if nsc else [0.0, 0.0]
        if self.end_coordinate is not None:
            nec = normalize_coordinate(self.end_coordinate, canvas_w, canvas_h)
            d["end_coordinate"] = list(nec) if nec else [0.0, 0.0]
        if self.drag_chunks:
            d["drag_chunks"] = [
                list(normalize_coordinate(c, canvas_w, canvas_h) or (0.0, 0.0))
                for c in self.drag_chunks
            ]
        d["coordinate_system"] = "normalized_0_1"
        return d

    @classmethod
    def from_normalized(
        cls,
        data: Dict[str, Any],
        canvas_w: int = 1280,
        canvas_h: int = 800,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> "Action":
        """Constructs Action from normalized [0.0, 1.0] payload."""
        if width is not None:
            canvas_w = width
        if height is not None:
            canvas_h = height
        data_copy = dict(data)
        for ckey in ("coordinate", "coords", "point", "position", "pos", "location", "target"):
            if ckey in data_copy and data_copy[ckey] is not None:
                data_copy[ckey] = denormalize_coordinate(data_copy[ckey], canvas_w=canvas_w, canvas_h=canvas_h)
        if "x" in data_copy and "y" in data_copy and data_copy["x"] is not None and data_copy["y"] is not None:
            denorm = denormalize_coordinate((data_copy["x"], data_copy["y"]), canvas_w=canvas_w, canvas_h=canvas_h)
            if denorm is not None:
                data_copy["coordinate"] = denorm
                data_copy["x"], data_copy["y"] = denorm
        for skey in ("start_coordinate", "start_coords", "start_point", "from", "start"):
            if skey in data_copy and data_copy[skey] is not None:
                data_copy[skey] = denormalize_coordinate(data_copy[skey], canvas_w=canvas_w, canvas_h=canvas_h)
        if "x1" in data_copy and "y1" in data_copy and data_copy["x1"] is not None and data_copy["y1"] is not None:
            denorm = denormalize_coordinate((data_copy["x1"], data_copy["y1"]), canvas_w=canvas_w, canvas_h=canvas_h)
            if denorm is not None:
                data_copy["start_coordinate"] = denorm
                data_copy["x1"], data_copy["y1"] = denorm
        for ekey in ("end_coordinate", "end_coords", "end_point", "to", "end"):
            if ekey in data_copy and data_copy[ekey] is not None:
                data_copy[ekey] = denormalize_coordinate(data_copy[ekey], canvas_w=canvas_w, canvas_h=canvas_h)
        if "x2" in data_copy and "y2" in data_copy and data_copy["x2"] is not None and data_copy["y2"] is not None:
            denorm = denormalize_coordinate((data_copy["x2"], data_copy["y2"]), canvas_w=canvas_w, canvas_h=canvas_h)
            if denorm is not None:
                data_copy["end_coordinate"] = denorm
                data_copy["x2"], data_copy["y2"] = denorm
        if "drag_chunks" in data_copy and isinstance(data_copy["drag_chunks"], (list, tuple)):
            data_copy["drag_chunks"] = [
                d_c
                for c in data_copy["drag_chunks"]
                for d_c in [denormalize_coordinate(c, canvas_w=canvas_w, canvas_h=canvas_h)]
                if d_c is not None
            ]
        data_copy["coordinate_system"] = "pixels"
        return cls.from_dict(data_copy, canvas_w=canvas_w, canvas_h=canvas_h)

    def to_dict(self) -> Dict[str, Any]:
        kind_str = self.kind.value if isinstance(self.kind, ActionKind) else str(self.kind)
        d: Dict[str, Any] = {"kind": kind_str}
        if self.coordinate is not None:
            d["coordinate"] = list(self.coordinate)
        if self.start_coordinate is not None:
            d["start_coordinate"] = list(self.start_coordinate)
        if self.end_coordinate is not None:
            d["end_coordinate"] = list(self.end_coordinate)
        if self.text is not None:
            d["text"] = self.text
        if self.keys is not None:
            d["keys"] = list(self.keys)
        if self.scroll_delta is not None:
            d["scroll_delta"] = list(self.scroll_delta)
        d["duration"] = self.duration
        if self.reason:
            d["reason"] = self.reason
        if self.press_enter:
            d["press_enter"] = self.press_enter
        d["confidence"] = self.confidence
        if self.clear_first:
            d["clear_first"] = self.clear_first
        if self.drag_chunks is not None:
            d["drag_chunks"] = [list(c) for c in self.drag_chunks]
        if self.hid_codes is not None:
            d["hid_codes"] = list(self.hid_codes)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @staticmethod
    def _parse_coordinate_pair(
        val_x: Any,
        val_y: Any,
        canvas_w: int = 1280,
        canvas_h: int = 800,
        is_1000_grid: Optional[bool] = None,
    ) -> Optional[Tuple[int, int]]:
        """Parses coordinates, automatically converting normalized [0.0, 1.0] floats or [0, 1000] grid to pixels."""
        if val_x is None or val_y is None:
            return None
        try:
            fx = float(val_x)
            fy = float(val_y)
        except (ValueError, TypeError):
            return None

        if math.isnan(fx) or math.isnan(fy) or math.isinf(fx) or math.isinf(fy):
            return None

        def _is_float(val: Any) -> bool:
            if isinstance(val, float):
                return True
            if hasattr(val, "dtype") and "float" in str(getattr(val, "dtype", "")):
                return True
            if isinstance(val, str):
                s = val.strip().lower()
                return "." in s or ("e" in s and not s.startswith("0x"))
            return False

        is_float = _is_float(val_x) or _is_float(val_y)

        # 1. Standard normalized [0.0, 1.0] coordinates
        if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0 and (is_float or 0.0 < fx < 1.0 or 0.0 < fy < 1.0):
            ix = int(round(fx * canvas_w))
            iy = int(round(fy * canvas_h))
        # 2. Frontier VAM normalized [0, 1000] grid (e.g. Nemotron, Qwen-VL, Molmo emitting coordinates on 1000 grid or exceeding canvas bounds)
        elif (
            (is_1000_grid is True and 0.0 <= fx <= 1000.0 and 0.0 <= fy <= 1000.0)
            or (
                is_1000_grid is not False
                and (fx > 1.0 or fy > 1.0)
                and (fx > canvas_w or fy > canvas_h)
                and 0.0 <= fx <= 1000.0
                and 0.0 <= fy <= 1000.0
            )
        ):
            ix = int(round((fx / 1000.0) * canvas_w))
            iy = int(round((fy / 1000.0) * canvas_h))
        else:
            ix = int(fx)
            iy = int(fy)

        ix = min(canvas_w, max(0, ix))
        iy = min(canvas_h, max(0, iy))
        return ix, iy

    @classmethod
    def _extract_coord(
        cls,
        val: Any,
        is_1000_grid: Optional[bool] = None,
        canvas_w: int = 1280,
        canvas_h: int = 800,
    ) -> Optional[Tuple[int, int]]:
        """Safely extracts coordinate pair from list, tuple, dict, or string."""
        if val is None:
            return None
        if isinstance(val, dict):
            vx = val.get("x", val.get("0", val.get("x1")))
            vy = val.get("y", val.get("1", val.get("y1")))
            return cls._parse_coordinate_pair(vx, vy, canvas_w=canvas_w, canvas_h=canvas_h, is_1000_grid=is_1000_grid)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return cls._parse_coordinate_pair(val[0], val[1], canvas_w=canvas_w, canvas_h=canvas_h, is_1000_grid=is_1000_grid)
        if isinstance(val, str):
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
            if len(nums) >= 2:
                return cls._parse_coordinate_pair(nums[0], nums[1], canvas_w=canvas_w, canvas_h=canvas_h, is_1000_grid=is_1000_grid)
        return None

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        canvas_w: int = 1280,
        canvas_h: int = 800,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Action:
        if width is not None:
            canvas_w = width
        if height is not None:
            canvas_h = height

        # 1. Unwrap nested parameter dictionaries (parameters, arguments, args, params, input, choices, message)
        flat_data = dict(data)
        if "choices" in flat_data and isinstance(flat_data["choices"], list) and flat_data["choices"]:
            first_choice = flat_data["choices"][0]
            if isinstance(first_choice, dict) and "message" in first_choice:
                flat_data = dict(first_choice["message"])
        if "message" in flat_data and isinstance(flat_data["message"], dict):
            flat_data = dict(flat_data["message"])
        if "tool_calls" in flat_data and isinstance(flat_data["tool_calls"], list) and flat_data["tool_calls"]:
            flat_data = dict(flat_data["tool_calls"][0])
        if "function" in flat_data and isinstance(flat_data["function"], dict):
            fn = flat_data["function"]
            fn_name = fn.get("name")
            fn_args = fn.get("arguments", fn.get("parameters", fn.get("args", {})))
            if isinstance(fn_args, str):
                try:
                    norm_args = re.sub(r'\bTrue\b', 'true', fn_args)
                    norm_args = re.sub(r'\bFalse\b', 'false', norm_args)
                    norm_args = re.sub(r'\bNone\b', 'null', norm_args)
                    fn_args = json.loads(norm_args)
                except Exception:
                    try:
                        norm_args = norm_args.replace("'", '"')
                        fn_args = json.loads(norm_args)
                    except Exception:
                        fn_args = {}
            if isinstance(fn_args, dict):
                flat_data = dict(fn_args)
                flat_data["action"] = fn_name

        for nested_key in ("parameters", "arguments", "args", "params", "input"):
            nested_val = flat_data.get(nested_key)
            if isinstance(nested_val, dict):
                for k, v in nested_val.items():
                    if k not in flat_data or flat_data[k] is None or flat_data[k] in ("computer", "agent", "tool"):
                        flat_data[k] = v
            elif isinstance(nested_val, str):
                try:
                    norm_val = re.sub(r'\bTrue\b', 'true', nested_val)
                    norm_val = re.sub(r'\bFalse\b', 'false', norm_val)
                    norm_val = re.sub(r'\bNone\b', 'null', norm_val)
                    parsed_val = json.loads(norm_val)
                    if isinstance(parsed_val, dict):
                        for k, v in parsed_val.items():
                            if k not in flat_data or flat_data[k] is None or flat_data[k] in ("computer", "agent", "tool"):
                                flat_data[k] = v
                except Exception:
                    try:
                        norm_val = norm_val.replace("'", '"')
                        parsed_val = json.loads(norm_val)
                        if isinstance(parsed_val, dict):
                            for k, v in parsed_val.items():
                                if k not in flat_data or flat_data[k] is None or flat_data[k] in ("computer", "agent", "tool"):
                                    flat_data[k] = v
                    except Exception:
                        pass
        data = flat_data

        raw_kind = str(data.get("kind", data.get("action", data.get("name", data.get("tool", "wait"))))).lower().strip()
        
        # Map common aliases from diverse frontier models
        alias_map = {
            "click": ActionKind.LEFT_CLICK,
            "left_click": ActionKind.LEFT_CLICK,
            "mouse_click": ActionKind.LEFT_CLICK,
            "primary_click": ActionKind.LEFT_CLICK,
            "right_click": ActionKind.RIGHT_CLICK,
            "secondary_click": ActionKind.RIGHT_CLICK,
            "double_click": ActionKind.DOUBLE_CLICK,
            "triple_click": ActionKind.TRIPLE_CLICK,
            "middle_click": ActionKind.MIDDLE_CLICK,
            "mouse_move": ActionKind.MOUSE_MOVE,
            "move": ActionKind.MOUSE_MOVE,
            "hover": ActionKind.MOUSE_MOVE,
            "cursor_move": ActionKind.MOUSE_MOVE,
            "left_click_drag": ActionKind.LEFT_CLICK_DRAG,
            "drag": ActionKind.LEFT_CLICK_DRAG,
            "scroll": ActionKind.SCROLL,
            "mouse_scroll": ActionKind.SCROLL,
            "wheel": ActionKind.SCROLL,
            "type": ActionKind.TYPE,
            "type_text": ActionKind.TYPE_TEXT,
            "input_text": ActionKind.TYPE_TEXT,
            "write": ActionKind.TYPE,
            "key": ActionKind.KEY,
            "press": ActionKind.KEY,
            "press_key": ActionKind.KEY,
            "hotkey": ActionKind.KEY,
            "shortcut": ActionKind.KEY,
            "key_press": ActionKind.KEY,
            "keydown": ActionKind.KEY,
            "wait": ActionKind.WAIT,
            "sleep": ActionKind.WAIT,
            "delay": ActionKind.WAIT,
            "stop": ActionKind.STOP,
            "finish": ActionKind.STOP,
            "done": ActionKind.STOP,
            "complete": ActionKind.STOP,
        }
        try:
            kind = ActionKind(raw_kind)
        except (ValueError, KeyError):
            kind = alias_map.get(raw_kind, ActionKind.WAIT)

        # Handle button specification and click count overrides (e.g. Anthropic format {"action": "mouse_click", "button": "right"})
        btn = str(data.get("button", "")).lower().strip()
        raw_cc = data.get("click_count", 0)
        try:
            cc_val = int(raw_cc) if raw_cc is not None else 0
        except (ValueError, TypeError):
            cc_val = 0
        dbl_click = bool(data.get("double_click", False)) or cc_val == 2
        trp_click = bool(data.get("triple_click", False)) or cc_val == 3
        clear_first_flag = bool(data.get("clear_first", False))

        if kind in (ActionKind.LEFT_CLICK, ActionKind.WAIT) and raw_kind in ("click", "mouse_click", "left_click", "button_click"):
            if trp_click:
                kind = ActionKind.TRIPLE_CLICK
            elif dbl_click:
                kind = ActionKind.DOUBLE_CLICK
            elif btn in ("right", "secondary"):
                kind = ActionKind.RIGHT_CLICK
            elif btn in ("middle", "wheel"):
                kind = ActionKind.MIDDLE_CLICK
            else:
                kind = ActionKind.LEFT_CLICK

        is_1000: Optional[bool] = None
        if (
            data.get("grid") in (1000, "1000")
            or str(data.get("coordinate_system", "")).lower() in ("1000", "nemotron", "qwen", "molmo")
            or data.get("scale") in (1000, "1000")
            or data.get("is_1000_grid") is True
        ):
            is_1000 = True
        elif (
            str(data.get("coordinate_system", "")).lower() in ("pixel", "pixels", "raw", "screen", "normalized_0_1")
            or data.get("is_1000_grid") is False
        ):
            is_1000 = False

        coord = None
        for key_name in ("coordinate", "coords", "point", "position", "pos", "location", "target"):
            if key_name in data and data[key_name] is not None:
                coord = cls._extract_coord(data[key_name], is_1000_grid=is_1000, canvas_w=canvas_w, canvas_h=canvas_h)
                if coord is not None:
                    break
        if coord is None and "x" in data and "y" in data and data["x"] is not None and data["y"] is not None:
            coord = cls._parse_coordinate_pair(data["x"], data["y"], is_1000_grid=is_1000, canvas_w=canvas_w, canvas_h=canvas_h)

        start_c = None
        for s_key in ("start_coordinate", "start_coords", "start_point", "from", "start"):
            if s_key in data and data[s_key] is not None:
                start_c = cls._extract_coord(data[s_key], is_1000_grid=is_1000, canvas_w=canvas_w, canvas_h=canvas_h)
                if start_c is not None:
                    break
        if start_c is None and "x1" in data and "y1" in data:
            start_c = cls._parse_coordinate_pair(data["x1"], data["y1"], is_1000_grid=is_1000, canvas_w=canvas_w, canvas_h=canvas_h)

        end_c = None
        for e_key in ("end_coordinate", "end_coords", "end_point", "to", "end"):
            if e_key in data and data[e_key] is not None:
                end_c = cls._extract_coord(data[e_key], is_1000_grid=is_1000, canvas_w=canvas_w, canvas_h=canvas_h)
                if end_c is not None:
                    break
        if end_c is None and "x2" in data and "y2" in data:
            end_c = cls._parse_coordinate_pair(data["x2"], data["y2"], is_1000_grid=is_1000, canvas_w=canvas_w, canvas_h=canvas_h)

        # Fallback for left_click_drag when destination coordinate is passed as 'coordinate'
        if kind == ActionKind.LEFT_CLICK_DRAG and end_c is None and coord is not None:
            end_c = coord

        def _split_key_combo(s: str) -> List[str]:
            s_clean = s.strip().lower()
            if s_clean in ("-", "+"):
                return [s_clean]
            s_norm = re.sub(r'\s*[\+]\s*|\s+-\s+|(?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])', '|', s_clean)
            return [p.strip() for p in s_norm.split('|') if p.strip()]

        # Key parsing: normalize aliases (keys, key, hotkey, shortcut, press, value, combo)
        # And support Anthropic format (text: "win" for key action)
        keys = None
        raw_keys_val = None
        key_alias_list = ("keys", "key", "hotkey", "shortcut", "press", "combo")
        if kind == ActionKind.KEY:
            key_alias_list = ("keys", "key", "hotkey", "shortcut", "press", "combo", "value", "text")
        for k_alias in key_alias_list:
            if k_alias in data and data[k_alias] is not None:
                raw_keys_val = data[k_alias]
                break

        if raw_keys_val is not None:
            if isinstance(raw_keys_val, (list, tuple)):
                keys_list = []
                for item in raw_keys_val:
                    if isinstance(item, str):
                        for sub_k in _split_key_combo(item):
                            if sub_k:
                                keys_list.append(sub_k)
                    else:
                        sub_k_clean = str(item).strip().lower()
                        if sub_k_clean:
                            keys_list.append(sub_k_clean)
                keys = keys_list if keys_list else None
            elif isinstance(raw_keys_val, str):
                parts = _split_key_combo(raw_keys_val)
                keys = parts if parts else None
            else:
                s = str(raw_keys_val).strip().lower()
                keys = [s] if s else None

        # Extract HID codes if provided
        raw_hid = data.get("hid_codes", data.get("hid_code"))
        hid_list = None
        if raw_hid is not None:
            if isinstance(raw_hid, (list, tuple)):
                hid_list = [int(h) for h in raw_hid if h is not None]
            elif isinstance(raw_hid, (int, str)):
                try:
                    hid_list = [int(raw_hid)]
                except (ValueError, TypeError):
                    hid_list = None
        if kind == ActionKind.KEY and not keys and hid_list:
            resolved_keys = [hid_to_key(h) for h in hid_list if hid_to_key(h) is not None]
            if resolved_keys:
                keys = resolved_keys



        scroll_d = None
        try:
            if "scroll_delta" in data and data["scroll_delta"] is not None:
                sd = data["scroll_delta"]
                if isinstance(sd, (list, tuple)) and len(sd) >= 2:
                    scroll_d = (int(float(sd[0])), int(float(sd[1])))
                elif isinstance(sd, str):
                    nums = re.findall(r'[-+]?\d*\.?\d+', sd)
                    if len(nums) >= 2:
                        scroll_d = (int(float(nums[0])), int(float(nums[1])))
            elif "delta_y" in data or "delta_x" in data:
                scroll_d = (int(float(data.get("delta_x", 0))), int(float(data.get("delta_y", 0))))
            elif "direction" in data:
                direction = str(data["direction"]).lower().strip()
                amount_val = int(float(data.get("amount", data.get("clicks", 1)))) * 120
                if direction == "up":
                    scroll_d = (0, amount_val)
                elif direction == "down":
                    scroll_d = (0, -amount_val)
                elif direction == "right":
                    scroll_d = (amount_val, 0)
                elif direction == "left":
                    scroll_d = (-amount_val, 0)
        except (ValueError, TypeError):
            scroll_d = None

        # Text alias normalization (text, value, content, input, query, string, search, text_to_type)
        text_val = None
        for t_alias in ("text", "value", "content", "input", "query", "string", "search", "text_to_type"):
            if t_alias in data and data[t_alias] is not None:
                if isinstance(data[t_alias], (dict, list)):
                    continue
                if kind == ActionKind.KEY and t_alias in ("text", "value") and keys is not None:
                    continue
                text_val = data[t_alias]
                break

        duration_val = 0.0
        try:
            raw_dur = data.get("duration", data.get("seconds", data.get("time", 0.0)))
            if isinstance(raw_dur, str):
                raw_dur = re.sub(r"[^\d\.]", "", raw_dur)
            duration_val = float(raw_dur) if raw_dur else 0.0
        except (ValueError, TypeError):
            duration_val = 0.0
        press_enter_val = bool(data.get("press_enter", data.get("enter", False)))
        reason_str = str(data.get("reason", "")).strip()

        # Semantic fallback extraction from reason when model puts intent into reason
        if kind == ActionKind.KEY and not keys and reason_str:
            r_lower = reason_str.lower()
            # 1. Hotkey combination regex checked FIRST (e.g. ctrl+shift+esc, ctrl+s, win+r)
            hotkey_match = re.search(r'\b((?:ctrl|alt|shift|win|super)\s*[\+\-]\s*[a-z0-9]+(?:\s*[\+\-]\s*[a-z0-9]+)*)\b', r_lower)
            if hotkey_match:
                keys = _split_key_combo(hotkey_match.group(1))
            elif any(term in r_lower for term in ("start menu", "windows key", "win key", "win button", "super key", "open start", "start button", "press win", "press windows")):
                keys = ["win"]
            elif re.search(r'\b(?:enter|return)\b', r_lower):
                keys = ["enter"]
            elif re.search(r'\b(?:escape|esc)\b', r_lower):
                keys = ["esc"]
            elif re.search(r'\b(?:tab)\b', r_lower):
                keys = ["tab"]
            elif re.search(r'\b(?:backspace)\b', r_lower):
                keys = ["backspace"]

        if kind in (ActionKind.TYPE, ActionKind.TYPE_TEXT) and (text_val is None or str(text_val) == "") and reason_str:
            quote_match = re.search(r'''(?:type|search|input|write|enter)\s+(?:for\s+|:\s*)?['"]([^'"]+)['"]''', reason_str, re.IGNORECASE)
            if not quote_match:
                quote_match = re.search(r'''['"]([^'"]+)['"]''', reason_str)
            if quote_match:
                text_val = quote_match.group(1).strip()
            else:
                # Standard English extraction: 'Type <AppName> into search' or 'Type <AppName>'
                word_match = re.search(
                    r'(?:type|search|input|write|enter)\s+'
                    r'(?:for\s+|:\s*)?'
                    r'([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+)*?)'
                    r'(?:\s+(?:in|into|to|on|with|and)\b|\.|\s*$)',
                    reason_str,
                    re.IGNORECASE,
                )
                if word_match:
                    cand = word_match.group(1).strip()
                    if cand and cand.lower() not in ('the', 'a', 'an'):
                        text_val = cand

        if kind in (ActionKind.TYPE, ActionKind.TYPE_TEXT) and not press_enter_val and reason_str:
            r_low = reason_str.lower()
            if any(term in r_low for term in (
                "press enter", "hit enter", "with enter", "launch", "run", "search",
                "open", "execute", "dialog", "command", "find"
            )):
                press_enter_val = True

        # Guard against empty no-op actions
        if kind == ActionKind.KEY and not keys:
            kind = ActionKind.WAIT
            duration_val = max(0.2, duration_val)
            if not reason_str:
                reason_str = "Guarded against empty key action"
        elif kind in (ActionKind.TYPE, ActionKind.TYPE_TEXT) and (text_val is None or str(text_val) == "") and not press_enter_val:
            kind = ActionKind.WAIT
            duration_val = max(0.2, duration_val)
            if not reason_str:
                reason_str = "Guarded against empty type action"
        elif kind == ActionKind.MOUSE_MOVE and coord is None:
            kind = ActionKind.WAIT
            duration_val = max(0.2, duration_val)
            if not reason_str:
                reason_str = "Guarded against empty mouse_move action"
        elif kind == ActionKind.WAIT and duration_val <= 0.0:
            duration_val = 0.5

        # Extract drag chunks if provided
        drag_chunks_val = None
        for dc_key in ("drag_chunks", "chunks", "waypoints"):
            if dc_key in data and isinstance(data[dc_key], (list, tuple)):
                parsed_chunks = []
                for chunk_item in data[dc_key]:
                    cp = cls._extract_coord(chunk_item, is_1000_grid=is_1000, canvas_w=canvas_w, canvas_h=canvas_h)
                    if cp is not None:
                        parsed_chunks.append(cp)
                if parsed_chunks:
                    drag_chunks_val = parsed_chunks
                    break



        conf_val = 0.0
        try:
            if "confidence" in data and data["confidence"] is not None:
                conf_val = max(0.0, min(1.0, float(data["confidence"])))
                if math.isnan(conf_val) or math.isinf(conf_val):
                    conf_val = 0.0
        except (ValueError, TypeError):
            conf_val = 0.0

        meta_val = dict(data.get("metadata", {})) if isinstance(data.get("metadata"), dict) else {}

        return cls(
            kind=kind,
            coordinate=coord,
            start_coordinate=start_c,
            end_coordinate=end_c,
            text=str(text_val) if text_val is not None else None,
            keys=keys,
            scroll_delta=scroll_d,
            duration=duration_val,
            reason=reason_str,
            press_enter=press_enter_val,
            confidence=conf_val,
            clear_first=clear_first_flag,
            metadata=meta_val,
            drag_chunks=drag_chunks_val,
            hid_codes=hid_list,
        )


class ActionAdapter:
    """Normalizes diverse model wire formats into our internal deterministic Action schema.

    Supports:
    1. Anthropic Computer Use format ({"action": "key", "text": "Return"}, {"action": "mouse_move", "coordinate": [x, y]}, etc.)
    2. OpenAI / Tool-Calling format (nested function, parameters, arguments, args, input, tool_calls)
    3. AVCS Fast Action Grammar tokens (ACT:CLICK 292 768, ACT:TYPE <AppName>, ACT:KEY win, ACT:WAIT 0.5, ACT:STOP Done)
    4. General Agents API responses
    5. Fallback regex intent extraction from reason if payload fields are omitted.
    """

    ANTHROPIC_KEY_MAP = {
        "return": "enter",
        "enter": "enter",
        "backspace": "backspace",
        "tab": "tab",
        "space": "space",
        "escape": "esc",
        "page_up": "page_up",
        "page_down": "page_down",
        "super_l": "win",
        "super_r": "win",
        "super": "win",
        "ctrl_l": "ctrl",
        "ctrl_r": "ctrl",
        "alt_l": "alt",
        "alt_r": "alt",
        "shift_l": "shift",
        "shift_r": "shift",
    }

    @staticmethod
    def _split_comment_outside_quotes(text: str) -> Tuple[str, str]:
        """Splits trailing comment (# or //) while preserving # or // inside quoted strings."""
        in_quote = None
        for i, ch in enumerate(text):
            if ch in ('"', "'"):
                if in_quote == ch:
                    in_quote = None
                elif in_quote is None:
                    in_quote = ch
            elif in_quote is None:
                if ch == '#':
                    return text[:i].strip(), text[i + 1:].strip()
                if text[i:i + 2] == '//':
                    return text[:i].strip(), text[i + 2:].strip()
        return text.strip(), ""

    @classmethod
    def parse_grammar_token(cls, text: str) -> Optional[Action]:
        """Parses AVCS Fast Action Grammar tokens into internal Action objects."""
        s = text.strip()
        # Find ACT: token within text if embedded in reasoning
        m_act = re.search(r'\bACT:([A-Z_]+)(?:\s+(.*?))?(?:\n|$)', s, re.IGNORECASE)
        if not m_act:
            return None

        verb = m_act.group(1).upper()
        raw_args = (m_act.group(2) or "").strip()
        args_str, reason = cls._split_comment_outside_quotes(raw_args)

        if verb in ("CLICK", "LEFT_CLICK"):
            coords = re.findall(r'[-+]?\d*\.?\d+', args_str)
            if len(coords) >= 2:
                coord = Action._parse_coordinate_pair(coords[0], coords[1])
                return Action(kind=ActionKind.LEFT_CLICK, coordinate=coord, reason=reason)
            return Action(kind=ActionKind.LEFT_CLICK, reason=reason)

        elif verb == "RIGHT_CLICK":
            coords = re.findall(r'[-+]?\d*\.?\d+', args_str)
            if len(coords) >= 2:
                coord = Action._parse_coordinate_pair(coords[0], coords[1])
                return Action(kind=ActionKind.RIGHT_CLICK, coordinate=coord, reason=reason)
            return Action(kind=ActionKind.RIGHT_CLICK, reason=reason)

        elif verb == "DOUBLE_CLICK":
            coords = re.findall(r'[-+]?\d*\.?\d+', args_str)
            if len(coords) >= 2:
                coord = Action._parse_coordinate_pair(coords[0], coords[1])
                return Action(kind=ActionKind.DOUBLE_CLICK, coordinate=coord, reason=reason)
            return Action(kind=ActionKind.DOUBLE_CLICK, reason=reason)

        elif verb == "TRIPLE_CLICK":
            coords = re.findall(r'[-+]?\d*\.?\d+', args_str)
            if len(coords) >= 2:
                coord = Action._parse_coordinate_pair(coords[0], coords[1])
                return Action(kind=ActionKind.TRIPLE_CLICK, coordinate=coord, reason=reason)
            return Action(kind=ActionKind.TRIPLE_CLICK, reason=reason)

        elif verb in ("MOVE", "MOUSE_MOVE", "HOVER"):
            coords = re.findall(r'[-+]?\d*\.?\d+', args_str)
            if len(coords) >= 2:
                coord = Action._parse_coordinate_pair(coords[0], coords[1])
                return Action(kind=ActionKind.MOUSE_MOVE, coordinate=coord, reason=reason)
            return Action(kind=ActionKind.MOUSE_MOVE, reason=reason)

        elif verb in ("DRAG", "LEFT_CLICK_DRAG"):
            coords = re.findall(r'[-+]?\d*\.?\d+', args_str)
            if len(coords) >= 4:
                sc = Action._parse_coordinate_pair(coords[0], coords[1])
                ec = Action._parse_coordinate_pair(coords[2], coords[3])
                return Action(kind=ActionKind.LEFT_CLICK_DRAG, start_coordinate=sc, end_coordinate=ec, reason=reason)
            elif len(coords) >= 2:
                ec = Action._parse_coordinate_pair(coords[0], coords[1])
                return Action(kind=ActionKind.LEFT_CLICK_DRAG, end_coordinate=ec, reason=reason)
            return Action(kind=ActionKind.LEFT_CLICK_DRAG, reason=reason)

        elif verb == "SCROLL":
            nums = re.findall(r'[-+]?\d*\.?\d+', args_str)
            if len(nums) >= 4:
                coord = Action._parse_coordinate_pair(nums[0], nums[1])
                scroll_d = (int(float(nums[2])), int(float(nums[3])))
                return Action(kind=ActionKind.SCROLL, coordinate=coord, scroll_delta=scroll_d, reason=reason)
            elif len(nums) >= 2:
                scroll_d = (int(float(nums[0])), int(float(nums[1])))
                return Action(kind=ActionKind.SCROLL, scroll_delta=scroll_d, reason=reason)
            return Action(kind=ActionKind.SCROLL, scroll_delta=(0, -120), reason=reason)

        elif verb in ("TYPE", "TYPE_TEXT"):
            text_val = args_str
            # Handle quoted strings e.g. ACT:TYPE "hello world"
            q_m = re.match(r'^["\'](.*)["\']$', text_val)
            if q_m:
                text_val = q_m.group(1)
            act_kind = ActionKind.TYPE_TEXT if verb == "TYPE_TEXT" else ActionKind.TYPE
            return Action(kind=act_kind, text=text_val, reason=reason)

        elif verb == "KEY":
            # Can be "ctrl+s" or "win" or "ctrl shift esc"
            keys_list = [k.strip().lower() for k in re.split(r'[\s\+\-]+', args_str) if k.strip()]
            norm_keys = [cls.ANTHROPIC_KEY_MAP.get(k, k) for k in keys_list]
            return Action(kind=ActionKind.KEY, keys=norm_keys, reason=reason)

        elif verb == "WAIT":
            dur = 0.5
            nums = re.findall(r'[-+]?\d*\.?\d+', args_str)
            if nums:
                dur = float(nums[0])
            return Action(kind=ActionKind.WAIT, duration=dur, reason=reason)

        elif verb in ("STOP", "FINISH", "DONE"):
            stop_reason = args_str or reason or "Goal completed successfully"
            return Action(kind=ActionKind.STOP, reason=stop_reason)

        return None

    @classmethod
    def from_reason_fallback(cls, reason: str) -> Optional[Action]:
        """Extracts intent if payload fields were omitted by model."""
        if not reason or not isinstance(reason, str):
            return None
        r = reason.strip()
        r_lower = r.lower()

        # 1. Flexible coordinate extraction for click/move actions
        coord = None
        coord_match = re.search(r'(?:at|to|pos|coords?|position)?\s*[:=\[\(]?\s*(?:x\s*[:=]\s*)?(\d{1,4})\s*[,x\s]\s*(?:y\s*[:=]\s*)?(\d{1,4})\s*[\)\]]?', r, re.IGNORECASE)
        if coord_match:
            coord = (int(coord_match.group(1)), int(coord_match.group(2)))

        if any(term in r_lower for term in ("double click", "double-click", "double_click")):
            return Action(kind=ActionKind.DOUBLE_CLICK, coordinate=coord, reason=r)
        if any(term in r_lower for term in ("right click", "right-click", "right_click", "context menu")):
            return Action(kind=ActionKind.RIGHT_CLICK, coordinate=coord, reason=r)
        if any(term in r_lower for term in ("triple click", "triple-click", "triple_click")):
            return Action(kind=ActionKind.TRIPLE_CLICK, coordinate=coord, reason=r)
        if coord and any(term in r_lower for term in ("click", "press", "tap", "select", "focus")):
            return Action(kind=ActionKind.LEFT_CLICK, coordinate=coord, reason=r)
        if coord and any(term in r_lower for term in ("move", "hover", "cursor to")):
            return Action(kind=ActionKind.MOUSE_MOVE, coordinate=coord, reason=r)

        # 2. Key combo or key press
        hotkey_match = re.search(r'\b((?:ctrl|alt|shift|win|super)\s*[\+\-]\s*[a-z0-9]+(?:\s*[\+\-]\s*[a-z0-9]+)*)\b', r_lower)
        if hotkey_match:
            keys = [k.strip() for k in re.split(r'[\+\-\s]+', hotkey_match.group(1)) if k.strip()]
            norm_keys = [cls.ANTHROPIC_KEY_MAP.get(k, k) for k in keys]
            return Action(kind=ActionKind.KEY, keys=norm_keys, reason=r)

        if any(term in r_lower for term in ("start menu", "windows key", "win key", "press win", "super key", "open start")):
            return Action(kind=ActionKind.KEY, keys=["win"], reason=r)
        if re.search(r'\b(?:enter|return)\b', r_lower):
            return Action(kind=ActionKind.KEY, keys=["enter"], reason=r)
        if re.search(r'\b(?:escape|esc)\b', r_lower):
            return Action(kind=ActionKind.KEY, keys=["esc"], reason=r)
        if re.search(r'\b(?:tab)\b', r_lower):
            return Action(kind=ActionKind.KEY, keys=["tab"], reason=r)
        if re.search(r'\b(?:backspace)\b', r_lower):
            return Action(kind=ActionKind.KEY, keys=["backspace"], reason=r)

        # 3. Type intent
        quote_match = re.search(r'''(?:type|search|input|write|enter)\s+(?:for\s+|:\s*)?['"]([^'"]+)['"]''', r, re.IGNORECASE)
        if not quote_match:
            quote_match = re.search(r'''['"]([^'"]+)['"]''', r)
        if quote_match:
            return Action(kind=ActionKind.TYPE, text=quote_match.group(1).strip(), reason=r)

        # Standard English extraction: 'Type <AppName> into search' or 'Type <AppName>'
        word_match = re.search(
            r'(?:type|search|input|write|enter)\s+'
            r'(?:for\s+|:\s*)?'
            r'([a-zA-Z0-9_\-\.]+(?:\s+[a-zA-Z0-9_\-\.]+)*?)'
            r'(?:\s+(?:in|into|to|on|with|and)\b|\.|\s*$)',
            r,
            re.IGNORECASE,
        )
        if word_match:
            cand = word_match.group(1).strip()
            if cand and cand.lower() not in ('the', 'a', 'an'):
                return Action(kind=ActionKind.TYPE, text=cand, reason=r)

        # 4. Stop intent
        if any(term in r_lower for term in ("task complete", "goal complete", "finished", "all done", "goal achieved")):
            return Action(kind=ActionKind.STOP, reason=r)

        # 5. Wait intent
        wait_m = re.search(r'wait\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?', r_lower)
        if wait_m:
            return Action(kind=ActionKind.WAIT, duration=float(wait_m.group(1)), reason=r)

        return None

    @classmethod
    def from_response(cls, payload: Any) -> Action:
        """Alias for adapt: normalizes model response content into an Action."""
        return cls.adapt(payload)

    @classmethod
    def from_llm_json(cls, payload: Any) -> Action:
        """Parses LLM JSON payload (raw string, markdown codeblock, or dict) into an Action."""
        return cls.adapt(payload)

    @classmethod
    def adapt(cls, payload: Any) -> Action:
        """Master normalization method translating any model wire format into Action."""
        if isinstance(payload, Action):
            return payload

        # 0. List or tuple payload: adapt first action
        if isinstance(payload, (list, tuple)):
            if not payload:
                return Action(kind=ActionKind.WAIT, duration=0.5, reason="Empty action sequence")
            return cls.adapt(payload[0])

        # 1. String payload: could be Grammar token, JSON string, or reasoning text
        if isinstance(payload, str):
            s = payload.strip()
            # Fast check for Grammar token
            grammar_action = cls.parse_grammar_token(s)
            if grammar_action is not None:
                return grammar_action

            # Strip reasoning tags and markdown codeblocks if present
            s_clean = re.sub(r'<(?:think|thought|reasoning|extra_thought)>[\s\S]*?</(?:think|thought|reasoning|extra_thought)>', '', s, flags=re.IGNORECASE).strip()
            if "```" in s_clean:
                code_m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s_clean, re.IGNORECASE)
                if code_m:
                    s_clean = code_m.group(1).strip()

            # Try parsing as JSON string (exact or nested object)
            first_brace = s_clean.find("{")
            last_brace = s_clean.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                cand = s_clean[first_brace : last_brace + 1]
                cand_normalized = re.sub(r'\bTrue\b', 'true', cand)
                cand_normalized = re.sub(r'\bFalse\b', 'false', cand_normalized)
                cand_normalized = re.sub(r'\bNone\b', 'null', cand_normalized)
                clean_cand = re.sub(r",\s*([}\]])", r"\1", cand_normalized)
                try:
                    data = json.loads(clean_cand)
                    if isinstance(data, dict):
                        return cls.adapt(data)
                except Exception:
                    try:
                        clean_cand_q = clean_cand.replace("'", '"')
                        data = json.loads(clean_cand_q)
                        if isinstance(data, dict):
                            return cls.adapt(data)
                    except Exception:
                        pass

            first_bracket = s_clean.find("[")
            last_bracket = s_clean.rfind("]")
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                cand_arr = s_clean[first_bracket : last_bracket + 1]
                cand_arr_normalized = re.sub(r'\bTrue\b', 'true', cand_arr)
                cand_arr_normalized = re.sub(r'\bFalse\b', 'false', cand_arr_normalized)
                cand_arr_normalized = re.sub(r'\bNone\b', 'null', cand_arr_normalized)
                clean_arr = re.sub(r",\s*([}\]])", r"\1", cand_arr_normalized)
                try:
                    data = json.loads(clean_arr)
                    if isinstance(data, list):
                        return cls.adapt(data)
                except Exception:
                    try:
                        clean_arr_q = clean_arr.replace("'", '"')
                        data = json.loads(clean_arr_q)
                        if isinstance(data, list):
                            return cls.adapt(data)
                    except Exception:
                        pass

            # Fallback intent extraction from string
            fallback = cls.from_reason_fallback(s)
            if fallback is not None:
                return fallback

            return Action(kind=ActionKind.WAIT, duration=0.5, reason=s)

        # 2. Dict payload
        if isinstance(payload, dict):
            raw = dict(payload)

            # Anthropic API format: {"content": [{"type": "tool_use", "name": "computer", "input": {...}}]}
            if "content" in raw and isinstance(raw["content"], list):
                for block in raw["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        raw = dict(block.get("input", {}))
                        if "name" in block and block["name"] != "computer" and "action" not in raw:
                            raw["action"] = block["name"]
                        break
                    elif isinstance(block, dict) and block.get("type") == "text":
                        txt = block.get("text", "")
                        g_act = cls.parse_grammar_token(txt)
                        if g_act is not None:
                            return g_act

            # OpenAI Completion Choices format: {"choices": [{"message": {...}}]}
            if "choices" in raw and isinstance(raw["choices"], list) and raw["choices"]:
                first_choice = raw["choices"][0]
                if isinstance(first_choice, dict) and "message" in first_choice:
                    raw = dict(first_choice["message"])
            if "message" in raw and isinstance(raw["message"], dict):
                raw = dict(raw["message"])

            # OpenAI Tool Calls / Function Call unwrapping
            if "tool_calls" in raw and isinstance(raw["tool_calls"], list) and raw["tool_calls"]:
                raw = dict(raw["tool_calls"][0])
            if "function" in raw and isinstance(raw["function"], dict):
                fn = raw["function"]
                fn_name = fn.get("name")
                fn_args = fn.get("arguments", fn.get("parameters", fn.get("args", {})))
                if isinstance(fn_args, str):
                    try:
                        norm_args = re.sub(r'\bTrue\b', 'true', fn_args)
                        norm_args = re.sub(r'\bFalse\b', 'false', norm_args)
                        norm_args = re.sub(r'\bNone\b', 'null', norm_args)
                        fn_args = json.loads(norm_args)
                    except Exception:
                        try:
                            norm_args = norm_args.replace("'", '"')
                            fn_args = json.loads(norm_args)
                        except Exception:
                            fn_args = {}
                if isinstance(fn_args, dict):
                    raw = dict(fn_args)
                    raw["action"] = fn_name
            if "function_call" in raw and isinstance(raw["function_call"], dict):
                fn = raw["function_call"]
                fn_name = fn.get("name")
                fn_args = fn.get("arguments", {})
                if isinstance(fn_args, str):
                    try:
                        norm_args = re.sub(r'\bTrue\b', 'true', fn_args)
                        norm_args = re.sub(r'\bFalse\b', 'false', norm_args)
                        norm_args = re.sub(r'\bNone\b', 'null', norm_args)
                        fn_args = json.loads(norm_args)
                    except Exception:
                        try:
                            norm_args = norm_args.replace("'", '"')
                            fn_args = json.loads(norm_args)
                        except Exception:
                            fn_args = {}
                if isinstance(fn_args, dict):
                    raw = dict(fn_args)
                    raw["action"] = fn_name

            # Anthropic Computer Tool wrapper: {"name": "computer", "input": {...}}
            if raw.get("name") in ("computer", "agent", "tool") and isinstance(raw.get("input"), dict):
                raw = dict(raw["input"])

            # Anthropic Computer Use specific actions
            raw_action = str(raw.get("action", raw.get("kind", ""))).lower().strip()
            if raw_action == "screenshot":
                return Action(kind=ActionKind.WAIT, duration=0.5, reason=raw.get("reason", "Anthropic screenshot request"))
            elif raw_action == "cursor_position":
                return Action(kind=ActionKind.WAIT, duration=0.2, reason=raw.get("reason", "Anthropic cursor position query"))
            elif raw_action == "key":
                if "text" in raw and "keys" not in raw:
                    key_text = str(raw["text"]).strip().lower()
                    parts = [p.strip() for p in re.split(r'[\+\-\s]+', key_text) if p.strip()]
                    raw["keys"] = [cls.ANTHROPIC_KEY_MAP.get(p, p) for p in parts]
                elif "keys" in raw and isinstance(raw["keys"], list):
                    norm_k = []
                    for k in raw["keys"]:
                        for p in re.split(r'[\+\-\s]+', str(k).strip().lower()):
                            if p.strip():
                                norm_k.append(cls.ANTHROPIC_KEY_MAP.get(p.strip(), p.strip()))
                    raw["keys"] = norm_k

            # Parse via Action.from_dict
            action = Action.from_dict(raw)

            # Fallback intent recovery: if kind is WAIT but reason specifies clear intent
            if (
                action.kind == ActionKind.WAIT
                and action.reason
                and action.reason not in (
                    "Guarded against empty key action",
                    "Guarded against empty type action",
                    "Guarded against empty mouse_move action",
                )
                and not re.search(r'\b(?:wait|waiting|awaiting|settle|settling|pause|delay)\b', action.reason, re.IGNORECASE)
            ):
                recovered = cls.from_reason_fallback(action.reason)
                if recovered is not None:
                    return recovered

            return action

        # 3. Object with to_dict or dict attribute
        if hasattr(payload, "to_dict") and callable(getattr(payload, "to_dict")):
            return cls.adapt(payload.to_dict())
        if hasattr(payload, "__dict__"):
            return cls.adapt(vars(payload))

        return Action(kind=ActionKind.WAIT, duration=0.5, reason=str(payload))


