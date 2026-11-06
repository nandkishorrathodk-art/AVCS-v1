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
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK_DRAG = "left_click_drag"
    SCROLL = "scroll"
    TYPE = "type"
    KEY = "key"
    WAIT = "wait"
    STOP = "stop"


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

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind.value}
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
        if self.duration:
            d["duration"] = self.duration
        if self.reason:
            d["reason"] = self.reason
        if self.press_enter:
            d["press_enter"] = self.press_enter
        return d

    @staticmethod
    def _parse_coordinate_pair(val_x: Any, val_y: Any, canvas_w: int = 1280, canvas_h: int = 800) -> Optional[Tuple[int, int]]:
        """Parses coordinates, automatically converting normalized [0.0, 1.0] floats to pixels."""
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
        # 2. Frontier VAM normalized [0, 1000] grid (e.g. Nemotron, Qwen-VL, Molmo emitting fy > canvas_h)
        elif (canvas_h < fy <= 1000.0 and 0.0 <= fx <= 1000.0):
            ix = int(round((fx / 1000.0) * canvas_w))
            iy = int(round((fy / 1000.0) * canvas_h))
        else:
            ix = int(fx)
            iy = int(fy)

        ix = min(canvas_w, max(0, ix))
        iy = min(canvas_h, max(0, iy))
        return ix, iy

    @classmethod
    def _extract_coord(cls, val: Any) -> Optional[Tuple[int, int]]:
        """Safely extracts coordinate pair from list, tuple, dict, or string."""
        if val is None:
            return None
        if isinstance(val, dict):
            vx = val.get("x", val.get("0", val.get("x1")))
            vy = val.get("y", val.get("1", val.get("y1")))
            return cls._parse_coordinate_pair(vx, vy)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return cls._parse_coordinate_pair(val[0], val[1])
        if isinstance(val, str):
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
            if len(nums) >= 2:
                return cls._parse_coordinate_pair(nums[0], nums[1])
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Action:
        # 1. Unwrap nested parameter dictionaries (parameters, arguments, args, params, input)
        flat_data = dict(data)
        for nested_key in ("parameters", "arguments", "args", "params", "input"):
            nested_val = flat_data.get(nested_key)
            if isinstance(nested_val, dict):
                for k, v in nested_val.items():
                    if k not in flat_data or flat_data[k] is None or flat_data[k] in ("computer", "agent", "tool"):
                        flat_data[k] = v
            elif isinstance(nested_val, str):
                try:
                    parsed_val = json.loads(nested_val)
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
            "mouse_move": ActionKind.MOUSE_MOVE,
            "move": ActionKind.MOUSE_MOVE,
            "hover": ActionKind.MOUSE_MOVE,
            "cursor_move": ActionKind.MOUSE_MOVE,
            "left_click_drag": ActionKind.LEFT_CLICK_DRAG,
            "drag": ActionKind.LEFT_CLICK_DRAG,
            "scroll": ActionKind.SCROLL,
            "type": ActionKind.TYPE,
            "type_text": ActionKind.TYPE,
            "input_text": ActionKind.TYPE,
            "write": ActionKind.TYPE,
            "key": ActionKind.KEY,
            "press": ActionKind.KEY,
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
        kind = alias_map.get(raw_kind, ActionKind.WAIT)

        coord = None
        for key_name in ("coordinate", "coords", "point", "position", "pos", "location", "target"):
            if key_name in data and data[key_name] is not None:
                coord = cls._extract_coord(data[key_name])
                if coord is not None:
                    break
        if coord is None and "x" in data and "y" in data and data["x"] is not None and data["y"] is not None:
            coord = cls._parse_coordinate_pair(data["x"], data["y"])

        start_c = None
        for s_key in ("start_coordinate", "start_coords", "start_point", "from", "start"):
            if s_key in data and data[s_key] is not None:
                start_c = cls._extract_coord(data[s_key])
                if start_c is not None:
                    break
        if start_c is None and "x1" in data and "y1" in data:
            start_c = cls._parse_coordinate_pair(data["x1"], data["y1"])

        end_c = None
        for e_key in ("end_coordinate", "end_coords", "end_point", "to", "end"):
            if e_key in data and data[e_key] is not None:
                end_c = cls._extract_coord(data[e_key])
                if end_c is not None:
                    break
        if end_c is None and "x2" in data and "y2" in data:
            end_c = cls._parse_coordinate_pair(data["x2"], data["y2"])

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

        scroll_d = None
        if "scroll_delta" in data and data["scroll_delta"] and len(data["scroll_delta"]) >= 2:
            sd = data["scroll_delta"]
            scroll_d = (int(float(sd[0])), int(float(sd[1])))
        elif "delta_y" in data or "delta_x" in data:
            scroll_d = (int(float(data.get("delta_x", 0))), int(float(data.get("delta_y", 0))))

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

        duration_val = float(data.get("duration", data.get("seconds", data.get("time", 0.0))))
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

        if kind == ActionKind.TYPE and (text_val is None or str(text_val) == "") and reason_str:
            quote_match = re.search(r'''(?:type|search|input|write|enter)\s+(?:for\s+|:\s*)?['"]([^'"]+)['"]''', reason_str, re.IGNORECASE)
            if not quote_match:
                quote_match = re.search(r'''['"]([^'"]+)['"]''', reason_str)
            if quote_match:
                text_val = quote_match.group(1)
            else:
                # 1. Hinglish prefix: 'Notepad type karo'
                before_karo = re.search(r'([a-zA-Z0-9_\-\.]+)\s+(?:type|search|input|write|enter|khojo|likho)\s+(?:karo|karein|kijiye|kar|do)\b', reason_str, re.IGNORECASE)
                if before_karo:
                    text_val = before_karo.group(1)
                else:
                    # 2. Hinglish infix: 'Type karo Notepad'
                    after_karo = re.search(r'(?:type|search|input|write|enter)\s+(?:karo|karein|kijiye|kar)\s+([a-zA-Z0-9_\-\.]+)', reason_str, re.IGNORECASE)
                    if after_karo:
                        text_val = after_karo.group(1)
                    else:
                        # 3. Standard English: 'Type Notepad into search' or 'Type Notepad'
                        word_match = re.search(r'(?:type|search|input|write|enter)\s+(?:for\s+|:\s*)?([a-zA-Z0-9_\-\.]+)(?:\s+in|\s+into|\s+and|\s+to|\.|\s*$)', reason_str, re.IGNORECASE)
                        if word_match and word_match.group(1).lower() not in ('karo', 'karein', 'kijiye', 'kar', 'the', 'a', 'an'):
                            text_val = word_match.group(1)

        if kind == ActionKind.TYPE and not press_enter_val and reason_str:
            if "press enter" in reason_str.lower() or "hit enter" in reason_str.lower():
                press_enter_val = True

        # Guard against empty no-op actions
        if kind == ActionKind.KEY and not keys:
            kind = ActionKind.WAIT
            duration_val = max(0.2, duration_val)
            if not reason_str:
                reason_str = "Guarded against empty key action"
        elif kind == ActionKind.TYPE and (text_val is None or str(text_val) == "") and not press_enter_val:
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
        )

