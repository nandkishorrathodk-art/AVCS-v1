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
                    fn_args = json.loads(fn_args)
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
            "mouse_scroll": ActionKind.SCROLL,
            "wheel": ActionKind.SCROLL,
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

        scroll_d = None
        if "scroll_delta" in data and data["scroll_delta"] and len(data["scroll_delta"]) >= 2:
            sd = data["scroll_delta"]
            scroll_d = (int(float(sd[0])), int(float(sd[1])))
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


class ActionAdapter:
    """Normalizes diverse model wire formats into our internal deterministic Action schema.

    Supports:
    1. Anthropic Computer Use format ({"action": "key", "text": "Return"}, {"action": "mouse_move", "coordinate": [x, y]}, etc.)
    2. OpenAI / Tool-Calling format (nested function, parameters, arguments, args, input, tool_calls)
    3. AVCS Fast Action Grammar tokens (ACT:CLICK 292 768, ACT:TYPE Notepad, ACT:KEY win, ACT:WAIT 0.5, ACT:STOP Done)
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

        elif verb == "TYPE":
            text_val = args_str
            # Handle quoted strings e.g. ACT:TYPE "hello world"
            q_m = re.match(r'^["\'](.*)["\']$', text_val)
            if q_m:
                text_val = q_m.group(1)
            return Action(kind=ActionKind.TYPE, text=text_val, reason=reason)

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
            return Action(kind=ActionKind.TYPE, text=quote_match.group(1), reason=r)
        
        # Hinglish and standard sentence typing
        type_word_match = re.search(r'(?:type|search|input|write)\s+(?:for\s+|karo\s+|kijiye\s+)?([a-zA-Z0-9_\-\.\s]+?)(?:\s+(?:in|into|to|on|with|aur|and)\s+|\s*$)', r, re.IGNORECASE)
        if type_word_match:
            candidate = type_word_match.group(1).strip()
            if candidate.lower() not in ("karo", "kijiye", "kar", "the", "a", "an", "on", "in", "it"):
                return Action(kind=ActionKind.TYPE, text=candidate, reason=r)

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
    def adapt(cls, payload: Any) -> Action:
        """Master normalization method translating any model wire format into Action."""
        if isinstance(payload, Action):
            return payload

        # 1. String payload: could be Grammar token, JSON string, or reasoning text
        if isinstance(payload, str):
            s = payload.strip()
            # Fast check for Grammar token
            grammar_action = cls.parse_grammar_token(s)
            if grammar_action is not None:
                return grammar_action

            # Try parsing as JSON string
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    data = json.loads(s)
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
                        fn_args = json.loads(fn_args)
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
                        fn_args = json.loads(fn_args)
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
            if action.kind == ActionKind.WAIT and action.reason and action.reason not in (
                "Guarded against empty key action",
                "Guarded against empty type action",
                "Guarded against empty mouse_move action",
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


