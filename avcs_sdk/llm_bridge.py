"""[5] llm_bridge.py: Local Single-Brain VLA Bridge (AVCS <---> Local VLA Engine).

Connects the real-time visual narrator, foveal vision streams, and motor hands
strictly through the self-contained local VLA engine (Qwen2.5-VL-3B + GUI-Actor +
ParallelActionHeads). Zero external cloud LLM endpoints or outbound HTTP APIs.

Invariants:
- Pure-pixel security: zero OCR, zero DOM, zero UIA.
- 100% offline local VLA: zero external cloud calls (OpenAI, Anthropic, Gemini, api.parthlabs.in).
- Pure ASCII: 100% pure ASCII characters.
- Coordinate privacy: zero coordinate arrays or bounding boxes in dialogue prompts.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .narrator import LiveCommentary

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_local_url(url: str) -> bool:
    """Return True only when url points to a local loopback host."""
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme in ("http", "https", "ws", "wss") and host in _LOCAL_HOSTS


def encode_patch_to_base64(patch: np.ndarray) -> str:
    """Encode numpy image patch (H, W, 3) or (H, W) into base64 PNG string."""
    if not isinstance(patch, np.ndarray):
        raise TypeError("patch must be a numpy ndarray")

    if patch.ndim == 2:
        img = Image.fromarray(patch.astype(np.uint8), mode="L")
    elif patch.ndim == 3:
        if patch.shape[2] == 1:
            img = Image.fromarray(patch[:, :, 0].astype(np.uint8), mode="L")
        elif patch.shape[2] == 3:
            img = Image.fromarray(patch.astype(np.uint8), mode="RGB")
        elif patch.shape[2] == 4:
            img = Image.fromarray(patch.astype(np.uint8), mode="RGBA")
        else:
            img = Image.fromarray(patch[:, :, :3].astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported patch dimensions: {patch.shape}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def summarize_foveal_patches(patches: List[np.ndarray]) -> str:
    """Summarize dense visual statistics across foveal image patches without coordinates."""
    if not patches:
        return ""

    summaries = []
    for idx, patch in enumerate(patches, 1):
        if not isinstance(patch, np.ndarray) or patch.size == 0 or patch.ndim < 2:
            continue
        h, w = patch.shape[:2]
        if h < 2 or w < 2:
            continue
        gray = patch if patch.ndim == 2 else np.mean(patch[:, :, :3], axis=2)
        mean_lum = float(np.mean(gray))
        std_contrast = float(np.std(gray))

        # Spatial gradient magnitude for structural edge and detail density
        dy, dx = np.gradient(gray.astype(float))
        edge_density = float(np.mean(np.hypot(dx, dy)))

        if edge_density > 15.0 and std_contrast > 30.0:
            texture_category = "dense_syntax_or_code"
        elif edge_density > 8.0:
            texture_category = "structured_grid_or_table"
        elif std_contrast > 15.0:
            texture_category = "graphical_control_or_icon"
        else:
            texture_category = "flat_or_blank"

        summaries.append(
            f"Patch {idx} ({w}x{h} px): {texture_category} (luminance: {mean_lum:.1f}, contrast: {std_contrast:.1f}, edge_density: {edge_density:.1f})"
        )

    if not summaries:
        return ""
    return "[Dense Foveal Vision Statistics]:\n" + "\n".join(f"- {s}" for s in summaries)


_ACTION_MAP: Dict[str, str] = {
    "left_click": "click",
    "click": "click",
    "primary_click": "click",
    "mouse_click": "click",
    "right_click": "right_click",
    "secondary_click": "right_click",
    "double_click": "double_click",
    "triple_click": "triple_click",
    "mouse_move": "mouse_move",
    "move": "mouse_move",
    "hover": "mouse_move",
    "cursor_move": "mouse_move",
    "left_click_drag": "left_click_drag",
    "drag": "left_click_drag",
    "drag_start": "left_click_drag",
    "scroll": "scroll",
    "mouse_scroll": "scroll",
    "wheel": "scroll",
    "key": "hotkey",
    "hotkey": "hotkey",
    "press": "hotkey",
    "press_key": "hotkey",
    "shortcut": "hotkey",
    "key_press": "hotkey",
    "type": "type",
    "type_text": "type",
    "input_text": "type",
    "write": "type",
    "wait": "wait",
    "sleep": "wait",
    "delay": "wait",
    "stop": "finish",
    "finish": "finish",
    "done": "finish",
    "complete": "finish",
    "abort": "abort",
    "switch_app": "switch_app",
    "recover": "recover",
}
_VALID_COMMANDS = frozenset(_ACTION_MAP.values())
_POINTER_COMMANDS = frozenset(
    {"click", "right_click", "double_click", "triple_click", "mouse_move", "left_click_drag"}
)


def _map_action_name(raw_action: str) -> str:
    """Map raw VLA/runtime action primitive names onto canonical LLMCommand actions."""
    return _ACTION_MAP.get(str(raw_action or "").strip().lower(), "wait")


def _coerce_coord(value: Any) -> Optional[Tuple[int, int]]:
    """Convert a 2-element sequence into an integer (x, y) tuple, or None."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        fx, fy = float(value[0]), float(value[1])
        if not (np.isfinite(fx) and np.isfinite(fy)):
            return None
        return (int(round(fx)), int(round(fy)))
    except (TypeError, ValueError):
        return None


def _coerce_keys(value: Any) -> Optional[List[str]]:
    """Normalize key names from string or sequence into a non-empty list."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split("+") if p.strip()]
        return parts or None
    if isinstance(value, (list, tuple)):
        parts = [str(k).strip() for k in value if str(k).strip()]
        return parts or None
    return None


def _coerce_scroll_delta(value: Any) -> Optional[Tuple[int, int]]:
    """Normalize scroll delta (dx, dy) or scalar vertical delta."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not np.isfinite(float(value)):
            return None
        return (0, int(round(float(value))))
    return _coerce_coord(value)


@dataclass(frozen=True)
class LLMCommand:
    """Action directive emitted by the local VLA engine."""
    action: str                  # "click", "right_click", "double_click", "triple_click", "mouse_move", "left_click_drag", "scroll", "type", "hotkey", "wait", "finish", "abort", "switch_app", "recover"
    target: Optional[str] = None # Description of target element
    text: Optional[str] = None   # Text to type
    reason: Optional[str] = None # Cognitive rationale
    source: str = "local_vla"    # "local_vla" or "heuristic_fallback"
    coordinate: Optional[Tuple[int, int]] = None
    start_coordinate: Optional[Tuple[int, int]] = None
    end_coordinate: Optional[Tuple[int, int]] = None
    keys: Optional[List[str]] = None
    scroll_delta: Optional[Tuple[int, int]] = None
    confidence: float = 0.0


class LLMBridge:
    """Local single-brain VLA bridge between AVCS perception and motor control.

    All external cloud LLM endpoints (OpenAI, Anthropic, Gemini, api.parthlabs.in)
    have been removed. Perception, action prediction, and dialogue route strictly
    through the local VLA engine (in-process ``EmbeddedVAMRuntime`` or a local
    loopback ``FastVLAInferenceEngine`` server).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        vision_enabled: bool = False,
        vla_engine: Optional[Any] = None,
    ) -> None:
        self.api_key = ""
        raw_url = (
            base_url
            or os.environ.get("AVCS_VLA_SERVER_URL")
            or os.environ.get("AVCS_LOCAL_VAM_ENDPOINT")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        if not _is_local_url(raw_url):
            raise ValueError(
                f"External cloud LLM URLs are disabled ({raw_url!r}); "
                "AVCS operates 100% locally via the local VLA engine."
            )
        self.base_url = raw_url
        self.model_name = (
            model_name
            or os.environ.get("AVCS_LOCAL_VAM_MODEL")
            or "Qwen2.5-VL-3B-Instruct"
        )
        self.credential_source = "local_vla"
        self.timeout = timeout
        self.vision_enabled = vision_enabled
        self._vla_engine = vla_engine
        self._vla_engine_attempted = False
        self._embedded_runtime: Optional[Any] = None
        self._embedded_attempted = False
        self.conversation_history: List[Dict[str, Any]] = []

    def _get_local_vla_engine(self) -> Optional[Any]:
        """Lazily initialize FastVLAInferenceEngine for local single-brain VLA inference."""
        if self._vla_engine is not None:
            return self._vla_engine
        if self._vla_engine_attempted:
            return None
        self._vla_engine_attempted = True
        try:
            from avcs.runtime.vla_engine import FastVLAInferenceEngine
            self._vla_engine = FastVLAInferenceEngine()
            return self._vla_engine
        except Exception:
            return None

    def _get_embedded_runtime(self) -> Optional[Any]:
        """Lazily initialize the in-process EmbeddedVAMRuntime for local inference."""
        if self._embedded_runtime is not None:
            return self._embedded_runtime
        if self._embedded_attempted:
            return None
        self._embedded_attempted = True
        try:
            from avcs.models.local_model import EmbeddedVAMRuntime
            self._embedded_runtime = EmbeddedVAMRuntime()
            return self._embedded_runtime
        except Exception:
            return None

    def _call_local_vla(
        self,
        user_goal: str,
        commentary: LiveCommentary,
        fovea_patches: Optional[List[np.ndarray]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> Optional[LLMCommand]:
        """Run local single-brain VLA inference without any external cloud calls."""
        # Encode frame or representative patch for in-process VLA runtime
        img_array = frame if isinstance(frame, np.ndarray) and frame.size > 0 else None
        if img_array is None and fovea_patches:
            for patch in fovea_patches:
                if isinstance(patch, np.ndarray) and patch.size > 0:
                    img_array = patch
                    break

        image_bytes = b""
        if img_array is not None:
            buf = io.BytesIO()
            if img_array.ndim == 2:
                Image.fromarray(img_array.astype(np.uint8), mode="L").convert("RGB").save(buf, format="JPEG")
            else:
                Image.fromarray(img_array[:, :, :3].astype(np.uint8), mode="RGB").save(buf, format="JPEG")
            image_bytes = buf.getvalue()

        # 1. FastVLAInferenceEngine (injected or lazily initialized)
        engine = self._get_local_vla_engine()
        if engine is not None and img_array is not None:
            try:
                if hasattr(engine, "act") and callable(engine.act):
                    res = engine.act(img_array, user_goal)
                elif hasattr(engine, "infer") and callable(engine.infer):
                    res = engine.infer(img_array, user_goal)
                else:
                    res = None

                if isinstance(res, dict):
                    raw_act = str(res.get("action") or res.get("action_name") or "wait").strip().lower()
                    mapped = _map_action_name(raw_act)
                    coord = _coerce_coord(res.get("coordinate"))
                    start_coord = _coerce_coord(res.get("start_coordinate"))
                    end_coord = _coerce_coord(res.get("end_coordinate"))
                    keys = _coerce_keys(res.get("keys"))
                    scroll_delta = _coerce_scroll_delta(res.get("scroll_delta"))
                    text_val = str(res["text"]) if res.get("text") is not None else None
                    target_val = (
                        str(res["target"])
                        if res.get("target") is not None
                        else (user_goal if mapped in _POINTER_COMMANDS else None)
                    )
                    raw_conf = res.get("peak_probability", res.get("confidence", 0.0))
                    try:
                        conf_val = float(raw_conf) if raw_conf is not None and np.isfinite(float(raw_conf)) else 0.0
                    except (TypeError, ValueError):
                        conf_val = 0.0
                    return LLMCommand(
                        action=mapped,
                        target=target_val,
                        text=text_val,
                        reason=str(res.get("reason") or f"Local FastVLAInferenceEngine predicted {raw_act}"),
                        source="local_vla",
                        coordinate=coord,
                        start_coordinate=start_coord,
                        end_coordinate=end_coord,
                        keys=keys,
                        scroll_delta=scroll_delta,
                        confidence=conf_val,
                    )
            except Exception:
                pass

        # 2. In-process EmbeddedVAMRuntime (zero-network local CPU neural weights)
        runtime = self._get_embedded_runtime()
        if runtime is not None and image_bytes:
            try:
                act_obj, _lat = runtime.predict(image_bytes, user_goal, step_index=len(self.conversation_history) // 2 + 1)
                act_kind = getattr(act_obj.kind, "value", str(act_obj.kind)).strip().lower()
                mapped = _map_action_name(act_kind)
                return LLMCommand(
                    action=mapped,
                    target=user_goal if mapped in _POINTER_COMMANDS else None,
                    text=getattr(act_obj, "text", None),
                    reason=getattr(act_obj, "reason", None) or "Local EmbeddedVAMRuntime directive",
                    source="local_vla",
                    coordinate=_coerce_coord(getattr(act_obj, "coordinate", None)),
                    start_coordinate=_coerce_coord(getattr(act_obj, "start_coordinate", None)),
                    end_coordinate=_coerce_coord(getattr(act_obj, "end_coordinate", None)),
                    keys=_coerce_keys(getattr(act_obj, "keys", None)),
                    scroll_delta=_coerce_scroll_delta(getattr(act_obj, "scroll_delta", None)),
                    confidence=float(getattr(act_obj, "confidence", 0.0) or 0.0),
                )
            except Exception:
                pass

        return None

    @staticmethod
    def _parse_command(content: str, goal: str) -> LLMCommand:
        """Extract a single JSON directive emitted by the local VLA model."""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"model reply contained no JSON object: {content!r}")

        data: Dict[str, Any] = json.loads(match.group(0))
        raw_action = str(data.get("action", "")).strip().lower()
        if raw_action not in _ACTION_MAP and raw_action not in _VALID_COMMANDS:
            raise ValueError(f"model returned an unknown action: {raw_action!r}")
        action = _map_action_name(raw_action)

        target = data.get("target")
        text = data.get("text")
        return LLMCommand(
            action=action,
            target=str(target) if target else None,
            text=str(text) if text else None,
            reason=str(data.get("reason") or f"local VLA directive for goal: {goal}"),
            source="local_vla",
            coordinate=_coerce_coord(data.get("coordinate")),
            start_coordinate=_coerce_coord(data.get("start_coordinate")),
            end_coordinate=_coerce_coord(data.get("end_coordinate")),
            keys=_coerce_keys(data.get("keys")),
            scroll_delta=_coerce_scroll_delta(data.get("scroll_delta")),
            confidence=float(data.get("peak_probability", data.get("confidence", 0.0)) or 0.0),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan_next_step(
        self,
        user_goal: str,
        commentary: LiveCommentary,
        fovea_patches: Optional[List[np.ndarray]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> LLMCommand:
        """Route visual commentary and patches through the local VLA engine."""
        prompt_sections = [
            f"Goal: {user_goal}",
            f"Current Screen State:\n{commentary.to_llm_prompt()}",
        ]

        if fovea_patches:
            summary = summarize_foveal_patches(fovea_patches)
            if summary:
                prompt_sections.append(summary)

        prompt = "\n\n".join(prompt_sections)

        command = self._call_local_vla(
            user_goal=user_goal,
            commentary=commentary,
            fovea_patches=fovea_patches,
            frame=frame,
        )
        if command is None:
            command = self._heuristic_fallback(user_goal, commentary, fovea_patches)

        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": command.reason or command.action})
        return command

    def describe_screen(
        self,
        frame_or_patches: Union[np.ndarray, List[np.ndarray]],
        prompt: str = "Describe what is visible on this screen in detail.",
    ) -> str:
        """Produce a strictly local pixel-derived summary of screen frame or patches.

        When vision_enabled=False, raises PermissionError to preserve privacy.
        External cloud HTTP requests are permanently disabled.
        """
        if not self.vision_enabled:
            raise PermissionError("Screen image egress disabled. Set vision_enabled=True to inspect pixels locally.")

        if isinstance(frame_or_patches, np.ndarray):
            patches = [frame_or_patches]
        elif isinstance(frame_or_patches, (list, tuple)):
            patches = list(frame_or_patches)
        else:
            raise TypeError("frame_or_patches must be a numpy ndarray or a list of numpy ndarrays")

        summary = summarize_foveal_patches(patches) or "Local frame contains uniform/blank visual content."
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": summary})
        return summary

    def _heuristic_fallback(
        self,
        user_goal: str,
        commentary: LiveCommentary,
        fovea_patches: Optional[List[np.ndarray]] = None,
    ) -> LLMCommand:
        """Offline planner used only when no model is reachable.

        Results are tagged ``source="heuristic_fallback"`` so callers never
        mistake this for a real model decision.
        """
        goal_lower = user_goal.lower()

        if commentary.has_popup:
            return LLMCommand(
                action="click",
                target="Dismiss or OK button",
                reason="Dismissing unexpected screen dialog.",
                source="heuristic_fallback",
            )

        if not commentary.is_stable and commentary.surprise_score > 0.35:
            return LLMCommand(
                action="wait",
                reason="Screen content is actively loading or rendering.",
                source="heuristic_fallback",
            )

        if "switch" in goal_lower or "focus" in goal_lower:
            return LLMCommand(
                action="switch_app",
                target=user_goal.replace("switch to", "").replace("focus", "").strip(),
                reason="Switching active desktop window context.",
                source="heuristic_fallback",
            )

        if "type" in goal_lower or "write" in goal_lower:
            return LLMCommand(
                action="type",
                text=user_goal.replace("type", "").replace("write", "").strip(),
                reason="Executing user requested typing operation.",
                source="heuristic_fallback",
            )

        return LLMCommand(
            action="finish",
            reason="Goal completed or waiting for next user instruction.",
            source="heuristic_fallback",
        )


def describe_screen(
    frame_or_patches: Union[np.ndarray, List[np.ndarray]],
    prompt: str = "Describe what is visible on this screen in detail.",
    bridge: Optional[LLMBridge] = None,
) -> str:
    """Module-level screen describer using LLMBridge.

    When vision_enabled=True on the bridge, sends pixels to the LLM.
    When vision_enabled=False (or if bridge is omitted), raises PermissionError.
    """
    if bridge is None:
        bridge = LLMBridge(vision_enabled=False)
    return bridge.describe_screen(frame_or_patches, prompt=prompt)
