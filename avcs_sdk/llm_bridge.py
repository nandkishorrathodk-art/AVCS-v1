"""[5] llm_bridge.py: Live 2-Way Communication Bridge (AVCS <---> LLM Stream).

Connects the real-time visual narrator, foveal vision streams, and motor hands
to System 2 LLMs (Gemini Flash, Claude Sonnet, or Meta Llama 3.2). Enables the
LLM to guide execution with multimodal foveal high-acuity patch transport and
dense visual statistics.

Transport is a plain OpenAI-compatible ``/chat/completions`` POST over the
standard library, supporting both text and multimodal image payloads.

Invariants:
- Pure-pixel security: zero OCR, zero DOM, zero UIA.
- Application agnosticism: zero hardcoded proprietary application names.
- Pure ASCII: 100% pure ASCII characters.
- Coordinate privacy: zero coordinate arrays or bounding boxes in prompts.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from .narrator import LiveCommentary

DEFAULT_BASE_URL = "https://api.parthlabs.in/v1"


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


@dataclass(frozen=True)
class LLMCommand:
    """Action directive emitted by System 2 LLM."""
    action: str                  # "click", "type", "wait", "finish", "abort", "switch_app", "hotkey", "recover"
    target: Optional[str] = None # Description of target element
    text: Optional[str] = None   # Text to type
    reason: Optional[str] = None # Cognitive rationale
    source: str = "llm"          # "llm" or "heuristic_fallback"


class LLMBridge:
    """The 2-way streaming dialogue bridge between AVCS and System 2."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        vision_enabled: bool = False,
    ) -> None:
        """``vision_enabled`` defaults to False to honor the System 1/System 2
        boundary in ``docs/AVCS_ENGINEERING_RULES.md``: screen images, crops and
        their encodings must not be sent to System 2. Enabling it is an explicit
        opt-in that a caller (and a product/privacy decision) must make, not a
        default a caller inherits silently.
        """
        self.api_key = (
            api_key
            or os.environ.get("AVC_API_KEY")
            or os.environ.get("AVCS_API_KEY")
            or os.environ.get("AVCS_LLM_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.base_url = (
            base_url
            or os.environ.get("AVC_BASE_URL")
            or os.environ.get("AVCS_BASE_URL")
            or os.environ.get("AVCS_LLM_BASE_URL")
            or os.environ.get("AVCS_LLM_ENDPOINT")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model_name = (
            model_name
            or os.environ.get("AVC_MODEL")
            or os.environ.get("AVCS_LLM_MODEL")
            or ("avc-reasoner-v1" if "api.parthlabs.in" in self.base_url else "gemini-flash-lite-latest")
        )
        self.timeout = timeout
        self.vision_enabled = vision_enabled
        self.conversation_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Real model transport
    # ------------------------------------------------------------------ #
    def _call_llm(
        self,
        prompt: str,
        fovea_patches: Optional[List[np.ndarray]] = None,
    ) -> str:
        """POST the prompt and optional multimodal foveal patches to OpenAI-compatible endpoint."""
        if not self.api_key:
            raise RuntimeError("no API key configured for LLMBridge")

        if fovea_patches and len(fovea_patches) > 0:
            content_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for patch in fovea_patches:
                try:
                    b64_png = encode_patch_to_base64(patch)
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                    })
                except Exception:
                    pass
            messages = [{"role": "user", "content": content_items}]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_command(content: str, goal: str) -> LLMCommand:
        """Extract the single JSON directive the model was asked to return."""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"model reply contained no JSON object: {content!r}")

        data: Dict[str, Any] = json.loads(match.group(0))
        action = str(data.get("action", "")).strip().lower()
        valid_actions = {"click", "type", "wait", "finish", "abort", "switch_app", "hotkey", "recover"}
        if action not in valid_actions:
            raise ValueError(f"model returned an unknown action: {action!r}")

        target = data.get("target")
        text = data.get("text")
        return LLMCommand(
            action=action,
            target=str(target) if target else None,
            text=str(text) if text else None,
            reason=str(data.get("reason") or f"model directive for goal: {goal}"),
            source="llm",
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan_next_step(
        self,
        user_goal: str,
        commentary: LiveCommentary,
        fovea_patches: Optional[List[np.ndarray]] = None,
    ) -> LLMCommand:
        """Sends live commentary and optional foveal patches to LLM and parses the next action directive."""
        prompt_sections = [
            f"Goal: {user_goal}",
            f"Current Screen State:\n{commentary.to_llm_prompt()}",
        ]

        if fovea_patches:
            summary = summarize_foveal_patches(fovea_patches)
            if summary:
                prompt_sections.append(summary)

        prompt_sections.append(
            "What should AVCS do next? Reply ONLY with a single JSON object in this format:\n"
            '{"action": "click"|"type"|"wait"|"finish"|"switch_app", "target": "element_name", "text": "text_if_type", "reason": "short explanation"}'
        )
        prompt = "\n\n".join(prompt_sections)

        if not self.api_key:
            return self._heuristic_fallback(user_goal, commentary, fovea_patches)

        try:
            content = self._call_llm(prompt, fovea_patches=fovea_patches)
            command = self._parse_command(content, user_goal)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[LLMBridge] Model request failed ({exc}); using local heuristic planner.")
            return self._heuristic_fallback(user_goal, commentary, fovea_patches)

        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": content})
        return command

    def describe_screen(
        self,
        frame_or_patches: Union[np.ndarray, List[np.ndarray]],
        prompt: str = "Describe what is visible on this screen in detail.",
    ) -> str:
        """Query LLM directly for natural language description of screen frame or patches.

        When vision_enabled=True, encodes frame or foveal patches to base64 PNGs and queries
        /v1/chat/completions directly for natural language description without forcing command
        action JSON parsing.
        When vision_enabled=False, raises PermissionError to preserve privacy.
        """
        if not self.vision_enabled:
            raise PermissionError("Screen image egress disabled. Set vision_enabled=True to send pixels.")

        if isinstance(frame_or_patches, np.ndarray):
            patches = [frame_or_patches]
        elif isinstance(frame_or_patches, (list, tuple)):
            patches = list(frame_or_patches)
        else:
            raise TypeError("frame_or_patches must be a numpy ndarray or a list of numpy ndarrays")

        if not self.api_key:
            raise RuntimeError("no API key configured for LLMBridge")

        content_items: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for patch in patches:
            if isinstance(patch, np.ndarray) and patch.size > 0:
                b64_png = encode_patch_to_base64(patch)
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                })

        messages = [{"role": "user", "content": content_items}]
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
            }
        ).encode("utf-8")

        if self.base_url.endswith("/v1"):
            endpoint = f"{self.base_url}/chat/completions"
        elif "/v1" in self.base_url:
            endpoint = f"{self.base_url}/chat/completions"
        else:
            endpoint = f"{self.base_url}/v1/chat/completions"

        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        content = str(body["choices"][0]["message"]["content"]).strip()
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": content})
        return content

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
