"""Local Vision-Action Model (VAM) Edge Engine Integration.

Provides direct sub-500ms local inference via local vLLM, TensorRT-LLM,
Ollama, or SGLang server running lightweight edge models (e.g. Qwen2.5-VL-3B/7B,
SmolVLM2-2.2B, or fine-tuned Ace-Control weights) in FP8/INT4 precision.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from avcs.ace.actions import Action, ActionAdapter, ActionKind
from avcs.ace.client import AceClientError

logger = logging.getLogger("avcs.ace.local_model")


class LocalVAMClient:
    """Local edge Vision-Action Model client for zero-cloud sub-500ms desktop autopilot."""

    CANONICAL_WIDTH = 1280
    CANONICAL_HEIGHT = 800

    SYSTEM_PROMPT = (
        "You are Ace-Control-Medium, a real-time computer use foundation model running locally on desktop.\n"
        "Input: Desktop observation on a canonical 1280x800 display and user goal.\n"
        "Output: Direct physical action token or compact JSON with ZERO chain-of-thought commentary.\n\n"
        "Available Actions:\n"
        "- ACT:CLICK <x> <y>        (left click at canonical coordinate [0-1280, 0-800])\n"
        "- ACT:RIGHT_CLICK <x> <y>  (context menu click)\n"
        "- ACT:DOUBLE_CLICK <x> <y> (double click to open/launch)\n"
        "- ACT:TYPE <text>          (type text into active element)\n"
        "- ACT:KEY <key_name>       (press key like win, enter, tab, escape, ctrl+s)\n"
        "- ACT:WAIT <seconds>       (wait for UI load/animation)\n"
        "- ACT:STOP                 (goal successfully accomplished)\n\n"
        "Output ONLY the single next action token."
    )

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 5.0,
        max_tokens: int = 32,
        temperature: float = 0.0,
        allow_offline_mock: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        raw_endpoint = (
            endpoint
            or os.environ.get("AVCS_LOCAL_VAM_ENDPOINT")
            or "http://127.0.0.1:8000/v1"
        )
        self.endpoint = raw_endpoint.rstrip("/")
        if not self.endpoint.endswith("/v1"):
            self.endpoint = f"{self.endpoint}/v1"

        self.model = (
            model
            or os.environ.get("AVCS_LOCAL_VAM_MODEL")
            or "Qwen2.5-VL-3B-Instruct"
        )
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.allow_offline_mock = allow_offline_mock
        self.extra_headers = extra_headers or {}

    def check_health(self) -> bool:
        """Pings local VAM server health or models endpoint to verify readiness."""
        test_urls = [
            f"{self.endpoint}/models",
            f"{self.endpoint.rsplit('/v1', 1)[0]}/health",
            f"{self.endpoint.rsplit('/v1', 1)[0]}/api/tags",  # Ollama compatibility
        ]
        headers = {"User-Agent": "AVCS-LocalVAM/1.0", **self.extra_headers}

        for url in test_urls:
            try:
                resp = requests.get(url, headers=headers, timeout=1.0)
                if resp.status_code in (200, 204):
                    logger.info(f"Local VAM engine verified active at {url}")
                    return True
            except Exception:
                continue

        logger.debug(f"Local VAM engine not responding at {self.endpoint}")
        return False

    def predict_action(
        self,
        image_bytes: bytes,
        goal: str,
        history: Optional[List[Dict[str, Any]]] = None,
        step_index: int = 1,
        format: str = "jpeg",
    ) -> Tuple[Action, float]:
        """Infers next desktop action from visual frame using local VAM engine."""
        t0 = time.time()

        # Build prompt messages
        b64_img = base64.b64encode(image_bytes).decode("ascii")
        img_mime = f"image/{format.lower()}" if format.lower() in ("jpeg", "webp", "png") else "image/jpeg"

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Goal: {goal}\nCanonical Resolution: {self.CANONICAL_WIDTH}x{self.CANONICAL_HEIGHT}\nStep: {step_index}",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img_mime};base64,{b64_img}",
                },
            },
        ]

        if history:
            recent_acts = [
                f"Step {h.get('step', i+1)}: {h.get('kind', 'action')} "
                f"coord={h.get('coordinate', 'none')} reason={h.get('reason', '')}"
                for i, h in enumerate(history[-3:])
            ]
            user_content.append({
                "type": "text",
                "text": f"Recent Actions:\n" + "\n".join(recent_acts),
            })

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AVCS-LocalVAM/1.0",
            **self.extra_headers,
        }

        url = f"{self.endpoint}/chat/completions"

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0

            if resp.status_code != 200:
                raise AceClientError(
                    f"Local VAM endpoint returned HTTP {resp.status_code}: {resp.text[:300]}"
                )

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise AceClientError("Local VAM returned response with empty choices array.")

            msg = choices[0].get("message", {})
            content = msg.get("content", "").strip()

            # Parse through universal ActionAdapter
            action = ActionAdapter.from_response(content)
            logger.info(
                f"Local VAM ({self.model}) predicted: {action.kind.value} "
                f"{action.coordinate or ''} in {latency_ms:.1f}ms"
            )
            return action, latency_ms

        except (requests.RequestException, requests.Timeout) as e:
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0

            if self.allow_offline_mock:
                logger.warning(
                    f"Local VAM server offline at {url} ({e}). Generating fallback heuristic action."
                )
                action = self._offline_fallback_action(goal, step_index)
                return action, latency_ms

            raise AceClientError(
                f"Local VAM engine connection failed at {url}: {e}. "
                "Ensure local vLLM or TensorRT-LLM server is running or set allow_offline_mock=True."
            ) from e

    def _offline_fallback_action(self, goal: str, step_index: int) -> Action:
        """Deterministic offline fallback for edge cases when local engine server is initializing."""
        g_lower = goal.lower()
        if "notepad" in g_lower:
            if step_index == 1:
                return Action(kind=ActionKind.KEY, keys=["win"], reason="Open Start Menu to search Notepad")
            elif step_index == 2:
                return Action(kind=ActionKind.TYPE, text="notepad", press_enter=True, reason="Type Notepad and launch")
            else:
                return Action(kind=ActionKind.STOP, reason="Notepad launched successfully")

        if "start" in g_lower or "menu" in g_lower:
            if step_index == 1:
                return Action(kind=ActionKind.KEY, keys=["win"], reason="Open Start Menu")
            return Action(kind=ActionKind.STOP, reason="Start Menu opened")

        if step_index > 2:
            return Action(kind=ActionKind.STOP, reason=f"Goal completed: {goal}")

        return Action(kind=ActionKind.WAIT, duration=0.5, reason=f"Awaiting UI response for '{goal}'")
