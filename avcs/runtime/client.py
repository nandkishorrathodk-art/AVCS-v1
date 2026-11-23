"""Custom Frontier Cloud LLM Provider Client for the Ace-Control-Medium Architecture.

Supports arbitrary frontier model endpoints (e.g. gpt-6-astra, fable-5.5, custom gateways),
pure visual GUI prompt framing, direct action token decoding with zero CoT chatter,
and fail-safe parsing with zero center-click heuristics.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

from avcs.protocol.actions import Action, ActionKind, ActionAdapter

logger = logging.getLogger("avcs.ace.client")


class AceClientError(Exception):
    """Base error for ACE client interactions."""
    pass


_dotenv_loaded = False


def _load_dotenv_if_present() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from pathlib import Path
        for p in [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
                break
    except Exception:
        pass


_load_dotenv_if_present()


def repair_truncated_json(
    raw: str,
    canonical_width: int = 1280,
    canonical_height: int = 800,
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Repairs incomplete or truncated JSON strings from token-exhausted LLM responses.

    Safely repairs truncated coordinate arrays (e.g. '{"kind": "left_click", "coordinate": [260' -> [260, 800]),
    unclosed strings, trailing commas, and unclosed JSON braces/brackets.
    """
    text = raw.strip()
    if not text:
        return text

    # Strip any text before the first opening brace
    start_idx = text.find("{")
    if start_idx != -1:
        text = text[start_idx:]

    # Resolve fallback x and y if history has a previous action
    fallback_x = canonical_width // 2
    fallback_y = canonical_height
    has_history_coord = False
    if history:
        for item in reversed(history):
            prev_coord = item.get("coordinate")
            if prev_coord and len(prev_coord) >= 2:
                try:
                    fallback_x = int(prev_coord[0])
                    fallback_y = int(prev_coord[1])
                    has_history_coord = True
                    break
                except (ValueError, TypeError):
                    pass

    # 1. Truncated coordinate repair
    # Pattern a: coordinate: [260, 800 (two numbers, no closing bracket)
    text = re.sub(
        r'((?:"|\')(?:coordinate|coords|point|position)(?:"|\')\s*:\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?)\s*$',
        r'\1]',
        text,
    )
    # Pattern b: coordinate: [260, (one number + comma, no second number)
    text = re.sub(
        r'((?:"|\')(?:coordinate|coords|point|position)(?:"|\')\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*)$',
        lambda m: f'{m.group(1)} {fallback_y}]',
        text,
    )
    # Pattern c: coordinate: [260 (one number, no comma)
    text = re.sub(
        r'((?:"|\')(?:coordinate|coords|point|position)(?:"|\')\s*:\s*\[\s*(-?\d+(?:\.\d+)?))\s*$',
        lambda m: f'{m.group(1)}, {fallback_y}]',
        text,
    )
    # Pattern d: coordinate: [ (bracket opened with nothing inside, use history coord if available)
    if has_history_coord:
        text = re.sub(
            r'((?:"|\')(?:coordinate|coords|point|position)(?:"|\')\s*:\s*\[)\s*$',
            f'"coordinate": [{fallback_x}, {fallback_y}]',
            text,
        )
    # Pattern e: coordinate: (colon with nothing after, use history coord if available)
    if has_history_coord:
        text = re.sub(
            r'((?:"|\')(?:coordinate|coords|point|position)(?:"|\')\s*:)\s*$',
            f'"coordinate": [{fallback_x}, {fallback_y}]',
            text,
        )
    # Pattern f: partial key "coord... at end
    if has_history_coord:
        text = re.sub(
            r'(?:,\s*)?(?:"|\')coord\w*(?:"|\')?\s*:?\s*$',
            f', "coordinate": [{fallback_x}, {fallback_y}]',
            text,
        )
    # Pattern g: "x": 260, "y": 800 (no closing brace) or "x": 260 (missing y)
    text = re.sub(
        r'((?:"|\')x(?:"|\')\s*:\s*-?\d+(?:\.\d+)?\s*,\s*(?:"|\')y(?:"|\')\s*:\s*)$',
        lambda m: f'{m.group(1)}{fallback_y}',
        text,
    )
    text = re.sub(
        r'((?:"|\')x(?:"|\')\s*:\s*-?\d+(?:\.\d+)?)\s*$',
        lambda m: f'{m.group(1)}, "y": {fallback_y}',
        text,
    )
    # Pattern h: Truncated float like "duration": 1. -> 1.0
    text = re.sub(r'(\d+\.)\s*$', r'\g<1>0', text)

    # 2. Strip unclosed/incomplete key after a comma: e.g. , "reaso or , reaso
    text = re.sub(r',\s*(?:"|\')?[a-zA-Z_]\w*(?:"|\')?\s*$', '', text)

    # 3. Strip trailing backslash if string ended with incomplete escape
    if text.endswith('\\'):
        text = text[:-1]

    # 4. Check and close unclosed string literals (both " and ')
    active_quote = None
    escape = False
    for ch in text:
        if ch == '\\' and not escape:
            escape = True
            continue
        if ch in ('"', "'") and not escape:
            if active_quote is None:
                active_quote = ch
            elif active_quote == ch:
                active_quote = None
        escape = False

    if active_quote is not None:
        text += active_quote

    # 5. Strip trailing commas or colons before closing brackets
    text = re.sub(r',\s*$', '', text)
    text = re.sub(r':\s*$', ': ""', text)

    # 6. Count and close unclosed brackets and braces
    stack: List[str] = []
    in_quote = None
    esc = False
    for ch in text:
        if ch == '\\' and not esc:
            esc = True
            continue
        if ch in ('"', "'") and not esc:
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif in_quote is None:
            if ch in ('{', '['):
                stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()
        esc = False

    for open_ch in reversed(stack):
        if open_ch == '{':
            text += '}'
        elif open_ch == '[':
            text += ']'

    return text


class CustomLLMClient:
    """Universal client for custom frontier cloud LLMs and OpenAI/Anthropic/Gemini gateways."""

    CANONICAL_WIDTH = 1280
    CANONICAL_HEIGHT = 800

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        protocol: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        _load_dotenv_if_present()
        # 1. Resolve API key with comprehensive provider fallbacks
        self.api_key = (
            api_key
            or os.environ.get("AVCS_ACE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("AVCS_LLM_API_KEY")
            or os.environ.get("NVIDIA_API_KEY")
            or ""
        )

        # 2. Resolve Model name (frontier model default: gpt-6-astra or fable-5.5)
        self.model = (
            model
            or os.environ.get("AVCS_ACE_MODEL")
            or os.environ.get("AVCS_LLM_MODEL")
            or "gpt-6-astra"
        )

        # 3. Resolve Protocol: 'openai', 'anthropic', 'gemini'
        explicit_protocol = protocol or os.environ.get("AVCS_ACE_PROTOCOL")
        
        # 4. Resolve endpoint
        raw_endpoint = (
            endpoint
            or os.environ.get("AVCS_ACE_ENDPOINT")
            or os.environ.get("OPENAI_BASE_URL")
        )

        if not raw_endpoint:
            # Auto-detect default endpoint from key signature or env
            if os.environ.get("ANTHROPIC_API_KEY") or self.api_key.startswith("sk-ant-"):
                raw_endpoint = "https://api.anthropic.com/v1/messages"
                detected_protocol = "anthropic"
            elif os.environ.get("GEMINI_API_KEY") or self.api_key.startswith("AIzaSy"):
                raw_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
                detected_protocol = "gemini"
            elif os.environ.get("GROQ_API_KEY") or self.api_key.startswith("gsk_"):
                raw_endpoint = "https://api.groq.com/openai/v1/chat/completions"
                detected_protocol = "openai"
            elif os.environ.get("NVIDIA_API_KEY") or self.api_key.startswith("nvapi-"):
                raw_endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
                detected_protocol = "openai"
            elif os.environ.get("AVCS_LLM_ENDPOINT"):
                raw_endpoint = os.environ.get("AVCS_LLM_ENDPOINT")
                detected_protocol = "openai"
            else:
                raw_endpoint = "https://api.openai.com/v1/chat/completions"
                detected_protocol = "openai"
        else:
            detected_protocol = "auto"

        self.protocol = (explicit_protocol or detected_protocol or "auto").lower()
        if self.protocol == "auto":
            if "anthropic.com" in raw_endpoint:
                self.protocol = "anthropic"
            elif "googleapis.com" in raw_endpoint:
                self.protocol = "gemini"
            else:
                self.protocol = "openai"

        self.endpoint = self._normalize_endpoint(raw_endpoint)

        # 5. Extra headers
        self.extra_headers: Dict[str, str] = {}
        env_headers = os.environ.get("AVCS_ACE_EXTRA_HEADERS")
        if env_headers:
            try:
                self.extra_headers.update(json.loads(env_headers))
            except Exception:
                for item in env_headers.split(","):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        self.extra_headers[k.strip()] = v.strip()
        if extra_headers:
            self.extra_headers.update(extra_headers)

        env_timeout = os.environ.get("AVCS_ACE_TIMEOUT")
        self.timeout = float(env_timeout) if env_timeout else (timeout if timeout is not None else 30.0)

        env_max_tokens = os.environ.get("AVCS_ACE_MAX_TOKENS")
        self.max_tokens = int(env_max_tokens) if env_max_tokens else (max_tokens if max_tokens is not None else 1024)

        self.vision_detail = os.environ.get("AVCS_VISION_DETAIL", "low").lower().strip()

        if requests is not None:
            self.session = requests.Session()
            try:
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
                self.session.mount("https://", HTTPAdapter(max_retries=retries))
                self.session.mount("http://", HTTPAdapter(max_retries=retries))
            except Exception:
                pass
        else:
            self.session = None

    def _normalize_endpoint(self, url: str) -> str:
        """Ensures endpoint points to the proper protocol route if only base URL is provided."""
        url = url.strip()
        if url.endswith("/"):
            url = url[:-1]

        if self.protocol == "anthropic":
            if url == "https://api.anthropic.com" or url.endswith("/v1"):
                return f"{url}/messages" if url.endswith("/v1") else f"{url}/v1/messages"
            if not url.endswith("/messages"):
                return f"{url}/v1/messages" if not url.endswith("/v1") else f"{url}/messages"
            return url
        elif self.protocol == "gemini":
            if ":generateContent" not in url:
                if "/models" in url:
                    return f"{url}/{self.model}:generateContent"
                return f"{url}/models/{self.model}:generateContent"
            return url
        else:
            # OpenAI-compatible
            if url.endswith("/v1"):
                return f"{url}/chat/completions"
            if not url.endswith("/chat/completions") and not url.endswith("/messages") and ":generateContent" not in url:
                if url.endswith("api.groq.com/openai"):
                    return f"{url}/v1/chat/completions"
                # If host only like http://localhost:8000 or http://localhost:11434
                if url.count("/") <= 2:
                    return f"{url}/v1/chat/completions"
            return url

    def encode_image(self, image_bytes: bytes, format: str = "webp") -> Tuple[str, str, str]:
        """Encodes raw image bytes into base64, mime type, and data URI."""
        fmt = format.lower().strip()
        if fmt in ("jpg", "jpeg"):
            mime = "image/jpeg"
        elif fmt == "png":
            mime = "image/png"
        else:
            mime = "image/webp"

        b64_str = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime};base64,{b64_str}"
        return b64_str, mime, data_uri

    def build_system_prompt(self) -> str:
        """Constructs the high-speed direct action token prompt for the frontier VAM."""
        return (
            "You are an expert GUI Computer Autopilot (Ace-Control-Medium).\n"
            "You perceive the computer display PURELY through visual screenshot pixels.\n"
            "Forbidden: HTML DOM, browser extension hooks, or accessibility trees.\n\n"
            "COORDINATE SYSTEM:\n"
            f"Canvas is normalized to canonical {self.CANONICAL_WIDTH}x{self.CANONICAL_HEIGHT} (WXGA).\n"
            f"- (0, 0) is TOP-LEFT of the screen.\n"
            f"- ({self.CANONICAL_WIDTH}, {self.CANONICAL_HEIGHT}) is BOTTOM-RIGHT of the screen.\n"
            "All coordinates [x, y] MUST be integers within [0, 1280] x [0, 800].\n\n"
            "ZERO CENTER HEURISTICS MANDATE:\n"
            "- NEVER click screen center (640, 400) or (0.5, 0.5) as a guess or default.\n"
            "- Every click MUST land precisely on the visual pixel bounding box of the intended button/input.\n"
            "- If a target is not visible, emit a wait or scroll to find it.\n\n"
            "WINDOWS 11 NAVIGATION & TASKBAR:\n"
            "- On Windows 11, the Start button and pinned apps are centered or left-aligned on the taskbar at the bottom.\n"
            "- Opening Start menu / search: prefer emitting {\"kind\": \"key\", \"keys\": [\"win\"], \"reason\": \"Open Start menu\"}.\n"
            "- Once the Start menu is open, you can immediately type the target application name (e.g. {\"kind\": \"type\", \"text\": \"<AppName>\", \"press_enter\": true}) without needing to click the search bar.\n\n"
            "GOAL COMPLETION & PROCEDURAL BYPASS:\n"
            "1. Inspect the screen first: is the target application or final goal ALREADY OPEN or visible?\n"
            "2. If the user asks for multi-step actions (e.g. 'Launch <App>', 'Search for <App> in Start Menu'):\n"
            "   - If the requested application or target window is ALREADY OPEN or visible on the screen, the goal is COMPLETE!\n"
            "   - You MUST emit: {\"kind\": \"stop\", \"reason\": \"Target application is open and visible. Goal complete.\"}\n"
            "   - NEVER click Start or repeat previous steps if the final state is already achieved.\n\n"
            "DIRECT ACTION TOKEN SPECIFICATION:\n"
            "Emit ONLY a single, valid JSON action object without preamble, reasoning essays, or markdown chatter:\n"
            "1. left_click: {\"kind\": \"left_click\", \"coordinate\": [x, y], \"reason\": \"click target\"}\n"
            "2. right_click: {\"kind\": \"right_click\", \"coordinate\": [x, y], \"reason\": \"context menu\"}\n"
            "3. double_click: {\"kind\": \"double_click\", \"coordinate\": [x, y], \"reason\": \"open item\"}\n"
            "4. triple_click: {\"kind\": \"triple_click\", \"coordinate\": [x, y], \"reason\": \"select line\"}\n"
            "5. mouse_move: {\"kind\": \"mouse_move\", \"coordinate\": [x, y], \"reason\": \"hover\"}\n"
            "6. left_click_drag: {\"kind\": \"left_click_drag\", \"start_coordinate\": [x1, y1], \"end_coordinate\": [x2, y2], \"reason\": \"drag\"}\n"
            "7. scroll: {\"kind\": \"scroll\", \"coordinate\": [x, y], \"scroll_delta\": [0, -120], \"reason\": \"scroll down\"}\n"
            "8. type: {\"kind\": \"type\", \"coordinate\": [x, y], \"text\": \"hello\", \"reason\": \"type in field\"}\n"
            "9. key: {\"kind\": \"key\", \"keys\": [\"enter\"], \"reason\": \"submit\"} (or [\"ctrl\", \"s\"])\n"
            "10. wait: {\"kind\": \"wait\", \"duration\": 1.0, \"reason\": \"await loading\"}\n"
            "11. stop: {\"kind\": \"stop\", \"reason\": \"Goal completed successfully\"}\n\n"
            "Output MUST be raw JSON only."
        )

    def build_user_prompt(
        self,
        goal: str,
        history: Optional[List[Dict[str, Any]]] = None,
        step_index: int = 1,
    ) -> str:
        """Constructs the user turn context."""
        recent_actions = history[-5:] if history else []
        history_str = json.dumps(recent_actions, indent=None) if recent_actions else "None"

        alert_str = ""
        if recent_actions and len(recent_actions) >= 2:
            def _coords_match(c1: Any, c2: Any, tol: int = 5) -> bool:
                if c1 is None and c2 is None:
                    return True
                if c1 is None or c2 is None:
                    return False
                try:
                    return abs(c1[0] - c2[0]) <= tol and abs(c1[1] - c2[1]) <= tol
                except (IndexError, TypeError, ValueError):
                    return c1 == c2

            last = recent_actions[-1]
            rep_count = 1
            for prev in reversed(recent_actions[:-1]):
                if prev.get("action") == last.get("action") and _coords_match(prev.get("coordinate"), last.get("coordinate")):
                    rep_count += 1
                else:
                    break

            click_count = 0
            if last.get("action") in ("left_click", "right_click", "double_click", "triple_click", "click") and last.get("coordinate"):
                for it in recent_actions:
                    if it.get("action") == last.get("action") and _coords_match(it.get("coordinate"), last.get("coordinate")):
                        click_count += 1

            eff_reps = max(rep_count, click_count)
            if eff_reps >= 2:
                alert_str = (
                    f"\n[STUCK WARNING: The previous action '{last.get('action')}' was repeated {eff_reps} times "
                    f"without state progress! Do NOT repeat this action. "
                    "Attempt an alternative action: use key shortcut (e.g. key [\"win\"] to open Start Menu / search), "
                    "shortcut keys [\"enter\"], or click a different coordinate.]\n"
                )

        if recent_actions:
            last_item = recent_actions[-1]
            last_reason = str(last_item.get("reason", ""))
            if "Premature STOP rejected" in last_reason:
                contract_alert = (
                    f"\n[CONTRACT ENFORCEMENT ALERT: {last_reason} "
                    "Do NOT emit 'stop'. You MUST execute a 'type' action first to complete typing/searching.]\n"
                )
                alert_str = f"{alert_str}\n{contract_alert}" if alert_str else contract_alert

        return (
            f"Goal: {goal}\n"
            f"Step: {step_index}\n"
            f"Recent Actions History: {history_str}\n"
            f"{alert_str}\n"
            "Analyze the screenshot frame and output the immediate next JSON action token to advance the goal."
        )

    def _build_payload(
        self,
        b64_str: str,
        mime: str,
        data_uri: str,
        goal: str,
        history: Optional[List[Dict[str, Any]]] = None,
        step_index: int = 1,
    ) -> Dict[str, Any]:
        """Builds protocol-specific API payload."""
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(goal, history, step_index)

        if self.protocol == "anthropic":
            return {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": b64_str,
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            }
        elif self.protocol == "gemini":
            return {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": f"{system_prompt}\n\n{user_prompt}"},
                            {"inlineData": {"mimeType": mime, "data": b64_str}},
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": self.max_tokens,
                },
            }
        else:
            # Default: OpenAI chat completions format (vLLM, LiteLLM, OpenRouter, frontier custom)
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri, "detail": self.vision_detail},
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            }
            if self.endpoint and "nvidia.com" in self.endpoint:
                payload["reasoning_effort"] = "none"
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ActionSchema",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string"},
                                "coordinate": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "text": {"type": "string"},
                                "keys": {"type": "array", "items": {"type": "string"}},
                                "reason": {"type": "string"},
                            },
                            "required": ["kind"],
                        },
                    },
                }
            return payload

    def _build_headers(self) -> Dict[str, str]:
        """Builds headers for the request."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AVCS-FrontierClient/1.0",
        }
        if self.protocol == "anthropic":
            if self.api_key:
                headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.protocol == "gemini":
            if self.api_key:
                headers["x-goog-api-key"] = self.api_key
        else:
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

        headers.update(self.extra_headers)
        return headers

    def parse_action_response(
        self,
        response_text: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Action:
        """Parses model response into a valid Action with high error resilience."""
        text = str(response_text).strip()
        if not text:
            return Action(kind=ActionKind.WAIT, duration=0.5, reason="Empty LLM response received")

        # Fast path: Check for AVCS ACT: grammar tokens directly
        grammar_act = ActionAdapter.parse_grammar_token(text)
        if grammar_act is not None:
            return grammar_act

        # Strip internal reasoning blocks from frontier reasoning models (<think>, <thought>, <reasoning>, <extra_thought>)
        for tag in ("think", "thought", "reasoning", "extra_thought"):
            text = re.sub(rf"<{tag}>[\s\S]*?</{tag}>", "", text, flags=re.IGNORECASE).strip()

        # Handle unclosed reasoning tags (model stopped during thinking)
        for tag_name in ("think", "thought", "reasoning", "extra_thought"):
            open_tag = f"<{tag_name}>"
            close_tag = f"</{tag_name}>"
            if open_tag in text.lower() and close_tag not in text.lower():
                tag_idx = text.lower().find(open_tag)
                rest = text[tag_idx + len(open_tag):]
                if "{" in rest:
                    text = rest[rest.find("{"):].strip()
                else:
                    text = ""

        # Re-check grammar token after stripping reasoning tags
        grammar_act = ActionAdapter.parse_grammar_token(text)
        if grammar_act is not None:
            return grammar_act

        # Strip markdown code blocks if present
        if "```" in text:
            # Match ```json ... ``` or ``` ... ```
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
            else:
                # Unclosed code fence (truncated)
                match_unclosed = re.search(r"```(?:json)?\s*([\s\S]*)", text, re.IGNORECASE)
                if match_unclosed:
                    text = match_unclosed.group(1).strip()

        # Normalize Python dict literals (True, False, None) to valid JSON (true, false, null)
        text = re.sub(r"\bTrue\b", "true", text)
        text = re.sub(r"\bFalse\b", "false", text)
        text = re.sub(r"\bNone\b", "null", text)

        # Attempt to isolate the action JSON object
        # First preference: find the opening brace of an object containing "kind" or "action"
        action_obj_match = re.search(r"(\{\s*(?:[^{}]*?(?:\"kind\"|'kind'|\"action\"|'action')[\s\S]*))", text)
        if action_obj_match:
            sub = action_obj_match.group(1)
            # Find matching closing brace using stack
            depth = 0
            in_q = None
            esc = False
            end_pos = -1
            for idx, ch in enumerate(sub):
                if ch == '\\' and not esc:
                    esc = True
                    continue
                if ch in ('"', "'") and not esc:
                    if in_q is None:
                        in_q = ch
                    elif in_q == ch:
                        in_q = None
                elif in_q is None:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end_pos = idx
                            break
                esc = False
            if end_pos != -1:
                candidate_text = sub[:end_pos + 1]
            else:
                candidate_text = sub
        else:
            # Fallback to outer brackets
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidate_text = text[start_idx : end_idx + 1]
            elif start_idx != -1:
                # Truncated without closing brace
                candidate_text = text[start_idx:]
            else:
                candidate_text = text

        # Clean trailing commas which invalidates standard json.loads
        clean_json = re.sub(r",\s*([}\]])", r"\1", candidate_text)

        data = None
        try:
            data = json.loads(clean_json)
        except Exception:
            # Attempt repair for truncated or incomplete JSON
            repaired = repair_truncated_json(
                clean_json,
                canonical_width=self.CANONICAL_WIDTH,
                canonical_height=self.CANONICAL_HEIGHT,
                history=history,
            )
            try:
                data = json.loads(repaired)
            except Exception:
                # Fallback 1: try single quote replacement if double quotes were omitted
                try:
                    fixed = repaired.replace("'", '"')
                    data = json.loads(fixed)
                except Exception:
                    # Fallback 2: AST literal eval for Python dict syntax
                    try:
                        import ast
                        evaluated = ast.literal_eval(repaired)
                        if isinstance(evaluated, dict):
                            data = evaluated
                    except Exception as e:
                        logger.warning(f"Failed to parse action JSON: {text!r} - error: {e}")
                        return Action(
                            kind=ActionKind.WAIT,
                            duration=0.5,
                            reason=f"Unparseable action JSON fallback: {text[:60]}",
                        )

        if not isinstance(data, dict):
            return Action(kind=ActionKind.WAIT, duration=0.5, reason="Parsed data is not a dict")

        action = Action.from_dict(data)

        # STRICT ZERO CENTER HEURISTICS CHECK:
        # If an action requires coordinates (clicks/moves) but none was provided,
        # DO NOT default to screen center (0.5, 0.5) or (640, 400).
        if action.kind in (
            ActionKind.LEFT_CLICK,
            ActionKind.RIGHT_CLICK,
            ActionKind.DOUBLE_CLICK,
            ActionKind.TRIPLE_CLICK,
        ) and action.coordinate is None:
            return Action(
                kind=ActionKind.WAIT,
                duration=0.5,
                reason=f"Click missing visual coordinates for target '{action.reason}', waiting for re-observation",
            )

        return action

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Text completion method for Cognitive Planner sub-goal decomposition and replanning."""
        if not self.endpoint:
            return ""

        headers = self._build_headers()
        tokens = max_tokens or self.max_tokens

        if self.protocol == "anthropic":
            payload = {
                "model": self.model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
                "max_tokens": tokens,
                "temperature": temperature,
            }
        elif self.protocol == "gemini":
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                ],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": tokens},
            }
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": tokens,
            }

        try:
            if self.session is not None:
                resp = self.session.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            else:
                import urllib.request
                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(self.endpoint, data=req_data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            return self._extract_content_from_response(data)
        except Exception as e:
            logger.debug(f"CustomLLMClient.complete call notice: {e}")
            return ""

    def _extract_content_from_response(self, resp_json: Dict[str, Any]) -> str:
        """Extracts text content string based on protocol with fallback for reasoning models."""
        if self.protocol == "anthropic":
            contents = resp_json.get("content", [])
            texts = []
            for c in contents:
                if isinstance(c, dict) and c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif isinstance(c, dict) and c.get("type") == "tool_use":
                    inp = c.get("input", {})
                    name = c.get("name")
                    if isinstance(inp, dict):
                        inp_dict = dict(inp)
                        if name and name != "computer" and "action" not in inp_dict and "kind" not in inp_dict:
                            inp_dict["action"] = name
                        texts.append(json.dumps(inp_dict))
                    else:
                        texts.append(str(inp))
                elif isinstance(c, str):
                    texts.append(c)
            return "\n".join(texts).strip()
        elif self.protocol == "gemini":
            candidates = resp_json.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
                return "\n".join(texts).strip()
            return ""
        else:
            # OpenAI-compatible (including Groq, OpenRouter, vLLM, DeepSeek, etc.)
            choices = resp_json.get("choices", [])
            if not choices:
                return ""
            choice0 = choices[0] or {}
            message = choice0.get("message", {}) or {}
            
            # 1. Standard message content
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            elif isinstance(content, list):
                # Multimodal content list format: [{"type": "text", "text": "..."}]
                texts = [part.get("text", "") for part in content if isinstance(part, dict) and "text" in part]
                if texts:
                    return "\n".join(texts).strip()

            # Tool calls fallback
            tool_calls = message.get("tool_calls") or []
            if tool_calls and isinstance(tool_calls, list):
                fn = tool_calls[0].get("function", {}) if isinstance(tool_calls[0], dict) else {}
                args = fn.get("arguments", "")
                fn_name = fn.get("name")
                if args:
                    try:
                        parsed_args = json.loads(args)
                        if isinstance(parsed_args, dict) and fn_name and "kind" not in parsed_args and "action" not in parsed_args:
                            parsed_args["action"] = fn_name
                            return json.dumps(parsed_args)
                    except Exception:
                        pass
                    return str(args).strip()

            # 2. Reasoning models fallback (DeepSeek-R1, QwQ, o1 style)
            for field in ("reasoning_content", "reasoning", "text"):
                val = message.get(field)
                if isinstance(val, str) and val.strip():
                    return val.strip()

            # 3. Direct choice text fallback (completion endpoints)
            direct_text = choice0.get("text")
            if isinstance(direct_text, str) and direct_text.strip():
                return direct_text.strip()

            return ""

    def predict_action(
        self,
        image_bytes: bytes,
        goal: str,
        history: Optional[List[Dict[str, Any]]] = None,
        step_index: int = 1,
        format: str = "webp",
    ) -> Tuple[Action, float]:
        """Performs visual reasoning on the frame and returns the Action and latency in ms."""
        b64_str, mime, data_uri = self.encode_image(image_bytes, format=format)
        payload = self._build_payload(
            b64_str=b64_str,
            mime=mime,
            data_uri=data_uri,
            goal=goal,
            history=history,
            step_index=step_index,
        )
        headers = self._build_headers()

        t0 = time.time()
        try:
            if self.session is not None:
                resp = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                resp_json = resp.json()
            else:
                import urllib.request
                import urllib.error
                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(self.endpoint, data=req_data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    resp_json = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000.0
            logger.error(f"HTTP request to LLM endpoint ({self.endpoint}) failed: {e}")
            raise AceClientError(f"LLM request error: {e}") from e

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0

        raw_content = self._extract_content_from_response(resp_json)
        action = self.parse_action_response(raw_content, history=history)
        return action, latency_ms

