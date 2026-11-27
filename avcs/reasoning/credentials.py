"""Credential resolution for the AVCS local VLA reasoning endpoint.

Resolves the local loopback endpoint, optional API key, and model name for an
OpenAI-compatible local server (such as a local Qwen2.5-VL-3B VLA server or
local Ollama daemon). External cloud LLM credentials, vendor auto-detection
(``NVIDIA_API_KEY``, ``OPENAI_API_KEY``), and platform-managed remote key
refresh requests are disabled in 100% self-contained local VLA mode.

Nothing here prints, logs, or returns the raw key to a caller that did not ask
for it; ``mask()`` / ``mask_key()`` are the only renderings used in messages.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen2.5-VL-3B-Instruct"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_cache_lock = threading.Lock()
_cached_key: Optional[str] = None
_cached_at: float = 0.0
_cached_models: Optional[List[str]] = None


def mask(key: Optional[str]) -> str:
    """Render a key for logs without exposing it."""
    if not key:
        return "None"
    if len(key) <= 8:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


mask_key = mask


def _clean(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip()
    return value or None


def _is_local_endpoint(endpoint: Optional[str]) -> bool:
    """Return True only when ``endpoint`` is a loopback HTTP(S) URL."""
    raw = _clean(endpoint)
    if not raw:
        return False
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    return parsed.hostname.lower() in _LOOPBACK_HOSTS


def _fetch_json(url: str, headers: Dict[str, str], timeout: float) -> object:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_managed_key(
    refresh_url: Optional[str] = None,
    session_key: Optional[str] = None,
    timeout: float = 20.0,
) -> Optional[str]:
    """Disabled in 100% self-contained local VLA mode; always returns ``None``."""
    return None


def list_models(endpoint: str, api_key: str, timeout: float = 20.0) -> List[str]:
    """Model ids a local loopback endpoint advertises. ``[]`` on failure or non-loopback."""
    if not _is_local_endpoint(endpoint):
        return []
    global _cached_models
    with _cache_lock:
        if _cached_models is not None:
            return list(_cached_models)
    try:
        obj = _fetch_json(
            f"{endpoint.rstrip('/')}/models",
            {"Authorization": f"Bearer {api_key}"},
            timeout,
        )
        ids = [str(m.get("id")) for m in (obj or {}).get("data", []) if m.get("id")]
    except Exception:
        ids = []
    with _cache_lock:
        _cached_models = ids
    return list(ids)


def extract_message_text(data: object) -> str:
    """Pull the assistant text out of an OpenAI-compatible response body.

    Reasoning models (DeepSeek-R1-style, and the managed proxy's
    ``deepseek-v4-*``) put their visible answer in ``content`` but may return
    ``content: null`` when the token budget is spent on hidden reasoning first.
    Returning ``None`` there crashes the caller with an opaque
    ``'NoneType' object has no attribute 'strip'``. Fall through the known fields
    so a truncated-but-successful reply still yields text.
    """
    try:
        choices = (data or {}).get("choices") or []
        message = (choices[0] or {}).get("message") or {} if choices else {}
    except Exception:
        message = {}

    for field in ("content", "reasoning_content", "reasoning", "text"):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            inner = value.get("text") or value.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
    return ""


@dataclass(frozen=True)
class LlmCredentials:
    """Resolved connection details for an OpenAI-compatible endpoint."""

    api_key: str
    endpoint: str
    model: str
    source: str  # "user_env" | "managed" | "none"

    @property
    def live(self) -> bool:
        return bool(self.api_key)


_PREFERRED_MODEL_HINTS = ("flash", "mini", "small", "instruct", "chat")


def _pick_model(models: List[str]) -> Optional[str]:
    if not models:
        return None
    for hint in _PREFERRED_MODEL_HINTS:
        for mid in models:
            if hint in mid.lower():
                return mid
    return models[0]


def resolve_llm_credentials(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    *,
    timeout: float = 20.0,
) -> LlmCredentials:
    """Resolve local VLA (api_key, endpoint, model) from arguments or local env.

    External cloud endpoints, vendor API key env vars (``NVIDIA_API_KEY``,
    ``OPENAI_API_KEY``), and remote managed-key refresh URLs are disabled.
    Any non-loopback endpoint is ignored and replaced with ``DEFAULT_BASE_URL``.
    """
    cand_endpoint = _clean(endpoint) or _clean(os.environ.get("AVCS_LLM_ENDPOINT"))
    safe_endpoint = cand_endpoint if _is_local_endpoint(cand_endpoint) else DEFAULT_BASE_URL
    resolved_model = _clean(model) or _clean(os.environ.get("AVCS_LLM_MODEL")) or DEFAULT_MODEL

    if api_key is not None and not api_key.strip():
        return LlmCredentials(
            api_key="",
            endpoint=safe_endpoint,
            model=resolved_model,
            source="none",
        )

    key = _clean(api_key) or _clean(os.environ.get("AVCS_LLM_API_KEY"))
    source = "user_env" if key else "none"

    return LlmCredentials(
        api_key=key or "",
        endpoint=safe_endpoint,
        model=resolved_model,
        source=source,
    )
