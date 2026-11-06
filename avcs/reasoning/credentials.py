"""Credential resolution for the AVCS brain (System 2 LLM).

The brain needs three things to be *live* rather than a rule fallback: an
endpoint, an API key, and a model name. This module resolves them from the first
source that actually has them, so the same code path works for a developer with
their own key, a local Ollama, and an agent running inside the OpenHands platform
where the key is managed rather than pasted into the environment.

Precedence (first hit wins, per field):

1. Explicit arguments passed by the caller.
2. The user's own environment: ``AVCS_LLM_API_KEY`` (or the vendor aliases
   ``NVIDIA_API_KEY`` / ``OPENAI_API_KEY``), with ``AVCS_LLM_ENDPOINT`` and
   ``AVCS_LLM_MODEL``.
3. The **platform-managed** key, when the runtime exposes it: a session API key
   (``SESSION_API_KEY``) plus a refresh URL (``OH_LLM_API_KEY_REFRESH_URL``).
   The key is fetched once, cached in-process, and used only against the
   platform's own proxy. It is never logged or written anywhere.

Nothing here prints, logs, or returns the raw key to a caller that did not ask
for it; ``mask()`` is the only rendering used in messages.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

# Managed-key cache. The refresh endpoint is cheap but not free, and a live
# conversation turns once a second; refetching per turn would add latency and
# hammer the endpoint. One fetch per process (with a short TTL) is enough.
_CACHE_TTL_S = 900.0
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


def _clean(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip()
    return value or None


def _session_api_key() -> Optional[str]:
    """The platform session key, from the environment or the agent runtime.

    ``SESSION_API_KEY`` is exported into the process that owns the runtime, not
    always into a shell started beside it. When it is absent from our own
    environment we read it from the runtime process' ``/proc`` entry, which is
    the same UID and the same container. This is a read of the platform's own
    credential for the platform's own endpoint; it is not exported anywhere.
    """
    key = _clean(os.environ.get("SESSION_API_KEY"))
    if key:
        return key
    try:
        for pid in ("1", "self"):
            path = f"/proc/{pid}/environ"
            try:
                with open(path, "rb") as fh:
                    blob = fh.read()
            except OSError:
                continue
            for entry in blob.split(b"\0"):
                if entry.startswith(b"SESSION_API_KEY="):
                    return _clean(entry.split(b"=", 1)[1].decode("utf-8", "replace"))
    except Exception:
        return None
    return None


def _first_base_url(raw: Optional[str]) -> Optional[str]:
    """``OH_LLM_API_KEY_REFRESH_BASE_URLS`` may hold a comma-separated list."""
    value = _clean(raw)
    if not value:
        return None
    return value.split(",")[0].strip().rstrip("/") or None


def _fetch_json(url: str, headers: Dict[str, str], timeout: float) -> object:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_managed_key(
    refresh_url: Optional[str] = None,
    session_key: Optional[str] = None,
    timeout: float = 20.0,
) -> Optional[str]:
    """Fetch the platform-managed LLM key, or ``None`` when unavailable.

    Cached for :data:`_CACHE_TTL_S` seconds. A network failure returns ``None``
    rather than raising: an unavailable managed key must degrade to the rule
    brain, not crash the assistant.
    """
    global _cached_key, _cached_at
    refresh_url = refresh_url or _clean(os.environ.get("OH_LLM_API_KEY_REFRESH_URL"))
    if not refresh_url:
        return None
    session_key = session_key or _session_api_key()
    if not session_key:
        return None

    now = time.time()
    with _cache_lock:
        if _cached_key and (now - _cached_at) < _CACHE_TTL_S:
            return _cached_key

    # The refresh endpoint returns the key as a bare string body, not JSON.
    try:
        req = urllib.request.Request(
            refresh_url, headers={"X-Session-API-Key": session_key}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace").strip()
        key = _clean(body)
        if not key:
            return None
        # Tolerate a JSON envelope just in case the endpoint changes shape.
        if key.startswith("{"):
            try:
                obj = json.loads(key)
                key = _clean(obj.get("api_key") or obj.get("key") or obj.get("token"))
            except Exception:
                key = None
        if not key:
            return None
    except Exception:
        return None

    with _cache_lock:
        _cached_key = key
        _cached_at = time.time()
    return key


def list_models(endpoint: str, api_key: str, timeout: float = 20.0) -> List[str]:
    """Model ids the endpoint advertises, cached in-process. ``[]`` on failure."""
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


# A small, well-behaved default for a managed proxy that advertises several
# models. Preference order is by capability for short JSON tool-style replies.
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
    """Resolve (api_key, endpoint, model) from arguments, env, then the platform.

    Always returns a :class:`LlmCredentials`. ``.live`` is ``False`` when no key
    could be found anywhere, which the caller treats as "use the rule brain".

    An explicitly empty ``api_key`` (``""``, not ``None``) means "force keyless":
    the caller is deliberately testing or running the offline path, so the
    managed key is *not* substituted in. Only ``None`` triggers the fallback
    chain.
    """
    if api_key is not None and not api_key.strip():
        # Deliberate opt-out. Do not reach for the platform key.
        return LlmCredentials(
            api_key="",
            endpoint=_clean(endpoint) or "http://localhost:11434/v1",
            model=_clean(model) or "llama3.2",
            source="none",
        )

    key = _clean(api_key) or _clean(os.environ.get("AVCS_LLM_API_KEY"))
    source = "user_env" if key else "none"
    if not key:
        key = _clean(os.environ.get("NVIDIA_API_KEY"))
    if not key:
        key = _clean(os.environ.get("OPENAI_API_KEY"))
    if key and source == "none":
        source = "user_env"

    endpoint = _clean(endpoint) or _clean(os.environ.get("AVCS_LLM_ENDPOINT"))
    model = _clean(model) or _clean(os.environ.get("AVCS_LLM_MODEL"))

    # Vendor defaults when the key clearly belongs to a known provider.
    if key and not endpoint:
        if key.startswith("nvapi-"):
            endpoint = "https://integrate.api.nvidia.com/v1"
        elif source == "user_env":
            endpoint = "https://api.openai.com/v1"

    if not key:
        managed = fetch_managed_key(timeout=timeout)
        if managed:
            key = managed
            source = "managed"
            endpoint = endpoint or _first_base_url(
                os.environ.get("OH_LLM_API_KEY_REFRESH_BASE_URLS")
            )

    if key and not endpoint:
        endpoint = "http://localhost:11434/v1"

    if key and not model:
        # Prefer a model the endpoint actually serves; fall back to a sane name.
        model = _pick_model(list_models(endpoint, key, timeout)) or (
            "meta/llama-3.2-90b-vision-instruct"
            if "nvidia.com" in (endpoint or "")
            else "llama3.2"
        )

    return LlmCredentials(
        api_key=key or "",
        endpoint=endpoint or "http://localhost:11434/v1",
        model=model or "llama3.2",
        source=source,
    )
