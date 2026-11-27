"""Text-only transport for local System 1 observations to a System 2 service."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import requests


_OBSERVATION_FIELDS = (
    "timestamp",
    "frame_sequence",
    "primary_category",
    "stability",
    "surprise_score",
    "description",
    "visible_entities",
    "has_active_modal",
    "is_stable",
    "confidence",
    "uncertainty",
)
_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\b(password|passcode|otp|one[- ]time code)\s*[:=]\s*\S+"),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{5,}\d)(?!\w)"),
)


def _redact_text(value: str) -> str:
    for pattern in _SENSITIVE_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def _observation_dict(observation: Any) -> Dict[str, Any]:
    if callable(getattr(observation, "to_dict", None)):
        source = observation.to_dict()
    elif isinstance(observation, Mapping):
        source = observation
    else:
        raise TypeError("observation must be a mapping or expose to_dict()")
    if not isinstance(source, Mapping):
        raise TypeError("observation.to_dict() must return a mapping")

    result: Dict[str, Any] = {}
    for field in _OBSERVATION_FIELDS:
        if field not in source:
            continue
        value = source[field]
        if field in {"timestamp", "surprise_score", "confidence", "uncertainty"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"observation.{field} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"observation.{field} must be a finite number")
            result[field] = float(value)
        elif field == "frame_sequence":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("observation.frame_sequence must be a non-negative integer")
            result[field] = value
        elif field == "visible_entities":
            if not isinstance(value, (list, tuple)) or any(
                not isinstance(item, str) for item in value
            ):
                raise ValueError("observation.visible_entities must be a list of strings")
            result[field] = [_redact_text(item[:160]) for item in value[:20]]
        elif field in {"has_active_modal", "is_stable"}:
            if not isinstance(value, bool):
                raise ValueError(f"observation.{field} must be a boolean")
            result[field] = value
        elif field in {"primary_category", "stability", "description"}:
            if not isinstance(value, str):
                raise ValueError(f"observation.{field} must be a string")
            result[field] = _redact_text(value[:2000])
    return result


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _validate_loopback_endpoint(endpoint_url: str) -> str:
    endpoint = (endpoint_url or "").strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("endpoint_url must not contain embedded credentials")
    hostname = parsed.hostname.lower()
    if hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"External non-loopback endpoints are disabled in 100% local VLA mode; "
            f"only loopback hosts (127.0.0.1, localhost, ::1) are permitted, got {hostname!r}"
        )
    return endpoint


class TextOnlySystem2Client:
    """Send a minimized text observation to a local loopback System 2 endpoint.

    This client has no screen-capture, image-encoding, or embedding interface.
    It sends only an allowlisted JSON schema and accepts a text goal response.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        allow_insecure_http: bool = False,
    ) -> None:
        self.endpoint = _validate_loopback_endpoint(endpoint_url)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key

    def request_goal(
        self, task: str, observation: Any, *, consent_confirmed: bool = False
    ) -> str:
        """Send a text-only request after the caller confirms the user's consent."""
        if consent_confirmed is not True:
            raise PermissionError(
                "System 2 sharing requires explicit user consent from the calling application"
            )
        payload = self.prepare_payload(task, observation)
        endpoint = _validate_loopback_endpoint(self.endpoint)
        response = requests.post(
            f"{endpoint}/v1/system2/goal",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        goal = data.get("goal") if isinstance(data, dict) else None
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("System 2 response must contain a non-empty text 'goal'")
        return goal.strip()

    @staticmethod
    def prepare_payload(task: str, observation: Any) -> Dict[str, Any]:  # noqa: UP006
        """Build the exact redacted, allowlisted payload without sending it."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        return {
            "task": _redact_text(task.strip()[:4000]),
            "observation": _observation_dict(observation),
        }


System2Client = TextOnlySystem2Client
