"""AVCS Cloud Client for External Projects (Project X).

This client sends local screen frames to the configured AVCS inference server,
receives structured results, and can execute pointer actions locally.

Usage in any external project:
    from avcs.server.client import AVCSCloudClient

    client = AVCSCloudClient(
        endpoint_url="https://avcs-api.yourcompany.com",
        api_key="your_secret_api_key",
    )
    state = client.see()
    print("Detected Scene:", state["category"])
    affordances = client.get_affordances("clickable")
    if affordances:
        client.click(affordances[0][:2])
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
import requests

from forge.vision.capture import open_capture
from forge.vision.cortex.motor import get_default_driver


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


class AVCSCloudClient:
    """Lightweight client for local loopback AVCS inference server."""

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        source: Optional[Any] = None,
        force_virtual_motor: bool = False,
        timeout: float = 5.0,
        allow_insecure_http: bool = False,
    ) -> None:
        self.endpoint = _validate_loopback_endpoint(endpoint_url)
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key

        # Local capture and motor driver (runs on client machine)
        if source is None:
            self.source = open_capture(prefer_real=True, threaded=True)
        else:
            self.source = source

        self.driver = get_default_driver(force_virtual=force_virtual_motor)
        self._last_state: Optional[Dict[str, Any]] = None
        self._last_frame: Optional[np.ndarray] = None

    def _encode_image(self, frame: np.ndarray, quality: int = 85) -> str:
        """Compresses RGB uint8 numpy array to base64 JPEG (<50 KB) for high-speed transfer."""
        arr = frame
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        pil_img = Image.fromarray(arr)
        if pil_img.mode not in ("RGB", "L"):
            pil_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _url(self, path: str) -> str:
        endpoint = _validate_loopback_endpoint(self.endpoint)
        return f"{endpoint}{path}"

    def ping(self) -> Dict[str, Any]:
        """Checks local AVCS server health and model readiness."""
        resp = requests.get(self._url("/v1/health"), headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def see(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Captures local screen, sends to local AVCS server, and returns visual perception state."""
        if frame is None:
            frame = self.source.grab()

        self._last_frame = frame
        b64_img = self._encode_image(frame)

        payload = {
            "image_b64": b64_img,
            "action_type": "all",
            "top_k": 5,
        }

        resp = requests.post(
            self._url("/v1/perceive"),
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._last_state = data
        return data

    def get_affordances(self, action_type: str = "clickable") -> List[Tuple[float, float, str, float]]:
        """Returns interactive affordance hotspots from the most recent perception state."""
        if self._last_state is None:
            self.see()
        hotspots = self._last_state.get("affordances", [])
        return [tuple(h) for h in hotspots if action_type == "all" or h[2] == action_type]

    def foveate(self, target: Tuple[float, float]) -> Dict[str, Any]:
        """Requests high-acuity foveal inspection for target coordinate from local server."""
        if self._last_frame is None:
            self.see()

        b64_img = self._encode_image(self._last_frame)
        payload = {"image_b64": b64_img, "target": list(target)}
        resp = requests.post(self._url("/v1/foveate"), json=payload, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def ground(
        self, query: str, frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Request a candidate target point; the response is not authorization to act."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Grounding query cannot be empty")
        if frame is None:
            frame = self.source.grab()
        self._last_frame = frame
        payload = {
            "image_b64": self._encode_image(frame),
            "query": clean_query,
        }
        resp = requests.post(
            self._url("/v1/ground"),
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "target_coord" not in data:
            raise ValueError("Grounding response is missing the target_coord field")
        return data

    def verify_focus(self, target: Tuple[float, float], reference_frame: Optional[np.ndarray] = None) -> bool:
        """Verifies if input element at target visually possesses active typing focus (fail-closed)."""
        if self._last_frame is None:
            self.see()

        payload = {
            "current_image_b64": self._encode_image(self._last_frame),
            "reference_image_b64": self._encode_image(reference_frame) if reference_frame is not None else None,
            "target": list(target),
        }
        resp = requests.post(self._url("/v1/verify_focus"), json=payload, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or type(data.get("is_focused")) is not bool:
            raise ValueError("Focus response must contain a boolean is_focused field")
        return data["is_focused"]

    def move_to(self, target: Tuple[float, float], realtime: bool = True) -> bool:
        """Requests minimum-jerk trajectory from local server and executes smooth movement on local OS."""
        cur_pos = (
            self.driver.get_position_normalized()
            if hasattr(self.driver, "get_position_normalized")
            else (0.5, 0.5)
        )

        payload = {"start": list(cur_pos), "target": list(target)}
        resp = requests.post(self._url("/v1/plan_trajectory"), json=payload, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        traj_data = resp.json()

        waypoints = traj_data.get("waypoints", [])
        if not waypoints:
            self.driver.move_to_normalized(target[0], target[1])
            return True

        if realtime:
            t0 = time.perf_counter()
            for pt in waypoints:
                target_t = pt["timestamp"]
                while (time.perf_counter() - t0) < target_t:
                    time.sleep(0.001)
                self.driver.move_to_normalized(pt["x"], pt["y"])
        else:
            self.driver.move_to_normalized(target[0], target[1])

        return True

    def click(self, target: Tuple[float, float], button: str = "left") -> bool:
        """Moves cursor with minimum-jerk kinematics and clicks target."""
        self.move_to(target=target, realtime=True)
        time.sleep(0.02)
        self.driver.click(button=button)
        return True

    def type_into(
        self,
        target: Tuple[float, float],
        text: str,
        clear_first: bool = True,
        enforce_focus: bool = True,
    ) -> bool:
        """Clicks target and types text with fail-closed focus verification."""
        self.click(target=target)
        time.sleep(0.05)

        if enforce_focus:
            # Re-perceive to check for blinking caret
            self.see()
            is_focused = self.verify_focus(target=target)
            if not is_focused:
                raise RuntimeError(f"Cannot type into {target}: visual focus verification failed closed.")

        if clear_first:
            self.driver.press_hotkey("ctrl", "a")
            self.driver.press_key("backspace")

        self.driver.type_text(text)
        return True

    def close(self) -> None:
        """Closes local capture resources."""
        if hasattr(self.source, "close"):
            try:
                self.source.close()
            except Exception:
                pass
