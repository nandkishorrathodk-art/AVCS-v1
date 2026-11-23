"""AVCS Thin-Client Streaming Engine for Remote VLA Actuation.

Provides low-latency communication with the AVCS VLA model server:
1. Persistent WebSocket transport (/ws/act) with raw binary Turbo JPEG payloads.
2. Low-latency Turbo JPEG encoding (<15ms, <35KB).
3. Automatic HTTP POST (/act) fallback.
4. Telemetry tracking for capture, compression, network roundtrip, and server latency.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from avcs.runtime.turbo_jpeg import encode_turbo_jpeg

try:
    import websocket
except ImportError:
    websocket = None

logger = logging.getLogger("avcs.runtime.vla_client")


class VLAClient:
    """Thin client for streaming desktop observations to the AVCS VLA cloud server."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        use_websocket: bool = True,
        jpeg_quality: int = 75,
        timeout_s: float = 15.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.use_websocket = use_websocket
        self.jpeg_quality = jpeg_quality
        self.timeout_s = timeout_s
        self._ws: Optional[Any] = None

    @staticmethod
    def _strip_endpoint_suffixes(url: str) -> str:
        clean = url.rstrip("/")
        for suffix in ("/ws/act", "/act", "/ws"):
            if clean.endswith(suffix):
                clean = clean[:-len(suffix)].rstrip("/")
                break
        return clean

    def _resolve_ws_url(self, instruction: str = "", width: int = 1280, height: int = 800) -> str:
        base = self._strip_endpoint_suffixes(self.server_url)
        if base.startswith("http://"):
            ws_base = "ws://" + base[7:]
        elif base.startswith("https://"):
            ws_base = "wss://" + base[8:]
        elif base.startswith("ws://") or base.startswith("wss://"):
            ws_base = base
        else:
            ws_base = "ws://" + base

        ws_url = f"{ws_base}/ws/act"
        params = {"width": str(width), "height": str(height)}
        if instruction:
            params["instruction"] = instruction
        return f"{ws_url}?{urllib.parse.urlencode(params)}"

    def _resolve_http_url(self) -> str:
        base = self._strip_endpoint_suffixes(self.server_url)
        if base.startswith("ws://"):
            http_base = "http://" + base[5:]
        elif base.startswith("wss://"):
            http_base = "https://" + base[6:]
        elif base.startswith("http://") or base.startswith("https://"):
            http_base = base
        else:
            http_base = "http://" + base

        return f"{http_base}/act"

    def _ensure_ws_connected(self, instruction: str = "", width: int = 1280, height: int = 800) -> Any:
        if self._ws is not None:
            try:
                # Check connection status
                if hasattr(self._ws, "connected") and self._ws.connected:
                    return self._ws
            except Exception:
                self.close()

        if websocket is None:
            raise RuntimeError("websocket-client package is required for WebSocket transport: pip install websocket-client")

        ws_url = self._resolve_ws_url(instruction, width, height)
        logger.info(f"Connecting persistent WebSocket to {ws_url}...")
        self._ws = websocket.create_connection(ws_url, timeout=self.timeout_s)
        return self._ws

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def __enter__(self) -> "VLAClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def act(
        self,
        frame: Union[np.ndarray, Image.Image],
        instruction: str = "Perform a left click on the primary target",
        width: int = 1280,
        height: int = 800,
    ) -> Dict[str, Any]:
        """Dispatches screen frame to VLA server and receives predicted action."""
        # 1. Compress with low-latency Turbo JPEG
        t_enc_start = time.time()
        jpeg_bytes = encode_turbo_jpeg(frame, quality=self.jpeg_quality)
        t_enc_ms = (time.time() - t_enc_start) * 1000.0

        if self.use_websocket:
            try:
                return self._act_websocket(jpeg_bytes, instruction, width, height, t_enc_ms)
            except Exception as exc:
                logger.warning(f"WebSocket execution failed ({exc}); falling back to HTTP /act...")
                self.close()

        return self._act_http(jpeg_bytes, instruction, width, height, t_enc_ms)

    def _act_websocket(
        self,
        jpeg_bytes: bytes,
        instruction: str,
        width: int,
        height: int,
        t_enc_ms: float,
    ) -> Dict[str, Any]:
        ws = self._ensure_ws_connected(instruction, width, height)

        # Build framed packet: [4-byte header_len] [header JSON] [jpeg binary]
        header_bytes = json.dumps({
            "instruction": instruction,
            "width": width,
            "height": height,
        }).encode("utf-8")
        packet = len(header_bytes).to_bytes(4, byteorder="big") + header_bytes + jpeg_bytes

        t_net_start = time.time()
        ws.send_binary(packet)
        raw_msg = ws.recv()
        t_roundtrip_ms = (time.time() - t_net_start) * 1000.0

        result = json.loads(raw_msg)
        result["compression_ms"] = round(t_enc_ms, 2)
        result["roundtrip_ms"] = round(t_roundtrip_ms, 2)
        result["payload_bytes"] = len(jpeg_bytes)
        return result

    def _act_http(
        self,
        jpeg_bytes: bytes,
        instruction: str,
        width: int,
        height: int,
        t_enc_ms: float,
    ) -> Dict[str, Any]:
        b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
        payload = json.dumps({
            "instruction": instruction,
            "image_base64": b64_str,
            "width": width,
            "height": height,
        }).encode("utf-8")

        http_url = self._resolve_http_url()
        req = urllib.request.Request(
            http_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t_net_start = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        t_roundtrip_ms = (time.time() - t_net_start) * 1000.0

        result["compression_ms"] = round(t_enc_ms, 2)
        result["roundtrip_ms"] = round(t_roundtrip_ms, 2)
        result["payload_bytes"] = len(jpeg_bytes)
        return result
