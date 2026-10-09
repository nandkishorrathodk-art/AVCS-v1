"""AVCS Multi-Agent Fleet Stream Aggregator & Telemetry Server.

Allows operations and QA teams to monitor concurrent virtual AI agent screens
in real time, providing low-latency MJPEG feeds and Prometheus metrics.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from avcs import ScreenEye, CortexState, SyntheticScreen


@dataclass
class FleetAgentNode:
    """Represents an active virtual agent screen node in the fleet."""
    agent_id: str
    name: str
    eye: ScreenEye
    fps: int = 25
    frames_processed: int = 0
    tokens_saved: int = 0
    latest_state: Optional[CortexState] = None
    latest_jpeg_bytes: Optional[bytes] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class AVCSFleetManager:
    """Central fleet aggregator managing concurrent virtual agent perception streams."""

    def __init__(self) -> None:
        self.agents: Dict[str, FleetAgentNode] = {}
        self.primary_agent_id: Optional[str] = None
        self._running = False
        self._threads: List[threading.Thread] = []

    def register_agent(
        self,
        agent_id: str,
        name: str,
        eye: Optional[ScreenEye] = None,
        fps: int = 25,
    ) -> FleetAgentNode:
        """Registers a new agent into the monitored fleet."""
        node_eye = eye or ScreenEye(source=SyntheticScreen(width=640, height=360), fps=fps)
        node = FleetAgentNode(agent_id=agent_id, name=name, eye=node_eye, fps=fps)
        self.agents[agent_id] = node
        if self.primary_agent_id is None:
            self.primary_agent_id = agent_id
        return node

    def start_fleet(self) -> None:
        """Starts perception tick loops across all fleet agents."""
        self._running = True
        for agent_id, node in self.agents.items():
            t = threading.Thread(target=self._agent_loop, args=(node,), daemon=True)
            self._threads.append(t)
            t.start()

    def stop_fleet(self) -> None:
        """Stops all agent stream loops."""
        self._running = False

    def _agent_loop(self, node: FleetAgentNode) -> None:
        tick_interval = 1.0 / max(1, node.fps)
        while self._running:
            t0 = time.time()
            try:
                frame = node.eye.source.grab()
                state = node.eye.perceive_now(frame=frame)

                # Render JPEG bytes for stream
                pil_img = Image.fromarray(frame)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=80)
                jpeg_data = buf.getvalue()

                with node.lock:
                    node.latest_state = state
                    node.latest_jpeg_bytes = jpeg_data
                    node.frames_processed += 1
                    node.tokens_saved += 1600 if state.stability_state in ["animating", "transitioning"] else 0
            except Exception as e:
                pass

            dt = time.time() - t0
            sleep_needed = tick_interval - dt
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    def get_agent_jpeg(self, agent_id: Optional[str] = None) -> Optional[bytes]:
        """Returns latest JPEG frame bytes for specified agent or primary agent."""
        target_id = agent_id or self.primary_agent_id
        if target_id and target_id in self.agents:
            with self.agents[target_id].lock:
                return self.agents[target_id].latest_jpeg_bytes
        return None

    def get_fleet_telemetry(self) -> Dict[str, Any]:
        """Returns aggregated JSON telemetry across all active fleet nodes."""
        fleet_data = []
        for aid, node in self.agents.items():
            with node.lock:
                st = node.latest_state
                fleet_data.append({
                    "agent_id": node.agent_id,
                    "name": node.name,
                    "frames_processed": node.frames_processed,
                    "tokens_saved": node.tokens_saved,
                    "fps": node.fps,
                    "primary_category": st.primary_category if st else "unknown",
                    "stability_state": st.stability_state if st else "unknown",
                    "surprise_score": round(st.surprise_score, 4) if st else 0.0,
                })
        return {
            "total_agents": len(self.agents),
            "primary_agent": self.primary_agent_id,
            "agents": fleet_data,
        }

    def get_prometheus_metrics(self) -> str:
        """Exports fleet telemetry in standard Prometheus exposition format."""
        lines = [
            "# HELP avcs_agent_frames_total Total video frames processed by visual cortex",
            "# TYPE avcs_agent_frames_total counter",
        ]
        for aid, node in self.agents.items():
            lines.append(f'avcs_agent_frames_total{{agent="{aid}"}} {node.frames_processed}')

        lines.extend([
            "# HELP avcs_agent_tokens_saved Cumulative VLM tokens saved by reflex filter",
            "# TYPE avcs_agent_tokens_saved counter",
        ])
        for aid, node in self.agents.items():
            lines.append(f'avcs_agent_tokens_saved{{agent="{aid}"}} {node.tokens_saved}')

        lines.extend([
            "# HELP avcs_agent_surprise Friston predictive surprise score",
            "# TYPE avcs_agent_surprise gauge",
        ])
        for aid, node in self.agents.items():
            score = round(node.latest_state.surprise_score, 4) if node.latest_state else 0.0
            lines.append(f'avcs_agent_surprise{{agent="{aid}"}} {score}')

        return "\n".join(lines) + "\n"


class FleetHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving dashboard, MJPEG streams, and REST telemetry."""

    fleet_manager: Optional[AVCSFleetManager] = None
    web_dir: Path = Path(__file__).resolve().parent / "web"

    def do_GET(self) -> None:
        fm = self.fleet_manager
        if fm is None:
            self.send_error(503, "Fleet manager not initialized")
            return

        if self.path == "/" or self.path == "/index.html":
            # Serve index.html
            html_path = self.web_dir / "index.html"
            if html_path.is_file():
                content = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Dashboard HTML not found")

        elif self.path == "/stream.mjpg":
            # Serve primary MJPEG stream
            self._stream_mjpeg(agent_id=None)

        elif self.path.startswith("/api/fleet/") and self.path.endswith("/stream.mjpg"):
            # Serve specific agent MJPEG stream: /api/fleet/<agent_id>/stream.mjpg
            parts = self.path.strip("/").split("/")
            agent_id = parts[2]
            self._stream_mjpeg(agent_id=agent_id)

        elif self.path == "/api/telemetry" or self.path == "/api/fleet":
            # Return fleet JSON telemetry
            data = json.dumps(fm.get_fleet_telemetry()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif self.path == "/metrics":
            # Prometheus metrics endpoint
            data = fm.get_prometheus_metrics().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_error(404, "Endpoint not found")

    def _stream_mjpeg(self, agent_id: Optional[str] = None) -> None:
        fm = self.fleet_manager
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        for _ in range(50):  # Stream chunk loop
            data = fm.get_agent_jpeg(agent_id=agent_id)
            if data is not None:
                try:
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
            time.sleep(0.04)  # ~25 FPS

