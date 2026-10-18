"""Real-time local neuro-streaming telemetry.

The packet schema carries visual and sensorimotor telemetry; achievable rate
depends on the capture source, perception workload, and subscribers.
"""
from __future__ import annotations

import collections
import dataclasses
import struct
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np

from forge.vision.cortex.model import CortexState
from forge.vision.cortex.sdk import ScreenEye

COMPACT_FMT_40 = "<dI5fB7x"
COMPACT_FMT_33 = "<dI5fB"


@dataclasses.dataclass(frozen=True)
class NeuroPacket:
    """Visual and sensorimotor telemetry packet with optional capture provenance."""
    timestamp: float
    seq: int
    fixation: Tuple[float, float]
    motor_flow: Tuple[float, float]
    contact_acquired: bool
    primary_category: str
    stability_state: str
    surprise_score: float
    inference_ms: float
    action_hotspots: List[Tuple[float, float, str, float]] = dataclasses.field(default_factory=list)
    visual_latent: Optional[np.ndarray] = None
    source_frame_sequence: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        packet = {
            "timestamp": self.timestamp,
            "seq": self.seq,
            "fixation": [round(self.fixation[0], 4), round(self.fixation[1], 4)],
            "motor_flow": [round(self.motor_flow[0], 4), round(self.motor_flow[1], 4)],
            "contact_acquired": self.contact_acquired,
            "primary_category": self.primary_category,
            "stability_state": self.stability_state,
            "surprise_score": round(self.surprise_score, 4),
            "inference_ms": round(self.inference_ms, 2),
            "action_hotspots": [
                [round(x, 4), round(y, 4), act, round(conf, 3)]
                for x, y, act, conf in self.action_hotspots
            ],
        }
        if self.source_frame_sequence is not None:
            packet["source_frame_sequence"] = self.source_frame_sequence
        return packet

    COMPACT_FMT_40 = "<dI5fB7x"
    COMPACT_FMT_33 = "<dI5fB"

    def pack_compact(self) -> bytes:
        """Serializes core coordinates and flow into 40-byte C-ABI aligned binary for IPC."""
        return struct.pack(
            self.COMPACT_FMT_40,
            float(self.timestamp),
            int(self.seq),
            float(self.fixation[0]),
            float(self.fixation[1]),
            float(self.motor_flow[0]),
            float(self.motor_flow[1]),
            float(self.surprise_score),
            1 if self.contact_acquired else 0,
        )

    def to_compact(self) -> bytes:
        """Serializes core coordinates and flow into 40-byte C-ABI aligned binary for IPC."""
        return self.pack_compact()

    @classmethod
    def unpack_compact(cls, data: bytes) -> Dict[str, Any]:
        """Unpacks compact binary stream telemetry (supports 40-byte aligned and legacy 33-byte)."""
        fmt = cls.COMPACT_FMT_40 if len(data) >= 40 else cls.COMPACT_FMT_33
        size = struct.calcsize(fmt)
        if len(data) < size:
            raise ValueError(f"Data length {len(data)} is too short for format {fmt} (needs {size} bytes)")
        ts, seq, fx, fy, flowx, flowy, surprise, contact = struct.unpack(fmt, data[:size])
        return {
            "timestamp": ts,
            "seq": seq,
            "fixation": (fx, fy),
            "motor_flow": (flowx, flowy),
            "surprise_score": surprise,
            "contact_acquired": bool(contact),
        }

    @classmethod
    def from_compact(
        cls,
        data: bytes,
        primary_category: str = "unknown",
        stability_state: str = "stable",
        inference_ms: float = 0.0,
        source_frame_sequence: Optional[int] = None,
    ) -> NeuroPacket:
        """Constructs a typed NeuroPacket directly from compact binary data."""
        unpacked = cls.unpack_compact(data)
        return cls(
            timestamp=unpacked["timestamp"],
            seq=unpacked["seq"],
            fixation=unpacked["fixation"],
            motor_flow=unpacked["motor_flow"],
            contact_acquired=unpacked["contact_acquired"],
            primary_category=primary_category,
            stability_state=stability_state,
            surprise_score=unpacked["surprise_score"],
            inference_ms=inference_ms,
            source_frame_sequence=source_frame_sequence,
        )


class NeuroStreamBus:
    """Thread-safe circular ring buffer and low-contention pub-sub broadcaster for telemetry."""

    def __init__(self, buffer_size: int = 120) -> None:
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        self._buffer: Deque[NeuroPacket] = collections.deque(maxlen=buffer_size)
        self._subscribers: Tuple[Callable[[NeuroPacket], None], ...] = ()
        self._packet_count = 0
        self._last_tick_time = time.time()
        self._fps = 0.0

    def publish(self, packet: NeuroPacket) -> None:
        with self._lock:
            self._buffer.append(packet)
            self._packet_count += 1
            now = packet.timestamp
            dt = now - self._last_tick_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps = 0.9 * self._fps + 0.1 * inst_fps if self._fps > 0 else inst_fps
            self._last_tick_time = now

        # Non-blocking lock-free subscriber iteration under GIL (copy-on-write immutable tuple)
        for sub in self._subscribers:
            try:
                sub(packet)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[NeuroPacket], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers = self._subscribers + (callback,)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers = tuple(cb for cb in self._subscribers if cb != callback)

        return unsubscribe

    def get_latest(self) -> Optional[NeuroPacket]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_window(self, count: int = 10) -> List[NeuroPacket]:
        with self._lock:
            items = list(self._buffer)
            return items[-count:] if len(items) >= count else items

    def get_since(self, seq: int) -> List[NeuroPacket]:
        """Returns all buffered packets with sequence greater than seq."""
        with self._lock:
            return [p for p in self._buffer if p.seq > seq]

    def clear(self) -> None:
        """Clears ring buffer and resets packet counters."""
        with self._lock:
            self._buffer.clear()
            self._packet_count = 0
            self._fps = 0.0

    def to_binary_stream(self, count: Optional[int] = None) -> bytes:
        """Serializes the last `count` packets into contiguous 40-byte binary stream."""
        with self._lock:
            pkts = list(self._buffer)
            if count is not None:
                pkts = pkts[-count:]
            return b"".join(p.pack_compact() for p in pkts)

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def packet_count(self) -> int:
        with self._lock:
            return self._packet_count

    @property
    def dropped_count(self) -> int:
        """Returns number of packets dropped due to ring buffer capacity overflow."""
        with self._lock:
            return max(0, self._packet_count - self.buffer_size)


class RealTimeNeuroStreamer:
    """Connects AVCS ScreenEye or HumanEye perception ticks directly to the high-rate NeuroStreamBus."""

    def __init__(self, eye: Any, bus: Optional[NeuroStreamBus] = None) -> None:
        self.eye = eye
        self.bus = bus or NeuroStreamBus()
        self._seq = 0
        self._active = False
        self._is_human_eye = hasattr(eye, "observe") and not hasattr(eye, "on_change")
        if not self._is_human_eye and hasattr(self.eye, "on_change"):
            self._setup_listener()
        elif hasattr(self.eye, "attach_stream_bus"):
            self.eye.attach_stream_bus(self.bus)

    def _create_packet(self, state: Any) -> NeuroPacket:
        return NeuroPacket(
            timestamp=getattr(state, "timestamp", time.time()),
            seq=self._seq,
            fixation=getattr(state, "gaze_fixation", getattr(state, "fixation", (0.5, 0.5))),
            motor_flow=getattr(state, "motor_flow", (0.0, 0.0)),
            contact_acquired=getattr(state, "contact_acquired", False),
            primary_category=getattr(state, "primary_category", "unknown"),
            stability_state=getattr(state, "stability", getattr(state, "stability_state", "stable")),
            surprise_score=getattr(state, "surprise_score", 0.0),
            inference_ms=getattr(state, "latency_ms", getattr(state, "inference_ms", 0.0)),
            action_hotspots=list(getattr(state, "affordance_hotspots", getattr(state, "action_hotspots", [])) or []),
            visual_latent=getattr(state, "visual_latent", None),
        )

    def _setup_listener(self) -> None:
        def on_screen_state(state: Any) -> None:
            self._seq += 1
            packet = self._create_packet(state)
            self.bus.publish(packet)

        if hasattr(self.eye, "on_change"):
            self.eye.on_change(on_screen_state)

    def start(self) -> RealTimeNeuroStreamer:
        self._active = True
        if hasattr(self.eye, "start"):
            self.eye.start()
        return self

    def stop(self) -> None:
        self._active = False
        if hasattr(self.eye, "stop"):
            self.eye.stop()

    def step_once(self) -> NeuroPacket:
        """Forces a single synchronous perception tick and broadcasts the resulting packet."""
        prev_seq = self._seq
        if hasattr(self.eye, "perceive_now"):
            state = self.eye.perceive_now()
        elif hasattr(self.eye, "observe"):
            state = self.eye.observe()
        else:
            raise NotImplementedError("eye must have perceive_now() or observe()")

        latest = self.bus.get_latest()
        if latest is not None and self._seq > prev_seq:
            return latest

        self._seq += 1
        packet = self._create_packet(state)
        self.bus.publish(packet)
        return packet

    def tick(self) -> NeuroPacket:
        """Alias for step_once() to execute a single synchronous streaming tick."""
        return self.step_once()

