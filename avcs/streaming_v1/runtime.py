"""Bounded latest-frame observation runtime, independent of capture backends."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
import threading
import time
from typing import Callable, Protocol

import numpy as np
import torch
from PIL import Image

from avcs.streaming_v1.model import CortexState, StreamingVisualCortexV1


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    timestamp: float
    rgb: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("frame sequence must be a non-negative integer.")
        if not math.isfinite(self.timestamp):
            raise ValueError("frame timestamp must be finite.")
        if not isinstance(self.rgb, np.ndarray) or self.rgb.dtype != np.uint8:
            raise TypeError("frame must be a uint8 RGB numpy array.")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or min(self.rgb.shape[:2]) < 32:
            raise ValueError("frame must have shape (height>=32, width>=32, 3).")
        snapshot = self.rgb.copy()
        snapshot.setflags(write=False)
        object.__setattr__(self, "rgb", snapshot)


@dataclass(frozen=True)
class ObservationUpdate:
    sequence: int
    timestamp: float
    updated_at: float
    timestamp_delta: float | None
    gap_reset: bool
    latent: np.ndarray
    change_score: float
    change_map: np.ndarray
    motion_grid_displacement: np.ndarray
    model_latency_ms: float
    capture_to_update_ms: float


class CaptureSource(Protocol):
    def open(self) -> object: ...
    def grab(self) -> np.ndarray: ...
    def close(self) -> None: ...


class StreamingPermissions:
    """Fail-closed tri-state security and privacy controls for streaming perception.

    Separate independent controls:
      1. enable_observation: bool = False
      2. enable_online_learning: bool = False
      3. authorize_checkpoint_save: bool = False

    All three fail closed and raise PermissionError if accessed while False.
    """

    def __init__(
        self,
        enable_observation: bool = False,
        enable_online_learning: bool = False,
        authorize_checkpoint_save: bool = False,
    ) -> None:
        self.enable_observation = bool(enable_observation)
        self.enable_online_learning = bool(enable_online_learning)
        self.authorize_checkpoint_save = bool(authorize_checkpoint_save)

    def assert_observation_allowed(self) -> None:
        if not self.enable_observation:
            raise PermissionError(
                "Live observation is disabled (enable_observation=False). "
                "Explicit authorization is required before capturing screen pixels."
            )

    def assert_learning_allowed(self) -> None:
        if not self.enable_online_learning:
            raise PermissionError(
                "Online parameter learning is disabled (enable_online_learning=False). "
                "Model parameters must remain frozen."
            )

    def assert_save_allowed(self) -> None:
        if not self.authorize_checkpoint_save:
            raise PermissionError(
                "Checkpoint persistence is disabled (authorize_checkpoint_save=False). "
                "Writing weights to disk is prohibited without explicit authorization."
            )


class LatestFrameMailbox:
    """Thread-safe decoupled 1-slot latest-value mailbox buffer.

    Dynamics:
    - Bounded capacity of exactly 1 frame.
    - Decoupled background capture and inference.
    - Atomic frame swap with thread condition variable.
    - Drop-tail policy: if a frame is unread when a newer frame arrives,
      the unread frame is dropped immediately, and replaced_count is incremented.
    - Strictly zero queue backlog and zero stale frame lag.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._packet: FramePacket | None = None
        self._closed = False
        self.replaced_count = 0
        self._last_sequence: int | None = None
        self._last_timestamp: float | None = None

    def put(self, packet: FramePacket) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("frame buffer is closed.")
            if self._last_sequence is not None and (
                packet.sequence <= self._last_sequence
                or packet.timestamp <= self._last_timestamp
            ):
                raise ValueError("out-of-order frame rejected by latest-frame buffer.")
            if self._packet is not None:
                self.replaced_count += 1
            self._packet = packet
            self._last_sequence = packet.sequence
            self._last_timestamp = packet.timestamp
            self._condition.notify_all()

    def get(self, timeout: float | None = None) -> FramePacket | None:
        with self._condition:
            if self._packet is None and not self._closed:
                self._condition.wait(timeout)
            packet = self._packet
            self._packet = None
            return packet

    def take(self, timeout: float | None = None) -> FramePacket | None:
        """Alias for get() providing take() semantics."""
        return self.get(timeout=timeout)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def capacity(self) -> int:
        return 1

    def is_empty(self) -> bool:
        with self._condition:
            return self._packet is None


# Backward-compatible alias
LatestFrameBuffer = LatestFrameMailbox


class StreamingObserver:
    """Continuously run a learned state model over a live, bounded frame stream."""

    def __init__(
        self,
        model: StreamingVisualCortexV1,
        *,
        input_size: tuple[int, int] = (128, 128),
        max_frame_age_seconds: float = 1.0,
        reset_after_seconds: float = 2.0,
        device: str = "cpu",
        on_update: Callable[[ObservationUpdate], None] | None = None,
        permissions: StreamingPermissions | None = None,
    ) -> None:
        if min(input_size) < 32:
            raise ValueError("input dimensions must each be at least 32.")
        if not math.isfinite(max_frame_age_seconds) or max_frame_age_seconds <= 0:
            raise ValueError("max_frame_age_seconds must be positive and finite.")
        if not math.isfinite(reset_after_seconds) or reset_after_seconds <= 0:
            raise ValueError("reset_after_seconds must be positive and finite.")
        self.model = model.to(torch.device(device)).eval()
        self.device = torch.device(device)
        self.input_size = input_size
        self.max_frame_age_seconds = max_frame_age_seconds
        self.reset_after_seconds = reset_after_seconds
        self.on_update = on_update
        self.permissions = permissions
        self.frames = LatestFrameMailbox()
        self._state: CortexState | None = None
        self._last_sequence: int | None = None
        self._last_timestamp: float | None = None
        self._last_processed_timestamp: float | None = None
        self._last_frame_digest: bytes | None = None
        self._updates: deque[ObservationUpdate] = deque(maxlen=2048)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._source: CaptureSource | None = None
        self.captured_count = 0
        self.processed_count = 0
        self.duplicate_count = 0
        self.out_of_order_count = 0
        self.stale_count = 0
        self.inference_error_count = 0
        self.capture_recovery_count = 0
        self.last_error: str | None = None

    @property
    def updates(self) -> tuple[ObservationUpdate, ...]:
        with self._lock:
            return tuple(self._updates)

    @property
    def dropped_count(self) -> int:
        return self.frames.replaced_count

    def process(self, packet: FramePacket, *, now: float | None = None) -> ObservationUpdate | None:
        """Process one ordered packet; stale, duplicate, or reordered inputs abstain."""
        arrival = time.monotonic() if now is None else now
        digest = hashlib.sha256(memoryview(packet.rgb)).digest()
        with self._lock:
            if self._last_sequence is not None and (
                packet.sequence <= self._last_sequence
                or packet.timestamp <= self._last_timestamp
            ):
                self.out_of_order_count += 1
                return None
            if self._last_frame_digest == digest:
                self.duplicate_count += 1
                self._last_sequence = packet.sequence
                self._last_timestamp = packet.timestamp
                return None
            age = max(0.0, arrival - packet.timestamp)
            if age > self.max_frame_age_seconds:
                self.stale_count += 1
                self._last_sequence = packet.sequence
                self._last_timestamp = packet.timestamp
                return None
            delta = (
                None
                if self._last_processed_timestamp is None
                else packet.timestamp - self._last_processed_timestamp
            )
            gap_reset = delta is not None and delta > self.reset_after_seconds
            if gap_reset:
                self._state = None
            prepared = self._prepare(packet.rgb)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    output, next_state = self.model.forward_step(
                        prepared,
                        self._state,
                        timestamp_delta=delta or 0.0,
                    )
            except Exception as exc:
                self._state = None
                self.inference_error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise
            model_ms = (time.perf_counter() - started) * 1000.0
            updated_at = time.monotonic()
            update = ObservationUpdate(
                sequence=packet.sequence,
                timestamp=packet.timestamp,
                updated_at=updated_at,
                timestamp_delta=delta,
                gap_reset=gap_reset,
                latent=output["latent"][0].detach().cpu().numpy().copy(),
                change_score=float(output["change_score"][0].item()),
                change_map=torch.sigmoid(output["change_logits"][0, 0]).detach().cpu().numpy().copy(),
                motion_grid_displacement=output["motion_grid_displacement"][0].detach().cpu().numpy().copy(),
                model_latency_ms=model_ms,
                capture_to_update_ms=max(0.0, (updated_at - packet.timestamp) * 1000.0),
            )
            self._state = next_state
            self._last_sequence = packet.sequence
            self._last_timestamp = packet.timestamp
            self._last_processed_timestamp = packet.timestamp
            self._last_frame_digest = digest
            self.processed_count += 1
            self.last_error = None
            self._updates.append(update)
        if self.on_update is not None:
            self.on_update(update)
        return update

    def reset(self) -> None:
        with self._lock:
            self._state = None
            self._last_sequence = None
            self._last_timestamp = None
            self._last_processed_timestamp = None
            self._last_frame_digest = None

    def observe(self, packet: FramePacket, *, now: float | None = None) -> ObservationUpdate | None:
        """Observe a frame packet with fail-closed observation permission assertion."""
        if self.permissions is not None:
            self.permissions.assert_observation_allowed()
        return self.process(packet, now=now)

    def start(self, source: CaptureSource, *, target_hz: float = 10.0) -> None:
        if not math.isfinite(target_hz) or target_hz <= 0:
            raise ValueError("target_hz must be positive and finite.")
        if self.permissions is not None:
            self.permissions.assert_observation_allowed()
        with self._lock:
            if self._capture_thread is not None and self._capture_thread.is_alive():
                raise RuntimeError("observer is already running.")
            self._stop.clear()
            self.frames = LatestFrameMailbox()
            self._source = source
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(source, target_hz),
                daemon=True,
                name="avcs-v1-capture",
            )
            self._inference_thread = threading.Thread(
                target=self._inference_loop,
                daemon=True,
                name="avcs-v1-inference",
            )
            self._inference_thread.start()
            self._capture_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.frames.close()
        source = self._source
        if source is not None:
            source.close()
        for thread in (self._capture_thread, self._inference_thread):
            if thread is not None:
                thread.join(timeout)
                if thread.is_alive():
                    raise TimeoutError(f"AVCS {thread.name} did not stop within {timeout}s.")
        self._capture_thread = None
        self._inference_thread = None
        self._source = None

    def _prepare(self, rgb: np.ndarray) -> torch.Tensor:
        width, height = self.input_size
        image = Image.fromarray(rgb, mode="RGB")
        source_width, source_height = image.size
        scale = min(width / source_width, height / source_height)
        resized_size = (
            max(1, round(source_width * scale)),
            max(1, round(source_height * scale)),
        )
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize(resized_size, resampling.BILINEAR)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(image, ((width - resized_size[0]) // 2, (height - resized_size[1]) // 2))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _capture_loop(self, source: CaptureSource, target_hz: float) -> None:
        interval = 1.0 / target_hz
        sequence = 0
        backoff = 0.25
        while not self._stop.is_set():
            try:
                source.open()
                self.reset()
                while not self._stop.is_set():
                    started = time.monotonic()
                    rgb = source.grab()
                    timestamp = time.monotonic()
                    packet = FramePacket(sequence, timestamp, rgb)
                    self.frames.put(packet)
                    with self._lock:
                        self.captured_count += 1
                    sequence += 1
                    backoff = 0.25
                    self._stop.wait(max(0.0, interval - (time.monotonic() - started)))
            except EOFError:
                break
            except Exception as exc:
                if self._stop.is_set():
                    break
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.capture_recovery_count += 1
                self.reset()
                try:
                    source.close()
                except Exception as close_exc:
                    self.last_error += f"; close failed: {type(close_exc).__name__}: {close_exc}"
                self._stop.wait(backoff)
                backoff = min(5.0, backoff * 2.0)
        self.frames.close()
        try:
            source.close()
        except Exception as exc:
            self.last_error = f"capture close failed: {type(exc).__name__}: {exc}"

    def _inference_loop(self) -> None:
        while True:
            packet = self.frames.get(timeout=0.1)
            if packet is None:
                if self.frames.closed:
                    break
                continue
            try:
                self.process(packet)
            except Exception:
                continue
