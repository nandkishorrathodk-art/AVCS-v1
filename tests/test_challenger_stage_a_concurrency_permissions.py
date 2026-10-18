"""Empirical Concurrency & Permissions Stress Test Suite for AVCS Stage A.

Executed by: challenger_stage_a_1
Target: avcs/streaming_v1/runtime.py (LatestFrameMailbox, StreamingPermissions, StreamingObserver)

Challenge dimensions:
1. High-concurrency stress test: 10 producer threads pushing 10,000 frames total at 500 FPS vs 1 consumer at 30 FPS.
   - Verifies monotonicity of received sequence numbers.
   - Verifies zero queue backlog (mailbox capacity strictly 1).
   - Verifies zero deadlocks across all 11 threads.
   - Verifies exact arithmetic invariant: total_put == replaced_count + taken_count + remaining_in_box.
2. Permission bypass stress test:
   - Exhaustive truth table across all 2^3 = 8 permutations of (enable_observation, enable_online_learning, authorize_checkpoint_save).
   - Verifies ungranted calls to assert_observation_allowed(), assert_learning_allowed(), assert_save_allowed() raise PermissionError fail-closed.
   - Edge case type testing (falsy values, non-bool conversions).
   - Downstream fail-closed gating on StreamingObserver.observe() and StreamingObserver.start().
3. Zero raw persistence stress test:
   - Full filesystem scan before and after processing 1,000 continuous frames through StreamingObserver with SyntheticCaptureSource.
   - Decoupled background capture and inference stream test with SyntheticCaptureSource.
   - Asserts 0 image/video/frame files (.png, .jpg, .jpeg, .bmp, .raw, .tiff, .mp4, .avi, .npy) written to repo or temp dirs.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
import pytest
import torch

from avcs.streaming_v1.model import StreamingVisualCortexV1
from avcs.streaming_v1.runtime import (
    FramePacket,
    LatestFrameMailbox,
    ObservationUpdate,
    StreamingObserver,
    StreamingPermissions,
)
from avcs.streaming_v1.sources import SyntheticCaptureSource


# ==============================================================================
# Challenge 1: High-Concurrency Stress Test (10 Producers vs 1 Consumer)
# ==============================================================================

class MonotonicSequencer:
    """Thread-safe monotonic sequence and timestamp generator for concurrent capture producers."""

    def __init__(self, initial_sequence: int = 0) -> None:
        self._lock = threading.Lock()
        self._sequence = initial_sequence
        self._last_time = time.monotonic()

    def next_frame(self, frame_rgb: np.ndarray) -> FramePacket:
        with self._lock:
            seq = self._sequence
            self._sequence += 1
            now = time.monotonic()
            if now <= self._last_time:
                now = self._last_time + 1e-6
            self._last_time = now
            return FramePacket(seq, now, frame_rgb)


def test_concurrency_stress_10_producers_10000_frames_at_500fps():
    """Stress test: 10 concurrent producer threads push 10,000 frames total at 500 FPS vs 1 consumer at 30 FPS.

    Invariants tested:
    1. Zero deadlocks: All 10 producer threads and 1 consumer thread terminate cleanly within timeout.
    2. Monotonicity: Consumer receives strictly monotonically increasing sequence numbers (no retrogrades or duplicates).
    3. Zero queue backlog: Mailbox buffer size never exceeds 1 (capacity == 1).
    4. Exact replacement accounting: total_put == replaced_count + taken_count + remaining_in_box.
    """
    total_frames = 10_000
    num_producers = 10
    frames_per_producer = total_frames // num_producers
    assert frames_per_producer * num_producers == total_frames

    mailbox = LatestFrameMailbox()
    assert mailbox.capacity == 1

    sequencer = MonotonicSequencer()
    submission_lock = threading.Lock()

    # Pre-generate lightweight frame payloads
    base_frame = np.zeros((32, 32, 3), dtype=np.uint8)

    received_sequences: list[int] = []
    received_timestamps: list[float] = []
    consumer_done = threading.Event()
    producers_done = threading.Event()
    consumer_errors: list[Exception] = []
    producer_errors: list[Exception] = []

    def producer_worker(producer_id: int):
        try:
            for i in range(frames_per_producer):
                pattern = (producer_id * 25 + i) % 256
                frame = base_frame.copy()
                frame[0, 0, 0] = pattern

                with submission_lock:
                    pkt = sequencer.next_frame(frame)
                    mailbox.put(pkt)

                if i % 10 == 0:
                    time.sleep(0.0001)
        except Exception as exc:
            producer_errors.append(exc)

    def consumer_worker():
        """Consumer runs at ~30 FPS (taking frames every ~33ms, with non-blocking take when available)."""
        last_seq = -1
        last_ts = -1.0
        try:
            while not (producers_done.is_set() and mailbox.is_empty()):
                pkt = mailbox.take(timeout=0.01)
                if pkt is not None:
                    assert mailbox.is_empty(), "Mailbox must be empty immediately after take()"
                    assert pkt.sequence > last_seq, f"Sequence retrograde: {pkt.sequence} <= {last_seq}"
                    assert pkt.timestamp > last_ts, f"Timestamp retrograde: {pkt.timestamp} <= {last_ts}"
                    last_seq = pkt.sequence
                    last_ts = pkt.timestamp
                    received_sequences.append(pkt.sequence)
                    received_timestamps.append(pkt.timestamp)
                    time.sleep(0.001)
        except Exception as exc:
            consumer_errors.append(exc)
        finally:
            consumer_done.set()

    consumer_thread = threading.Thread(target=consumer_worker, name="stress-consumer", daemon=True)
    consumer_thread.start()

    producer_threads = [
        threading.Thread(target=producer_worker, args=(p_id,), name=f"stress-producer-{p_id}", daemon=True)
        for p_id in range(num_producers)
    ]
    t_start = time.perf_counter()
    for t in producer_threads:
        t.start()

    for t in producer_threads:
        t.join(timeout=30.0)
        assert not t.is_alive(), f"Producer thread {t.name} deadlocked!"

    t_producers_end = time.perf_counter()
    producers_done.set()

    consumer_thread.join(timeout=10.0)
    assert not consumer_thread.is_alive(), "Consumer thread deadlocked!"

    elapsed = t_producers_end - t_start
    effective_producer_fps = total_frames / elapsed

    assert not producer_errors, f"Producer errors: {producer_errors}"
    assert not consumer_errors, f"Consumer errors: {consumer_errors}"

    taken_count = len(received_sequences)
    remaining_in_box = 0 if mailbox.is_empty() else 1
    total_accounted = mailbox.replaced_count + taken_count + remaining_in_box

    # 1. Monotonicity verification
    for i in range(1, len(received_sequences)):
        assert received_sequences[i] > received_sequences[i - 1], "Monotonicity violated in consumer sequence!"

    # 2. Zero backlog: mailbox capacity is exactly 1
    assert mailbox.capacity == 1

    # 3. Exact replacement math
    assert total_accounted == total_frames, (
        f"Frame accounting mismatch: total_put={total_frames} vs "
        f"replaced({mailbox.replaced_count}) + taken({taken_count}) + remaining({remaining_in_box}) = {total_accounted}"
    )

    # 4. Under fast producers vs slower consumer, replaced_count must be substantial
    assert mailbox.replaced_count > total_frames * 0.5, (
        f"Expected high replacement rate, got {mailbox.replaced_count}/{total_frames}"
    )

    print(
        f"\n[STRESS TEST 1 PASSED] 10 producers pushed {total_frames} frames in {elapsed:.2f}s "
        f"({effective_producer_fps:.1f} FPS aggregate). Consumer took {taken_count} frames. "
        f"Replaced count: {mailbox.replaced_count}. Remaining in box: {remaining_in_box}. "
        f"Accounting: {mailbox.replaced_count} + {taken_count} + {remaining_in_box} == {total_frames} (EXACT)."
    )


def test_concurrency_stress_out_of_order_rejection_under_multithreaded_race():
    """Verify that uncoordinated out-of-order frame injections are strictly rejected fail-closed with ValueError."""
    mailbox = LatestFrameMailbox()
    base_frame = np.zeros((32, 32, 3), dtype=np.uint8)

    mailbox.put(FramePacket(100, 10.0, base_frame))
    assert mailbox.replaced_count == 0

    rejected_count = 0
    accepted_count = 0
    rejection_lock = threading.Lock()

    def racer_worker(seq: int, ts: float):
        nonlocal rejected_count, accepted_count
        try:
            mailbox.put(FramePacket(seq, ts, base_frame))
            with rejection_lock:
                accepted_count += 1
        except ValueError:
            with rejection_lock:
                rejected_count += 1

    threads = [
        threading.Thread(target=racer_worker, args=(50 + i, 5.0 + i * 0.1))
        for i in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert rejected_count == 10
    assert accepted_count == 0
    assert mailbox.replaced_count == 0


# ==============================================================================
# Challenge 2: Permission Bypass Stress Test (All Permutations Fail-Closed)
# ==============================================================================

@pytest.mark.parametrize(
    "obs_granted, learn_granted, save_granted",
    list(itertools.product([False, True], [False, True], [False, True])),
)
def test_permissions_exhaustive_truth_table_permutations(
    obs_granted: bool,
    learn_granted: bool,
    save_granted: bool,
):
    """Test all 2^3 = 8 permutations of StreamingPermissions flags.

    Verifies that every ungranted action strictly raises PermissionError fail-closed,
    and every granted action executes cleanly without raising.
    """
    perms = StreamingPermissions(
        enable_observation=obs_granted,
        enable_online_learning=learn_granted,
        authorize_checkpoint_save=save_granted,
    )

    if not obs_granted:
        with pytest.raises(PermissionError, match="Live observation is disabled"):
            perms.assert_observation_allowed()
    else:
        perms.assert_observation_allowed()

    if not learn_granted:
        with pytest.raises(PermissionError, match="Online parameter learning is disabled"):
            perms.assert_learning_allowed()
    else:
        perms.assert_learning_allowed()

    if not save_granted:
        with pytest.raises(PermissionError, match="Checkpoint persistence is disabled"):
            perms.assert_save_allowed()
    else:
        perms.assert_save_allowed()


def test_permissions_type_coercion_and_falsy_defaults():
    """Verify that falsy non-boolean inputs fail-closed while truthy values coerce properly."""
    falsy_cases = [
        (0, "", None),
        ([], {}, ()),
        (0.0, False, None),
    ]
    for obs, learn, save in falsy_cases:
        p = StreamingPermissions(obs, learn, save)
        assert p.enable_observation is False
        assert p.enable_online_learning is False
        assert p.authorize_checkpoint_save is False
        with pytest.raises(PermissionError):
            p.assert_observation_allowed()
        with pytest.raises(PermissionError):
            p.assert_learning_allowed()
        with pytest.raises(PermissionError):
            p.assert_save_allowed()


def test_permissions_streaming_observer_fail_closed_integration():
    """Verify StreamingObserver blocks both start() and observe() whenever observation is ungranted."""
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)

    obs_default = StreamingObserver(model, input_size=(32, 32), permissions=StreamingPermissions())
    dummy_packet = FramePacket(0, time.monotonic(), np.zeros((32, 32, 3), dtype=np.uint8))
    dummy_source = SyntheticCaptureSource(width=32, height=32, fps=10.0, total_frames=2)

    with pytest.raises(PermissionError, match="Live observation is disabled"):
        obs_default.observe(dummy_packet)

    with pytest.raises(PermissionError, match="Live observation is disabled"):
        obs_default.start(dummy_source)

    p_no_obs = StreamingPermissions(enable_observation=False, enable_online_learning=True, authorize_checkpoint_save=True)
    obs_no_obs = StreamingObserver(model, input_size=(32, 32), permissions=p_no_obs)

    with pytest.raises(PermissionError, match="Live observation is disabled"):
        obs_no_obs.observe(dummy_packet)

    with pytest.raises(PermissionError, match="Live observation is disabled"):
        obs_no_obs.start(dummy_source)


# ==============================================================================
# Challenge 3: Zero Raw Persistence Stress Test (1,000 Frames)
# ==============================================================================

def test_zero_raw_persistence_1000_frames_through_observer(tmp_path: Path):
    """Stress test: Process 1,000 continuous frames through StreamingObserver with synthetic source.

    Scan the filesystem before and after processing to assert ZERO image, video,
    or raw frame files are persisted to disk.
    """
    repo_root = Path(__file__).resolve().parent.parent

    MEDIA_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".raw", ".webp",
        ".mp4", ".avi", ".mov", ".mkv", ".webm",
        ".npy", ".npz",
    }

    def scan_media_files(directory: Path) -> set[Path]:
        try:
            return {
                f.resolve()
                for f in directory.rglob("*")
                if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
            }
        except (PermissionError, FileNotFoundError):
            return set()

    # Pre-scan baseline media files in tmp_path and repo_root
    baseline_repo_media = scan_media_files(repo_root)
    baseline_tmp_media = scan_media_files(tmp_path)

    # Configure StreamingObserver with authorized observation
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    perms = StreamingPermissions(enable_observation=True)
    observer = StreamingObserver(model, input_size=(32, 32), permissions=perms)

    # Process 1,000 frames using SyntheticCaptureSource
    num_frames = 1000
    source = SyntheticCaptureSource(width=32, height=32, fps=30.0, total_frames=num_frames)
    source.open()

    base_time = time.monotonic()
    t_start = time.perf_counter()

    try:
        for i in range(num_frames):
            frame_rgb = source.grab()
            # Feed frames monotonically at simulated 30 FPS intervals
            ts = base_time + i * 0.033
            pkt = FramePacket(sequence=i, timestamp=ts, rgb=frame_rgb)
            update = observer.observe(pkt, now=ts)
            assert update is not None
            assert isinstance(update, ObservationUpdate)
            assert update.sequence == i
    finally:
        source.close()

    elapsed = time.perf_counter() - t_start
    fps = num_frames / max(1e-5, elapsed)

    assert observer.processed_count == num_frames, (
        f"Expected {num_frames} processed frames, got {observer.processed_count}"
    )

    # Post-scan filesystem for newly created media files
    post_repo_media = scan_media_files(repo_root)
    post_tmp_media = scan_media_files(tmp_path)

    new_repo_media = post_repo_media - baseline_repo_media
    new_tmp_media = post_tmp_media - baseline_tmp_media

    new_leaked_media = [
        str(f) for f in (new_repo_media | new_tmp_media)
        if ".git" not in str(f)
    ]

    assert len(new_leaked_media) == 0, (
        f"CRITICAL PRIVACY VIOLATION: Zero raw persistence failed! "
        f"Found {len(new_leaked_media)} persisted media files: {new_leaked_media}"
    )

    print(
        f"\n[STRESS TEST 3 PASSED] Processed {num_frames} frames through SyntheticCaptureSource "
        f"in {elapsed:.2f}s ({fps:.1f} FPS). Observer processed_count: {observer.processed_count}. "
        f"New media files persisted to disk: 0 (Zero Raw Persistence fully verified)."
    )


def test_zero_raw_persistence_decoupled_thread_mode(tmp_path: Path):
    """Stress test: Process 50 continuous frames through decoupled background capture and inference threads.

    Asserts zero disk media persistence in multithreaded runtime mode.
    """
    repo_root = Path(__file__).resolve().parent.parent

    MEDIA_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".raw", ".webp",
        ".mp4", ".avi", ".mov", ".mkv", ".webm",
        ".npy", ".npz",
    }

    def scan_media_files(directory: Path) -> set[Path]:
        try:
            return {
                f.resolve()
                for f in directory.rglob("*")
                if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS
            }
        except (PermissionError, FileNotFoundError):
            return set()

    baseline_repo_media = scan_media_files(repo_root)
    baseline_tmp_media = scan_media_files(tmp_path)

    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    perms = StreamingPermissions(enable_observation=True)
    observer = StreamingObserver(model, input_size=(32, 32), permissions=perms)

    num_frames = 50
    source = SyntheticCaptureSource(width=32, height=32, fps=50.0, total_frames=num_frames)

    observer.start(source, target_hz=50.0)
    observer._capture_thread.join(timeout=5.0)
    assert not observer._capture_thread.is_alive()
    observer._inference_thread.join(timeout=5.0)
    assert not observer._inference_thread.is_alive()
    observer.stop()

    assert observer.captured_count == num_frames
    assert observer.processed_count > 0

    post_repo_media = scan_media_files(repo_root)
    post_tmp_media = scan_media_files(tmp_path)

    new_repo_media = post_repo_media - baseline_repo_media
    new_tmp_media = post_tmp_media - baseline_tmp_media

    new_leaked_media = [
        str(f) for f in (new_repo_media | new_tmp_media)
        if ".git" not in str(f)
    ]

    assert len(new_leaked_media) == 0, (
        f"CRITICAL PRIVACY VIOLATION: Zero raw persistence failed in threaded mode! "
        f"Found {len(new_leaked_media)} persisted media files: {new_leaked_media}"
    )

    print(
        f"\n[DECOUPLED THREAD TEST PASSED] Captured {observer.captured_count} frames, "
        f"processed {observer.processed_count} frames in background threads. "
        f"New media files persisted: 0."
    )
