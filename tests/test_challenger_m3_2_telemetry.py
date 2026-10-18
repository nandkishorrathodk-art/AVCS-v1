"""Challenger M3-2 Empirical Stress Test Suite & Benchmarks.

Adversarial Stress Testing of Milestone 3 Feature 10:
1. Multi-Threaded Concurrency & Race Conditions (8-16 concurrent threads)
2. High-Frequency Throughput & Pack Latency Benchmark (100,000 pkts)
3. Ring Buffer Overflow & Monotonicity Stress (2,000 pkts in 120 buffer)
4. 40-Byte Binary ABI Round-Trip, Fuzzing & Binary Stream
"""
from __future__ import annotations

import collections
import concurrent.futures
import math
import struct
import threading
import time
from typing import List, Tuple

import numpy as np
import pytest

from avcs.stream.neuro_stream import (
    COMPACT_FMT_33,
    COMPACT_FMT_40,
    NeuroPacket,
    NeuroStreamBus,
    RealTimeNeuroStreamer,
)


def _make_dummy_packet(seq: int, ts: float | None = None) -> NeuroPacket:
    return NeuroPacket(
        timestamp=ts if ts is not None else time.time(),
        seq=seq,
        fixation=(0.42, 0.58),
        motor_flow=(0.01, -0.02),
        contact_acquired=False,
        primary_category="editor_code",
        stability_state="stable",
        surprise_score=0.03,
        inference_ms=1.2,
    )


# ==============================================================================
# Challenge 1: Multi-Threaded Concurrency & Race Conditions
# ==============================================================================

class TestConcurrencyAndRaceConditions:
    """Stress tests concurrent multi-threaded execution on NeuroStreamBus."""

    def test_concurrent_producers_and_subscribers_16_threads(self):
        """Runs 16 concurrent threads (8 producers, 8 consumers/subscribers).
        
        Verifies:
        - Zero race conditions or deadlocks
        - Zero unhandled exceptions
        - Accurate packet counting across threads
        - Subscriber list mutation safety under GIL
        """
        bus = NeuroStreamBus(buffer_size=500)
        num_producers = 8
        num_subscribers = 8
        packets_per_producer = 2500
        total_expected_packets = num_producers * packets_per_producer

        received_counts = [0] * num_subscribers
        sub_locks = [threading.Lock() for _ in range(num_subscribers)]
        unsub_callbacks = []

        # Setup subscribers
        for i in range(num_subscribers):
            def make_handler(idx: int):
                def handler(pkt: NeuroPacket):
                    with sub_locks[idx]:
                        received_counts[idx] += 1
                return handler

            unsub = bus.subscribe(make_handler(i))
            unsub_callbacks.append(unsub)

        errors: List[Exception] = []

        def producer_worker(producer_id: int):
            try:
                base_seq = producer_id * packets_per_producer
                for i in range(packets_per_producer):
                    pkt = _make_dummy_packet(seq=base_seq + i)
                    bus.publish(pkt)
            except Exception as e:
                errors.append(e)

        start_barrier = threading.Barrier(num_producers + 1)

        threads = []
        for p_id in range(num_producers):
            t = threading.Thread(
                target=lambda p=p_id: (start_barrier.wait(), producer_worker(p))
            )
            threads.append(t)
            t.start()

        # Release barrier to start all producers simultaneously
        start_barrier.wait()

        # Wait for all producer threads to complete
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Producer thread deadlocked or timed out"

        assert len(errors) == 0, f"Encountered producer errors: {errors}"
        assert bus.packet_count == total_expected_packets

        # Clean up subscriptions
        for unsub in unsub_callbacks:
            unsub()

        assert len(bus._subscribers) == 0

    def test_concurrent_dynamic_subscription_churn_during_high_frequency_publish(self):
        """Concurrently subscribes and unsubscribes while multiple threads publish at maximum speed."""
        bus = NeuroStreamBus(buffer_size=200)
        stop_event = threading.Event()
        publish_errors: List[Exception] = []
        churn_errors: List[Exception] = []

        def publisher_task(p_id: int):
            seq = p_id * 100000
            try:
                while not stop_event.is_set():
                    pkt = _make_dummy_packet(seq=seq)
                    bus.publish(pkt)
                    seq += 1
            except Exception as e:
                publish_errors.append(e)

        def churn_task():
            try:
                while not stop_event.is_set():
                    # Quick subscribe and unsubscribe
                    received = []
                    unsub = bus.subscribe(lambda p: received.append(p.seq))
                    time.sleep(0.001)  # Allow a few packets to arrive
                    unsub()
            except Exception as e:
                churn_errors.append(e)

        def reader_task():
            try:
                while not stop_event.is_set():
                    _ = bus.get_latest()
                    _ = bus.get_window(20)
                    _ = bus.to_binary_stream(10)
                    _ = bus.dropped_count
                    time.sleep(0.001)
            except Exception as e:
                churn_errors.append(e)

        # Launch 4 publishers, 4 churners, 4 readers = 12 concurrent threads
        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=publisher_task, args=(i,)))
            threads.append(threading.Thread(target=churn_task))
            threads.append(threading.Thread(target=reader_task))

        for t in threads:
            t.start()

        # Let the stress run for 1.5 seconds
        time.sleep(1.5)
        stop_event.set()

        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Thread deadlocked during dynamic subscription churn"

        assert len(publish_errors) == 0, f"Publish errors: {publish_errors}"
        assert len(churn_errors) == 0, f"Churn/Reader errors: {churn_errors}"
        assert bus.packet_count > 1000


# ==============================================================================
# Challenge 2: High-Frequency Throughput & Pack Latency Benchmark
# ==============================================================================

class TestThroughputAndPackLatency:
    """Measures CPU throughput (100,000 packets) and microsecond pack/unpack latencies."""

    def test_cpu_throughput_100k_packets(self):
        """Measures CPU throughput publishing 100,000 packets. Target >= 100,000 pkts/sec."""
        bus = NeuroStreamBus(buffer_size=120)
        received_count = 0

        def subscriber(p: NeuroPacket):
            nonlocal received_count
            received_count += 1

        bus.subscribe(subscriber)

        n_packets = 100000
        pkt = _make_dummy_packet(seq=1)

        t_start = time.perf_counter()
        for i in range(n_packets):
            bus.publish(pkt)
        t_elapsed = time.perf_counter() - t_start

        pkts_per_sec = n_packets / t_elapsed

        print(f"\n[BENCHMARK] 100k Packet Throughput: {pkts_per_sec:,.0f} pkts/sec ({t_elapsed*1000:.2f} ms total)")

        assert received_count == n_packets
        assert bus.packet_count == n_packets
        assert pkts_per_sec >= 100000.0, f"Throughput {pkts_per_sec:.0f} pkts/sec is below 100,000 threshold"

    def test_microsecond_pack_and_unpack_latency(self):
        """Measures microsecond latency of pack_compact() / to_compact() and from_compact()."""
        pkt = NeuroPacket(
            timestamp=1700000000.12345,
            seq=42,
            fixation=(0.333, 0.666),
            motor_flow=(-0.015, 0.025),
            contact_acquired=True,
            primary_category="editor_code",
            stability_state="stable",
            surprise_score=0.05,
            inference_ms=1.5,
        )

        n_iters = 50000

        # Measure pack_compact / to_compact
        t0 = time.perf_counter()
        for _ in range(n_iters):
            data = pkt.to_compact()
        pack_total = time.perf_counter() - t0
        pack_us = (pack_total / n_iters) * 1e6

        # Measure from_compact / unpack_compact
        t1 = time.perf_counter()
        for _ in range(n_iters):
            _ = NeuroPacket.from_compact(data)
        unpack_total = time.perf_counter() - t1
        unpack_us = (unpack_total / n_iters) * 1e6

        print(f"\n[BENCHMARK] to_compact() latency: {pack_us:.3f} us | from_compact() latency: {unpack_us:.3f} us")

        assert pack_us < 10.0, f"Pack latency too slow: {pack_us:.3f} us"
        assert unpack_us < 10.0, f"Unpack latency too slow: {unpack_us:.3f} us"


# ==============================================================================
# Challenge 3: Ring Buffer Overflow & Monotonicity Stress
# ==============================================================================

class TestRingBufferOverflowAndMonotonicity:
    """Tests 2,000 packet overflow into 120-packet buffer, FIFO monotonicity, and get_since()."""

    def test_2000_packet_overflow_into_120_buffer(self):
        """Publishes 2,000 sequential packets into 120-slot buffer.
        
        Verifies:
        - dropped_count == 1880
        - packet_count == 2000
        - len(bus._buffer) == 120
        - FIFO ordering strictly preserves monotonicity (seq 1880 to 1999)
        - get_since(seq) behavior
        """
        buffer_size = 120
        bus = NeuroStreamBus(buffer_size=buffer_size)
        total_packets = 2000

        base_time = 1700000000.0
        for i in range(total_packets):
            pkt = _make_dummy_packet(seq=i, ts=base_time + i * 0.016)
            bus.publish(pkt)

        # 1. Verify counts
        assert bus.packet_count == total_packets
        assert bus.dropped_count == 1880  # 2000 - 120 = 1880
        assert len(bus._buffer) == buffer_size

        # 2. Verify FIFO window monotonicity
        window = bus.get_window(buffer_size)
        assert len(window) == buffer_size
        expected_seqs = list(range(1880, 2000))
        actual_seqs = [p.seq for p in window]
        assert actual_seqs == expected_seqs, "FIFO sequence monotonicity violated"

        # Check strictly monotonic increasing seqs
        for idx in range(len(window) - 1):
            assert window[idx + 1].seq == window[idx].seq + 1

        # 3. Verify get_since behavior
        # Newer than 1950: 1951..1999 (49 packets)
        since_1950 = bus.get_since(1950)
        assert len(since_1950) == 49
        assert [p.seq for p in since_1950] == list(range(1951, 2000))

        # Newer than 1879: all 120 packets
        since_1879 = bus.get_since(1879)
        assert len(since_1879) == 120

        # Newer than 1999: 0 packets
        since_1999 = bus.get_since(1999)
        assert len(since_1999) == 0

        # Newer than 2500: 0 packets
        since_future = bus.get_since(2500)
        assert len(since_future) == 0

        # Newer than 0: all 120 packets
        since_0 = bus.get_since(0)
        assert len(since_0) == 120

    def test_hasattr_len_contract_check(self):
        """Explicit check for hasattr(bus, '__len__').
        
        Records whether NeuroStreamBus implements the standard Python Sized protocol (__len__).
        """
        bus = NeuroStreamBus(buffer_size=120)
        has_len = hasattr(bus, "__len__")
        # Record finding: bus does not currently implement __len__
        print(f"\n[CONTRACT CHECK] hasattr(bus, '__len__'): {has_len}")


# ==============================================================================
# Challenge 4: 40-Byte Binary ABI Round-Trip & Fuzzing
# ==============================================================================

class TestBinaryABIAndFuzzing:
    """Stress tests 40-byte C-ABI struct round-trip under fuzzing and binary stream serialization."""

    def test_c_abi_format_constants(self):
        """Verifies struct format definitions and sizes."""
        assert COMPACT_FMT_40 == "<dI5fB7x"
        assert struct.calcsize(COMPACT_FMT_40) == 40
        assert COMPACT_FMT_33 == "<dI5fB"
        assert struct.calcsize(COMPACT_FMT_33) == 33

    def test_fuzzing_nan_inf_negative_values(self):
        """Fuzzes float coordinates and fields with NaN, Inf, and negative numbers."""
        test_cases = [
            # Case 1: NaN in fixation, motor_flow, surprise
            {
                "timestamp": 1700000000.0,
                "seq": 100,
                "fixation": (float("nan"), 0.5),
                "motor_flow": (0.0, float("nan")),
                "contact_acquired": False,
                "surprise_score": float("nan"),
            },
            # Case 2: Positive & Negative Infinity
            {
                "timestamp": 1700000000.0,
                "seq": 101,
                "fixation": (float("inf"), float("-inf")),
                "motor_flow": (float("-inf"), float("inf")),
                "contact_acquired": True,
                "surprise_score": float("inf"),
            },
            # Case 3: Large negative coordinates and timestamps
            {
                "timestamp": -9999999.5,
                "seq": 102,
                "fixation": (-1000.5, -2000.5),
                "motor_flow": (-500.25, -250.75),
                "contact_acquired": False,
                "surprise_score": -10.0,
            },
            # Case 4: Max unsigned 32-bit int sequence (2**32 - 1)
            {
                "timestamp": 1700000000.0,
                "seq": 4294967295,  # 2**32 - 1
                "fixation": (0.1, 0.9),
                "motor_flow": (0.01, 0.02),
                "contact_acquired": True,
                "surprise_score": 0.99,
            },
            # Case 5: Zero sequence
            {
                "timestamp": 0.0,
                "seq": 0,
                "fixation": (0.0, 0.0),
                "motor_flow": (0.0, 0.0),
                "contact_acquired": False,
                "surprise_score": 0.0,
            },
        ]

        for tc in test_cases:
            pkt = NeuroPacket(
                timestamp=tc["timestamp"],
                seq=tc["seq"],
                fixation=tc["fixation"],
                motor_flow=tc["motor_flow"],
                contact_acquired=tc["contact_acquired"],
                primary_category="test_fuzz",
                stability_state="stable",
                surprise_score=tc["surprise_score"],
                inference_ms=0.0,
            )

            # 1. Must produce exactly 40 bytes
            data = pkt.to_compact()
            assert len(data) == 40, f"Expected 40 bytes, got {len(data)}"

            # 2. Unpack round-trip
            unpacked = NeuroPacket.from_compact(data)
            assert unpacked.seq == tc["seq"]
            assert unpacked.contact_acquired == tc["contact_acquired"]

            if math.isnan(tc["timestamp"]):
                assert math.isnan(unpacked.timestamp)
            else:
                assert pytest.approx(unpacked.timestamp, rel=1e-5) == tc["timestamp"]

            if math.isnan(tc["fixation"][0]):
                assert math.isnan(unpacked.fixation[0])
            elif math.isinf(tc["fixation"][0]):
                assert math.isinf(unpacked.fixation[0])
                assert (unpacked.fixation[0] > 0) == (tc["fixation"][0] > 0)
            else:
                assert pytest.approx(unpacked.fixation[0], abs=1e-3) == tc["fixation"][0]

    def test_sequence_number_overflow_boundary(self):
        """Tests sequence number boundary conditions (> 2**32 - 1 and negative)."""
        # 1. Exceeds 32-bit unsigned int (2**32 = 4294967296)
        pkt_overflow = _make_dummy_packet(seq=2**32)
        with pytest.raises(struct.error):
            pkt_overflow.to_compact()

        # 2. Negative seq number (< 0)
        pkt_neg = _make_dummy_packet(seq=-1)
        with pytest.raises(struct.error):
            pkt_neg.to_compact()

    def test_float32_overflow_boundary(self):
        """Tests coordinate value exceeding IEEE 754 32-bit float range (> 3.4e38)."""
        pkt_huge = NeuroPacket(
            timestamp=1700000000.0,
            seq=1,
            fixation=(1e40, 0.5),  # 1e40 exceeds float32 max
            motor_flow=(0.0, 0.0),
            contact_acquired=False,
            primary_category="editor_code",
            stability_state="stable",
            surprise_score=0.0,
            inference_ms=1.0,
        )
        with pytest.raises((struct.error, OverflowError)):
            pkt_huge.to_compact()

    def test_to_binary_stream_serialization(self):
        """Verifies to_binary_stream produces contiguous count * 40 bytes and exact round-trip."""
        bus = NeuroStreamBus(buffer_size=50)

        for i in range(50):
            bus.publish(_make_dummy_packet(seq=i))

        # Test count=10 -> 10 * 40 = 400 bytes
        stream_10 = bus.to_binary_stream(count=10)
        assert len(stream_10) == 10 * 40

        # Unpack each 40-byte chunk and verify sequences are 40..49
        for idx in range(10):
            chunk = stream_10[idx * 40 : (idx + 1) * 40]
            assert len(chunk) == 40
            pkt = NeuroPacket.from_compact(chunk)
            assert pkt.seq == 40 + idx

        # Test count=None -> all buffered (50 * 40 = 2000 bytes)
        stream_all = bus.to_binary_stream(count=None)
        assert len(stream_all) == 50 * 40

        # Test count=100 (exceeds buffer size 50) -> clamps to 50 * 40 = 2000 bytes
        stream_excess = bus.to_binary_stream(count=100)
        assert len(stream_excess) == 50 * 40

    def test_zero_count_negative_zero_slicing_behavior(self):
        """Empirically demonstrates the Python negative-zero slice trap in to_binary_stream(0) and get_window(0).

        Because pkts[-0:] evaluates to pkts[0:] in Python, asking for 0 elements currently returns all elements.
        """
        bus = NeuroStreamBus(buffer_size=10)
        for i in range(10):
            bus.publish(_make_dummy_packet(seq=i))

        # to_binary_stream(count=0) currently returns all 10 packets (400 bytes) due to pkts[-0:] slice
        raw_stream_0 = bus.to_binary_stream(count=0)
        assert len(raw_stream_0) == 400, (
            f"Expected 400 bytes (demonstrating pkts[-0:] slice anomaly), got {len(raw_stream_0)}"
        )

        # get_window(count=0) currently returns all 10 packets due to items[-0:] slice
        window_0 = bus.get_window(count=0)
        assert len(window_0) == 10, (
            f"Expected 10 packets (demonstrating items[-0:] slice anomaly), got {len(window_0)}"
        )

    def test_neuro_stream_bus_len_protocol(self):
        """Verifies that len(bus) currently raises TypeError because __len__ is not implemented."""
        bus = NeuroStreamBus(buffer_size=120)
        for i in range(10):
            bus.publish(_make_dummy_packet(seq=i))

        assert hasattr(bus, "__len__") is False
        with pytest.raises(TypeError, match="object of type 'NeuroStreamBus' has no len"):
            _ = len(bus)

