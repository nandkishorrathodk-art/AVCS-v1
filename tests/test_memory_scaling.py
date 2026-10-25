"""Pytest test suite for memory scaling across continuous streaming ticks.

Validates that avcs/eye_stream.py and avcs/cortex/model.py exhibit strictly constant
O(1) memory scaling without leaks across continuous streaming ticks.
"""
from __future__ import annotations

import gc
import os
import psutil
import pytest
import torch

from avcs.cortex.model import StreamingVisualCortexV2
from avcs.eye_stream import EyeStream


def test_streaming_visual_cortex_v2_200_frames_constant_memory():
    """Execute 200 ticks of StreamingVisualCortexV2 and verify RSS delta < 5.0 MB."""
    process = psutil.Process(os.getpid())
    gc.collect()

    model = StreamingVisualCortexV2().eval()
    state = None

    # Warmup 20 frames
    with torch.inference_mode():
        for i in range(20):
            x = torch.rand(1, 4, 128, 128)
            out = model(x, prev_state=state, timestamp=float(i) * 0.033)
            state = out.temporal_state

    gc.collect()
    rss_warmup = process.memory_info().rss / (1024 * 1024)

    with torch.inference_mode():
        for tick in range(200):
            x = torch.rand(1, 4, 128, 128)
            out = model(x, prev_state=state, timestamp=float(20 + tick) * 0.033)
            state = out.temporal_state

    gc.collect()
    rss_end = process.memory_info().rss / (1024 * 1024)
    rss_delta = rss_end - rss_warmup

    assert rss_delta < 5.0, f"StreamingVisualCortexV2 memory leak: delta {rss_delta:.2f} MB >= 5.0 MB"


def test_eye_stream_200_frames_constant_memory():
    """Execute 200 ticks of EyeStream and verify RSS delta < 5.0 MB."""
    process = psutil.Process(os.getpid())
    gc.collect()

    stream = EyeStream(source="synthetic", load_weights=False)

    # Warmup 20 frames
    for _ in range(20):
        stream.step()

    gc.collect()
    rss_warmup = process.memory_info().rss / (1024 * 1024)

    for _ in range(200):
        obs = stream.step()
        assert obs is not None

    stream.close()
    gc.collect()
    rss_end = process.memory_info().rss / (1024 * 1024)
    rss_delta = rss_end - rss_warmup

    assert rss_delta < 5.0, f"EyeStream memory leak: delta {rss_delta:.2f} MB >= 5.0 MB"

