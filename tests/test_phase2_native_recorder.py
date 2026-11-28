"""Unit & integration tests for Phase 2 Zero-Lag Windows Native Recorder."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import sys
import time

import numpy as np
import pytest

from avcs.data.lerobot_v3 import QADoctor, load_lerobot_v3_shards
from avcs.data.native_recorder import (
    ACTION_DOUBLE_CLICK,
    ACTION_DRAG,
    ACTION_HOTKEY,
    ACTION_LEFT_CLICK,
    ACTION_RIGHT_CLICK,
    ACTION_SCROLL,
    ACTION_STOP,
    ACTION_TYPE,
    PreActionFrameRingBuffer,
    RawInputActionAssembler,
    RawInputEvent,
    RecorderConfig,
    ZeroLagWindowsRecorder,
    run_recorder_smoke_test,
    select_capture_backend,
)
from avcs.runtime.capture_adapter import DisplayGeometry, InteractiveSyntheticCaptureAdapter


def test_ring_buffer_binds_strictly_pre_action_frame_and_monotonic_ns():
    """Verifies PreActionFrameRingBuffer never binds post-action frames (<2ms margin) and stays monotonic."""
    ring = PreActionFrameRingBuffer(capacity=16, jpeg_quality=80)
    f1 = ring.push_frame(np.full((64, 64, 3), 30, dtype=np.uint8), present_ns=10_000_000)
    f2 = ring.push_frame(np.full((64, 64, 3), 90, dtype=np.uint8), present_ns=15_000_000)
    f3 = ring.push_frame(np.full((64, 64, 3), 180, dtype=np.uint8), present_ns=20_000_000)

    assert f1.present_ns < f2.present_ns < f3.present_ns

    # Action at 16.5ms must bind to f1 (10ms), NOT f2 (15ms -> 1.5ms margin < 2ms)
    bound_early = ring.bind_pre_action_frame(16_500_000, min_margin_ns=2_000_000)
    assert bound_early is not None and bound_early.seq == f1.seq

    # Action at 18.0ms cleanly binds to f2 (15ms -> 3.0ms margin >= 2ms)
    bound_exact = ring.bind_pre_action_frame(18_000_000, min_margin_ns=2_000_000)
    assert bound_exact is not None and bound_exact.seq == f2.seq

    # Action before any frame + 2ms returns None (fail-closed)
    assert ring.bind_pre_action_frame(11_000_000, min_margin_ns=2_000_000) is None


def test_raw_input_assembler_emits_canonical_actions_and_drag_trajectory():
    """Verifies clicks, double-clicks, drags, scrolls, hotkeys, and typing assemble cleanly."""
    ring = PreActionFrameRingBuffer(capacity=32)
    base = 100_000_000
    for i in range(6):
        ring.push_frame(np.full((120, 200, 3), 20 + i * 25, dtype=np.uint8), present_ns=base + i * 20_000_000)

    cfg = RecorderConfig(task_instruction="Test task", min_drag_px=12.0, min_move_px=20.0)
    geo = DisplayGeometry(0, 0, 1000, 500, 1.0, True)
    asm = RawInputActionAssembler(episode_id="ep_asm", geometry=geo, ring_buffer=ring, config=cfg)

    # Single click -> double click -> drag -> scroll -> typing -> hotkey -> stop
    events = [
        RawInputEvent(base + 15_000_000, "left_down", screen_xy=(100, 100)),
        RawInputEvent(base + 18_000_000, "left_up", screen_xy=(101, 100)),
        RawInputEvent(base + 30_000_000, "left_down", screen_xy=(101, 100)),
        RawInputEvent(base + 34_000_000, "left_up", screen_xy=(101, 100)),
        RawInputEvent(base + 50_000_000, "right_down", screen_xy=(250, 200)),
        RawInputEvent(base + 60_000_000, "left_down", screen_xy=(300, 250)),
        RawInputEvent(base + 65_000_000, "mouse_move", screen_xy=(400, 250)),
        RawInputEvent(base + 70_000_000, "left_up", screen_xy=(500, 250)),
        RawInputEvent(base + 82_000_000, "wheel", screen_xy=(500, 250), wheel_delta=(0.0, -1.0)),
        RawInputEvent(base + 92_000_000, "key_down", vk_code=ord("A"), char="a"),
        RawInputEvent(base + 96_000_000, "key_down", vk_code=ord("Z"), modifiers=("ctrl",)),
        RawInputEvent(base + 108_000_000, "stop", screen_xy=(500, 250)),
    ]
    for ev in events:
        asm.process_event(ev)

    kinds = [s.action_kind for s in asm.steps]
    assert kinds == [
        ACTION_LEFT_CLICK,
        ACTION_DOUBLE_CLICK,
        ACTION_RIGHT_CLICK,
        ACTION_DRAG,
        ACTION_SCROLL,
        ACTION_TYPE,
        ACTION_HOTKEY,
        ACTION_STOP,
    ]
    drag_step = asm.steps[3]
    assert len(drag_step.drag_trajectory_xy) == 3
    assert asm.steps[6].typed_text == "ctrl+z"
    qa = QADoctor.validate_episode(asm.steps)
    assert qa["valid"], f"QADoctor failed: {qa}"


def test_smoke_recorder_exports_clean_lerobot_v3_shards_and_manifest(tmp_path):
    """End-to-end deterministic ZeroLagWindowsRecorder run -> Cleaner -> QADoctor -> shard roundtrip."""
    out_dir = tmp_path / "phase2_smoke"
    res = run_recorder_smoke_test(output_dir=out_dir)

    assert res["capture_backend"] == "synthetic"
    assert res["qa_report"]["pass_rate"] == 1.0
    assert res["atomic_p95_sync_ms"] < 15.0
    assert res["capture_fps"] >= 30.0
    assert res["cleaned_step_count"] >= 4
    assert res["reloaded_episodes"] == 1
    assert res["reloaded_steps"] == res["cleaned_step_count"]
    assert len(res["audit_files"]) >= 1

    manifest = json.loads((out_dir / "meta" / "recording_manifest.json").read_text(encoding="utf-8"))
    assert manifest["zero_title_pure_rgb"] is True
    assert manifest["qa_report"]["pass_rate"] == 1.0

    reloaded = load_lerobot_v3_shards(out_dir)
    assert QADoctor.audit_dataset(reloaded)["pass_rate"] == 1.0


def test_native_recorder_linux_import_safety_and_backend_fallback(monkeypatch):
    """Ensures select_capture_backend falls back cleanly to synthetic on non-Windows without win32 calls."""
    monkeypatch.setattr(sys, "platform", "linux")
    name, adp = select_capture_backend("auto", width=320, height=200)
    assert name == "synthetic"
    frame = adp.grab()
    assert frame.shape == (200, 320, 3)
    adp.close()


def test_win_key_mapping_and_non_overwriting_shards(tmp_path):
    """Verifies Win-key/OEM mapping and auto-incrementing shard filenames without overwriting."""
    from avcs.data.native_recorder import _vk_to_label_and_char

    assert _vk_to_label_and_char(0x5B, ()) == (True, "win", "")
    assert _vk_to_label_and_char(0xBE, ()) == (False, "", ".")

    out_dir = tmp_path / "append_shards"
    r1 = run_recorder_smoke_test(output_dir=out_dir, episode_id="ep_first")
    r2 = run_recorder_smoke_test(output_dir=out_dir, episode_id="ep_second")
    reloaded = load_lerobot_v3_shards(out_dir)
    assert len(reloaded) == 2
    assert r1["shard_info"]["shards"][0] != r2["shard_info"]["shards"][0]

    # Verify true LeRobot v3.0 parquet+mp4 separation when cv2 & pyarrow are available
    try:
        import pyarrow.parquet as pq
        import cv2  # noqa: F401
    except Exception:
        return

    assert r1["shard_info"]["format"] == "parquet+mp4"
    assert len(r1["shard_info"]["videos"]) == 1
    assert Path(r1["shard_info"]["videos"][0]).is_file()
    tbl = pq.read_table(r1["shard_info"]["shards"][0])
    assert "image_b64" not in tbl.column_names
    assert "video_relpath" in tbl.column_names
    assert all(len(step.image_bytes) > 0 for step in reloaded[0])


