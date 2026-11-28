"""Unit & integration tests for Phase 3 Gate 3 trajectory overfit verification."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from avcs.data.lerobot_v3 import LeRobotV3TrajectoryEpisode, save_lerobot_v3_shards
from avcs.data.native_recorder import run_recorder_smoke_test
from avcs.train.overfit_check import audit_recorded_coverage, run_phase3_overfit_check


def _make_distinct_ui_jpeg(w: int, h: int, cx_norm: float, cy_norm: float, color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (w, h), color=(24, 28, 36))
    draw = ImageDraw.Draw(img)
    px = int(round(cx_norm * w))
    py = int(round(cy_norm * h))
    draw.rectangle((max(0, px - 48), max(0, py - 32), min(w, px + 48), min(h, py + 32)), fill=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_phase3_overfit_check_converges_on_lerobot_v3_mp4_episode(tmp_path):
    """Verifies >=98% bbox and >=99% action-type overfit on a recorded LeRobot v3 parquet+mp4 episode."""
    smoke_dir = tmp_path / "smoke_ep"
    run_recorder_smoke_test(output_dir=smoke_dir, episode_id="ep_gate3_smoke")
    cov_smoke = audit_recorded_coverage([smoke_dir])
    assert cov_smoke["episode_count"] == 1
    assert cov_smoke["qa_passed_episodes"] == 1

    # Golden visual trajectory with distinct visual states per action step saved as parquet+mp4
    t0 = 100_000_000
    specs = [
        (0, (0.20, 0.25), (230, 60, 60), "Click File menu"),
        (1, (0.80, 0.25), (60, 210, 80), "Right click workspace"),
        (4, (0.20, 0.75), (60, 120, 240), "Drag slider right"),
        (5, (0.80, 0.75), (240, 200, 50), "Scroll down panel"),
    ]
    steps = [
        LeRobotV3TrajectoryEpisode(
            episode_id="ep_golden_distinct",
            step_index=idx,
            video_frame_index=idx,
            frame_present_ns=t0 + idx * 50_000_000,
            action_event_ns=t0 + idx * 50_000_000 + 4_000_000,
            screen_width=640,
            screen_height=400,
            dpi_scale=1.0,
            task_instruction=instr,
            action_kind=act,
            target_xy=xy,
            pixel_diff_ratio=0.05,
            image_bytes=_make_distinct_ui_jpeg(640, 400, xy[0], xy[1], color),
        )
        for idx, (act, xy, color, instr) in enumerate(specs)
    ]
    rec_dir = tmp_path / "golden_ep"
    save_lerobot_v3_shards([steps], rec_dir)

    out_json = tmp_path / "gate3_report.json"
    res = run_phase3_overfit_check(
        [rec_dir],
        dry_run=True,
        steps=140,
        lr=3e-3,
        max_samples=16,
        canvas_hw=(400, 640),
        output_path=str(out_json),
    )
    assert out_json.is_file()
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["gate3_passed"] is True
    assert saved["action_type_acc_pct"] >= 99.0
    assert max(saved["sharp_bbox_acc_pct"], saved["subpixel_bbox_acc_pct"]) >= 98.0
    assert saved["final_loss"] < saved["initial_loss"]

