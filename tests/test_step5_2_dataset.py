"""Unit tests for Step 5.2 Multi-Source Dataset Preparation Engine."""
import json
import pytest
from PIL import Image

from avcs.data.prepare_vla_dataset import (
    ContaminationFirewall,
    ContaminationError,
    resize_and_bound_image,
    format_chatml_instance,
    VLADataEngine,
    MAX_PIXELS_CEILING,
)


def test_contamination_firewall_blocks_showdown_dev(tmp_path):
    # Create fake held-out CSV
    fake_csv = tmp_path / "fake_held_out.csv"
    fake_csv.write_text(
        "id,image,instruction,x1,y1,x2,y2,width,height\n"
        "leak_id_001,frames/sample_01.webp,click here,10,20,30,40,1280,800\n",
        encoding="utf-8",
    )
    firewall = ContaminationFirewall(str(fake_csv))

    # 1. Clean sample must pass
    firewall.verify_sample("clean_sample_999", "images/clean.jpg")

    # 2. Blocked ID must raise ContaminationError
    with pytest.raises(ContaminationError, match="leak_id_001"):
        firewall.verify_sample("leak_id_001", "images/other.jpg")

    # 3. Blocked image path must raise ContaminationError
    with pytest.raises(ContaminationError, match="sample_01.webp"):
        firewall.verify_sample("different_id", "frames/sample_01.webp")


def test_resize_and_bound_image_downscales_oversized():
    # Large 1920x1080 display (2,073,600 pixels > 1,024,000 ceiling)
    img = Image.new("RGB", (1920, 1080), color=(128, 128, 128))
    resized, sx, sy = resize_and_bound_image(img, max_pixels=MAX_PIXELS_CEILING)

    nw, nh = resized.size
    assert nw * nh <= MAX_PIXELS_CEILING
    assert nw % 28 == 0
    assert nh % 28 == 0
    assert sx < 1.0 and sy < 1.0


def test_resize_and_bound_image_leaves_undersized_untouched():
    # Canonical 1280x800 display (1,024,000 pixels == ceiling)
    img = Image.new("RGB", (1280, 800), color=(128, 128, 128))
    resized, sx, sy = resize_and_bound_image(img, max_pixels=MAX_PIXELS_CEILING)

    assert resized.size == (1280, 800)
    assert sx == 1.0 and sy == 1.0


def test_format_chatml_instance_schema():
    instance = format_chatml_instance(
        sample_id="test_001",
        image_rel_path="images/screen.jpg",
        instruction="Click the Submit button",
        action_dict={"action": "left_click", "coordinate": [640, 400]},
        width=1280,
        height=800,
    )
    assert instance["id"] == "test_001"
    assert instance["image"] == "images/screen.jpg"
    convs = instance["conversations"]
    assert len(convs) == 2
    assert convs[0]["role"] == "user"
    assert "<image>" in convs[0]["content"]
    assert "Instruction: Click the Submit button" in convs[0]["content"]
    assert "Screen Resolution: 1280x800" in convs[0]["content"]
    assert convs[1]["role"] == "assistant"
    # Assistant response is valid compact JSON
    parsed = json.loads(convs[1]["content"])
    assert parsed["action"] == "left_click"
    assert parsed["coordinate"] == [640, 400]


def test_vla_data_engine_end_to_end(tmp_path):
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "vla_test_output"
    engine = VLADataEngine(
        output_dir=str(out_dir),
        held_out_csv=str(repo_root / "data" / "showdown-clicks-dev" / "data.csv"),
        max_pixels=MAX_PIXELS_CEILING,
        seed=123,
    )

    # Ingest synthetic workflows
    ingested = engine.ingest_synthetic_desktop_workflows(count=50)
    assert ingested == 50

    # Ingest Windows 11 manifest
    ingested_win = engine.ingest_windows11_manifest(
        manifest_path=str(repo_root / "data" / "manifest.json"),
        screens_dir=str(repo_root / "data" / "screens"),
    )
    assert ingested_win > 0

    # Export splits (90/10)
    train_n, val_n = engine.export_splits(train_ratio=0.90)
    assert train_n + val_n == 50 + ingested_win
    assert train_n > val_n

    # Check files exist on disk
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"
    manifest_file = out_dir / "dataset_manifest.json"

    assert train_file.exists()
    assert val_file.exists()
    assert manifest_file.exists()

    # Verify line counts and json formatting
    with open(train_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
        assert len(lines) == train_n
        # Verify first entry
        assert "id" in lines[0]
        assert "image" in lines[0]
        assert "conversations" in lines[0]

    with open(manifest_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta["contamination_firewall"]["status"] == "PASSED"
        assert meta["contamination_firewall"]["leakage_count"] == 0
        assert meta["total_samples"] == train_n + val_n


def test_phase2_scroll_coalesce_recovery_pii_redact_and_segmentation(tmp_path):
    """Verifies new Phase 2 cleaner & shard functions without breaking existing Phase 2 contracts."""
    import io
    from avcs.data.lerobot_v3 import (
        LeRobotV3TrajectoryEpisode,
        QADoctor,
        TrajectoryCleaner,
        load_lerobot_v3_shards,
        run_phase2_gate2,
        save_lerobot_v3_shards,
    )

    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    raw_bytes = buf.getvalue()

    t0 = 1_000_000_000
    steps = [
        # Two rapid scroll bursts within 150ms -> coalesced into single (0, -240)
        LeRobotV3TrajectoryEpisode(
            episode_id="ep_s", step_index=0, video_frame_index=0,
            frame_present_ns=t0, action_event_ns=t0 + 3_000_000,
            screen_width=1920, screen_height=1080,
            action_kind=5, scroll_delta=(0.0, -120.0), pixel_diff_ratio=0.02, image_bytes=raw_bytes,
        ),
        LeRobotV3TrajectoryEpisode(
            episode_id="ep_s", step_index=1, video_frame_index=1,
            frame_present_ns=t0 + 100_000_000, action_event_ns=t0 + 150_000_000,
            screen_width=1920, screen_height=1080,
            action_kind=5, scroll_delta=(0.0, -120.0), pixel_diff_ratio=0.03,
        ),
        # Dead click (pixel_diff_ratio == 0.0)
        LeRobotV3TrajectoryEpisode(
            episode_id="ep_s", step_index=2, video_frame_index=2,
            frame_present_ns=t0 + 500_000_000, action_event_ns=t0 + 504_000_000,
            screen_width=1920, screen_height=1080,
            action_kind=0, pixel_diff_ratio=0.0,
        ),
    ]

    scrolled = TrajectoryCleaner.coalesce_scrolling(steps)
    assert len(scrolled) == 2
    assert scrolled[0].scroll_delta == (0.0, -240.0)

    non_dead, dropped = TrajectoryCleaner.quarantine_dead_clicks(scrolled)
    assert dropped == 1
    assert len(non_dead) == 1

    # PII redaction blacks out normalized box [0.0, 0.0, 0.5, 1.0]
    TrajectoryCleaner.redact_pii_regions(non_dead, redact_boxes_norm=[(0.0, 0.0, 0.5, 1.0)])
    with Image.open(io.BytesIO(non_dead[0].image_bytes)) as redacted:
        assert redacted.getpixel((20, 50))[0] < 20
        assert redacted.getpixel((170, 50))[0] > 230

    # Recovery retention tags self-corrections instead of dropping all
    rec_steps = [
        LeRobotV3TrajectoryEpisode(
            episode_id="ep_r", step_index=0, video_frame_index=0,
            frame_present_ns=t0, action_event_ns=t0 + 3_000_000,
            screen_width=1920, screen_height=1080, action_kind=0, pixel_diff_ratio=0.02,
        ),
        LeRobotV3TrajectoryEpisode(
            episode_id="ep_r", step_index=1, video_frame_index=1,
            frame_present_ns=t0 + 200_000_000, action_event_ns=t0 + 205_000_000,
            screen_width=1920, screen_height=1080, action_kind=6, typed_text="ctrl+z", pixel_diff_ratio=0.02,
        ),
        LeRobotV3TrajectoryEpisode(
            episode_id="ep_r", step_index=2, video_frame_index=2,
            frame_present_ns=t0 + 500_000_000, action_event_ns=t0 + 504_000_000,
            screen_width=1920, screen_height=1080, action_kind=9, pixel_diff_ratio=0.01,
        ),
    ]
    kept, q_cnt = TrajectoryCleaner.quarantine_misclicks(rec_steps, retain_recovery_fraction=1.0)
    assert q_cnt == 0
    assert kept[0].is_recovery_step and kept[1].is_recovery_step

    segs = TrajectoryCleaner.segment_episodes(kept + non_dead)
    assert len(segs) == 2

    # Frame count check & shard roundtrip & visual audit
    vf_ok = QADoctor.verify_video_frame_counts(kept, actual_video_frame_count=3)
    assert vf_ok["valid"] is True
    audit_files = QADoctor.render_visual_replay_audit(non_dead, tmp_path / "audit", max_frames=5)
    assert len(audit_files) == 1 and audit_files[0].exists()

    shard_info = save_lerobot_v3_shards([kept, non_dead], tmp_path / "shards", episodes_per_file=1)
    assert shard_info["episode_count"] == 2 and len(shard_info["shards"]) == 2
    reloaded = load_lerobot_v3_shards(tmp_path / "shards")
    assert len(reloaded) == 2
    assert reloaded[0][0].is_recovery_step is True

    # Small Gate 2 runner smoke check (fast 10-episode check)
    g2 = run_phase2_gate2(
        output_path=str(tmp_path / "g2.json"),
        audit_dir=str(tmp_path / "g2_audit"),
        num_episodes=10,
        steps_per_episode=5,
    )
    assert g2["gate2_passed"] is True

