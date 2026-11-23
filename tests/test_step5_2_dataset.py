"""Unit tests for Step 5.2 Multi-Source Dataset Preparation Engine."""
import json
from pathlib import Path
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
    out_dir = tmp_path / "vla_test_output"
    engine = VLADataEngine(
        output_dir=str(out_dir),
        held_out_csv=r"C:\Users\nandk\AVCS-v1\data\showdown-clicks-dev\data.csv",
        max_pixels=MAX_PIXELS_CEILING,
        seed=123,
    )

    # Ingest synthetic workflows
    ingested = engine.ingest_synthetic_desktop_workflows(count=50)
    assert ingested == 50

    # Ingest Windows 11 manifest
    ingested_win = engine.ingest_windows11_manifest(
        manifest_path=r"C:\Users\nandk\AVCS-v1\data\manifest.json",
        screens_dir=r"C:\Users\nandk\AVCS-v1\data\screens",
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
