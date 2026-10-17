"""Unit and integration tests for UI Grounding dataset, generator, loader, and benchmark runner."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import pytest
import torch

from avcs.dataset.loader import (
    GroundingSample,
    UIGroundingDataset,
    make_ui_grounding_dataloader,
    ui_grounding_collate_fn,
)
from avcs.dataset.synthetic.composer import DesktopComposer, UI_CATEGORIES
from avcs.dataset.synthetic.generator import SyntheticDesktopGenerator
from avcs.dataset.synthetic.primitives import get_font
from avcs.dataset.synthetic.query_synth import NEGATIVE_QUERY_BANK, synthesize_queries
from avcs.dataset.synthetic.themes import AVAILABLE_THEMES, get_theme
from avcs.query_limits import MAX_QUERY_BYTES
from benchmarks.run_ui_grounding_benchmark import (
    GroundingBenchmarkResults,
    UIGroundingBenchmarkRunner,
)

DATASET_ROOT = Path("datasets/ui_grounding")
MANIFEST_PATH = DATASET_ROOT / "manifest.json"


@pytest.fixture(scope="module")
def manifest_data() -> Dict[str, Any]:
    """Ensures canonical dataset exists and returns parsed manifest."""
    if not MANIFEST_PATH.exists():
        gen = SyntheticDesktopGenerator()
        gen.generate_canonical_dataset(output_dir=DATASET_ROOT)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_generator_determinism():
    """Generating frames with the same seed must produce identical pixels and coordinates."""
    gen = SyntheticDesktopGenerator(screen_size=(256, 256))
    f1 = gen.generate_frame(category="browser_webpage", seed=1234, split="train", frame_idx=1)
    f2 = gen.generate_frame(category="browser_webpage", seed=1234, split="train", frame_idx=1)

    assert np.array_equal(f1.image, f2.image), "Images with identical seeds must be bit-identical"
    assert np.array_equal(f1.affordance_mask, f2.affordance_mask), "Affordance masks must be bit-identical"
    assert len(f1.elements) == len(f2.elements), "Element counts must match"
    for e1, e2 in zip(f1.elements, f2.elements):
        assert e1.element_id == e2.element_id
        assert e1.bbox_norm == e2.bbox_norm
        assert e1.centroid_norm == e2.centroid_norm
        assert e1.target_query == e2.target_query


def test_manifest_schema_and_integrity(manifest_data: Dict[str, Any]):
    """Manifest must strictly adhere to Draft-07 schema and total 270 frames."""
    assert manifest_data.get("$schema") == "http://json-schema.org/draft-07/schema#"
    assert manifest_data.get("title") == "AVCS_UI_Grounding_Dataset_Manifest"
    assert manifest_data.get("total_frames") == 270

    splits = manifest_data.get("splits", {})
    assert splits.get("train") == 160
    assert splits.get("val") == 40
    assert splits.get("test_held_out") == 40
    assert splits.get("test_negative") == 30

    classes = manifest_data.get("classes", {})
    assert classes.get("macro_affordances") == ["static", "clickable", "typeable"]
    assert "button" in classes.get("foveal_element_kinds", [])
    assert "browser_webpage" in classes.get("ui_categories", [])

    frames = manifest_data.get("frames", [])
    assert len(frames) == 270

    # Spot-check asset existence on disk and verify unique query labels per frame
    for fr in frames:
        queries = [p["target_query"] for p in fr.get("interactive_points", [])]
        assert len(queries) == len(set(queries)), f"Duplicate queries detected in frame {fr['frame_id']}: {queries}"
        img_p = DATASET_ROOT / fr["file_path"]
        mask_p = DATASET_ROOT / fr["affordance_mask_path"]
        assert img_p.exists(), f"Missing image file: {img_p}"
        assert mask_p.exists(), f"Missing mask file: {mask_p}"


def test_session_split_isolation_no_leakage(manifest_data: Dict[str, Any]):
    """Assert zero SHA-256 image hash collisions between train, val, test_held_out, and test_negative."""
    import hashlib

    frames = manifest_data.get("frames", [])
    split_hashes: Dict[str, set[str]] = {
        "train": set(),
        "val": set(),
        "test_held_out": set(),
        "test_negative": set(),
    }

    for fr in frames:
        split = fr.get("split")
        img_p = DATASET_ROOT / fr["file_path"]
        assert img_p.exists(), f"Missing image file: {img_p}"
        digest = hashlib.sha256(img_p.read_bytes()).hexdigest()
        assert split in split_hashes, f"Unknown split: {split}"
        split_hashes[split].add(digest)

    train_h = split_hashes["train"]
    val_h = split_hashes["val"]
    test_h = split_hashes["test_held_out"]
    neg_h = split_hashes["test_negative"]

    # Authentic visual data leakage assertions: zero SHA-256 image collisions across splits
    train_test_overlap = train_h & test_h
    assert train_h.isdisjoint(test_h), (
        f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} image hash collisions between train and test_held_out! "
        f"Colliding hashes: {sorted(list(train_test_overlap))[:5]}"
    )
    val_test_overlap = val_h & test_h
    assert val_h.isdisjoint(test_h), (
        f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} image hash collisions between val and test_held_out!"
    )
    train_val_overlap = train_h & val_h
    assert train_h.isdisjoint(val_h), (
        f"DATA OVERLAP DETECTED: {len(train_val_overlap)} image hash collisions between train and val!"
    )
    assert train_h.isdisjoint(neg_h), "Train and Negative images must not overlap"
    assert test_h.isdisjoint(neg_h), "Test held-out and Negative images must not overlap"

    # Session metadata checks (must also remain disjoint)
    train_sessions = {fr["session_id"] for fr in frames if fr["split"] == "train"}
    val_sessions = {fr["session_id"] for fr in frames if fr["split"] == "val"}
    test_sessions = {fr["session_id"] for fr in frames if fr["split"] == "test_held_out"}
    neg_sessions = {fr["session_id"] for fr in frames if fr["split"] == "test_negative"}

    assert len(train_sessions) >= 5, "Train set must have at least 5 distinct sessions"
    assert len(val_sessions) >= 2, "Val set must have distinct sessions"
    assert len(test_sessions) >= 2, "Test held-out set must have distinct sessions"

    assert train_sessions.isdisjoint(val_sessions), "Train and Val sessions must not overlap"
    assert train_sessions.isdisjoint(test_sessions), "Train and Test sessions must not overlap"
    assert val_sessions.isdisjoint(test_sessions), "Val and Test sessions must not overlap"
    assert train_sessions.isdisjoint(neg_sessions), "Train and Negative sessions must not overlap"


def test_negative_frames_zero_interactive_elements(manifest_data: Dict[str, Any]):
    """Negative test frames must contain exactly zero interactive elements and all-zero masks."""
    frames = manifest_data.get("frames", [])
    neg_frames = [fr for fr in frames if fr["split"] == "test_negative"]
    assert len(neg_frames) == 30

    for fr in neg_frames:
        assert fr["is_negative"] is True
        assert len(fr["interactive_points"]) == 0, f"Negative frame {fr['frame_id']} must have 0 points"
        mask_path = DATASET_ROOT / fr["affordance_mask_path"]
        mask = np.load(mask_path)
        assert np.all(mask == 0), f"Negative frame {fr['frame_id']} mask must be completely 0"


def test_bounding_boxes_accurately_enclose_centroids(manifest_data: Dict[str, Any]):
    """For every interactive point, normalized bounding box must strictly enclose its centroid."""
    frames = manifest_data.get("frames", [])
    checked_count = 0

    for fr in frames:
        if fr.get("is_negative", False):
            continue
        for pt in fr.get("interactive_points", []):
            cx = float(pt["x"])
            cy = float(pt["y"])
            ymin, xmin, ymax, xmax = pt["bbox"]

            assert 0.0 <= cx <= 1.0, f"Centroid x out of bounds: {cx}"
            assert 0.0 <= cy <= 1.0, f"Centroid y out of bounds: {cy}"
            assert 0.0 <= xmin <= xmax <= 1.0, f"Invalid bbox x: [{xmin}, {xmax}]"
            assert 0.0 <= ymin <= ymax <= 1.0, f"Invalid bbox y: [{ymin}, {ymax}]"

            # Enclosure check with 1e-4 tolerance for rounding
            assert xmin - 1e-4 <= cx <= xmax + 1e-4, f"cx {cx} not enclosed by [{xmin}, {xmax}] in {pt['element_id']}"
            assert ymin - 1e-4 <= cy <= ymax + 1e-4, f"cy {cy} not enclosed by [{ymin}, {ymax}] in {pt['element_id']}"
            assert (xmax - xmin) > 0.0, "Bounding box width must be positive"
            assert (ymax - ymin) > 0.0, "Bounding box height must be positive"
            checked_count += 1

    assert checked_count > 500, f"Expected >500 elements checked, got {checked_count}"


def test_query_byte_limits_and_variations(manifest_data: Dict[str, Any]):
    """All queries must be non-empty and strictly bounded by MAX_QUERY_BYTES (96)."""
    frames = manifest_data.get("frames", [])
    for fr in frames:
        for pt in fr.get("interactive_points", []):
            tq = pt.get("target_query", "")
            assert tq, "Target query must not be empty"
            assert len(tq.encode("utf-8")) <= MAX_QUERY_BYTES
            for qv in pt.get("query_variations", []):
                assert qv, "Query variation must not be empty"
                assert len(qv.encode("utf-8")) <= MAX_QUERY_BYTES

        for nq in fr.get("negative_queries", []):
            assert nq, "Negative query must not be empty"
            assert len(nq.encode("utf-8")) <= MAX_QUERY_BYTES


def test_dataset_loader_and_tuple_unpacking():
    """UIGroundingDataset must support both dict access and 6-tuple unpacking."""
    dataset = UIGroundingDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=(256, 256),
    )
    assert len(dataset) == 160

    # 1. Dict access
    sample = dataset[0]
    assert isinstance(sample, GroundingSample)
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 256, 256)
    assert sample["image"].dtype == torch.float32
    assert 0.0 <= sample["image"].min() and sample["image"].max() <= 1.0
    assert isinstance(sample["query_str"], str)
    assert sample["target_centroid"].shape == (2,)
    assert sample["target_bbox"].shape == (4,)
    assert sample["label"].ndim == 0
    assert isinstance(sample["is_negative"], bool)

    # 2. Tuple unpacking
    image_t, q_str, c_norm, b_norm, lbl, is_neg = sample
    assert image_t.shape == (3, 256, 256)
    assert isinstance(q_str, str)
    assert c_norm.shape == (2,)
    assert b_norm.shape == (4,)
    assert lbl.ndim == 0
    assert is_neg is False


def test_negative_dataset_samples():
    """Loading test_negative split yields negative sample markers."""
    dataset = UIGroundingDataset(
        manifest_path=MANIFEST_PATH,
        split="test_negative",
    )
    assert len(dataset) == 30
    sample = dataset[0]
    assert sample.is_negative is True
    assert sample["has_target"].item() == 0.0
    assert torch.equal(sample["target_centroid"], torch.tensor([-1.0, -1.0]))
    assert torch.equal(sample["target_bbox"], torch.tensor([-1.0, -1.0, -1.0, -1.0]))
    assert sample["target_heatmap"].sum().item() == 0.0


def test_dataloader_batch_collation_and_throughput():
    """Vectorized collate_fn must produce batched tensors efficiently."""
    dataset = UIGroundingDataset(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=(256, 256),
    )
    assert len(dataset) == 40

    loader = make_ui_grounding_dataloader(dataset, batch_size=16, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3  # 16 + 16 + 8

    b0 = batches[0]
    assert b0["image"].shape == (16, 3, 256, 256)
    assert len(b0["query_str"]) == 16
    assert b0["target_centroid"].shape == (16, 2)
    assert b0["target_bbox"].shape == (16, 4)
    assert b0["label"].shape == (16,)
    assert b0["is_negative"].shape == (16,)
    assert b0["target_heatmap"].shape == (16, 64, 64)

    # Throughput benchmark: loading all 40 items
    t0 = time.perf_counter()
    count = 0
    for b in loader:
        count += b["image"].shape[0]
    dt = time.perf_counter() - t0
    rate = count / max(1e-5, dt)
    assert rate >= 8.0, f"DataLoader throughput {rate:.1f} samples/sec is too low"


def test_benchmark_runner_oracle_evaluation():
    """Benchmark runner must correctly compute metrics and pass oracle model."""
    runner = UIGroundingBenchmarkRunner(
        manifest_path=MANIFEST_PATH,
        confidence_threshold=0.35,
        tolerance_px=15.0,
    )

    class OracleModel:
        """Ground-truth oracle model for benchmark runner verification."""
        def predict_point(self, image_tensor: torch.Tensor, query: str):
            # If query is a negative query, return low confidence
            if any(nq.lower() in query.lower() for nq in ["checkout", "play video", "volume", "trash", "bluetooth"]):
                return {"point": (0.5, 0.5), "confidence": 0.1, "bbox": (0.45, 0.45, 0.55, 0.55)}
            # Return realistic high confidence
            return {"point": (0.5, 0.5), "confidence": 0.95, "bbox": (0.45, 0.45, 0.55, 0.55)}

    results = runner.evaluate_model(OracleModel(), split="test_held_out")
    assert isinstance(results, GroundingBenchmarkResults)
    assert results.total_positive_samples > 0
    assert results.total_blank_screens == 30
    assert isinstance(results.top_1_accuracy_pct, float)
    assert isinstance(results.median_centroid_error_px, float)
    assert isinstance(results.mean_iou, float)
    assert isinstance(results.negative_query_rejection_pct, float)
    assert isinstance(results.breakdown_by_category, dict)
    assert isinstance(results.breakdown_by_element_kind, dict)
    summary_text = results.summary()
    assert "UI Grounding Benchmark Results" in summary_text


def test_benchmark_runner_fail_closed_on_invalid_queries():
    """UIGroundingBenchmarkRunner._infer_model must fail closed to confidence 0.0 without crashing."""
    from avcs.goal_grounder import GoalConditionedGrounder

    runner = UIGroundingBenchmarkRunner(manifest_path=MANIFEST_PATH)
    model = GoalConditionedGrounder()
    dummy_img = torch.zeros(3, 256, 256)

    # Edge cases: whitespace, empty string, oversized (>96 bytes), None, non-string
    edge_cases = [
        ("whitespace", "   "),
        ("empty", ""),
        ("oversized", "A" * 200),
        ("none", None),
        ("non_str", 12345),
    ]

    for label, query in edge_cases:
        pt, conf, bbox = runner._infer_model(model, dummy_img, query)
        assert conf == 0.0, f"Failed closed check: {label} query yielded non-zero confidence {conf}"
        assert pt == (0.5, 0.5), f"Failed closed check: {label} query coordinate not (0.5, 0.5)"
        assert bbox == (0.46, 0.46, 0.54, 0.54), f"Failed closed check: {label} query bbox mismatch"

    # Valid queries must succeed and produce valid confidence
    valid_pt, valid_conf, valid_bbox = runner._infer_model(model, dummy_img, "click submit")
    assert isinstance(valid_conf, float)
    assert 0.0 <= valid_conf <= 1.0

    # Emoji query with alphanumeric text (<=96 bytes) must succeed without crashing
    emoji_pt, emoji_conf, emoji_bbox = runner._infer_model(model, dummy_img, chr(128269) + " search")
    assert isinstance(emoji_conf, float)

    # Mock/callable model must also fail closed upfront on oversized and emoji-only queries
    class MockBypassModel:
        def predict_point(self, img, q):
            return {"point": (0.8, 0.8), "confidence": 0.99, "bbox": (0.7, 0.7, 0.9, 0.9)}

    mock_model = MockBypassModel()
    for label, query in [("oversized_10000", "A" * 10000), ("emoji_only", "\U0001f50d\U0001f3af\U0001f680")]:
        pt, conf, bbox = runner._infer_model(mock_model, dummy_img, query)
        assert conf == 0.0, f"Mock model failed closed check for {label}: got {conf}"
        assert pt == (0.5, 0.5)
        assert bbox == (0.46, 0.46, 0.54, 0.54)


# ==============================================================================
# Agent 2: Real UI Demonstration Dataset Pipeline Tests
# ==============================================================================

from avcs.training.ui_dataset import (
    SUPPORTED_REAL_APPS,
    APP_CATEGORY_MAPPING,
    REAL_UI_DEMO_PROVENANCE,
    NEGATIVE_PROBE_BANK,
    UIDemoAction,
    UIDemoFrame,
    UIDemoSession,
    RealUIDemoGenerator,
    RealUIDemonstrationDataset,
    create_real_ui_splits,
    export_real_ui_dataset,
    ingest_demo_recorder_directory,
    verify_real_ui_manifest,
    make_real_ui_dataloader,
)


def test_real_ui_demo_generator_supported_apps():
    """RealUIDemoGenerator must generate valid demonstration frames for all 6 supported desktop apps."""
    gen = RealUIDemoGenerator(screen_size=(256, 256), grid_size=(16, 16))

    for idx, app_name in enumerate(SUPPORTED_REAL_APPS):
        frame = gen.generate_app_frame(
            app_name=app_name,
            seed=2000 + idx * 10,
            split="train",
            session_id=f"sess_test_{app_name.lower().replace(' ', '_')}",
            frame_idx=idx + 1,
        )

        assert isinstance(frame, UIDemoFrame)
        assert frame.app_name == app_name
        assert frame.category == APP_CATEGORY_MAPPING[app_name]
        assert frame.image.shape == (256, 256, 3)
        assert frame.image.dtype == np.uint8
        assert frame.affordance_mask.shape == (16, 16)
        assert frame.affordance_mask.dtype == np.int64
        assert np.isin(frame.affordance_mask, [0, 1, 2]).all()
        assert len(frame.interactive_points) > 0, f"Expected interactive elements in {app_name}"

        # Centroid enclosure and query limits
        for pt in frame.interactive_points:
            cx, cy = pt.point_norm
            ymin, xmin, ymax, xmax = pt.bbox_norm
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
            assert xmin - 1e-4 <= cx <= xmax + 1e-4
            assert ymin - 1e-4 <= cy <= ymax + 1e-4
            assert len(pt.target_query.encode("utf-8")) <= MAX_QUERY_BYTES
            for qv in pt.query_variations:
                assert len(qv.encode("utf-8")) <= MAX_QUERY_BYTES

        # Negative probes attached
        assert len(frame.negative_queries) >= 3
        for nq in frame.negative_queries:
            assert len(nq.encode("utf-8")) <= MAX_QUERY_BYTES


def test_real_ui_demo_app_disjoint_and_session_disjoint_splits(tmp_path: Path):
    """Demonstration dataset splits must be strictly app-disjoint and session-disjoint (zero leakage)."""
    gen = RealUIDemoGenerator(screen_size=(256, 256))

    sessions: List[UIDemoSession] = []
    # Build 2 distinct sessions per app
    for app_name in SUPPORTED_REAL_APPS:
        for s_idx in range(1, 3):
            sess_id = f"session_{app_name.lower().replace(' ', '_')}_{s_idx:02d}"
            sess = UIDemoSession(session_id=sess_id, app_name=app_name, task=f"Test {app_name}")
            for f_idx in range(1, 3):
                fr = gen.generate_app_frame(
                    app_name=app_name,
                    seed=3000 + len(sessions) * 100 + f_idx,
                    session_id=sess_id,
                    frame_idx=f_idx,
                )
                sess.frames.append(fr)
            sessions.append(sess)

    # Static blank sessions for negative partition
    neg_sess = UIDemoSession(session_id="session_blank_01", app_name="Desktop Wallpaper", task="Blank screen")
    for b_idx in range(1, 4):
        b_fr = gen.generate_static_blank_frame(
            negative_type="solid",
            seed=4000 + b_idx,
            session_id="session_blank_01",
            frame_idx=b_idx,
        )
        neg_sess.frames.append(b_fr)
    sessions.append(neg_sess)

    # Partition with Terminal and Settings held out
    splits = create_real_ui_splits(
        sessions=sessions,
        train_apps=["VS Code", "Notepad", "Browser", "Explorer"],
        held_out_test_apps=["Terminal", "Settings"],
    )

    train_apps = {fr.app_name for fr in splits["train"]}
    val_apps = {fr.app_name for fr in splits["val"]}
    test_apps = {fr.app_name for fr in splits["test_held_out"]}

    # 1. Strict App-Disjoint assertion
    assert train_apps.isdisjoint(test_apps), (
        f"APP LEAKAGE DETECTED: Train apps {train_apps} overlap with test apps {test_apps}"
    )
    assert "Terminal" in test_apps
    assert "Settings" in test_apps
    assert "Terminal" not in train_apps and "Settings" not in train_apps

    # 2. Strict Session-Disjoint assertions
    tr_sess = {fr.session_id for fr in splits["train"]}
    val_sess = {fr.session_id for fr in splits["val"]}
    test_sess = {fr.session_id for fr in splits["test_held_out"]}
    neg_sess_ids = {fr.session_id for fr in splits["test_negative"]}

    assert tr_sess.isdisjoint(val_sess), "Train and Val sessions must be disjoint"
    assert tr_sess.isdisjoint(test_sess), "Train and Test sessions must be disjoint"
    assert val_sess.isdisjoint(test_sess), "Val and Test sessions must be disjoint"
    assert tr_sess.isdisjoint(neg_sess_ids), "Train and Negative sessions must be disjoint"

    # Export to disk and verify zero image hash collisions
    manifest_p = export_real_ui_dataset(tmp_path / "real_demo_dataset", splits)
    audit_res = verify_real_ui_manifest(manifest_p, verbose=False)
    assert audit_res["verified"] is True


def test_real_ui_demo_negative_probes_and_static_blank_screens():
    """Static blank screens and negative probe queries must produce zero-hit negative samples."""
    gen = RealUIDemoGenerator(screen_size=(256, 256))

    # 1. Static blank screens
    for ntype in ["solid", "gradient", "wallpaper"]:
        b_fr = gen.generate_static_blank_frame(negative_type=ntype, seed=99)
        assert b_fr.is_negative is True
        assert len(b_fr.interactive_points) == 0
        assert np.all(b_fr.affordance_mask == 0)

    # 2. Negative probe queries in RealUIDemonstrationDataset
    fr_vscode = gen.generate_app_frame("VS Code", seed=10)
    fr_blank = gen.generate_static_blank_frame("wallpaper", seed=11)

    ds = RealUIDemonstrationDataset(
        frames=[fr_vscode, fr_blank],
        include_negative_queries=True,
        expand_element_queries=True,
    )

    # Check that negative samples are marked appropriately
    neg_samples = [s for s in ds if s.is_negative]
    assert len(neg_samples) > 0
    for s in neg_samples:
        assert s["has_target"].item() == 0.0
        assert torch.equal(s["target_centroid"], torch.tensor([-1.0, -1.0]))
        assert torch.equal(s["target_bbox"], torch.tensor([-1.0, -1.0, -1.0, -1.0]))
        assert s["target_heatmap"].sum().item() == 0.0


def test_real_ui_demo_raw_recorder_session_ingestion(tmp_path: Path):
    """Ingesting WindowsDemoRecorder session folder parses click and absence actions."""
    sess_dir = tmp_path / "test_recorder_session"
    sess_dir.mkdir(parents=True)

    manifest_data = {
        "schema_version": 1,
        "status": "stopped",
        "capture_scope": "cropped target window only",
        "keyboard_recording": False,
        "network_upload": False,
        "target_window_title": "Untitled - Notepad",
        "task": "Test click recording in Notepad",
        "sample_count": 2,
    }
    with open(sess_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    # Dummy image
    img = Image.new("RGB", (256, 256), color=(240, 240, 240))
    img.save(sess_dir / "frame_0001.png")
    img.save(sess_dir / "frame_0002.png")

    samples_lines = [
        json.dumps({
            "schema_version": 1,
            "task": "Test click recording in Notepad",
            "target_label": "File menu",
            "frame": "frame_0001.png",
            "action": {"type": "mouse_click", "button": "left", "point_norm": [0.12, 0.05]},
            "recorded_at_utc": "2026-10-17T00:00:00Z",
        }),
        json.dumps({
            "schema_version": 1,
            "task": "Test click recording in Notepad",
            "target_label": "Bluetooth toggle",
            "frame": "frame_0002.png",
            "action": {"type": "target_absent"},
            "recorded_at_utc": "2026-10-17T00:00:01Z",
        }),
    ]
    with open(sess_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(samples_lines) + "\n")

    sessions = ingest_demo_recorder_directory(tmp_path)
    assert len(sessions) == 1
    sess = sessions[0]
    assert sess.app_name == "Notepad"
    assert len(sess.frames) == 2

    f1 = sess.frames[0]
    assert f1.is_negative is False
    assert len(f1.interactive_points) == 1
    assert f1.interactive_points[0].target_query == "File menu"
    assert f1.interactive_points[0].point_norm == (0.12, 0.05)

    f2 = sess.frames[1]
    assert f2.is_negative is True
    assert len(f2.interactive_points) == 0


def test_real_ui_dataloader_batch_iteration_and_grounder_forward():
    """Real UI DataLoader must collate batches and execute forward pass through GoalConditionedGrounder."""
    from avcs.goal_grounder import GoalConditionedGrounder

    gen = RealUIDemoGenerator(screen_size=(256, 256))
    frames = [
        gen.generate_app_frame("VS Code", seed=101, frame_idx=1),
        gen.generate_app_frame("Notepad", seed=102, frame_idx=2),
        gen.generate_app_frame("Browser", seed=103, frame_idx=3),
        gen.generate_app_frame("Explorer", seed=104, frame_idx=4),
    ]

    dataset = RealUIDemonstrationDataset(frames=frames, preload_frames=True)
    assert len(dataset) == 4

    # 1. 6-tuple unpacking check
    image_t, q_str, c_norm, b_norm, lbl, is_neg = dataset[0]
    assert image_t.shape == (3, 256, 256)
    assert isinstance(q_str, str)
    assert c_norm.shape == (2,)
    assert b_norm.shape == (4,)
    assert lbl.ndim == 0
    assert is_neg is False

    # 2. Collation and grounder forward pass
    loader = make_real_ui_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert batch["image"].shape == (4, 3, 256, 256)
    assert batch["target_heatmap"].shape == (4, 64, 64)
    assert batch["query_tokens"].shape == (4, 96)

    model = GoalConditionedGrounder()
    with torch.no_grad():
        logits = model(batch["image"], batch["query_tokens"])

    assert logits.shape == (4, 1, 128, 128)


