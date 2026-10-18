from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avcs.streaming_v1.data import (
    SEMANTIC_EVENT_LABELS,
    SequenceDataset,
    generate_procedural_sequences,
    validate_manifest,
)
from avcs.streaming_v1.diagnose import audit_checkpoint_predictions
from avcs.streaming_v1.ingest import ingest_sequence_index
from avcs.streaming_v1.model import StreamingVisualCortexV1
from avcs.streaming_v1.runtime import FramePacket, LatestFrameBuffer, StreamingObserver
from avcs.streaming_v1.sources import NpzSequenceReplaySource, WindowsDesktopCaptureSource
from avcs.streaming_v1.training import (
    evaluate_streaming_checkpoint,
    load_streaming_checkpoint,
    temporal_losses,
    train_streaming_v1,
)


@pytest.fixture
def sequence_root(tmp_path: Path) -> Path:
    root = tmp_path / "sequences"
    generate_procedural_sequences(
        root,
        seed=71,
        train_recordings=2,
        validation_recordings=1,
        test_recordings=1,
        frames_per_recording=5,
        width=32,
        height=32,
        fps=5.0,
    )
    return root


def test_model_is_reproducible_and_has_incremental_state():
    torch.manual_seed(9)
    first = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    torch.manual_seed(9)
    second = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    assert all(
        torch.equal(first.state_dict()[key], second.state_dict()[key])
        for key in first.state_dict()
    )
    first.eval()
    frame = torch.rand(1, 3, 32, 40)
    with torch.inference_mode():
        initial, state = first.forward_step(frame)
        updated, next_state = first.forward_step(frame, state, timestamp_delta=0.1)
    assert initial["latent"].shape == updated["latent"].shape == (1, 16)
    assert updated["change_logits"].shape == (1, 1, 32, 40)
    assert updated["motion_grid_displacement"].shape == (1, 2, 4, 5)
    assert not torch.equal(state.hidden, next_state.hidden)
    assert updated["has_previous"].item()


def test_model_rejects_invalid_time_and_pixel_inputs():
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    with pytest.raises(ValueError, match="timestamp_delta"):
        model.forward_step(torch.zeros(1, 3, 32, 32), timestamp_delta=-1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.forward_step(torch.full((1, 3, 32, 32), 2.0))


def test_model_accepts_per_sample_irregular_timestamps():
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    frames = torch.rand(2, 4, 3, 32, 32)
    times = torch.tensor(
        [[0.0, 0.1, 0.3, 0.4], [1.0, 1.4, 1.6, 2.0]],
        dtype=torch.float64,
    )
    output = model.forward_sequence(frames, times)
    assert output["latent"].shape == (2, 4, 16)


def test_spatial_ablation_ignores_recurrent_state():
    model = StreamingVisualCortexV1(
        feature_dim=16,
        hidden_dim=16,
        latent_dim=16,
        temporal_mode="spatial",
    ).eval()
    frame = torch.rand(1, 3, 32, 32)
    other_state = model.forward_step(torch.rand_like(frame))[1]
    with torch.inference_mode():
        without_state, _ = model.forward_step(frame)
        with_state, _ = model.forward_step(frame, other_state, timestamp_delta=0.2)
    assert torch.equal(without_state["latent"], with_state["latent"])
    assert torch.equal(without_state["change_logits"], with_state["change_logits"])


def test_delta_temporal_head_and_recurrent_state_receive_gradients():
    model = StreamingVisualCortexV1(
        feature_dim=16,
        hidden_dim=16,
        latent_dim=16,
        temporal_mode="convgru_delta",
    )
    frames = torch.rand(2, 4, 3, 32, 32)
    times = torch.arange(4, dtype=torch.float64).repeat(2, 1) * 0.1
    batch = {
        "frames": frames,
        "timestamps": times,
        "change_masks": torch.zeros(2, 3, 32, 32),
        "motion_grid_displacement": torch.zeros(2, 3, 32, 32, 2),
        "motion_mask": torch.ones(2, 3, 32, 32),
    }
    losses = temporal_losses(model, batch, change_positive_weight=8.0)
    losses["total"].backward()
    assert model.change_head[0].in_channels == 64
    assert model.recurrent.gates.weight.grad is not None
    assert model.recurrent.gates.weight.grad.abs().sum().item() > 0


def test_recording_dataset_is_hashed_and_source_split(sequence_root: Path):
    validation = validate_manifest(sequence_root)
    assert validation["recording_counts"] == {"test": 1, "train": 2, "validation": 1}
    train = SequenceDataset(sequence_root, "train")
    test = SequenceDataset(sequence_root, "test")
    assert set(item["source_id"] for item in [train[i] for i in range(len(train))]).isdisjoint(
        {test[0]["source_id"]}
    )
    sample = train[0]
    assert sample["frames"].shape == (5, 3, 32, 32)
    assert sample["change_masks"].shape == (4, 32, 32)
    assert sample["motion_grid_displacement"].shape == (4, 32, 32, 2)
    assert sample["annotations"]["object_identity"] == "not_collected"
    pixels = sample["frames"].mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    expected_change = np.any(pixels[1:] != pixels[:-1], axis=-1)
    assert np.array_equal(sample["change_masks"].numpy().astype(bool), expected_change)

    record = json.loads((sequence_root / "manifest.json").read_text(encoding="utf-8"))["recordings"][0]
    path = sequence_root / record["path"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_manifest(sequence_root)


def test_ingestion_validates_rights_source_split_and_missing_annotations(
    sequence_root: Path,
    tmp_path: Path,
):
    source_manifest = json.loads(
        (sequence_root / "manifest.json").read_text(encoding="utf-8")
    )
    records = []
    for row in source_manifest["recordings"]:
        source_path = sequence_root / row["path"]
        source_id = hashlib.sha256(source_path.read_bytes()).hexdigest()
        records.append({
            "recording_id": row["recording_id"],
            "source_id": source_id,
            "split": row["split"],
            "path": str((sequence_root / row["path"]).relative_to(tmp_path)),
            "source_path": str(source_path.relative_to(tmp_path)),
            "annotations": {
                **row["annotations"],
                "object_identity": "ambiguous",
                "semantic_event_labels": "not_collected",
                "object_id_masks": "not_collected",
                "occlusion_masks": "not_collected",
            },
            "provenance": {
                "kind": "authorized_recording",
                "rights_evidence": "documented test consent reference",
                "permitted_uses": ["internal_training"],
            },
        })
    index_path = tmp_path / "ingest.json"
    index_path.write_text(
        json.dumps({
            "schema": "avcs.streaming.ingest-index.v2",
            "dataset_version": "test-ingest-1",
            "recordings": records,
        }),
        encoding="utf-8",
    )
    output = tmp_path / "ingested"
    result = ingest_sequence_index(index_path, output)
    assert result["recording_counts"] == {"test": 1, "train": 2, "validation": 1}
    ingested = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert ingested["rights"]["internal_training"] == "PENDING"
    assert ingested["recordings"][0]["annotations"]["object_identity"] == "ambiguous"
    assert ingested["recordings"][0]["provenance"]["source_sha256"] == records[0]["source_id"]
    assert ingested["semantic_event_labels"] == list(SEMANTIC_EVENT_LABELS)
    with pytest.raises(FileExistsError, match="overwrite"):
        ingest_sequence_index(index_path, output)

    records[0]["provenance"]["permitted_uses"] = ["research_only"]
    blocked_index = tmp_path / "blocked-ingest.json"
    blocked_index.write_text(
        json.dumps({
            "schema": "avcs.streaming.ingest-index.v2",
            "dataset_version": "blocked-test",
            "recordings": records,
        }),
        encoding="utf-8",
    )
    blocked_output = tmp_path / "blocked"
    with pytest.raises(PermissionError, match="documented internal-training rights"):
        ingest_sequence_index(blocked_index, blocked_output)
    assert not blocked_output.exists()

    records[0]["source_id"] = "0" * 64
    tampered_index = tmp_path / "tampered-source.json"
    tampered_index.write_text(
        json.dumps({
            "schema": "avcs.streaming.ingest-index.v2",
            "dataset_version": "tampered-source",
            "recordings": records,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 of the original source"):
        ingest_sequence_index(tampered_index, tmp_path / "tampered-source")


def test_ingestion_preserves_unknown_semantic_event_labels(
    sequence_root: Path,
    tmp_path: Path,
):
    source_manifest = json.loads(
        (sequence_root / "manifest.json").read_text(encoding="utf-8")
    )
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    rows = []
    for record in source_manifest["recordings"]:
        source_path = sequence_root / record["path"]
        prepared_path = prepared_root / f"{record['recording_id']}.npz"
        with np.load(source_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["semantic_event_labels"] = np.full(
            (arrays["frames"].shape[0] - 1, len(SEMANTIC_EVENT_LABELS)),
            -1,
            dtype=np.int8,
        )
        np.savez_compressed(prepared_path, **arrays)
        rows.append({
            "recording_id": record["recording_id"],
            "source_id": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "split": record["split"],
            "path": str(prepared_path.relative_to(tmp_path)),
            "source_path": str(source_path.relative_to(tmp_path)),
            "annotations": {
                **record["annotations"],
                "semantic_event_labels": "available",
                "object_id_masks": "not_collected",
                "occlusion_masks": "not_collected",
            },
            "provenance": {
                "kind": "licensed_recording",
                "rights_evidence": "test-only documented permission reference",
                "permitted_uses": ["internal_training"],
            },
        })
    index_path = tmp_path / "semantic-index.json"
    index_path.write_text(
        json.dumps({
            "schema": "avcs.streaming.ingest-index.v2",
            "dataset_version": "semantic-label-fixture",
            "recordings": rows,
        }),
        encoding="utf-8",
    )
    output = tmp_path / "semantic-dataset"
    ingest_sequence_index(index_path, output)
    sample = SequenceDataset(output, "train")[0]
    assert sample["semantic_event_labels"].dtype == torch.int8
    assert torch.all(sample["semantic_event_labels"] == -1)


def test_temporal_learning_objectives_are_finite(sequence_root: Path):
    dataset = SequenceDataset(sequence_root, "train")
    batch = {
        key: torch.stack([dataset[0][key], dataset[1][key]])
        for key in (
            "frames",
            "timestamps",
            "change_masks",
            "motion_grid_displacement",
            "motion_mask",
        )
    }
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    losses = temporal_losses(model, batch)
    assert set(losses) == {
        "total",
        "future_feature_prediction",
        "change_bce",
        "motion_endpoint_l1",
        "representation_variance",
    }
    assert all(torch.isfinite(value).item() for value in losses.values())


def test_unavailable_annotations_are_not_scored_as_negative_labels():
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    batch = {
        "frames": torch.rand(2, 4, 3, 32, 32),
        "timestamps": torch.arange(4, dtype=torch.float64).repeat(2, 1) * 0.1,
        "change_masks": torch.ones(2, 3, 32, 32),
        "motion_grid_displacement": torch.ones(2, 3, 32, 32, 2),
        "motion_mask": torch.ones(2, 3, 32, 32),
        "annotation_valid": [
            {"change_masks": False, "motion_mask": False, "motion_grid_displacement": False}
            for _ in range(2)
        ],
    }
    losses = temporal_losses(model, batch)
    assert losses["change_bce"].item() == 0.0
    assert losses["motion_endpoint_l1"].item() == 0.0
    assert torch.isfinite(losses["future_feature_prediction"])


def test_latest_frame_buffer_replaces_stale_unprocessed_frames():
    frames = np.zeros((32, 32, 3), dtype=np.uint8)
    buffer = LatestFrameBuffer()
    buffer.put(FramePacket(1, 1.0, frames))
    buffer.put(FramePacket(2, 2.0, frames + 1))
    assert buffer.replaced_count == 1
    assert buffer.get().sequence == 2
    with pytest.raises(ValueError, match="out-of-order"):
        buffer.put(FramePacket(1, 3.0, frames + 2))
    buffer.close()


def test_frame_packet_owns_immutable_pixel_snapshot():
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    packet = FramePacket(0, 1.0, source)
    source[:] = 255
    assert not packet.rgb.flags.writeable
    assert packet.rgb.max() == 0


def test_continuous_runtime_recovers_capture_and_stops_cleanly():
    class RecoverableSource:
        def __init__(self):
            self.opens = 0
            self.frames = 0

        def open(self):
            self.opens += 1
            return self

        def grab(self):
            if self.opens == 1:
                raise OSError("injected first-open capture fault")
            if self.frames >= 5:
                raise EOFError("replay complete")
            frame = np.full((32, 32, 3), self.frames * 20, dtype=np.uint8)
            self.frames += 1
            return frame

        def close(self):
            return None

    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    observer = StreamingObserver(model, input_size=(32, 32), max_frame_age_seconds=1.0)
    source = RecoverableSource()
    try:
        observer.start(source, target_hz=20)
        observer._capture_thread.join(timeout=4)
        observer._inference_thread.join(timeout=4)
        assert not observer._capture_thread.is_alive()
        assert not observer._inference_thread.is_alive()
        assert observer.capture_recovery_count == 1
        assert observer.captured_count == 5
        assert observer.processed_count > 0
        assert observer.last_error is None
    finally:
        observer.stop()
        torch.set_num_threads(old_num_threads)


def test_observer_skips_duplicate_reordered_and_stale_packets():
    torch.manual_seed(1)
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    observer = StreamingObserver(
        model,
        input_size=(32, 32),
        max_frame_age_seconds=0.5,
        reset_after_seconds=1.0,
    )
    first = np.zeros((32, 32, 3), dtype=np.uint8)
    changed = first.copy()
    changed[8:20, 8:20] = 255
    assert observer.process(FramePacket(0, 10.0, first), now=10.1) is not None
    assert observer.process(FramePacket(1, 10.1, first), now=10.2) is None
    assert observer.duplicate_count == 1
    assert observer.process(FramePacket(1, 10.2, changed), now=10.3) is None
    assert observer.out_of_order_count == 1
    stale = observer.process(FramePacket(2, 10.3, changed), now=11.0)
    assert stale is None
    assert observer.stale_count == 1
    recovered = observer.process(FramePacket(3, 12.0, changed), now=12.1)
    assert recovered is not None and recovered.gap_reset
    assert recovered.timestamp_delta == pytest.approx(2.0)


def test_replay_and_live_capture_require_explicit_scope(sequence_root: Path):
    record = json.loads((sequence_root / "manifest.json").read_text(encoding="utf-8"))["recordings"][0]
    source = NpzSequenceReplaySource(sequence_root / record["path"], realtime=False).open()
    try:
        assert source.grab().shape == (32, 32, 3)
        for _ in range(4):
            source.grab()
        with pytest.raises(EOFError):
            source.grab()
    finally:
        source.close()
    with pytest.raises(PermissionError, match="requires confirm_screen_capture"):
        WindowsDesktopCaptureSource()


def test_training_gate_requires_approval_for_exact_manifest(
    sequence_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("AVCS_RIGHTS_TRUST_CONFIG_SHA256", raising=False)
    output = tmp_path / "candidate.pt"
    with pytest.raises(RuntimeError, match="process-level"):
        train_streaming_v1(
            sequence_root,
            output,
            approval_path=tmp_path / "approval.json",
            trust_config_path=tmp_path / "trust.json",
        )
    assert not output.exists()


def test_training_rejects_missing_supervised_labels_explicitly(
    sequence_root: Path,
    tmp_path: Path,
):
    manifest_path = sequence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(row for row in manifest["recordings"] if row["split"] == "train")
    sequence_path = sequence_root / record["path"]
    with np.load(sequence_path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name].copy()
            for name in archive.files
            if name != "change_masks"
        }
    np.savez_compressed(sequence_path, **arrays)
    record["sha256"] = hashlib.sha256(sequence_path.read_bytes()).hexdigest()
    record["annotations"]["change_masks"] = "ambiguous"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "candidate.pt"
    with pytest.raises(ValueError, match="missing or ambiguous labels"):
        train_streaming_v1(
            sequence_root,
            output,
            approval_path=tmp_path / "approval.json",
            trust_config_path=tmp_path / "trust.json",
        )
    assert not output.exists()


def test_versioned_checkpoint_hash_and_evaluator_integrity(sequence_root: Path, tmp_path: Path):
    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    path = tmp_path / "untrained-fixture.pt"
    payload = {
        "format": "avcs.streaming_v1.checkpoint",
        "format_version": 1,
        "architecture": StreamingVisualCortexV1.model_name,
        "model_config": {
            "feature_dim": 16,
            "hidden_dim": 16,
            "latent_dim": 16,
            "max_state_gap_seconds": 2.0,
        },
        "model_state_dict": model.state_dict(),
        "lineage": {
            "initialization": "test fixture only",
            "seed": 0,
            "pretrained_weights_loaded": False,
            "source_checkpoint_hashes": [],
        },
    }
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(
        json.dumps({"checkpoint_sha256": digest}),
        encoding="utf-8",
    )
    loaded, _ = load_streaming_checkpoint(path)
    assert loaded.feature_dim == 16
    report = evaluate_streaming_checkpoint(
        path,
        sequence_root,
        split="test",
        change_threshold=0.3,
    )
    assert report["recording_count"] == 1
    assert report["frame_count"] == 5
    assert report["change_threshold"] == 0.3
    f1 = report["change_metrics"]["pixel_f1"]
    interval = report["recording_bootstrap_95pct_intervals"]["change_pixel_f1"]
    assert interval["low_95"] == pytest.approx(f1)
    assert interval["high_95"] == pytest.approx(f1)
    assert "exact RGB differences" in report["warning"]
    prediction_audit = audit_checkpoint_predictions(
        path,
        sequence_root,
        change_threshold=0.3,
    )
    assert prediction_audit["labelled_transitions"] == 4
    assert len(prediction_audit["recordings"][0]["transitions"]) == 4
    assert prediction_audit["raw_frames_or_prediction_maps_persisted"] is False
    assert "not semantic" in prediction_audit["pixel_label_semantics"]
    with pytest.raises(ValueError, match="change_threshold"):
        evaluate_streaming_checkpoint(path, sequence_root, change_threshold=1.0)

    path.with_suffix(".json").write_text(
        json.dumps({"checkpoint_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest"):
        load_streaming_checkpoint(path)


def test_evaluator_does_not_score_unlabelled_changes_as_negatives(
    sequence_root: Path,
    tmp_path: Path,
):
    manifest_path = sequence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(row for row in manifest["recordings"] if row["split"] == "test")
    sequence_path = sequence_root / record["path"]
    with np.load(sequence_path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name].copy()
            for name in archive.files
            if name != "change_masks"
        }
    np.savez_compressed(sequence_path, **arrays)
    record["sha256"] = hashlib.sha256(sequence_path.read_bytes()).hexdigest()
    record["annotations"]["change_masks"] = "not_collected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    model = StreamingVisualCortexV1(feature_dim=16, hidden_dim=16, latent_dim=16)
    path = tmp_path / "untrained-unlabelled-fixture.pt"
    torch.save(
        {
            "format": "avcs.streaming_v1.checkpoint",
            "format_version": 1,
            "architecture": StreamingVisualCortexV1.model_name,
            "model_config": {
                "feature_dim": 16,
                "hidden_dim": 16,
                "latent_dim": 16,
                "max_state_gap_seconds": 2.0,
            },
            "model_state_dict": model.state_dict(),
            "lineage": {
                "initialization": "test fixture only",
                "seed": 0,
                "pretrained_weights_loaded": False,
                "source_checkpoint_hashes": [],
            },
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps({
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    report = evaluate_streaming_checkpoint(path, sequence_root)
    assert report["change_metrics"]["pixel_f1"] is None
    assert report["change_metrics"]["labelled_recordings"] == 0
    assert report["change_metrics"]["unlabelled_recordings"] == 1
    assert report["test_recordings"][0]["change_f1"] is None
