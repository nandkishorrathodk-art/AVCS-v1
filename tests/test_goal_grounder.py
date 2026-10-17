import json
import time

import numpy as np
import pytest
import torch
from PIL import Image

from avcs.goal_grounder import (
    GROUNDING_CHECKPOINT_SCHEMA,
    GROUNDING_DATASET_PROVENANCE,
    LEGACY_GROUNDING_CHECKPOINT_SCHEMA,
    LEGACY_GROUNDING_DATASET_PROVENANCE,
    GoalConditionedGrounder,
    GroundingPrediction,
    GroundingStream,
    click_heatmap,
    encode_queries,
    letterbox_rgb,
    load_grounding_checkpoint,
    migrate_grounding_checkpoint,
)
from avcs.training.demonstrations import (
    DemonstrationClickDataset,
    discover_demonstrations,
    fingerprint_sessions,
    split_sessions,
)
from avcs.training.train_grounder import train_grounder


def test_query_encoder_uses_byte_tokens_and_rejects_empty_text():
    tokens = encode_queries(["Save", "é"])

    assert tokens.shape == (2, 96)
    assert tokens[0, :4].tolist() == [84, 98, 119, 102]
    assert tokens[1, :2].tolist() == [196, 170]
    with pytest.raises(ValueError, match="must not be empty"):
        encode_queries(["  "])
    with pytest.raises(ValueError, match="at most 4 UTF-8 bytes"):
        encode_queries(["button"], max_length=4)


def test_model_predict_requires_real_demonstration_checkpoint():
    model = GoalConditionedGrounder()
    with pytest.raises(RuntimeError, match="no demonstration-trained checkpoint"):
        model.predict(np.zeros((40, 80, 3), dtype=np.uint8), "save button")


def test_goal_conditioned_model_emits_dense_map_and_backpropagates():
    model = GoalConditionedGrounder(feature_dim=64, input_size=64)
    image = torch.rand(2, 3, 64, 64)
    query = encode_queries(["save button", "close window"])

    logits = model(image, query)
    assert logits.shape == (2, 1, 32, 32)
    logits.square().mean().backward()
    assert model.query_encoder.embedding.weight.grad is not None
    assert model.visual_encoder.stem.conv.weight.grad is not None


def test_letterbox_preserves_aspect_and_coordinate_round_trip():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, transform = letterbox_rgb(frame, input_size=100)

    assert tensor.shape == (3, 100, 100)
    assert transform.original_to_canvas((0.75, 0.25)) == pytest.approx((0.75, 0.375))
    assert transform.canvas_to_original((0.75, 0.375)) == pytest.approx((0.75, 0.25))


def test_click_heatmap_is_normalized_and_peaks_near_labeled_click():
    target = click_heatmap((0.7, 0.25), (32, 32))

    assert float(target.sum()) == pytest.approx(1.0)
    y, x = divmod(int(target.argmax()), target.shape[1])
    assert abs((x + 0.5) / 32 - 0.7) < 0.04
    assert abs((y + 0.5) / 32 - 0.25) < 0.04


def _write_session(root, name, point=(0.5, 0.5), include_absent=False):
    session = root / name
    (session / "frames").mkdir(parents=True)
    Image.new("RGB", (120, 60), color=(40, 50, 60)).save(session / "frames" / "frame.png")
    records = [{
        "schema_version": 1,
        "target_label": "Save button",
        "frame": "frames/frame.png",
        "action": {"type": "mouse_click", "button": "left", "point_norm": list(point)},
    }]
    if include_absent:
        Image.new("RGB", (120, 60), color=(0, 0, 0)).save(
            session / "frames" / "absent.png"
        )
        records.append({
            "schema_version": 1,
            "target_label": "Save button",
            "frame": "frames/absent.png",
            "action": {"type": "target_absent"},
        })
    (session / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "status": "stopped",
        "task": "click the labeled test button",
        "target_window_title": "AVCS disposable test fixture",
        "sample_count": len(records),
        "capture_scope": "cropped target window only",
        "keyboard_recording": False,
        "network_upload": False,
    }
    (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return session


def test_demonstration_split_keeps_recording_sessions_disjoint(tmp_path):
    _write_session(tmp_path, "session-a")
    _write_session(tmp_path, "session-b", (0.25, 0.75))
    _write_session(tmp_path, "session-c", (0.75, 0.25))
    _write_session(tmp_path, "session-d", (0.1, 0.9))
    _write_session(tmp_path, "session-e", (0.9, 0.1))
    discovered = discover_demonstrations(tmp_path)
    train, validation, test = split_sessions(discovered, seed=3)

    assert len(train) == 3
    assert len(validation) == len(test) == 1
    assert len({record.session_dir for record in train + validation + test}) == 5
    sample = DemonstrationClickDataset(train, input_size=64)[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["target_heatmap"].shape == (32, 32)
    assert sample["query_tokens"].shape == (96,)
    assert sample["target_present"].item() == 1.0


def test_dataset_loads_explicit_target_absence_without_fake_click_coordinates(tmp_path):
    _write_session(tmp_path, "session-absent", include_absent=True)
    records = discover_demonstrations(tmp_path)[tmp_path / "session-absent"]
    absent = next(record for record in records if not record.target_present)

    sample = DemonstrationClickDataset([absent], input_size=64)[0]

    assert absent.point_norm is None
    assert sample["target_present"].item() == 0.0
    assert sample["target_heatmap"].count_nonzero().item() == 0


def test_dataset_rejects_frame_path_traversal(tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("RGB", (10, 10)).save(outside)
    session = tmp_path / "session"
    session.mkdir()
    sample = {
        "schema_version": 1,
        "target_label": "button",
        "frame": "../outside.png",
        "action": {"type": "mouse_click", "button": "left", "point_norm": [0.5, 0.5]},
    }
    (session / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "stopped",
                "task": "fixture task",
                "target_window_title": "fixture",
                "sample_count": 1,
                "capture_scope": "cropped target window only",
                "keyboard_recording": False,
                "network_upload": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Frame path escapes"):
        discover_demonstrations(tmp_path)


def test_session_split_requires_multiple_recording_sessions(tmp_path):
    _write_session(tmp_path, "only-session")
    with pytest.raises(ValueError, match="At least 5 separate recording sessions"):
        split_sessions(discover_demonstrations(tmp_path))


def test_real_demo_trainer_learns_presence_and_calibrates_abstention(tmp_path):
    data_dir = tmp_path / "demos"
    for index in range(5):
        _write_session(data_dir, f"session-{index}", include_absent=True)

    report = train_grounder(
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "trained"),
        epochs=1,
        batch_size=2,
        input_size=64,
        seed=7,
        device_name="cpu",
    )

    model = load_grounding_checkpoint(tmp_path / "trained" / "goal_grounder_best.pt")
    assert report["target_presence_calibration"]["validation_false_activations"] == 0
    assert report["target_presence_threshold"] > 0
    assert report["abstention_action_authorized"] is False
    assert model.is_trained


def test_real_demo_trainer_refuses_data_without_absence_labels(tmp_path):
    data_dir = tmp_path / "positive-only"
    for index in range(5):
        _write_session(data_dir, f"session-{index}")

    with pytest.raises(ValueError, match="target-present and target-absent"):
        train_grounder(
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "must-not-train"),
            epochs=1,
            input_size=64,
            device_name="cpu",
        )


def test_checkpoint_loader_rejects_non_demonstration_provenance(tmp_path):
    checkpoint = tmp_path / "synthetic.pt"
    torch.save(
        {
            "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
            "model_type": "avcs_goal_conditioned_grounder",
            "dataset_provenance": "avcs_procedural_synthetic_desktop_v1",
            "training_samples": 100,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="dataset_provenance"):
        load_grounding_checkpoint(checkpoint)


def test_checkpoint_loader_accepts_demonstration_trainer_shape_and_predicts(tmp_path):
    source_model = GoalConditionedGrounder(input_size=64)
    with torch.no_grad():
        source_model.target_presence.weight.zero_()
        source_model.target_presence.bias.fill_(-10.0)
    checkpoint = tmp_path / "grounder.pt"
    torch.save(
        {
            "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
            "model_type": "avcs_goal_conditioned_grounder",
            "dataset_provenance": GROUNDING_DATASET_PROVENANCE,
            "training_samples": 12,
            "validation_samples": 3,
            "held_out_test_samples": 3,
            "target_presence_threshold": 0.5,
            "input_size": 64,
            "split_seed": 42,
            "model_state_dict": source_model.state_dict(),
        },
        checkpoint,
    )

    loaded = load_grounding_checkpoint(checkpoint)
    result = loaded.predict(np.zeros((48, 96, 3), dtype=np.uint8), "Save button")

    assert loaded.is_trained is True
    assert result.heatmap.shape == (32, 32)
    assert result.target_present is False
    assert result.target_present_probability < 0.5
    assert result.candidate_point_norm is None
    assert result.action_authorized is False


def test_checkpoint_loader_rejects_legacy_schema_even_with_demo_label(tmp_path):
    checkpoint = tmp_path / "legacy-mislabeled.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_type": "avcs_goal_conditioned_grounder",
            "dataset_provenance": GROUNDING_DATASET_PROVENANCE,
            "training_samples": 100,
            "validation_samples": 10,
            "model_state_dict": GoalConditionedGrounder(input_size=64).state_dict(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_grounding_checkpoint(checkpoint)


def test_checkpoint_loader_accepts_and_migrates_legacy_schema_1(tmp_path):
    source_model = GoalConditionedGrounder(input_size=64)
    raw_state = {
        k: v for k, v in source_model.state_dict().items() if not k.startswith("target_presence")
    }
    checkpoint = tmp_path / "legacy-valid.pt"
    torch.save(
        {
            "schema_version": LEGACY_GROUNDING_CHECKPOINT_SCHEMA,
            "model_type": "avcs_goal_conditioned_grounder",
            "dataset_provenance": LEGACY_GROUNDING_DATASET_PROVENANCE,
            "training_samples": 100,
            "validation_samples": 10,
            "held_out_test_samples": 5,
            "input_size": 64,
            "model_state_dict": raw_state,
        },
        checkpoint,
    )

    loaded = load_grounding_checkpoint(checkpoint)
    assert loaded.is_trained is True
    assert loaded.checkpoint_metadata["schema_version"] == GROUNDING_CHECKPOINT_SCHEMA
    assert loaded.checkpoint_metadata["legacy_schema_version"] == 1
    assert loaded.checkpoint_metadata["target_presence_threshold"] == 0.5

    result = loaded.predict(np.zeros((48, 96, 3), dtype=np.uint8), "Save button")
    assert result.heatmap.shape == (32, 32)
    assert result.target_present is True
    assert result.target_present_probability > 0.5
    assert result.action_authorized is False


def test_controller_loads_legacy_grounder_weights_best_pt():
    from avcs.controller import AVCSController
    from forge.vision.capture import SyntheticScreen

    source = SyntheticScreen(width=320, height=240)
    controller = AVCSController(
        source=source,
        weights_path="runs/weights/goal_grounder_best.pt",
        device="cpu",
        force_virtual_motor=True,
    )
    assert controller.is_grounder_trained is True
    assert controller.grounder is not None
    assert controller.load_grounder_weights() is True
    if hasattr(controller.source, "close"):
        controller.source.close()


def test_grounding_stream_reports_measured_per_frame_work_without_authorizing_actions(
    tmp_path, monkeypatch
):
    source_model = GoalConditionedGrounder(input_size=64)
    checkpoint = tmp_path / "trained-stream-fixture.pt"
    torch.save(
        {
            "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
            "model_type": "avcs_goal_conditioned_grounder",
            "dataset_provenance": GROUNDING_DATASET_PROVENANCE,
            "training_samples": 12,
            "validation_samples": 3,
            "held_out_test_samples": 3,
            "target_presence_threshold": 0.5,
            "input_size": 64,
            "model_state_dict": source_model.state_dict(),
        },
        checkpoint,
    )
    model = load_grounding_checkpoint(checkpoint)
    with pytest.raises(AttributeError):
        model.is_trained = True
    prediction = GroundingPrediction(
        candidate_point_norm=(0.5, 0.5),
        peak_probability=0.2,
        heatmap=torch.ones((32, 32)),
        action_authorized=False,
    )
    monkeypatch.setattr(model, "predict", lambda _frame, _query: prediction)
    stream = GroundingStream(model, metrics_window=3)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    timestamp = time.monotonic() - 1.0

    outputs = [
        stream.process_frame(frame, "save button", captured_at=timestamp + offset * 0.05)
        for offset in range(4)
    ]

    assert [result.sequence for result in outputs] == [1, 2, 3, 4]
    assert outputs[-1].prediction.action_authorized is False
    assert stream.metrics()["observed_input_fps"] == pytest.approx(20.0)
    assert stream.metrics()["p50_inference_latency_ms"] >= 0
    with pytest.raises(ValueError, match="must increase"):
        stream.process_frame(frame, "save button", captured_at=timestamp + 0.1)
    stream.reset()
    assert stream.metrics() == {
        "observed_input_fps": None,
        "p50_inference_latency_ms": None,
    }


def test_grounding_stream_refuses_untrained_model():
    with pytest.raises(RuntimeError, match="requires a demonstration-trained model"):
        GroundingStream(GoalConditionedGrounder())


def test_held_out_evaluator_uses_unchanged_disjoint_data_without_input_actions(
    tmp_path, monkeypatch
):
    from avcs.training import evaluate_grounder as evaluator

    for index in range(5):
        _write_session(
            tmp_path / "demos",
            f"session-{index}",
            (0.5, 0.5),
            include_absent=True,
        )
    sessions = discover_demonstrations(tmp_path / "demos")
    train, validation, test = split_sessions(sessions, seed=7)
    model_checkpoint = tmp_path / "fixture.pt"
    torch.save(
        {
            "split_seed": 7,
            "training_samples": len(train),
            "validation_samples": len(validation),
            "held_out_test_samples": len(test),
            "target_presence_threshold": 0.5,
            "training_session_fingerprint": fingerprint_sessions(tmp_path / "demos", train),
            "validation_session_fingerprint": fingerprint_sessions(
                tmp_path / "demos", validation
            ),
            "held_out_test_session_fingerprint": fingerprint_sessions(
                tmp_path / "demos", test
            ),
        },
        model_checkpoint,
    )

    class CenterModel(torch.nn.Module):
        input_size = 64

        @property
        def checkpoint_metadata(self):
            return {"target_presence_threshold": 0.5}

        def forward_with_presence(self, images, queries):
            logits = torch.zeros((images.shape[0], 1, 32, 32), device=images.device)
            logits[:, :, 15:17, 15:17] = 10.0
            presence_logits = images.mean(dim=(1, 2, 3)) * 20.0 - 2.0
            return logits, presence_logits

    monkeypatch.setattr(evaluator, "load_grounding_checkpoint", lambda _path: CenterModel())
    report = evaluator.evaluate_grounder(
        data_dir=str(tmp_path / "demos"),
        checkpoint_path=str(model_checkpoint),
    )

    assert report["held_out_test_samples"] == 2
    assert report["target_presence"]["false_activations"] == 0
    assert report["target_presence"]["false_activation_rate_wilson_95_percent"][1] > 0
    assert report["pck_1_percent_diagonal"] == 0
    assert report["pck_2_percent_diagonal"] == 1
    assert report["pck_5_percent_diagonal"] == 1
    assert report["baselines"]["screen_center"]["mean_normalized_point_distance"] == 0
    assert report["baselines"]["training_mean_click"]["mean_normalized_point_distance"] == 0
    assert report["model_beats_training_mean_baseline"] is False
    assert report["target_box_hit_rate"].startswith("not_measurable")
    assert report["llm_or_desktop_actions_used"] is False

    test[0].frame_path.write_bytes(test[0].frame_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="data changed since this checkpoint"):
        evaluator.evaluate_grounder(
            data_dir=str(tmp_path / "demos"),
            checkpoint_path=str(model_checkpoint),
        )
