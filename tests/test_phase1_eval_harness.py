"""Contract and functional tests for Phase 1 Evaluation Harness and Gate 1."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from avcs.eval import (
    OSWorldObservationContract,
    bootstrap_confidence_interval,
    generate_osworld_smoke_tasks,
    generate_synthetic_screenspot_pro_suite,
    generate_synthetic_showdown_suite,
    load_showdown_clicks_dev,
    run_benchmark,
    run_osworld_eval,
    run_phase1_gate,
    run_screenspot_pro,
    verify_mock_agent_lifecycle,
    verify_protocol_roundtrip,
)
from avcs.models import LocalVAMClient
from avcs.runtime.agent import AceAgent


def _offline_agent() -> AceAgent:
    return AceAgent(
        client=LocalVAMClient(
            endpoint="http://127.0.0.1:0/v1",
            model="test-offline-mock",
            allow_offline_mock=True,
        )  # type: ignore[arg-type]
    )


def test_phase1_configs_exist_and_parse():
    cfg_eval = Path("configs/phase1_eval.yaml")
    cfg_model = Path("configs/model_pure_rgb_vla.yaml")
    assert cfg_eval.exists()
    assert cfg_model.exists()
    text = cfg_eval.read_text(encoding="utf-8")
    assert 'mode: "pure_rgb"' in text
    assert "use_a11y_tree: false" in text


def test_protocol_deterministic_roundtrip_all_10_primitives():
    res = verify_protocol_roundtrip()
    assert res["passed"] is True
    assert res["primitives_verified"] == 10


def test_mock_agent_lifecycle_start_plan_reset_close():
    res = verify_mock_agent_lifecycle()
    assert res["passed"] is True
    assert res["errors"] == 0
    assert res["closed_guard_verified"] is True


def test_showdown_bootstrap_ci_and_csv_loader(tmp_path: Path):
    low, high = bootstrap_confidence_interval([True] * 8 + [False] * 2, n_resamples=400, seed=7)
    assert 0.4 <= low <= 0.8 <= high <= 1.0

    # Create a tiny synthetic showdown CSV + JPEG
    img_path = tmp_path / "shot.jpg"
    Image.new("RGB", (1920, 1080), color=(30, 40, 50)).save(img_path, format="JPEG")
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "id,instruction,x1,y1,x2,y2,width,height,image\n"
        "row1,Click Submit,960,540,1056,594,1920,1080,shot.jpg\n",
        encoding="utf-8",
    )

    samples = load_showdown_clicks_dev(str(csv_path))
    assert len(samples) == 1
    assert samples[0].target_bbox == (640, 400, 704, 440)  # Scaled to 1280x800
    assert samples[0].image_bytes is not None

    agent = _offline_agent()
    report = run_benchmark(agent=agent, samples=generate_synthetic_showdown_suite(count=10), tolerance_px=15.0)
    assert "bootstrap_95_ci" in report
    assert "strict_accuracy" in report
    assert "latency_p95_ms" in report
    assert report["sample_count"] == 10


def test_screenspot_pro_smoke_and_small_icon_recall():
    agent = _offline_agent()
    samples = generate_synthetic_screenspot_pro_suite(count=12)
    report = run_screenspot_pro(agent=agent, samples=samples, tolerance_px=15.0)
    assert report["observation_mode"] == "pure_rgb"
    assert report["sample_count"] == 12
    assert report["small_icon_count"] >= 8
    assert "small_icon_recall" in report
    assert "bootstrap_95_ci" in report
    assert "latency_p95_ms" in report


def test_osworld_pure_rgb_contract_and_30_task_smoke(tmp_path: Path):
    with pytest.raises(ValueError, match="forbidden"):
        OSWorldObservationContract(observation_mode="pure_rgb", use_a11y_tree=True)
    with pytest.raises(ValueError, match="pure_rgb"):
        OSWorldObservationContract(observation_mode="a11y_tree")

    agent = _offline_agent()
    tasks = generate_osworld_smoke_tasks(count=30)
    log_path = tmp_path / "osworld_rollouts.jsonl"
    report = run_osworld_eval(agent=agent, tasks=tasks, rollout_log_path=str(log_path))

    assert report["task_count"] == 30
    assert report["observation_mode"] == "pure_rgb"
    assert report["env_flags"]["require_a11y_tree"] is False
    assert report["deterministic_reset_rate"] == 100.0
    assert report["error_count"] == 0
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 30


def test_phase1_gate_runner_outputs_leaderboard(tmp_path: Path):
    out_file = tmp_path / "leaderboard.json"
    leaderboard = run_phase1_gate(output_path=str(out_file), smoke=True)
    assert leaderboard["gate1_passed"] is True
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["checks"]["protocol_roundtrip"]["primitives_verified"] == 10
    assert loaded["measured_local"]["osworld_verified_smoke"]["task_count"] == 30
    assert len(loaded["published_literature_baselines"]) >= 5
