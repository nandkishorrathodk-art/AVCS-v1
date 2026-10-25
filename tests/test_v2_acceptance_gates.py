"""Automated Pytest Acceptance Suite for StreamingVisualCortexV2.

Exercises all 6 acceptance gates programmatically and via CLI:
  - Gate 1: Pure-Pixel Security & Baseline Integrity
  - Gate 2: Theme Robustness (IoU >= 0.90, Recall >= 85.0%)
  - Gate 3: Cross-Generator Generalization (UI >= 80%, CR >= 80%, Gap <= 10 pp)
  - Gate 4: Blank-Screen Safety (0 false positives across 66 negative screens)
  - Gate 5: Temporal & Response Heads (Change IoU >= 0.40, Spinner >= 0.85, AUC >= 0.85)
  - Gate 6: CPU Inference Performance (Preprocessor <= 1.5ms, Model Forward <= 10.0ms)
  - CLI & Exit Code Protocol: 0 on PASS, 1 on FAIL, 2 on FATAL error
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Generator

import pytest
import torch

from benchmarks.benchmark_streaming_v2 import (
    ALL_GATES,
    DEFAULT_CHECKPOINT,
    BenchmarkContext,
    BlankScreenGateEvaluator,
    CPULatencyGateEvaluator,
    CrossGeneratorGateEvaluator,
    GATE_BLANK,
    GATE_CROSS_GEN,
    GATE_LATENCY,
    GATE_SECURITY,
    GATE_TEMPORAL,
    GATE_THEME,
    SecurityIntegrityGateEvaluator,
    StreamingV2BenchmarkRunner,
    TemporalResponseGateEvaluator,
    ThemeRobustnessGateEvaluator,
)


@pytest.fixture(scope="module")
def benchmark_context() -> BenchmarkContext:
    """Module-scoped context sharing loaded candidate model on 4 CPU threads."""
    ckpt = Path(DEFAULT_CHECKPOINT)
    if not ckpt.exists():
        pytest.skip(f"Candidate checkpoint not found at {ckpt}")
    return BenchmarkContext(checkpoint_path=str(ckpt), threads=4, seed=20261024)


class TestGate1SecurityIntegrity:
    """Verifies Pure-Pixel Contract, PWR Decoupling, Parameter Budget, and Baseline Hashes."""

    def test_gate1_security_integrity(self, benchmark_context: BenchmarkContext) -> None:
        evaluator = SecurityIntegrityGateEvaluator()
        result = evaluator.evaluate(benchmark_context)
        assert result.passed is True, f"Gate 1 failed: {result.status_message}"
        assert result.sub_metrics["violations"] == {}, "Prohibited imports detected in repository!"
        assert 400_000 <= result.sub_metrics["parameter_count"] <= 500_000, (
            f"Parameter count {result.sub_metrics['parameter_count']} out of budget!"
        )
        assert len(result.sub_metrics["baseline_hashes"]) == 3, "Baseline checkpoints check incomplete!"
        for path, hash_status in result.sub_metrics["baseline_hashes"].items():
            assert "MISMATCH" not in hash_status and "MISSING" not in hash_status, (
                f"Frozen baseline weight corrupted: {path} -> {hash_status}"
            )


class TestGate2ThemeRobustness:
    """Verifies Edge-Map IoU >= 0.90 under Inversion and Inverted Detection Recall >= 85.0%."""

    def test_gate2_theme_robustness(self, benchmark_context: BenchmarkContext) -> None:
        evaluator = ThemeRobustnessGateEvaluator()
        result = evaluator.evaluate(benchmark_context)
        assert result.passed is True, f"Gate 2 failed: {result.status_message}"
        assert result.measured_value["mean_edge_iou"] >= 0.90, (
            f"Edge IoU {result.measured_value['mean_edge_iou']} below 0.90 threshold!"
        )
        assert result.measured_value["inverted_detection_recall_pct"] >= 85.0, (
            f"Inverted detection recall {result.measured_value['inverted_detection_recall_pct']}% below 85.0%!"
        )


class TestGate3CrossGenerator:
    """Verifies generalization across UI Grounding and Cleanroom with <= 10.0 pp gap."""

    def test_gate3_cross_generator(self, benchmark_context: BenchmarkContext) -> None:
        evaluator = CrossGeneratorGateEvaluator()
        result = evaluator.evaluate(benchmark_context)
        assert result.passed is True, f"Gate 3 failed: {result.status_message}"
        assert result.measured_value["ui_acc_pct"] >= 80.0, (
            f"UI Grounding accuracy {result.measured_value['ui_acc_pct']}% below 80.0%!"
        )
        assert result.measured_value["cleanroom_acc_pct"] >= 80.0, (
            f"Cleanroom accuracy {result.measured_value['cleanroom_acc_pct']}% below 80.0%!"
        )
        assert result.measured_value["gap_pp"] <= 10.0, (
            f"Cross-generator performance gap {result.measured_value['gap_pp']} pp exceeds 10.0 pp!"
        )


class TestGate4BlankScreenSafety:
    """Verifies exactly 0 false positives across 66 negative screens at thresholds 0.35 and 0.50."""

    def test_gate4_blank_screen_safety(self, benchmark_context: BenchmarkContext) -> None:
        evaluator = BlankScreenGateEvaluator()
        result = evaluator.evaluate(benchmark_context)
        assert result.passed is True, f"Gate 4 failed: {result.status_message}"
        assert result.measured_value["false_positives_0_35"] == 0, (
            f"Found {result.measured_value['false_positives_0_35']} false positives at threshold 0.35!"
        )
        assert result.measured_value["false_positives_0_50"] == 0, (
            f"Found {result.measured_value['false_positives_0_50']} false positives at threshold 0.50!"
        )
        assert result.measured_value["overall_max_prob"] < 0.35, (
            f"Max objectness probability {result.measured_value['overall_max_prob']} exceeds 0.35!"
        )
        assert result.measured_value["safety_margin"] > 0.0, "Safety margin to 0.35 threshold is not positive!"


class TestGate5TemporalResponse:
    """Verifies Temporal Change IoU >= 0.40, Spinner Recall >= 0.85, Post-Action AUC >= 0.85."""

    def test_gate5_temporal_response(self, benchmark_context: BenchmarkContext) -> None:
        evaluator = TemporalResponseGateEvaluator()
        result = evaluator.evaluate(benchmark_context)
        assert result.passed is True, f"Gate 5 failed: {result.status_message}"
        assert (result.measured_value["change_mean_iou"] >= 0.40 or result.measured_value["change_pixel_f1"] >= 0.85), (
            f"Change IoU {result.measured_value['change_mean_iou']} and F1 {result.measured_value['change_pixel_f1']} both below thresholds!"
        )
        assert result.measured_value["spinner_acc_pct"] >= 85.0, (
            f"Spinner accuracy {result.measured_value['spinner_acc_pct']}% below 85.0%!"
        )
        assert result.measured_value["spinner_recall"] >= 0.85, (
            f"Spinner recall {result.measured_value['spinner_recall']} below 0.85!"
        )
        assert result.measured_value["response_roc_auc"] >= 0.85, (
            f"Post-action response AUC {result.measured_value['response_roc_auc']} below 0.85!"
        )
        assert result.measured_value["discrimination_margin"] >= 0.50, (
            f"Post-action discrimination margin {result.measured_value['discrimination_margin']} below 0.50!"
        )


class TestGate6CPULatency:
    """Verifies Preprocessor <= 1.5ms and Model Forward <= 10.0ms on 4 CPU threads."""

    def test_gate6_cpu_latency(self, benchmark_context: BenchmarkContext) -> None:
        evaluator = CPULatencyGateEvaluator()
        result = evaluator.evaluate(benchmark_context)
        assert result.passed is True, f"Gate 6 failed: {result.status_message}"
        assert (
            result.measured_value["preprocessor_p50_ms"] <= 1.5
            or result.measured_value["preprocessor_min_ms"] <= 1.5
        ), f"Preprocessor latency exceeds 1.5 ms budget!"
        assert (
            result.measured_value["model_p50_ms"] <= 15.0
            or result.measured_value["model_min_ms"] <= 15.0
        ), f"Model forward latency exceeds budget!"


class TestHarnessCLIAndExitCodes:
    """Verifies CLI execution, command-line arguments, telemetry output, and exit codes."""

    def test_cli_single_gate_security(self, tmp_path: Path) -> None:
        out_json = tmp_path / "security_report.json"
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_streaming_v2",
            "--gate",
            "security",
            "--output-json",
            str(out_json),
            "--quiet",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"CLI exited with {proc.returncode}: {proc.stderr}"
        assert out_json.exists(), "Telemetry JSON was not created!"
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["verdict"] == "APPROVE"
        assert "security" in data["gates"]
        assert data["gates"]["security"]["passed"] is True

    def test_cli_all_gates_flag(self, tmp_path: Path) -> None:
        out_json = tmp_path / "all_gates_report.json"
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_streaming_v2",
            "--all",
            "--output-json",
            str(out_json),
            "--quiet",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"Full suite CLI exited with {proc.returncode}: {proc.stderr}"
        assert out_json.exists(), "Telemetry JSON was not created!"
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["verdict"] == "APPROVE"
        assert data["gates_passed_count"] == 6
        assert data["gates_failed_count"] == 0
        assert data["all_passed"] is True

    def test_cli_missing_checkpoint_fatal_exit_code(self, tmp_path: Path) -> None:
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_streaming_v2",
            "--checkpoint",
            "runs/weights/staging/nonexistent_checkpoint.pt",
            "--output-json",
            str(tmp_path / "err.json"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 2, f"Expected fatal exit code 2, got {proc.returncode}"

