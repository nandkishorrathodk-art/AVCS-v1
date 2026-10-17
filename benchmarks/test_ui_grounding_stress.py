"""Empirical stress-test harness for DataLoader throughput and UIGroundingBenchmarkRunner fail-closed behavior."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader

from avcs.dataset.loader import UIGroundingDataset, ui_grounding_collate_fn
from avcs.goal_grounder import GoalConditionedGrounder
from avcs.query_limits import MAX_QUERY_BYTES
from benchmarks.run_ui_grounding_benchmark import (
    GroundingBenchmarkResults,
    UIGroundingBenchmarkRunner,
)


def run_dataloader_throughput_benchmark(
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    repetitions: int = 16,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Measures DataLoader throughput over 4,000+ samples with batch_size=32."""
    t0_init = time.perf_counter()
    ds_preloaded = UIGroundingDataset(manifest_path=manifest_path, preload_frames=True)
    t_init = time.perf_counter() - t0_init

    ds_concat = ConcatDataset([ds_preloaded] * repetitions)
    total_target_samples = len(ds_concat)

    loader = DataLoader(
        ds_concat,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=ui_grounding_collate_fn,
    )

    # Warmup
    _ = next(iter(loader))

    # Measurement run
    t0 = time.perf_counter()
    samples_counted = 0
    batches_counted = 0
    for batch in loader:
        samples_counted += batch["image"].shape[0]
        batches_counted += 1
    t1 = time.perf_counter()
    elapsed = t1 - t0
    throughput = samples_counted / max(1e-6, elapsed)

    # Benchmark without preload (small sample comparison)
    ds_disk = UIGroundingDataset(manifest_path=manifest_path, preload_frames=False)
    loader_disk = DataLoader(
        ConcatDataset([ds_disk] * 2),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ui_grounding_collate_fn,
    )
    t0_d = time.perf_counter()
    disk_samples = 0
    for b in loader_disk:
        disk_samples += b["image"].shape[0]
        if disk_samples >= 256:
            break
    elapsed_d = time.perf_counter() - t0_d
    disk_throughput = disk_samples / max(1e-6, elapsed_d)

    return {
        "dataset_init_time_s": t_init,
        "total_samples": samples_counted,
        "total_batches": batches_counted,
        "elapsed_time_s": elapsed,
        "throughput_samples_per_sec": throughput,
        "disk_unpreloaded_samples_per_sec": disk_throughput,
        "speedup_ratio": throughput / max(1e-6, disk_throughput),
    }


class MockPredictPointModel:
    def predict_point(self, img: torch.Tensor, query: str) -> Dict[str, Any]:
        return {"point": (0.5, 0.5), "confidence": 0.95, "bbox": (0.4, 0.4, 0.6, 0.6)}


class MockLocateModel:
    class LocateResult:
        target_coord = (0.5, 0.5)
        confidence = 0.95
        bbox = (0.4, 0.4, 0.6, 0.6)

    def locate(self, query: str) -> Any:
        return self.LocateResult()


class MockCallableModel:
    def __call__(self, img: torch.Tensor, query: str) -> Tuple[Any, ...]:
        return ((0.5, 0.5), 0.95, (0.4, 0.4, 0.6, 0.6))


def run_fail_closed_query_stress_test(
    runner: UIGroundingBenchmarkRunner,
) -> Dict[str, Any]:
    """Evaluates fail-closed behavior across edge cases on multiple model backends."""
    dummy_img = torch.zeros(3, 256, 256)
    nn_model = GoalConditionedGrounder()
    predict_point_model = MockPredictPointModel()
    locate_model = MockLocateModel()
    callable_model = MockCallableModel()

    test_queries = [
        ("empty_string", ""),
        ("whitespace_only", "   \t\n"),
        ("none_type", None),
        ("integer_12345", 12345),
        ("10000_char_string", "A" * 10000),
        ("non_ascii_emojis", "\U0001f50d\U0001f3af\U0001f680"),
    ]

    results_matrix: Dict[str, Dict[str, Any]] = {}

    for name, q in test_queries:
        pt_nn, c_nn, _ = runner._infer_model(nn_model, dummy_img, q)
        pt_pp, c_pp, _ = runner._infer_model(predict_point_model, dummy_img, q)
        pt_loc, c_loc, _ = runner._infer_model(locate_model, dummy_img, q)
        pt_call, c_call, _ = runner._infer_model(callable_model, dummy_img, q)

        results_matrix[name] = {
            "nn_model_conf": float(c_nn),
            "nn_fails_closed_zero": bool(c_nn == 0.0),
            "mock_predict_point_conf": float(c_pp),
            "mock_pp_fails_closed_zero": bool(c_pp == 0.0),
            "mock_locate_conf": float(c_loc),
            "mock_locate_fails_closed_zero": bool(c_loc == 0.0),
            "mock_callable_conf": float(c_call),
            "mock_callable_fails_closed_zero": bool(c_call == 0.0),
        }

    return results_matrix


class CenterCollapseMockModel:
    def predict_point(self, img: torch.Tensor, query: str) -> Dict[str, Any]:
        return {"point": (0.5, 0.5), "confidence": 0.95, "bbox": (0.46, 0.46, 0.54, 0.54)}


def run_center_collapse_benchmark(
    runner: UIGroundingBenchmarkRunner,
) -> GroundingBenchmarkResults:
    """Evaluates center collapse mock model against held-out benchmark."""
    model = CenterCollapseMockModel()
    return runner.evaluate_model(model, split="test_held_out")


def build_state_aware_oracle(
    manifest_path: str = "datasets/ui_grounding/manifest.json",
) -> Any:
    """Builds a state-aware oracle model achieving 100% Top-1 accuracy and 0 px error."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    oracle_table: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    data_dir = Path(manifest_path).parent

    for fr in manifest["frames"]:
        if fr.get("split") == "test_held_out":
            img_p = data_dir / fr["file_path"]
            with Image.open(img_p) as pil_img:
                arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            img_t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)
            fg = hashlib.sha256(img_t.numpy().tobytes()).hexdigest()

            for pt in fr.get("interactive_points", []):
                q = pt.get("target_query", "")
                key = (fg, q)
                if key not in oracle_table:
                    oracle_table[key] = []
                oracle_table[key].append({
                    "point": (float(pt["x"]), float(pt["y"])),
                    "confidence": 1.0,
                    "bbox": tuple(pt.get("bbox", [0.4, 0.4, 0.6, 0.6])),
                })

    class StateAwareOracleModel:
        def __init__(self) -> None:
            self.call_counts: Dict[Tuple[str, str], int] = {}

        def predict_point(self, img: torch.Tensor, query: str) -> Dict[str, Any]:
            fg = hashlib.sha256(img.numpy().tobytes()).hexdigest()
            key = (fg, query)
            idx = self.call_counts.get(key, 0)
            self.call_counts[key] = idx + 1
            if key in oracle_table and idx < len(oracle_table[key]):
                return oracle_table[key][idx]
            # Negative / ungroundable queries return 0.0 confidence
            return {"point": (0.5, 0.5), "confidence": 0.0, "bbox": (0.46, 0.46, 0.54, 0.54)}

    return StateAwareOracleModel()


def main() -> None:
    manifest_path = "datasets/ui_grounding/manifest.json"
    print("=================================================================")
    print("CHALLENGER EMPIRICAL VERIFICATION HARNESS")
    print("=================================================================\n")

    # 1. DataLoader throughput benchmark
    print("[1/4] Running DataLoader CPU Throughput Benchmark...")
    bench_data = run_dataloader_throughput_benchmark(manifest_path=manifest_path, repetitions=16, batch_size=32)
    print(f"  Preload Dataset Init: {bench_data['dataset_init_time_s']:.3f} s")
    print(f"  Total Samples Loaded: {bench_data['total_samples']} ({bench_data['total_batches']} batches)")
    print(f"  Elapsed Time:         {bench_data['elapsed_time_s']:.4f} s")
    print(f"  Throughput (Preloaded): {bench_data['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  Throughput (From Disk): {bench_data['disk_unpreloaded_samples_per_sec']:.2f} samples/sec")
    print(f"  Speedup Ratio:        {bench_data['speedup_ratio']:.2f}x\n")

    # 2. Runner Fail-Closed Stress Test
    print("[2/4] Running Runner Fail-Closed Stress Test...")
    runner = UIGroundingBenchmarkRunner(manifest_path=manifest_path)
    fc_matrix = run_fail_closed_query_stress_test(runner)
    for q_name, q_data in fc_matrix.items():
        print(f"  Query: {q_name:<20}")
        print(f"    NN Model:         conf={q_data['nn_model_conf']:.6f} (zero: {q_data['nn_fails_closed_zero']})")
        print(f"    Mock PredictPoint: conf={q_data['mock_predict_point_conf']:.2f} (zero: {q_data['mock_pp_fails_closed_zero']})")
        print(f"    Mock Locate:       conf={q_data['mock_locate_conf']:.2f} (zero: {q_data['mock_locate_fails_closed_zero']})")
        print(f"    Mock Callable:     conf={q_data['mock_callable_conf']:.2f} (zero: {q_data['mock_callable_fails_closed_zero']})")
    print()

    # 3. Center-Collapse Mock Model
    print("[3/4] Running Center-Collapse Mock Model Evaluation...")
    cc_res = run_center_collapse_benchmark(runner)
    print(f"  Center Collapse Detected: {cc_res.center_collapse_detected}")
    print(f"  Top-1 Accuracy:           {cc_res.top_1_accuracy_pct:.2f}%")
    print(f"  Passed All Gates:         {cc_res.passed_all_gates}\n")

    # 4. Mock Perfect Model
    print("[4/4] Running 100% Mock Perfect Model Evaluation...")
    oracle_model = build_state_aware_oracle(manifest_path=manifest_path)
    oracle_res = runner.evaluate_model(oracle_model, split="test_held_out")
    print(f"  Top-1 Accuracy:           {oracle_res.top_1_accuracy_pct:.2f}% ({oracle_res.top_1_hits}/{oracle_res.total_positive_samples})")
    print(f"  Median Centroid Error:    {oracle_res.median_centroid_error_px:.2f} px")
    print(f"  Mean Centroid Error:      {oracle_res.mean_centroid_error_px:.2f} px")
    print(f"  P95 Centroid Error:       {oracle_res.p95_centroid_error_px:.2f} px")
    print(f"  Mean IoU:                 {oracle_res.mean_iou:.4f}")
    print(f"  Passed All Gates:         {oracle_res.passed_all_gates}\n")

    print("=================================================================")
    print("VERIFICATION COMPLETED")
    print("=================================================================")


if __name__ == "__main__":
    main()
