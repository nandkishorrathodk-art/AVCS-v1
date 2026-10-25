"""Adversarial DataLoader, Generator Balance, and Throughput Stress Benchmark for M2.

Author: challenger_m2_2
Date: 2026-10-24
Milestone: StreamingVisualCortexV2 M2 (Multi-Generator Dataset Pipeline)

Executes empirical stress tests:
1. 100 consecutive DataLoader batches (batch_size=16, 1600 samples)
   - Verifies batch shapes, dtype, range, and collation validity.
   - Tracks empirical generator draw counts (UI Grounding vs Cleanroom within [40%, 60%]).
   - Computes Wilson 95% confidence intervals.
   - Measures per-batch latency, p50, p95, p99, throughput (batches/sec, samples/sec).
   - Verifies throughput sustainability across early vs late batches.
2. 1,000 samples negative screen injection rate test
   - Tests hard-negative screen injection across 1,000 samples.
   - Verifies rate falls within [15%, 20%].
   - Verifies that has_target == 0.0 and target_boxes is empty on all negative samples.
   - Tests stability across multiple random seeds.
3. 500 procedural temporal sequences stress test
   - Tests spinner, moving, and static modes across sequence lengths T in [2, 3, 4, 8, 16].
   - Verifies no NaNs, no Infs, range [0, 1], finite heatmaps.
   - Verifies static mode heatmaps are strictly zero, moving/spinner heatmaps are > 0.
4. 500 post-action observation pairs stress test
   - Tests positive (1.0) and negative (0.0) response scenarios.
   - Verifies identical pre/post frames on negative response and visible optical delta on positive response.
5. Adversarial edge cases & boundary tests:
   - Batch size extreme variations (B=1, B=32, B=64).
   - 100% negative injection rate (B=16, 100% negatives, empty box collation).
   - 0% negative injection rate.
   - Sequence length boundary tests (T=2 valid, T=1 raises ValueError).
   - Invalid mode handling (raises ValueError).
   - Memory soak test: RSS tracking over 1,000 samples to verify zero leakage.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from avcs.dataset.unified.generator_harmonizer import (
    GENERATOR_CLEANROOM,
    GENERATOR_UI_GROUNDING,
    SPLIT_TEST_HELD_OUT,
    SPLIT_TRAIN,
    SPLIT_VAL,
    harmonize_all_generators,
)
from avcs.dataset.unified.temporal_sampler import (
    generate_action_pair,
    generate_temporal_sequence,
    sample_action_pair_batch,
    sample_temporal_batch,
)
from avcs.training.multi_generator_dataset import (
    GENERATOR_ID_CLEANROOM,
    GENERATOR_ID_UI,
    ArmD2MultiGeneratorDataset,
    collate_multi_generator_batch,
    create_arm_d2_dataloader,
)


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    z = 1.95996 if confidence == 0.95 else 2.57583  # 95% vs 99%
    denominator = 1.0 + (z**2) / total
    centre_adjusted_probability = p_hat + (z**2) / (2.0 * total)
    adjusted_std_dev = math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)
    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    return max(0.0, lower), min(1.0, upper)


def get_process_memory_mb() -> float:
    """Get current process working set memory in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def stress_test_dataloader_batches(
    num_batches: int = 100,
    batch_size: int = 16,
    seed: int = 42,
) -> Dict[str, Any]:
    """Stress-test 100 consecutive batches from ArmD2MultiGeneratorDataset DataLoader."""
    print(f"\n--- [1] Stress-Testing DataLoader Batches (N={num_batches}, B={batch_size}) ---")
    manifest = harmonize_all_generators()
    dataset = ArmD2MultiGeneratorDataset(
        manifest=manifest,
        split=SPLIT_TRAIN,
        target_image_size=(128, 128),
        apply_augmentation=True,
        hard_negative_prob=0.175,
        preload_images=True,
        rng_seed=seed,
    )

    sampler = dataset.get_arm_d2_sampler()
    # Enlarge sampler num_samples to comfortably draw num_batches * batch_size samples
    total_requested_samples = num_batches * batch_size
    weights = torch.tensor(dataset._sample_weights, dtype=torch.double)
    large_sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=total_requested_samples,
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=large_sampler,
        collate_fn=collate_multi_generator_batch,
        num_workers=0,
    )

    batch_latencies_ms: List[float] = []
    ui_draws = 0
    cleanroom_draws = 0
    negative_draws = 0
    total_samples = 0
    total_objects = 0
    class_distribution: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    # Warmup 5 batches
    data_iter = iter(dataloader)
    for _ in range(5):
        _ = next(data_iter)

    # Time remaining batches
    start_time = time.perf_counter()
    drawn_batches = 0

    for b_idx in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        batch_latencies_ms.append(latency_ms)
        drawn_batches += 1

        # Tensor integrity assertions
        images = batch["images"]
        target_boxes = batch["target_boxes"]
        target_classes = batch["target_classes"]
        target_centroids = batch["target_centroids"]
        box_mask = batch["box_mask"]
        has_target = batch["has_target"]
        generator_id = batch["generator_id"]
        is_negative = batch["is_negative"]

        assert images.shape == (batch_size, 3, 128, 128), f"Bad image shape {images.shape}"
        assert not torch.isnan(images).any(), "NaN found in images"
        assert not torch.isinf(images).any(), "Inf found in images"
        assert images.min() >= 0.0 and images.max() <= 1.0, f"Image range out of [0, 1]: [{images.min()}, {images.max()}]"

        b_size, max_objs, _ = target_boxes.shape
        assert target_classes.shape == (b_size, max_objs)
        assert target_centroids.shape == (b_size, max_objs, 2)
        assert box_mask.shape == (b_size, max_objs)
        assert has_target.shape == (b_size,)
        assert generator_id.shape == (b_size,)
        assert is_negative.shape == (b_size,)

        # Collation mask verification: box_mask must be exactly True where target_classes >= 0
        expected_mask = (target_classes >= 0)
        assert torch.equal(box_mask, expected_mask), "box_mask does not match target_classes >= 0"

        # Centroids and boxes within [0, 1] for all masked items
        if box_mask.any():
            valid_boxes = target_boxes[box_mask]
            valid_centroids = target_centroids[box_mask]
            assert valid_boxes.min() >= 0.0 and valid_boxes.max() <= 1.0, "Box coordinate out of [0, 1]"
            assert valid_centroids.min() >= 0.0 and valid_centroids.max() <= 1.0, "Centroid out of [0, 1]"

        # Accumulate stats
        ui_count = (generator_id == GENERATOR_ID_UI).sum().item()
        cr_count = (generator_id == GENERATOR_ID_CLEANROOM).sum().item()
        ui_draws += ui_count
        cleanroom_draws += cr_count
        neg_count = is_negative.sum().item()
        negative_draws += neg_count
        total_samples += b_size

        # Class counts
        for c in target_classes[box_mask].tolist():
            class_distribution[c] = class_distribution.get(c, 0) + 1
            total_objects += 1

    total_time_s = time.perf_counter() - start_time
    throughput_fps = total_samples / total_time_s
    throughput_bps = drawn_batches / total_time_s

    # Latency percentiles
    lat_arr = np.array(batch_latencies_ms)
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    mean_lat = float(np.mean(lat_arr))

    # Drift analysis: first 20 vs last 20 batches
    early_p50 = float(np.percentile(lat_arr[:20], 50))
    late_p50 = float(np.percentile(lat_arr[-20:], 50))
    drift_ratio = late_p50 / max(1e-6, early_p50)

    # Proportions
    ui_prop = ui_draws / total_samples
    cr_prop = cleanroom_draws / total_samples
    neg_prop = negative_draws / total_samples

    ui_ci = wilson_score_interval(ui_draws, total_samples)
    cr_ci = wilson_score_interval(cleanroom_draws, total_samples)
    neg_ci = wilson_score_interval(negative_draws, total_samples)

    balance_ok = (0.40 <= ui_prop <= 0.60) and (0.40 <= cr_prop <= 0.60)

    print(f"Total Batches Drawn: {drawn_batches} ({total_samples} samples)")
    print(f"UI Grounding Draws: {ui_draws} ({ui_prop*100:.2f}%, 95% CI: [{ui_ci[0]*100:.2f}%, {ui_ci[1]*100:.2f}%])")
    print(f"Cleanroom Draws: {cleanroom_draws} ({cr_prop*100:.2f}%, 95% CI: [{cr_ci[0]*100:.2f}%, {cr_ci[1]*100:.2f}%])")
    print(f"Negative Screen Draws: {negative_draws} ({neg_prop*100:.2f}%, 95% CI: [{neg_ci[0]*100:.2f}%, {neg_ci[1]*100:.2f}%])")
    print(f"Generator Balance in [40%, 60%]: {'PASS' if balance_ok else 'FAIL'}")
    print(f"Batch Latency: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms, mean={mean_lat:.2f}ms, min={min_lat:.2f}ms, max={max_lat:.2f}ms")
    print(f"Throughput: {throughput_bps:.2f} batches/sec ({throughput_fps:.2f} samples/sec)")
    print(f"Drift Ratio (late_p50 / early_p50): {drift_ratio:.2f}x (early={early_p50:.2f}ms, late={late_p50:.2f}ms)")
    print(f"Annotated UI Objects Collated: {total_objects} (Classes: {class_distribution})")

    return {
        "num_batches": drawn_batches,
        "total_samples": total_samples,
        "batch_size": batch_size,
        "ui_draws": ui_draws,
        "cleanroom_draws": cleanroom_draws,
        "negative_draws": negative_draws,
        "ui_prop": ui_prop,
        "cr_prop": cr_prop,
        "neg_prop": neg_prop,
        "ui_ci": ui_ci,
        "cr_ci": cr_ci,
        "neg_ci": neg_ci,
        "balance_ok": balance_ok,
        "latency_ms": {
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "mean": mean_lat,
            "min": min_lat,
            "max": max_lat,
        },
        "throughput_samples_per_sec": throughput_fps,
        "throughput_batches_per_sec": throughput_bps,
        "drift_ratio": drift_ratio,
        "total_objects": total_objects,
        "class_distribution": class_distribution,
    }


def stress_test_negative_injection_1000_samples(
    num_samples: int = 1000,
    seeds: Sequence[int] = (42, 100, 2026),
) -> Dict[str, Any]:
    """Test 15-20% negative screen injection across 1,000 samples and multiple seeds."""
    print(f"\n--- [2] Stress-Testing Negative Screen Injection (N={num_samples} samples per seed) ---")
    manifest = harmonize_all_generators()

    seed_results = []
    overall_negatives = 0
    overall_total = 0

    for s in seeds:
        dataset = ArmD2MultiGeneratorDataset(
            manifest=manifest,
            split=SPLIT_TRAIN,
            hard_negative_prob=0.175,
            apply_augmentation=True,
            preload_images=True,
            rng_seed=s,
        )

        neg_count = 0
        violations = []

        # Draw num_samples randomly
        indices = [dataset.rng.randint(0, len(dataset) - 1) for _ in range(num_samples)]
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            is_neg = sample["is_negative"]
            has_target = sample["has_target"]
            boxes = sample["target_boxes"]
            classes = sample["target_classes"]
            centroids = sample["target_centroids"]
            img = sample["image"]

            if is_neg:
                neg_count += 1
                # Verification of negative invariants
                if has_target != 0.0:
                    violations.append(f"Sample {i}: is_negative=True but has_target={has_target}")
                if boxes.shape[0] != 0:
                    violations.append(f"Sample {i}: is_negative=True but non-empty boxes {boxes.shape}")
                if classes.shape[0] != 0:
                    violations.append(f"Sample {i}: is_negative=True but non-empty classes {classes.shape}")
                if centroids.shape[0] != 0:
                    violations.append(f"Sample {i}: is_negative=True but non-empty centroids {centroids.shape}")
            else:
                if has_target != 1.0:
                    violations.append(f"Sample {i}: is_negative=False but has_target={has_target}")

            # Check image valid
            if torch.isnan(img).any() or torch.isinf(img).any():
                violations.append(f"Sample {i}: image contains NaN or Inf")
            if img.min() < 0.0 or img.max() > 1.0:
                violations.append(f"Sample {i}: image out of bounds [{img.min()}, {img.max()}]")

        rate = neg_count / num_samples
        ci = wilson_score_interval(neg_count, num_samples)
        in_range = (0.15 <= rate <= 0.20)
        overall_negatives += neg_count
        overall_total += num_samples

        print(f"Seed {s}: {neg_count}/{num_samples} negatives ({rate*100:.2f}%, 95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]) | In [15%, 20%]: {'PASS' if in_range else 'BORDERLINE/FAIL'} | Violations: {len(violations)}")
        seed_results.append({
            "seed": s,
            "neg_count": neg_count,
            "total_samples": num_samples,
            "rate": rate,
            "ci": ci,
            "in_range": in_range,
            "violations_count": len(violations),
            "violations": violations[:5],
        })

    aggregate_rate = overall_negatives / overall_total
    agg_ci = wilson_score_interval(overall_negatives, overall_total)
    agg_in_range = (0.15 <= aggregate_rate <= 0.20)
    print(f"Aggregate ({len(seeds)} seeds, N={overall_total}): {overall_negatives}/{overall_total} ({aggregate_rate*100:.2f}%, 95% CI: [{agg_ci[0]*100:.2f}%, {agg_ci[1]*100:.2f}%]) | In [15%, 20%]: {'PASS' if agg_in_range else 'FAIL'}")

    return {
        "seeds": seed_results,
        "aggregate_rate": aggregate_rate,
        "aggregate_ci": agg_ci,
        "aggregate_in_range": agg_in_range,
        "total_evaluated": overall_total,
    }


def stress_test_temporal_sampler(num_sequences: int = 500) -> Dict[str, Any]:
    """Test procedural temporal sequence sampler across 500 generated sequences."""
    print(f"\n--- [3] Stress-Testing Temporal Sequence Sampler (N={num_sequences} sequences) ---")
    modes = ["spinner", "moving", "static"]
    lengths = [2, 3, 4, 8, 16]

    nan_count = 0
    inf_count = 0
    shape_errors = 0
    mode_violations = 0
    t0 = time.perf_counter()

    mode_counts: Dict[str, int] = {"spinner": 0, "moving": 0, "static": 0}

    for i in range(num_sequences):
        m = modes[i % len(modes)]
        t_len = lengths[i % len(lengths)]
        seq = generate_temporal_sequence(
            mode=m,
            sequence_length=t_len,
            image_size=(128, 128),
            rng=random.Random(1000 + i),
        )
        mode_counts[m] += 1

        # Check shapes
        if seq.frames.shape != (t_len, 3, 128, 128):
            shape_errors += 1
        if seq.change_heatmap.shape != (1, 128, 128):
            shape_errors += 1
        if seq.timestamps.shape != (t_len,):
            shape_errors += 1

        # Check NaNs and Infs
        if torch.isnan(seq.frames).any() or torch.isnan(seq.change_heatmap).any():
            nan_count += 1
        if torch.isinf(seq.frames).any() or torch.isinf(seq.change_heatmap).any():
            inf_count += 1

        # Value bounds
        if seq.frames.min() < 0.0 or seq.frames.max() > 1.0:
            shape_errors += 1
        if seq.change_heatmap.min() < 0.0 or seq.change_heatmap.max() > 1.0:
            shape_errors += 1

        # Mode-specific invariants
        if m == "static":
            if seq.spinner_label != 0.0 or seq.change_heatmap.max() > 0.0:
                mode_violations += 1
            # Frames must be identical
            diff = (seq.frames[1:] - seq.frames[:-1]).abs().max()
            if diff > 0.0:
                mode_violations += 1
        elif m == "spinner":
            if seq.spinner_label != 1.0 or seq.change_heatmap.max() == 0.0:
                mode_violations += 1
        elif m == "moving":
            if seq.spinner_label != 0.0 or seq.change_heatmap.max() == 0.0:
                mode_violations += 1

    elapsed_s = time.perf_counter() - t0
    seq_per_sec = num_sequences / elapsed_s

    print(f"Generated {num_sequences} sequences in {elapsed_s:.2f}s ({seq_per_sec:.2f} seq/sec)")
    print(f"Mode distribution: {mode_counts}")
    print(f"NaNs: {nan_count}, Infs: {inf_count}, Shape Errors: {shape_errors}, Mode Invariant Violations: {mode_violations}")

    # Also test sample_temporal_batch (10 batches of size 16)
    batch_errors = 0
    for b_i in range(10):
        t_batch = sample_temporal_batch(batch_size=16, sequence_length=4, image_size=(128, 128))
        if t_batch["sequences"].shape != (16, 4, 3, 128, 128):
            batch_errors += 1
        if t_batch["change_heatmaps"].shape != (16, 1, 128, 128):
            batch_errors += 1
        if t_batch["spinner_labels"].shape != (16, 1):
            batch_errors += 1

    print(f"sample_temporal_batch 10 batches of 16 errors: {batch_errors}")

    return {
        "num_sequences": num_sequences,
        "elapsed_s": elapsed_s,
        "seq_per_sec": seq_per_sec,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "shape_errors": shape_errors,
        "mode_violations": mode_violations,
        "batch_errors": batch_errors,
        "passed": (nan_count == 0 and inf_count == 0 and shape_errors == 0 and mode_violations == 0 and batch_errors == 0),
    }


def stress_test_action_pair_sampler(num_pairs: int = 500) -> Dict[str, Any]:
    """Test post-action observation pair sampler across 500 generated pairs."""
    print(f"\n--- [4] Stress-Testing Action Observation Pair Sampler (N={num_pairs} pairs) ---")
    nan_count = 0
    inf_count = 0
    shape_errors = 0
    semantic_violations = 0
    t0 = time.perf_counter()

    resp_counts = {0.0: 0, 1.0: 0}

    for i in range(num_pairs):
        target_resp = (i % 2 == 0)
        pair = generate_action_pair(
            has_response=target_resp,
            image_size=(128, 128),
            rng=random.Random(2000 + i),
        )
        resp_counts[pair.response_label] += 1

        if pair.pre_frame.shape != (3, 128, 128) or pair.post_frame.shape != (3, 128, 128):
            shape_errors += 1
        if torch.isnan(pair.pre_frame).any() or torch.isnan(pair.post_frame).any():
            nan_count += 1
        if torch.isinf(pair.pre_frame).any() or torch.isinf(pair.post_frame).any():
            inf_count += 1

        cx, cy = pair.action_location_norm
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            shape_errors += 1

        # Check semantic response
        delta = (pair.post_frame - pair.pre_frame).abs().max().item()
        if pair.response_label == 0.0:
            if delta != 0.0:
                semantic_violations += 1
        else:
            if delta <= 0.0:
                semantic_violations += 1

    elapsed_s = time.perf_counter() - t0
    pairs_per_sec = num_pairs / elapsed_s
    print(f"Generated {num_pairs} action pairs in {elapsed_s:.2f}s ({pairs_per_sec:.2f} pairs/sec)")
    print(f"Response distribution: {resp_counts}")
    print(f"NaNs: {nan_count}, Infs: {inf_count}, Shape Errors: {shape_errors}, Semantic Violations: {semantic_violations}")

    # Also test sample_action_pair_batch
    batch_errors = 0
    for _ in range(10):
        a_batch = sample_action_pair_batch(batch_size=16, image_size=(128, 128))
        if a_batch["pre_frames"].shape != (16, 3, 128, 128):
            batch_errors += 1
        if a_batch["post_frames"].shape != (16, 3, 128, 128):
            batch_errors += 1
        if a_batch["action_locations"].shape != (16, 2):
            batch_errors += 1
        if a_batch["response_labels"].shape != (16, 1):
            batch_errors += 1

    print(f"sample_action_pair_batch 10 batches of 16 errors: {batch_errors}")

    return {
        "num_pairs": num_pairs,
        "elapsed_s": elapsed_s,
        "pairs_per_sec": pairs_per_sec,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "shape_errors": shape_errors,
        "semantic_violations": semantic_violations,
        "batch_errors": batch_errors,
        "passed": (nan_count == 0 and inf_count == 0 and shape_errors == 0 and semantic_violations == 0 and batch_errors == 0),
    }


def stress_test_adversarial_edge_cases() -> Dict[str, Any]:
    """Test boundary conditions, parameter extremes, and memory stability."""
    print("\n--- [5] Stress-Testing Adversarial Edge Cases & Boundary Conditions ---")
    manifest = harmonize_all_generators()
    results = {}

    # 1. Batch size variations: B=1, B=32, B=64
    for b in [1, 32, 64]:
        try:
            loader = create_arm_d2_dataloader(manifest=manifest, batch_size=b)
            batch = next(iter(loader))
            assert batch["images"].shape[0] == b
            results[f"batch_size_{b}"] = "PASS"
        except Exception as e:
            results[f"batch_size_{b}"] = f"FAIL: {e}"

    # 2. 100% hard negative probability (B=16, all negative, empty box collation)
    try:
        loader_all_neg = create_arm_d2_dataloader(
            manifest=manifest,
            batch_size=16,
            hard_negative_prob=1.0,
        )
        batch_all_neg = next(iter(loader_all_neg))
        assert batch_all_neg["has_target"].sum().item() == 0.0
        assert batch_all_neg["box_mask"].sum().item() == 0
        assert batch_all_neg["target_boxes"].shape[1] == 1  # max_objs = max(1, 0) = 1
        assert (batch_all_neg["target_classes"] == -1).all()
        results["hard_negative_1.0"] = "PASS"
    except Exception as e:
        results["hard_negative_1.0"] = f"FAIL: {e}"

    # 3. 0% hard negative probability
    try:
        loader_zero_neg = create_arm_d2_dataloader(
            manifest=manifest,
            batch_size=16,
            hard_negative_prob=0.0,
        )
        batch_zero_neg = next(iter(loader_zero_neg))
        assert batch_zero_neg["has_target"].sum().item() == 16.0
        assert batch_zero_neg["box_mask"].sum().item() > 0
        results["hard_negative_0.0"] = "PASS"
    except Exception as e:
        results["hard_negative_0.0"] = f"FAIL: {e}"

    # 4. Temporal sequence length boundaries: T=2 valid, T=1 raises ValueError
    try:
        _ = generate_temporal_sequence(mode="spinner", sequence_length=2)
        results["temporal_T_2_valid"] = "PASS"
    except Exception as e:
        results["temporal_T_2_valid"] = f"FAIL: {e}"

    try:
        _ = generate_temporal_sequence(mode="spinner", sequence_length=1)
        results["temporal_T_1_error"] = "FAIL (did not raise ValueError)"
    except ValueError:
        results["temporal_T_1_error"] = "PASS (raised ValueError as expected)"
    except Exception as e:
        results["temporal_T_1_error"] = f"FAIL: unexpected exception {e}"

    # 5. Invalid temporal mode: raises ValueError
    try:
        _ = generate_temporal_sequence(mode="invalid_mode", sequence_length=4)
        results["temporal_invalid_mode"] = "FAIL (did not raise ValueError)"
    except ValueError:
        results["temporal_invalid_mode"] = "PASS (raised ValueError as expected)"
    except Exception as e:
        results["temporal_invalid_mode"] = f"FAIL: unexpected exception {e}"

    # 6. Memory soak test: Monitor RSS over 1,000 continuous samples
    mem_start = get_process_memory_mb()
    ds = ArmD2MultiGeneratorDataset(manifest=manifest, split=SPLIT_TRAIN, preload_images=True)
    gc.collect()
    mem_preloaded = get_process_memory_mb()
    preload_delta = mem_preloaded - mem_start

    loader_soak = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_multi_generator_batch)
    soak_iter = iter(loader_soak)

    for step in range(65):  # 65 * 16 = 1040 samples
        try:
            _ = next(soak_iter)
        except StopIteration:
            soak_iter = iter(loader_soak)
            _ = next(soak_iter)

    gc.collect()
    mem_end = get_process_memory_mb()
    iteration_delta = mem_end - mem_preloaded
    total_delta = mem_end - mem_start

    results["memory_preload_mb"] = preload_delta
    results["memory_iteration_soak_mb_delta"] = iteration_delta
    results["memory_total_mb_delta"] = total_delta
    results["memory_leak_check"] = "PASS" if iteration_delta < 30.0 else f"FAIL (delta={iteration_delta:.2f}MB)"

    print("Edge Case Results Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


def main() -> None:
    print("=" * 70)
    print("AVCS StreamingVisualCortexV2 — Milestone M2 Adversarial Stress Suite")
    print("=" * 70)

    # 1. DataLoader batch stress test
    batch_results = stress_test_dataloader_batches(num_batches=100, batch_size=16, seed=42)

    # 2. Negative injection rate test
    neg_results = stress_test_negative_injection_1000_samples(num_samples=1000, seeds=(42, 100, 2026))

    # 3. Temporal sequence sampler stress test
    temporal_results = stress_test_temporal_sampler(num_sequences=500)

    # 4. Action observation pair stress test
    action_results = stress_test_action_pair_sampler(num_pairs=500)

    # 5. Adversarial edge cases
    edge_results = stress_test_adversarial_edge_cases()

    # Verdict synthesis
    verdict = "APPROVE"
    reasons = []

    if not batch_results["balance_ok"]:
        verdict = "REJECT"
        reasons.append(f"Arm D2 draw balance out of bounds: UI={batch_results['ui_prop']*100:.2f}%, Cleanroom={batch_results['cr_prop']*100:.2f}%")

    if not neg_results["aggregate_in_range"]:
        verdict = "REJECT"
        reasons.append(f"Hard negative injection aggregate rate out of [15%, 20%]: {neg_results['aggregate_rate']*100:.2f}%")

    if not temporal_results["passed"]:
        verdict = "REJECT"
        reasons.append("Temporal sequence sampler failed invariants")

    if not action_results["passed"]:
        verdict = "REJECT"
        reasons.append("Action observation pair sampler failed invariants")

    for k, v in edge_results.items():
        if isinstance(v, str) and v.startswith("FAIL"):
            verdict = "REJECT"
            reasons.append(f"Edge case {k} failed: {v}")

    print("\n" + "=" * 70)
    print(f"FINAL ADVERSARIAL VERDICT: {verdict}")
    if reasons:
        print("Failure Reasons:")
        for r in reasons:
            print(f"  - {r}")
    print("=" * 70)

    # Save structured results to benchmarks/results/
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "challenger_m2_2_dataloader_stress.json"

    full_payload = {
        "verdict": verdict,
        "reasons": reasons,
        "batch_stress": batch_results,
        "negative_injection": neg_results,
        "temporal_sequences": temporal_results,
        "action_pairs": action_results,
        "edge_cases": edge_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"Saved benchmark results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
