"""AVC Command-Line Interface (CLI).

Provides developer tools for inspecting, benchmarking, and running real-time
perception with the Artificial Visual Cortex.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import avc
from avc import (
    AFFORDANCE_CLASSES,
    STABILITY_STATES,
    UI_CATEGORIES,
    ArtificialVisualCortex,
    CortexConfig,
    ScreenEye,
    load_cortex,
    realtime_config,
)


def print_info(weights_path: Optional[str] = None, api_key: Optional[str] = None, device: str = "cpu") -> None:
    """Print structural neural network summary and weight provenance."""
    print("=" * 65)
    print(" AVC -- Artificial Visual Cortex (Neural Network)")
    print(f" Package Version: {avc.__version__}")
    print("=" * 65)

    cortex = load_cortex(weights_path=weights_path, device=device, api_key=api_key)
    total_params = sum(p.numel() for p in cortex.parameters())
    trainable_params = sum(p.numel() for p in cortex.parameters() if p.requires_grad)

    print(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Inference Device     : {device}")
    print("\n[Cortical Streams]")
    print("  * Retinal Foveation : Saccadic dual-stream (Peripheral 256x256 + Foveal 256x256)")
    print("  * Cortical Backbone : Dual-path Inverted Residual (MBConv) -> (1, 128, 16, 16)")
    print(f"  * Dorsal Stream     : Dense Affordance Head -> {AFFORDANCE_CLASSES}")
    print(f"  * Ventral Stream    : 8-way UI Classifier -> {UI_CATEGORIES}")
    print("  * Language Gaze     : Learned Byte Text Conditioning -> (1, 2) normalized gaze")
    print(f"  * Predictive GRU    : 527K Temporal Memory -> {STABILITY_STATES}")
    print("=" * 65)


def run_benchmark(
    iterations: int = 50,
    realtime: bool = False,
    device: str = "cpu",
    weights_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Measure inference latency and throughput on the host hardware."""
    cfg = realtime_config() if realtime else CortexConfig()
    cortex = load_cortex(weights_path=weights_path, device=device, config=cfg, api_key=api_key)

    p_size = cfg.model.retina.peripheral_size
    f_size = cfg.model.retina.foveal_size
    p_tensor = torch.randn(1, 3, *p_size, device=device)
    f_tensor = torch.randn(1, 3, *f_size, device=device)

    print(f"\n[AVC Benchmark] Running {iterations} forward passes (retina: {p_size[0]}x{p_size[1]})...")

    # Warmup
    for _ in range(5):
        _ = cortex(p_tensor, f_tensor)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = cortex(p_tensor, f_tensor)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    latencies_np = np.array(latencies)
    p50 = np.percentile(latencies_np, 50)
    p95 = np.percentile(latencies_np, 95)
    min_lat = np.min(latencies_np)
    fps = 1000.0 / min_lat

    print("\nBenchmark Results:")
    print(f"  * Min Latency : {min_lat:.2f} ms")
    print(f"  * P50 Latency : {p50:.2f} ms")
    print(f"  * P95 Latency : {p95:.2f} ms")
    print(f"  * Peak FPS    : {fps:.1f} FPS")
    print("=" * 65)


def run_perceive(
    weights_path: Optional[str] = None,
    api_key: Optional[str] = None,
    device: str = "cpu",
    output_json: bool = True,
) -> None:
    """Grab a desktop frame and print sensory state."""
    with ScreenEye(device=device, weights_path=weights_path, api_key=api_key) as eye:
        state = eye.capture_and_perceive()

    if output_json:
        data = state.to_dict()
        print(json.dumps(data, indent=2))
    else:
        print("\n[Sensory Observation]")
        print(f"  * Primary Category   : {state.primary_category} ({state.category_confidence * 100:.1f}%)")
        print(f"  * Stability State    : {state.stability_state} (surprise: {state.surprise_score:.3f})")
        print(f"  * Action Hotspots    : {state.action_hotspots}")
        print(f"  * Text Density       : {state.text_density:.3f}")
        print(f"  * Latency            : {state.inference_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="AVC - Artificial Visual Cortex Developer Tool")
    parser.add_argument("--info", action="store_true", help="Print model parameters and architecture summary")
    parser.add_argument("--perceive", action="store_true", help="Capture screen and run neural perception")
    parser.add_argument("--benchmark", action="store_true", help="Run neural inference speed benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--realtime", action="store_true", help="Use 128x128 realtime retina preset")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights path (.pt or .avcs)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for encrypted .avcs vaults")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, cuda)")

    args = parser.parse_args()

    if args.info:
        print_info(weights_path=args.weights, api_key=args.api_key, device=args.device)
    elif args.benchmark:
        run_benchmark(
            iterations=args.iterations,
            realtime=args.realtime,
            device=args.device,
            weights_path=args.weights,
            api_key=args.api_key,
        )
    elif args.perceive:
        run_perceive(weights_path=args.weights, api_key=args.api_key, device=args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
