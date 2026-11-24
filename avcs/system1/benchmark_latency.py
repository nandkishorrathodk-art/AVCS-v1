"""Latency benchmark: can the one model run on a real-time screen clock?

A desktop screen delivers 60-144 frames per second, i.e. 16.7-6.9 ms per frame.
This measures the full loop step (preprocess + forward + gate) for several
configs so a real-time target is a measured choice, not a guess.

    python3 -m avcs.system1.benchmark_latency

Quote the *config* with the number: latency scales with the square of the input
size and roughly linearly with width, so "36 FPS" and "134 FPS" are the same
architecture at different widths, not different systems.
"""
from __future__ import annotations

import argparse
import time

import torch

from avcs.cortex.model import VLACortexConfig, VLACortexModel
from avcs.system1.model import OneModel, OneModelConfig

VLA_CONFIGS = (
    ("VLA-Standard (640x400 T=4)", VLACortexConfig(input_height=400, input_width=640, stage_channels=(32, 64, 128, 256), neck_channels=128), 4),
    ("VLA-SingleFrame (640x400 T=1)", VLACortexConfig(input_height=400, input_width=640, stage_channels=(32, 64, 128, 256), neck_channels=128), 1),
    ("VLA-Fast (640x400 T=4)", VLACortexConfig(input_height=400, input_width=640, stage_channels=(24, 48, 96, 192), neck_channels=96), 4),
    ("VLA-Fast (640x400 T=1)", VLACortexConfig(input_height=400, input_width=640, stage_channels=(24, 48, 96, 192), neck_channels=96), 1),
)

CONFIGS = (
    ("shipped 128px w128 z256 g16", OneModelConfig(input_size=128, width=128, latent_dim=256, grid=16)),
    ("128px w64 z128 g16", OneModelConfig(input_size=128, width=64, latent_dim=128, grid=16)),
    ("128px w48 z96 g16", OneModelConfig(input_size=128, width=48, latent_dim=96, grid=16)),
    ("96px  w64 z128 g12", OneModelConfig(input_size=96, width=64, latent_dim=128, grid=12)),
    ("64px  w64 z128 g8", OneModelConfig(input_size=64, width=64, latent_dim=128, grid=8)),
    ("64px  w32 z64 g8", OneModelConfig(input_size=64, width=32, latent_dim=64, grid=8)),
)


def bench(cfg: OneModelConfig, reps: int = 40) -> tuple[int, float]:
    torch.manual_seed(0)
    model = OneModel(cfg).eval()
    x = torch.randn(1, 3, cfg.input_size, cfg.input_size)
    for _ in range(5):
        with torch.no_grad():
            model(x, ["click Save button"])
    times = []
    for _ in range(reps):
        with torch.no_grad():
            t0 = time.perf_counter()
            model(x, ["click Save button"])
            times.append(time.perf_counter() - t0)
    # Minimum, not mean: on a loaded box the mean is scheduler noise.
    return model.total_parameters, min(times) * 1000.0


def bench_vla(cfg: VLACortexConfig, t_frames: int = 4, reps: int = 20) -> tuple[int, float]:
    torch.manual_seed(0)
    model = VLACortexModel(cfg).eval()
    x = torch.randn(1, t_frames, 3, cfg.input_height, cfg.input_width)
    goals = ["click Save button"]
    for _ in range(3):
        with torch.no_grad():
            model(x, goals)
    times = []
    for _ in range(reps):
        with torch.no_grad():
            t0 = time.perf_counter()
            model(x, goals)
            times.append(time.perf_counter() - t0)
    return model.total_parameters, min(times) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--vla", action="store_true", help="Benchmark Ace-style VLACortexModel")
    ap.add_argument("--all", action="store_true", help="Benchmark both System1 and VLACortexModel")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    run_vla = args.vla or args.all or not any([args.vla, args.all])
    run_sys1 = args.all or (not args.vla and not args.all)

    print(f"threads={args.threads}  reps={args.reps}  (minimum over reps)")

    if run_sys1 and not args.vla:
        print("\n=== System 1 (OneModel) Latency Benchmark ===")
        print(f"{'config':30s} {'params':>10s} {'ms':>7s} {'FPS':>6s} {'60fps':>6s} {'144fps':>7s}")
        for name, cfg in CONFIGS:
            params, ms = bench(cfg, args.reps)
            fps = 1000.0 / ms
            print(
                f"{name:30s} {params:>10,} {ms:>7.2f} {fps:>6.0f} "
                f"{'yes' if ms < 16.7 else 'no':>6s} {'yes' if ms < 6.9 else 'no':>7s}"
            )

    if run_vla or args.vla:
        print("\n=== Ace-Style VLACortexModel (640x400 Canonical) Benchmark ===")
        print(f"{'config':30s} {'params':>10s} {'ms':>7s} {'FPS':>6s} {'<50ms':>6s} {'<30ms':>7s}")
        for name, cfg, t_frames in VLA_CONFIGS:
            params, ms = bench_vla(cfg, t_frames=t_frames, reps=args.reps)
            fps = 1000.0 / ms
            print(
                f"{name:30s} {params:>10,} {ms:>7.2f} {fps:>6.0f} "
                f"{'yes' if ms < 50.0 else 'no':>6s} {'yes' if ms < 30.0 else 'no':>7s}"
            )


if __name__ == "__main__":
    main()
