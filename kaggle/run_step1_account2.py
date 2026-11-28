"""Kaggle Account #2 Step 2 Verification (~15-minute diagnostic):
1. Verify 0-generate() single-pass microsoft/GUI-Actor-3B-Qwen2.5-VL on showdown-clicks-dev
   (557 samples, max_pixels=1003520) -> check accuracy parity + latency speedup vs 1,686.7ms.
2. Run real CUDA Phase 3 Gate 3 trajectory overfit check on Windows demo_10..19
   using --model-backend gui_actor_3b (10 primitives, class-balanced).
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import zipfile


def run_live(cmd: list[str], log_path: str, env: dict[str, str]) -> None:
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (exit={rc}): {' '.join(cmd)}")


def find_demo_dirs() -> list[str]:
    demo_dirs: list[str] = []
    for i in range(10, 20):
        matches = sorted(glob.glob(f"/kaggle/input/**/demo_{i}", recursive=True))
        for m in matches:
            if os.path.isdir(os.path.join(m, "data")):
                demo_dirs.append(m)
                break
    return demo_dirs


def main() -> None:
    print("Step 2.0: Environment Check", flush=True)
    import torch

    print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count(), flush=True)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers>=4.45.0", "accelerate", "datasets", "qwen-vl-utils", "torchvision", "pyyaml", "pyarrow", "av"],
        check=True,
    )

    repo = "/kaggle/working/AVCS-v1"
    if not os.path.exists(repo):
        subprocess.run(
            ["git", "clone", "-b", "forge-bounded-agency", "https://github.com/nandkishorrathodk-art/AVCS-v1.git", repo],
            check=True,
        )
    os.chdir(repo)
    subprocess.run(["git", "fetch", "origin", "forge-bounded-agency"], check=True)
    subprocess.run(["git", "checkout", "forge-bounded-agency"], check=True)
    subprocess.run(["git", "pull", "origin", "forge-bounded-agency"], check=True)
    subprocess.run(["git", "log", "-n", "1", "--oneline"], check=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = repo

    print("\nStep 2A: Evaluating 0-generate single-pass microsoft/GUI-Actor-3B-Qwen2.5-VL (557 samples, max_pixels=1003520)...", flush=True)
    run_live(
        [
            sys.executable, "-u", "kaggle/eval_gui_actor_showdown.py",
            "--model-name", "microsoft/GUI-Actor-3B-Qwen2.5-VL",
            "--max-pixels", "1003520",
            "--output", "benchmarks/results/showdown_ms_gui_actor_1m_0gen.json",
        ],
        "/kaggle/working/step2a_ms_0gen.log",
        env,
    )

    demo_dirs = find_demo_dirs()
    print(f"\nStep 2B: Found {len(demo_dirs)} demo dirs: {demo_dirs}", flush=True)
    if demo_dirs:
        overfit_cmd = [
            sys.executable, "-u", "-m", "avcs.train.overfit_check",
            "--model-backend", "gui_actor_3b",
            "--dataset-paths", *demo_dirs,
            "--max-samples", "32",
            "--steps", "60",
            "--lr", "5e-4",
            "--micro-batch-size", "1",
            "--output", "benchmarks/results/phase3_gate3_overfit_gui_actor_3b.json",
        ]
        run_live(overfit_cmd, "/kaggle/working/step2b_overfit_gui_actor_3b.log", env)
    else:
        print("WARNING: No demo_10..19 dirs found under /kaggle/input/; skipping Step 2B.", flush=True)

    zip_path = "/kaggle/working/avcs_step2_verify_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in (
            "benchmarks/results/showdown_ms_gui_actor_1m_0gen.json",
            "benchmarks/results/phase3_gate3_overfit_gui_actor_3b.json",
        ):
            p = os.path.join(repo, rel)
            if os.path.exists(p):
                zf.write(p, arcname=rel)
        for log_fn in ("step2a_ms_0gen.log", "step2b_overfit_gui_actor_3b.log"):
            lp = os.path.join("/kaggle/working", log_fn)
            if os.path.exists(lp):
                zf.write(lp, arcname=f"logs/{log_fn}")
    print(f"\nDownload ready: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
