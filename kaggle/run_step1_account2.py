"""Kaggle Account #2 Step 1 (~15-minute diagnostic):
1. Benchmark microsoft/GUI-Actor-3B-Qwen2.5-VL on showdown-clicks-dev (557 samples, max_pixels=501760).
2. Run real CUDA Phase 3 Gate 3 trajectory overfit check on Windows demo_10..19 (10 primitives, class-balanced).
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


def find_demo_dirs() -> tuple[list[str], str | None]:
    demo_dirs: list[str] = []
    for i in range(10, 20):
        matches = sorted(glob.glob(f"/kaggle/input/**/demo_{i}", recursive=True))
        for m in matches:
            if os.path.isdir(os.path.join(m, "data")):
                demo_dirs.append(m)
                break
    ckpt_matches = sorted(glob.glob("/kaggle/input/**/gui_actor_step6_2_epoch_3.pt", recursive=True))
    ckpt = ckpt_matches[0] if ckpt_matches else None
    return demo_dirs, ckpt


def main() -> None:
    print("Step 1.0: Environment Check", flush=True)
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

    print("\nStep 1A: Evaluating microsoft/GUI-Actor-3B-Qwen2.5-VL (557 samples, max_pixels=501760)...", flush=True)
    run_live(
        [
            sys.executable, "-u", "kaggle/eval_gui_actor_showdown.py",
            "--model-name", "microsoft/GUI-Actor-3B-Qwen2.5-VL",
            "--max-pixels", "501760",
            "--output", "benchmarks/results/showdown_ms_gui_actor_501k.json",
        ],
        "/kaggle/working/step1a_ms_gui_actor.log",
        env,
    )

    demo_dirs, ckpt_path = find_demo_dirs()
    print(f"\nStep 1B: Found {len(demo_dirs)} demo dirs: {demo_dirs} | checkpoint={ckpt_path}", flush=True)
    if demo_dirs:
        overfit_cmd = [
            sys.executable, "-u", "-m", "avcs.train.overfit_check",
            "--dataset-paths", *demo_dirs,
            "--max-samples", "32",
            "--steps", "80",
            "--lr", "5e-4",
            "--micro-batch-size", "2",
            "--output", "benchmarks/results/phase3_gate3_overfit_cuda.json",
        ]
        if ckpt_path:
            overfit_cmd.extend(["--checkpoint", ckpt_path, "--micro-batch-size", "1"])
        run_live(overfit_cmd, "/kaggle/working/step1b_overfit_cuda.log", env)
    else:
        print("WARNING: No demo_10..19 dirs found under /kaggle/input/; skipping Step 1B.", flush=True)

    zip_path = "/kaggle/working/avcs_step1_account2_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in (
            "benchmarks/results/showdown_ms_gui_actor_501k.json",
            "benchmarks/results/phase3_gate3_overfit_cuda.json",
        ):
            p = os.path.join(repo, rel)
            if os.path.exists(p):
                zf.write(p, arcname=rel)
        for log_fn in ("step1a_ms_gui_actor.log", "step1b_overfit_cuda.log"):
            lp = os.path.join("/kaggle/working", log_fn)
            if os.path.exists(lp):
                zf.write(lp, arcname=f"logs/{log_fn}")
    print(f"\nDownload ready: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
