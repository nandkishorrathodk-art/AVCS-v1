"""End-to-End CPU Trainer for AVCS Neural Cross-Attention UI Grounder.

Trains GoalConditionedGrounder on the canonical UI grounding dataset
using spatial cross-entropy and calibrated background/target-absence penalty.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from avcs.dataset.loader import (
    GroundingSample,
    UIGroundingDataset,
    make_ui_grounding_dataloader,
)
from avcs.dataset.synthetic.generator import SyntheticDesktopGenerator
from avcs.dataset.synthetic.themes import AVAILABLE_THEMES
from avcs.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    GROUNDING_CHECKPOINT_SCHEMA,
    GROUNDING_MODEL_TYPE,
    GoalConditionedGrounder,
    encode_queries,
)
from benchmarks.run_ui_grounding_benchmark import UIGroundingBenchmarkRunner

SYNTHETIC_UI_DATASET_PROVENANCE = "avcs_procedural_synthetic_desktop_v1"


def create_synthetic_blank_samples(
    count: int = 36,
    seed_start: int = 1000,
    image_size: Tuple[int, int] = (256, 256),
    queries: Optional[Sequence[str]] = None,
) -> List[GroundingSample]:
    """Generates synthetic blank screen and wallpaper samples with zero interactive targets."""
    if queries is None:
        queries = [
            "click button",
            "search input",
            "close window",
            "select item",
            "open menu",
        ]

    gen = SyntheticDesktopGenerator(screen_size=image_size)
    types = ["solid"] * (count // 3) + ["gradient"] * (count // 3) + ["wallpaper"] * (count - 2 * (count // 3))
    samples: List[GroundingSample] = []

    for i, ntype in enumerate(types):
        seed = seed_start + i
        theme = AVAILABLE_THEMES[i % len(AVAILABLE_THEMES)]
        frame = gen.generate_frame(
            category="desktop_canvas",
            theme_name=theme,
            is_negative=True,
            negative_type=ntype,
            seed=seed,
            split="train",
            session_id=f"synth_neg_{i // 10:02d}",
            frame_idx=i + 1,
        )
        img_arr = np.ascontiguousarray(frame.image.transpose(2, 0, 1))
        img_t = torch.from_numpy(img_arr).float().div_(255.0)

        for q in queries:
            try:
                q_tok = encode_queries([q])[0]
            except Exception:
                q_tok = torch.zeros(64, dtype=torch.long)
            sample = GroundingSample({
                "_img_rel": f"synthetic_neg_{i}_{ntype}",
                "image": img_t,
                "query_str": q,
                "target_centroid": torch.tensor([-1.0, -1.0], dtype=torch.float32),
                "target_bbox": torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32),
                "label": torch.tensor(0, dtype=torch.long),
                "is_negative": True,
                "has_target": torch.tensor(0.0, dtype=torch.float32),
                "query_tokens": q_tok,
                "frame_id": f"synth_neg_{i:04d}",
                "category": "desktop_canvas",
            })
            samples.append(sample)

    return samples


def train_epoch(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Dict[str, float]:
    """Runs a single training or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    total_hits = 0
    total_pos_samples = 0
    total_neg_samples = 0
    total_neg_rejections = 0
    all_errors_px = []

    grad_ctx = torch.enable_grad if is_train else torch.no_grad
    with grad_ctx():
        for batch in loader:
            images = batch["image"].to(device)
            q_tokens = batch["query_tokens"].to(device)
            target_heatmaps = batch["target_heatmap"].to(device)
            target_centroids = batch["target_centroid"].to(device)
            has_targets = batch["has_target"].to(device)

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            logits = model(images, q_tokens)
            B, _, Hm, Wm = logits.shape

            # Spatial log-softmax over Hm x Wm grid (128x128)
            log_probs = logits[:, 0].flatten(1).log_softmax(dim=1)
            probs = log_probs.exp()

            # Positive loss: cross-entropy with target Gaussian heatmap
            pos_ce = -(target_heatmaps.flatten(1) * log_probs).sum(dim=1)  # (B,)

            # Expected coordinate loss via differentiable DSNT
            yy, xx = torch.meshgrid(
                (torch.arange(Hm, device=device, dtype=torch.float32) + 0.5) / Hm,
                (torch.arange(Wm, device=device, dtype=torch.float32) + 0.5) / Wm,
                indexing="ij",
            )
            probs_2d = probs.view(B, Hm, Wm)
            exp_x = (probs_2d * xx.unsqueeze(0)).sum(dim=(-2, -1))
            exp_y = (probs_2d * yy.unsqueeze(0)).sum(dim=(-2, -1))
            coord_dist = torch.hypot(exp_x - target_centroids[:, 0], exp_y - target_centroids[:, 1]) * 256.0

            # Compute differentiable 5x5 local probability mass
            probs_4d = probs.view(B, 1, Hm, Wm)
            local_mass_map = F.avg_pool2d(probs_4d, kernel_size=5, stride=1, padding=2) * 25.0
            max_local_mass = local_mass_map.amax(dim=(-2, -1)).squeeze(1)  # (B,)

            # Negative loss component 1: Calibrated Peak Local Mass Margin Penalty
            # Benchmark confidence threshold is 0.35; enforce safe margin at 0.12
            margin_thresh = 0.12
            mass_margin_pen = (
                max_local_mass * 12.0
                + F.relu(max_local_mass - margin_thresh) * 25.0
                + max_local_mass.pow(2) * 15.0
            )

            # Negative loss component 2: Spatial Negative Entropy Regularization (KL to Uniform)
            # D_KL(probs || Uniform) = sum(p * log p) + log(Hm * Wm) >= 0
            kl_uniform = (probs * log_probs).sum(dim=1) + math.log(Hm * Wm)
            entropy_reg = kl_uniform * 2.5

            neg_loss = mass_margin_pen + entropy_reg

            # Positive margin loss: ensure positive samples peak above 0.40
            pos_margin_loss = F.relu(0.40 - max_local_mass) * 2.0
            pos_loss = pos_ce + coord_dist * 0.1 + pos_margin_loss

            loss_per_sample = torch.where(has_targets > 0.5, pos_loss, neg_loss)
            loss = loss_per_sample.mean()

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                assert optimizer is not None
                optimizer.step()

            # Compute predictions & metrics
            pred_indices = probs.argmax(dim=1)
            pred_rows = torch.div(pred_indices, Wm, rounding_mode="floor")
            pred_cols = pred_indices.remainder(Wm)
            pred_x = (pred_cols.float() + 0.5) / Wm
            pred_y = (pred_rows.float() + 0.5) / Hm

            for i in range(B):
                total_samples += 1
                conf = float(max_local_mass[i].detach())
                if has_targets[i] > 0.5:
                    total_pos_samples += 1
                    tx, ty = float(target_centroids[i, 0]), float(target_centroids[i, 1])
                    px, py = float(pred_x[i]), float(pred_y[i])
                    # Error in pixels on 256x256 display
                    err_px = math.hypot((px - tx) * 256.0, (py - ty) * 256.0)
                    all_errors_px.append(err_px)
                    if err_px <= 15.0 and conf >= 0.35:
                        total_hits += 1
                else:
                    total_neg_samples += 1
                    if conf < 0.35:
                        total_neg_rejections += 1

            total_loss += float(loss.detach()) * B

    med_err = float(np.median(all_errors_px)) if all_errors_px else 0.0
    acc_pct = (total_hits / max(1, total_pos_samples)) * 100.0
    neg_rej_pct = (total_neg_rejections / max(1, total_neg_samples)) * 100.0

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy_pct": acc_pct,
        "neg_rejection_pct": neg_rej_pct,
        "median_error_px": med_err,
        "samples": total_samples,
    }


def train_ui_grounder(
    manifest_path: str = "datasets/ui_grounding/manifest.json",
    output_dir: str = "runs/weights",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1.5e-4,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Trains the UI grounder on the procedural synthetic dataset with calibrated target-absence loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    dataset_provenance = manifest.get("dataset_provenance")
    if dataset_provenance != SYNTHETIC_UI_DATASET_PROVENANCE:
        raise ValueError(
            "train_ui_grounder only accepts the procedural synthetic UI dataset; "
            f"got provenance {dataset_provenance!r}"
        )

    dev = torch.device(device)
    print(f"=== Initializing UI Grounder Training on {dev} ===", flush=True)
    t0 = time.perf_counter()

    # 1. Datasets & Loaders
    print("Loading training and validation datasets (preloading in-memory)...", flush=True)
    train_dataset = UIGroundingDataset(
        manifest_path=manifest_path,
        split="train",
        expand_element_queries=True,
        include_negative_queries=True,
        preload_frames=True,
    )
    val_dataset = UIGroundingDataset(
        manifest_path=manifest_path,
        split="val",
        expand_element_queries=True,
        include_negative_queries=True,
        preload_frames=True,
    )

    # Augment training set with synthetic blank & wallpaper target-absence samples
    print("Synthesizing calibrated wallpaper and blank desktop negative controls...", flush=True)
    synth_train_blanks = create_synthetic_blank_samples(count=36, seed_start=1000)
    synth_val_blanks = create_synthetic_blank_samples(count=12, seed_start=2000)
    train_dataset._sample_records.extend(synth_train_blanks)
    val_dataset._sample_records.extend(synth_val_blanks)

    print(
        f"Train samples: {len(train_dataset)} (incl. {len(synth_train_blanks)} synthetic blank/wallpaper controls), "
        f"Val samples: {len(val_dataset)} (incl. {len(synth_val_blanks)} controls)",
        flush=True,
    )

    train_loader = make_ui_grounding_dataloader(train_dataset, batch_size=batch_size, shuffle=True, heatmap_size=(128, 128))
    val_loader = make_ui_grounding_dataloader(val_dataset, batch_size=batch_size, shuffle=False, heatmap_size=(128, 128))

    # 2. Model & Optimizer
    model = GoalConditionedGrounder(feature_dim=192, input_size=DEFAULT_INPUT_SIZE).to(dev)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: GoalConditionedGrounder ({num_params:,} parameters)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_path / "goal_grounder_best.pt"
    best_val_acc = -1.0
    best_val_err = 999.0
    best_val_rej = -1.0
    history = []

    if best_ckpt_path.exists():
        try:
            bak_path = best_ckpt_path.with_suffix(".pt.prev")
            if not bak_path.exists():
                import shutil
                shutil.copy2(best_ckpt_path, bak_path)
            ckpt_data = torch.load(best_ckpt_path, map_location=dev, weights_only=True)
            model.load_state_dict(ckpt_data["model_state_dict"])
            print(f"Loaded weights from {best_ckpt_path} to fine-tune with negative query balancing.", flush=True)
        except Exception as exc:
            print(f"Could not resume from {best_ckpt_path}: {exc}", flush=True)

    print("\n--- Training Loop Starting ---", flush=True)
    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        train_res = train_epoch(model, train_loader, optimizer, dev)
        val_res = train_epoch(model, val_loader, None, dev)
        scheduler.step()
        ep_time = time.perf_counter() - ep_t0

        history.append({
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "train_acc": train_res["accuracy_pct"],
            "train_rej": train_res["neg_rejection_pct"],
            "val_loss": val_res["loss"],
            "val_acc": val_res["accuracy_pct"],
            "val_rej": val_res["neg_rejection_pct"],
            "val_med_err": val_res["median_error_px"],
            "time_sec": ep_time,
        })

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_time:.1f}s] | "
            f"Train Loss: {train_res['loss']:.4f}, Acc: {train_res['accuracy_pct']:.1f}%, Rej: {train_res['neg_rejection_pct']:.1f}% | "
            f"Val Loss: {val_res['loss']:.4f}, Val Acc: {val_res['accuracy_pct']:.1f}%, Rej: {val_res['neg_rejection_pct']:.1f}%, "
            f"Val Med Err: {val_res['median_error_px']:.2f} px",
            flush=True,
        )

        # Checkpoint selection: require high val accuracy and rejection
        val_acc = val_res["accuracy_pct"]
        val_rej = val_res["neg_rejection_pct"]
        val_err = val_res["median_error_px"]
        is_best = False

        if best_val_acc < 0:
            is_best = True
        elif val_acc >= 85.0 and val_rej >= 95.0:
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_err < best_val_err):
                is_best = True
        elif val_acc > best_val_acc:
            is_best = True

        if is_best:
            best_val_acc = val_acc
            best_val_err = val_err
            best_val_rej = val_rej

            checkpoint = {
                "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
                "model_type": GROUNDING_MODEL_TYPE,
                "dataset_provenance": dataset_provenance,
                "training_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "held_out_test_samples": 40,
                "input_size": DEFAULT_INPUT_SIZE,
                "split_seed": seed,
                "best_val_accuracy_pct": best_val_acc,
                "best_val_median_error_px": best_val_err,
                "best_val_rejection_pct": best_val_rej,
                "model_state_dict": model.state_dict(),
            }
            torch.save(checkpoint, best_ckpt_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {best_val_acc:.1f}%, Rej: {best_val_rej:.1f}%, Med Err: {best_val_err:.2f}px)", flush=True)

    total_time = time.perf_counter() - t0
    print(f"\nTraining completed in {total_time:.1f} seconds.", flush=True)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%, Rejection: {best_val_rej:.2f}%, Median Error: {best_val_err:.2f} px", flush=True)

    # 3. Independent Held-Out Benchmark Evaluation
    print("\n=== Running Frozen Held-Out Benchmark Evaluation ===", flush=True)
    runner = UIGroundingBenchmarkRunner(
        manifest_path=manifest_path,
        device=str(dev),
        confidence_threshold=0.35,
    )

    # Load best checkpoint into evaluation model
    best_model = GoalConditionedGrounder(input_size=DEFAULT_INPUT_SIZE).to(dev)
    best_model.load_state_dict(torch.load(best_ckpt_path, map_location=dev, weights_only=True)["model_state_dict"])
    best_model.eval()

    benchmark_results = runner.evaluate_model(best_model, split="test_held_out")
    print(benchmark_results.summary(), flush=True)

    results_dict = {
        "training_time_sec": total_time,
        "best_val_acc": best_val_acc,
        "best_val_med_err": best_val_err,
        "best_val_rej": best_val_rej,
        "benchmark": benchmark_results.to_dict(),
    }
    report_file = out_path / "ui_grounding_training_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved evaluation report to {report_file}")

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AVCS Neural UI Grounder on CPU")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--output-dir", type=str, default="runs/weights")
    args = parser.parse_args()

    train_ui_grounder(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
