"""Train AVCS goal-conditioned grounding on recorded UI click demonstrations."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from avcs.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    GROUNDING_CHECKPOINT_SCHEMA,
    GROUNDING_MODEL_TYPE,
    GoalConditionedGrounder,
)
from avcs.training.demonstrations import (
    DemonstrationClickDataset,
    discover_demonstrations,
    fingerprint_sessions,
    split_sessions,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_canvas_points(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("grounder logits must have shape (batch, 1, height, width)")
    height, width = logits.shape[-2:]
    indices = logits[:, 0].flatten(1).argmax(dim=1)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices.remainder(width)
    x = (columns.to(torch.float32) + 0.5) / width
    y = (rows.to(torch.float32) + 0.5) / height
    return torch.stack((x, y), dim=1)


def _batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    original_points: torch.Tensor,
    letterbox: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probabilities = logits[:, 0].flatten(1).log_softmax(dim=1)
    loss = -(targets.flatten(1) * log_probabilities).sum(dim=1).mean()
    canvas_points = _decode_canvas_points(logits)
    original_x = (canvas_points[:, 0] - letterbox[:, 0]) / letterbox[:, 2]
    original_y = (canvas_points[:, 1] - letterbox[:, 1]) / letterbox[:, 3]
    decoded = torch.stack((original_x, original_y), dim=1)
    normalized_error = torch.linalg.vector_norm(decoded - original_points, dim=1).mean()
    return loss, normalized_error


def _run_epoch(
    model: GoalConditionedGrounder,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],  # noqa: UP045
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_error = 0.0
    total_samples = 0
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch in loader:
            image = batch["image"].to(device)
            queries = batch["query_tokens"].to(device)
            target = batch["target_heatmap"].to(device)
            points = batch["point_norm"].to(device)
            letterbox = batch["letterbox"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image, queries)
            loss, error = _batch_metrics(logits, target, points, letterbox)
            if not torch.isfinite(loss) or not torch.isfinite(error):
                raise RuntimeError("Grounding epoch produced non-finite metrics")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = image.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_error += float(error.detach()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("A training or validation split contained no samples")
    return {
        "loss": total_loss / total_samples,
        "mean_normalized_click_distance": total_error / total_samples,
        "samples": float(total_samples),
    }


def train_grounder(
    *,
    data_dir: str,
    output_dir: str = "runs/real_ui_grounder",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    input_size: int = DEFAULT_INPUT_SIZE,
    seed: int = 42,
    device_name: Optional[str] = None,  # noqa: UP045
    overwrite: bool = False,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if input_size < 64 or input_size % 16:
        raise ValueError("input_size must be at least 64 and divisible by 16")

    sessions = discover_demonstrations(data_dir)
    train_records, validation_records, test_records = split_sessions(sessions, seed=seed)
    split_fingerprints = {
        "training_session_fingerprint": fingerprint_sessions(data_dir, train_records),
        "validation_session_fingerprint": fingerprint_sessions(data_dir, validation_records),
        "held_out_test_session_fingerprint": fingerprint_sessions(data_dir, test_records),
    }
    train_data = DemonstrationClickDataset(train_records, input_size=input_size)
    validation_data = DemonstrationClickDataset(validation_records, input_size=input_size)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else 2,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
    )

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(seed)
    model = GoalConditionedGrounder(input_size=input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "goal_grounder_best.pt"
    report_path = output_path / "training_report.json"
    if not overwrite and (checkpoint_path.exists() or report_path.exists()):
        raise FileExistsError(
            f"Training outputs already exist in {output_path}; use a new directory "
            "or pass --overwrite."
        )

    history: list[dict[str, float]] = []
    best_validation_loss = math.inf
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        validation_metrics = _run_epoch(model, validation_loader, None, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_mean_normalized_click_distance": train_metrics[
                    "mean_normalized_click_distance"
                ],
                "validation_loss": validation_metrics["loss"],
                "validation_mean_normalized_click_distance": validation_metrics[
                    "mean_normalized_click_distance"
                ],
            }
        )
        print(
            f"epoch {epoch:03d}/{epochs}: "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            "val_mean_normalized_click_distance="
            f"{validation_metrics['mean_normalized_click_distance']:.5f}"
        )
        if math.isfinite(validation_metrics["loss"]) and (
            validation_metrics["loss"] < best_validation_loss
        ):
            best_validation_loss = validation_metrics["loss"]
            checkpoint = {
                "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
                "model_type": GROUNDING_MODEL_TYPE,
                "dataset_provenance": "recorded_ui_click_demonstrations",
                "training_samples": len(train_records),
                "validation_samples": len(validation_records),
                "held_out_test_samples": len(test_records),
                "input_size": input_size,
                "split_seed": seed,
                **split_fingerprints,
                "model_state_dict": model.state_dict(),
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)

    if not math.isfinite(best_validation_loss) or not checkpoint_path.is_file():
        raise RuntimeError("No finite validation checkpoint was produced")

    report = {
        "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": "recorded_ui_click_demonstrations",
        "session_split": "recording_session_disjoint",
        **split_fingerprints,
        "recording_sessions": len(sessions),
        "split_seed": seed,
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "held_out_test_samples": len(test_records),
        "held_out_test_metrics": "not_evaluated",
        "input_size": input_size,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "best_validation_loss": best_validation_loss,
        "validation_metric": "mean normalized Euclidean click-point distance",
        "target_hit_rate": "not_measured; recorder has click points, not target bounds",
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the AVCS pixel-and-goal grounder from recorded UI clicks."
    )
    parser.add_argument("--data-dir", required=True, help="Root containing recorder sessions.")
    parser.add_argument("--output-dir", default="runs/real_ui_grounder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:  # noqa: UP045
    args = build_parser().parse_args(argv)
    report = train_grounder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        seed=args.seed,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(
        f"Saved demonstration-trained checkpoint to "
        f"{Path(args.output_dir) / 'goal_grounder_best.pt'}"
    )
    print(
        f"Held-out sessions: {report['held_out_test_samples']} samples; "
        "test metrics were not evaluated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
