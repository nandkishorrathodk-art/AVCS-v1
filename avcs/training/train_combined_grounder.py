"""Combined-generator trainer for the AVCS pixel-and-goal grounder.

Trains the canonical ``avcs.goal_grounder.GoalConditionedGrounder`` on more
than one frame generator at once, then reports per-generator held-out accuracy.
The measured motivation is in AGENTS.md ("Cross-generator grounding"): a model
trained on a single generator is accurate at home and near-chance abroad, and
training on both generators removes the gap at no cost to the home split.

The clean-room arm is gated behind the signed owner approval bundle. When that
bundle (or an intact corpus) is unavailable the clean-room arm is refused, not
silently skipped, so a "combined" checkpoint can never be produced from data
that did not actually pass the rights and integrity gates.
"""
from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from avcs.dataset.cleanroom_v1.dataset import CleanroomSystem1Dataset
from avcs.dataset.cleanroom_v1.generator import validate_dataset
from avcs.dataset.loader import GroundingSample, UIGroundingDataset, ui_grounding_collate_fn
from avcs.goal_grounder import (
    DEFAULT_INPUT_SIZE,
    GROUNDING_CHECKPOINT_SCHEMA,
    GROUNDING_MODEL_TYPE,
    MULTI_GENERATOR_GROUNDING_DATASET_PROVENANCE,
    GoalConditionedGrounder,
    encode_queries,
)
from avcs.training.train_system1_clean_v1 import (
    RightsGateError,
    validate_rights_attestation,
)

SYNTHETIC_UI_DATASET_PROVENANCE = "avcs_procedural_synthetic_desktop_v1"
HEATMAP_SIZE = (128, 128)
HIT_TOLERANCE_PX = 15.0
GENERATOR_UI = "ui_grounding"
GENERATOR_CLEANROOM = "cleanroom_system1_v1"
SAMPLING_POLICIES = ("natural", "balanced")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _cleanroom_sample_to_grounding_sample(sample: Dict[str, Any]) -> GroundingSample:
    """Adapt a CleanroomSystem1Dataset item to the shape the collate expects."""
    positive = float(sample["target_present"]) > 0.5
    point = sample["target_point"].float() if positive else torch.zeros(2, dtype=torch.float32)
    query = str(sample["query"])
    try:
        query_tokens = encode_queries([query])[0]
    except Exception:
        query_tokens = torch.zeros(96, dtype=torch.long)
    return GroundingSample(
        {
            "_img_rel": str(sample.get("frame_id", "")),
            "image": sample["image"].float(),
            "query_str": query,
            "target_centroid": point,
            "target_bbox": sample["target_box"].float() if positive else torch.zeros(4),
            "label": torch.tensor(1 if positive else 0, dtype=torch.long),
            "is_negative": not positive,
            "has_target": torch.tensor(float(positive), dtype=torch.float32),
            "query_tokens": query_tokens,
        }
    )


def load_generator_samples(
    *,
    generator: str,
    ui_manifest_path: Optional[str] = None,
    cleanroom_dir: Optional[str] = None,
    cleanroom_attestation: Optional[str] = None,
    approval_path: Optional[str] = None,
    trust_config_path: Optional[str] = None,
    expected_trust_config_sha256: Optional[str] = None,
    split: str = "train",
    validate_hashes: bool = True,
) -> Tuple[List[GroundingSample], Dict[str, Any]]:
    """Load one generator's samples, enforcing that generator's own gates."""
    if generator == GENERATOR_UI:
        if not ui_manifest_path:
            raise ValueError("ui_grounding requires --ui-manifest")
        manifest = json.loads(Path(ui_manifest_path).read_text(encoding="utf-8"))
        provenance = manifest.get("dataset_provenance")
        if provenance != SYNTHETIC_UI_DATASET_PROVENANCE:
            raise ValueError(
                f"ui_grounding manifest has unexpected provenance {provenance!r}"
            )
        dataset = UIGroundingDataset(
            manifest_path=ui_manifest_path,
            split=split,
            expand_element_queries=True,
            include_negative_queries=True,
            preload_frames=True,
        )
        samples = [dataset[index] for index in range(len(dataset))]
        return samples, {"provenance": provenance, "samples": len(samples)}

    if generator == GENERATOR_CLEANROOM:
        if not cleanroom_dir:
            raise ValueError("cleanroom_system1_v1 requires --cleanroom-dir")
        if not cleanroom_attestation:
            raise RightsGateError(
                "The clean-room arm requires the signed owner attestation record; "
                "refusing to train on it without one."
            )
        corpus_dir = Path(cleanroom_dir)
        validation = validate_dataset(corpus_dir, verify_generator_sources=True)
        manifest_sha256 = validation["manifest_sha256"]
        validate_rights_attestation(
            cleanroom_attestation,
            manifest_sha256,
            approval_path=approval_path,
            trust_config_path=trust_config_path,
            expected_trust_config_sha256=expected_trust_config_sha256,
        )
        if validation["rights_status"] != "PENDING_OWNER_ATTESTATION":
            raise RightsGateError("Unexpected rights state in generated corpus manifest.")
        manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        if any(manifest.get("legacy_inputs_consumed", {}).values()):
            raise RightsGateError("Corpus manifest reports legacy input consumption.")
        dataset = CleanroomSystem1Dataset(corpus_dir, split, validate_hashes=validate_hashes)
        samples = [
            _cleanroom_sample_to_grounding_sample(dataset[index])
            for index in range(len(dataset))
        ]
        return samples, {
            "provenance": manifest.get("schema"),
            "samples": len(samples),
            "manifest_sha256": manifest_sha256,
        }

    raise ValueError(f"Unknown generator: {generator!r}")


def _evaluate(
    model: GoalConditionedGrounder,
    samples: Sequence[GroundingSample],
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """Top-1-within-tolerance accuracy and rejection rate on one generator."""
    loader = DataLoader(
        list(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: ui_grounding_collate_fn(batch, heatmap_size=HEATMAP_SIZE),
        num_workers=0,
    )
    model.eval()
    hits = 0
    positives = 0
    rejections = 0
    negatives = 0
    errors: List[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["query_tokens"].to(device))
            probs = logits[:, 0].flatten(1).softmax(dim=1)
            _, _, hm, wm = logits.shape
            peak, index = probs.max(dim=1)
            rows = torch.div(index, wm, rounding_mode="floor")
            cols = index.remainder(wm)
            pred_x = (cols.float() + 0.5) / wm
            pred_y = (rows.float() + 0.5) / hm
            targets = batch["target_centroid"]
            has_target = batch["has_target"]
            for i in range(len(batch["image"])):
                if float(has_target[i]) > 0.5:
                    positives += 1
                    error = math.hypot(
                        (float(pred_x[i]) - float(targets[i, 0])) * DEFAULT_INPUT_SIZE,
                        (float(pred_y[i]) - float(targets[i, 1])) * DEFAULT_INPUT_SIZE,
                    )
                    errors.append(error)
                    if error <= HIT_TOLERANCE_PX:
                        hits += 1
                else:
                    negatives += 1
                    if float(peak[i]) < 0.35:
                        rejections += 1
    return {
        "samples": len(samples),
        "positives": positives,
        "negatives": negatives,
        "top1_accuracy_pct": round(100.0 * hits / positives, 2) if positives else 0.0,
        "rejection_rate_pct": round(100.0 * rejections / negatives, 2) if negatives else 0.0,
        "median_error_px": round(float(np.median(errors)), 2) if errors else 0.0,
    }


def _calibrate_presence_threshold(
    model: GoalConditionedGrounder,
    samples: Sequence[GroundingSample],
    device: torch.device,
    batch_size: int,
) -> float:
    """Pick the peak-probability threshold maximising balanced accuracy."""
    loader = DataLoader(
        list(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: ui_grounding_collate_fn(batch, heatmap_size=HEATMAP_SIZE),
        num_workers=0,
    )
    peaks: List[float] = []
    labels: List[bool] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["query_tokens"].to(device))
            probs = logits[:, 0].flatten(1).softmax(dim=1)
            peak, _ = probs.max(dim=1)
            peaks.extend(float(value) for value in peak)
            labels.extend(float(value) > 0.5 for value in batch["has_target"])
    if not peaks or all(labels) or not any(labels):
        return 0.5
    candidates = sorted(set(peaks))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        true_positive = sum(
            1 for peak, label in zip(peaks, labels) if label and peak >= threshold
        )
        true_negative = sum(
            1 for peak, label in zip(peaks, labels) if not label and peak < threshold
        )
        positives = sum(labels)
        negatives = len(labels) - positives
        score = 0.5 * (true_positive / positives + true_negative / negatives)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def _generator_weights(
    policy: str,
    ui_count: int,
    cleanroom_count: int,
) -> Optional[List[float]]:
    """Per-sample weights implementing the requested sampling policy."""
    if policy == "natural":
        return None
    if policy != "balanced":
        raise ValueError(f"Unknown sampling policy: {policy!r}")
    # Equalise expected draws per generator so a small corpus is not drowned by
    # a large one. With weights total/(2*count) and num_samples=total, each
    # generator contributes half the draws per epoch.
    total = ui_count + cleanroom_count
    ui_weight = total / (2.0 * ui_count) if ui_count else 0.0
    cleanroom_weight = total / (2.0 * cleanroom_count) if cleanroom_count else 0.0
    return [ui_weight] * ui_count + [cleanroom_weight] * cleanroom_count


def train_combined_grounder(
    *,
    output_dir: str,
    ui_manifest_path: Optional[str] = "datasets/ui_grounding/manifest.json",
    cleanroom_dir: Optional[str] = None,
    cleanroom_attestation: Optional[str] = None,
    approval_path: Optional[str] = None,
    trust_config_path: Optional[str] = None,
    expected_trust_config_sha256: Optional[str] = None,
    generators: Sequence[str] = (GENERATOR_UI, GENERATOR_CLEANROOM),
    epochs: int = 12,
    batch_size: int = 16,
    learning_rate: float = 1.5e-4,
    input_size: int = DEFAULT_INPUT_SIZE,
    feature_dim: int = 128,
    sampling: str = "natural",
    seed: int = 7,
    device_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Train on every requested generator and report per-generator held-out metrics."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if input_size < 64 or input_size % 16:
        raise ValueError("input_size must be at least 64 and divisible by 16")
    if sampling not in SAMPLING_POLICIES:
        raise ValueError(f"sampling must be one of {SAMPLING_POLICIES}")
    generators = tuple(generators)
    if not generators:
        raise ValueError("At least one generator is required")

    output_path = Path(output_dir).expanduser()
    checkpoint_path = output_path / "goal_grounder_best.pt"
    report_path = output_path / "training_report.json"
    if not overwrite and (checkpoint_path.exists() or report_path.exists()):
        raise FileExistsError(
            f"Training outputs already exist in {output_path}; use a new directory "
            "or pass --overwrite."
        )
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(seed)
    started = time.perf_counter()

    train_samples: List[GroundingSample] = []
    validation_samples: List[GroundingSample] = []
    provenance: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    for generator in generators:
        common = {
            "generator": generator,
            "ui_manifest_path": ui_manifest_path,
            "cleanroom_dir": cleanroom_dir,
            "cleanroom_attestation": cleanroom_attestation,
            "approval_path": approval_path,
            "trust_config_path": trust_config_path,
            "expected_trust_config_sha256": expected_trust_config_sha256,
        }
        train_part, train_meta = load_generator_samples(split="train", **common)
        validation_split = "val" if generator == GENERATOR_UI else "validation"
        validation_part, _ = load_generator_samples(split=validation_split, **common)
        train_samples.extend(train_part)
        validation_samples.extend(validation_part)
        provenance[generator] = train_meta
        counts[generator] = len(train_part)

    if len(train_samples) == 0:
        raise ValueError("The combined training split is empty")

    weights = _generator_weights(
        sampling,
        counts.get(GENERATOR_UI, 0),
        counts.get(GENERATOR_CLEANROOM, 0),
    )
    sampler = (
        WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        if weights
        else None
    )

    train_loader = DataLoader(
        train_samples,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda batch: ui_grounding_collate_fn(batch, heatmap_size=HEATMAP_SIZE),
        num_workers=0,
    )
    if len(validation_samples) == 0:
        raise ValueError("The combined validation split is empty")
    val_loader = DataLoader(
        validation_samples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: ui_grounding_collate_fn(batch, heatmap_size=HEATMAP_SIZE),
        num_workers=0,
    )
    model = GoalConditionedGrounder(feature_dim=feature_dim, input_size=input_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    from avcs.training.train_ui_grounder import train_epoch

    history: List[Dict[str, float]] = []
    best_validation_loss = math.inf
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        validation_metrics = train_epoch(model, val_loader, None, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
            }
        )
        print(
            f"epoch {epoch:03d}/{epochs}: train_loss={train_metrics['loss']:.5f} "
            f"validation_loss={validation_metrics['loss']:.5f}",
            flush=True,
        )
        if math.isfinite(validation_metrics["loss"]) and (
            validation_metrics["loss"] < best_validation_loss
        ):
            best_validation_loss = validation_metrics["loss"]
            checkpoint = {
                "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
                "model_type": GROUNDING_MODEL_TYPE,
                "dataset_provenance": MULTI_GENERATOR_GROUNDING_DATASET_PROVENANCE,
                "target_presence_threshold": _calibrate_presence_threshold(
                    model, validation_samples, device, batch_size
                ),
                "target_presence_calibrated": True,
                "training_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "input_size": input_size,
                "feature_dim": feature_dim,
                "split_seed": seed,
                "generators": list(generators),
                "generator_provenance": provenance,
                "generator_train_counts": counts,
                "sampling_policy": sampling,
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                },
                "model_state_dict": model.state_dict(),
            }
            temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_checkpoint)
            temporary_checkpoint.replace(checkpoint_path)

    if not math.isfinite(best_validation_loss) or not checkpoint_path.is_file():
        raise RuntimeError("No finite validation checkpoint was produced")

    held_out: Dict[str, Any] = {}
    for generator in generators:
        common = {
            "generator": generator,
            "ui_manifest_path": ui_manifest_path,
            "cleanroom_dir": cleanroom_dir,
            "cleanroom_attestation": cleanroom_attestation,
            "approval_path": approval_path,
            "trust_config_path": trust_config_path,
            "expected_trust_config_sha256": expected_trust_config_sha256,
        }
        test_split = "test_held_out" if generator == GENERATOR_UI else "test_heldout"
        samples, _ = load_generator_samples(split=test_split, **common)
        held_out[generator] = _evaluate(model, samples, device, batch_size)

    report = {
        "schema_version": GROUNDING_CHECKPOINT_SCHEMA,
        "model_type": GROUNDING_MODEL_TYPE,
        "dataset_provenance": MULTI_GENERATOR_GROUNDING_DATASET_PROVENANCE,
        "generators": list(generators),
        "generator_provenance": provenance,
        "generator_train_counts": counts,
        "sampling_policy": sampling,
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "input_size": input_size,
        "feature_dim": feature_dim,
        "seed": seed,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "best_validation_loss": best_validation_loss,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "held_out": held_out,
        "abstention_action_authorized": False,
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the AVCS grounder on one or more frame generators and report "
            "per-generator held-out accuracy."
        )
    )
    parser.add_argument("--ui-manifest", default="datasets/ui_grounding/manifest.json")
    parser.add_argument("--cleanroom-dir", default=None)
    parser.add_argument("--cleanroom-attestation", default=None)
    parser.add_argument("--approval-path", default=None)
    parser.add_argument("--trust-config-path", default=None)
    parser.add_argument("--expected-trust-config-sha256", default=None)
    parser.add_argument(
        "--generators",
        nargs="+",
        default=[GENERATOR_UI, GENERATOR_CLEANROOM],
        choices=[GENERATOR_UI, GENERATOR_CLEANROOM],
    )
    parser.add_argument("--output-dir", default="runs/combined_grounder")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--sampling", default="natural", choices=list(SAMPLING_POLICIES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = train_combined_grounder(
        output_dir=args.output_dir,
        ui_manifest_path=args.ui_manifest,
        cleanroom_dir=args.cleanroom_dir,
        cleanroom_attestation=args.cleanroom_attestation,
        approval_path=args.approval_path,
        trust_config_path=args.trust_config_path,
        expected_trust_config_sha256=args.expected_trust_config_sha256,
        generators=args.generators,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        feature_dim=args.feature_dim,
        sampling=args.sampling,
        seed=args.seed,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps({k: report[k] for k in ("held_out", "checkpoint_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
