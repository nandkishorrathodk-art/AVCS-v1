"""Rights-gated, random-initialization-only System 1 training entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import __version__ as PILLOW_VERSION
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from avcs.dataset.cleanroom_v1.dataset import (
    CleanroomSystem1Dataset,
    encode_cleanroom_queries,
)
from avcs.dataset.cleanroom_v1.generator import _sha256, validate_dataset
from avcs.perception.system1_v1 import AVCSSystem1V1
from avcs.training.cleanroom_rights import (
    RightsApprovalError,
    validate_signed_rights_approval,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "datasets" / "cleanroom_system1_v1" / "manifest.json"
DEFAULT_ATTESTATION = REPOSITORY_ROOT / "docs" / "CLEANROOM_DATA_RIGHTS_ATTESTATION.md"
DEFAULT_APPROVAL = REPOSITORY_ROOT / "docs" / "CLEANROOM_RIGHTS_APPROVAL.json"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "runs"
    / "weights"
    / "clean_lineage_candidates"
    / "avcs_system1_clean_v1.pt"
)
BASELINE_CHECKPOINT = (
    REPOSITORY_ROOT / "runs" / "weights" / "candidates" / "avcs_system1_v1.pt"
)
TRAINED_MODULE_PREFIXES = (
    "stem.",
    "stage1.",
    "stage2.",
    "stage3.",
    "lat2.",
    "lat3.",
    "lat4.",
    "smooth2.",
    "smooth3.",
    "cat_head.",
    "head_p2.",
    "head_p3.",
    "head_p4.",
    "grounder.",
    "word_recognizer.",
)


RightsGateError = RightsApprovalError


def collate_cleanroom_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor targets while preserving optional metadata such as absent IDs."""
    if not samples:
        raise ValueError("Cannot collate an empty clean-room batch")
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples[1:]):
        raise ValueError("Clean-room samples in a batch must have identical fields")

    batch: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        tensor_values = [value for value in values if isinstance(value, torch.Tensor)]
        if tensor_values:
            if len(tensor_values) != len(values):
                raise TypeError(f"Mixed tensor and metadata values for clean-room field {key!r}")
            batch[key] = torch.stack(tensor_values)
        else:
            batch[key] = values
    return batch


def validate_rights_attestation(
    attestation_path: str | Path,
    manifest_sha256: str,
    *,
    approval_path: str | Path | None = None,
    trust_config_path: str | Path | None = None,
    expected_trust_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the human record's corpus binding and cryptographic authorization."""
    path = Path(attestation_path)
    if not path.is_file():
        raise RightsGateError(f"Rights attestation record is missing: {path}")
    text = path.read_text(encoding="utf-8")
    expected_binding = f"Dataset manifest SHA-256: {manifest_sha256}"
    if expected_binding not in text.splitlines():
        raise RightsGateError("Human attestation record does not identify the exact dataset manifest.")
    if "Rights decision: APPROVED" not in text.splitlines():
        raise RightsGateError("Human attestation record decision is not explicitly APPROVED.")
    if "Unresolved rights blockers: NONE" not in text.splitlines():
        raise RightsGateError("Human attestation record still has unresolved rights blockers.")
    if approval_path is None or trust_config_path is None:
        raise RightsGateError(
            "Signed approval, separately pinned trust configuration, and trust-config SHA-256 are required."
        )
    signed_rights = validate_signed_rights_approval(
        approval_path,
        manifest_sha256,
        trust_config_path,
        expected_trust_config_sha256,
    )
    return {
        **signed_rights,
        "attestation_record_sha256": _sha256(path),
    }


def assert_safe_output_path(output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    baseline = BASELINE_CHECKPOINT.resolve()
    if output == baseline:
        raise ValueError("The frozen baseline checkpoint is never a valid training output.")
    legacy_candidates = (REPOSITORY_ROOT / "runs" / "weights" / "candidates").resolve()
    if output == legacy_candidates or legacy_candidates in output.parents:
        raise ValueError("Training output must not be placed in the legacy candidate directory.")
    if output.suffix.lower() != ".pt":
        raise ValueError("Training output must be a .pt checkpoint path.")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint or sidecar: {output}")
    report_path = output.with_suffix(".training.json")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite training report: {report_path}")
    return output


def _state_fingerprint(model: AVCSSystem1V1) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def instantiate_from_random_initialization(seed: int) -> tuple[AVCSSystem1V1, str]:
    """Construct via module initializers while making checkpoint loads impossible."""
    torch.manual_seed(seed)
    original_load = torch.load

    def reject_checkpoint_load(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Clean-lineage initialization forbids all checkpoint loads.")

    torch.load = reject_checkpoint_load  # type: ignore[assignment]
    try:
        model = AVCSSystem1V1()
    finally:
        torch.load = original_load
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(f"Unexpected frozen parameters in fresh model: {frozen[:8]}")
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise RuntimeError("Fresh model initialization contains non-finite parameters.")
    return model, _state_fingerprint(model)


def _build_grounding_targets(
    points: torch.Tensor,
    present: torch.Tensor,
    size: int = 128,
) -> torch.Tensor:
    batch = points.shape[0]
    yy, xx = torch.meshgrid(
        (torch.arange(size, dtype=torch.float32, device=points.device) + 0.5) / size,
        (torch.arange(size, dtype=torch.float32, device=points.device) + 0.5) / size,
        indexing="ij",
    )
    targets = torch.zeros((batch, size, size), dtype=torch.float32, device=points.device)
    for index in range(batch):
        if present[index] <= 0.5:
            continue
        x, y = points[index]
        dist = ((xx - x) * size).square() + ((yy - y) * size).square()
        heat = torch.exp(-dist / (2.0 * 1.5**2))
        targets[index] = heat / heat.sum().clamp_min(1e-12)
    return targets


def _balanced_affordance_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel_loss = F.cross_entropy(logits, target, reduction="none")
    positive = target.eq(1)
    negative = ~positive
    if positive.any():
        return 0.75 * pixel_loss[positive].mean() + 0.25 * pixel_loss[negative].mean()
    return pixel_loss.mean()


def compute_cleanroom_losses(
    model: AVCSSystem1V1,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Return finite losses for every learnable System 1 branch."""
    images = batch["image"]
    proposals = model.forward_proposals(images)
    category_loss = F.cross_entropy(proposals["cat_logits"], batch["category"])

    proposal_losses: list[torch.Tensor] = []
    for level in ("p2", "p3", "p4"):
        output = proposals[level]
        affordance_target = batch[f"{level}_aff"]
        box_target = batch[f"{level}_box"].permute(0, 2, 3, 1)
        kind_target = batch[f"{level}_kind"]
        positive = affordance_target.eq(1)

        affordance_loss = _balanced_affordance_loss(output[:, :2], affordance_target)
        box_prediction = torch.sigmoid(output[:, 2:6]).permute(0, 2, 3, 1)
        if positive.any():
            box_loss = F.smooth_l1_loss(box_prediction[positive], box_target[positive])
            kind_loss = F.cross_entropy(output[:, 6:10].permute(0, 2, 3, 1)[positive], kind_target[positive])
        else:
            box_loss = output[:, 2:6].sum() * 0.0
            kind_loss = output[:, 6:10].sum() * 0.0
        proposal_losses.append(affordance_loss + box_loss + kind_loss)
    proposal_loss = torch.stack(proposal_losses).sum()

    query_tokens = encode_cleanroom_queries(batch["query"]).to(images.device)
    heatmap_logits, presence_logits = model.forward_grounding(images, query_tokens)
    present = batch["target_present"]
    presence_loss = F.binary_cross_entropy_with_logits(presence_logits, present)
    heat_targets = _build_grounding_targets(batch["target_point"], present)
    flat_logits = heatmap_logits.flatten(1)
    positive = present > 0.5
    if positive.any():
        positive_heat_loss = -(
            heat_targets.flatten(1)[positive] * F.log_softmax(flat_logits[positive], dim=1)
        ).sum(dim=1).mean()
    else:
        positive_heat_loss = flat_logits.sum() * 0.0
    if (~positive).any():
        negative_heat_loss = F.softplus(flat_logits[~positive]).mean()
    else:
        negative_heat_loss = flat_logits.sum() * 0.0
    grounding_loss = presence_loss + positive_heat_loss + 0.1 * negative_heat_loss

    word_logits = model.forward_word_recognition(batch["word_crops"].flatten(0, 1))
    word_loss = F.cross_entropy(word_logits, batch["word_ids"].flatten())
    return {
        "category": category_loss,
        "proposals": proposal_loss,
        "grounding": grounding_loss,
        "words": word_loss,
        "total": category_loss + proposal_loss + grounding_loss + word_loss,
    }


def assert_learning_signal(model: AVCSSystem1V1) -> dict[str, float]:
    """Fail if any claimed trainable branch has no finite non-zero gradient."""
    summary: dict[str, float] = {}
    named_parameters = list(model.named_parameters())
    for prefix in TRAINED_MODULE_PREFIXES:
        parameters = [p for name, p in named_parameters if name.startswith(prefix) and p.requires_grad]
        if not parameters:
            raise RuntimeError(f"Expected trainable module has no parameters: {prefix}")
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError(f"No loss reached trainable module: {prefix}")
        if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
            raise RuntimeError(f"Non-finite gradient in trainable module: {prefix}")
        magnitude = sum(float(gradient.detach().abs().sum().item()) for gradient in gradients)
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            raise RuntimeError(f"No effective non-zero learning signal for: {prefix}")
        summary[prefix] = magnitude
    return summary


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _epoch(model: AVCSSystem1V1, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    gradient_audit_done = False
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for raw_batch in loader:
            batch = _move_batch(raw_batch, torch.device("cpu"))
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = compute_cleanroom_losses(model, batch)
            if any(not torch.isfinite(value).item() for value in losses.values()):
                raise RuntimeError("Non-finite loss encountered during clean-room training.")
            if training:
                losses["total"].backward()
                if not gradient_audit_done:
                    assert_learning_signal(model)
                    gradient_audit_done = True
                optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())
            count += 1
    if count == 0:
        raise ValueError("Cannot train or validate with an empty loader.")
    if training and not gradient_audit_done:
        raise RuntimeError("No training minibatch completed its gradient audit.")
    return {name: value / count for name, value in totals.items()}


def _source_revision() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            "git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain", "--",
            "avcs/training/train_system1_clean_v1.py",
            "avcs/dataset/cleanroom_v1",
            "avcs/perception/system1_v1.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "git_commit": head.stdout.strip() if head.returncode == 0 else None,
        "relevant_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def train_clean_candidate(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    attestation_path: str | Path = DEFAULT_ATTESTATION,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    approval_path: str | Path | None = None,
    trust_config_path: str | Path | None = None,
    expected_trust_config_sha256: str | None = None,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    seed: int = 20261018,
    early_stopping_patience: int | None = 3,
    early_stopping_min_delta: float = 0.0,
    command: str | None = None,
) -> dict[str, Any]:
    """Train only after signed rights approval of the exact corpus manifest."""
    output = assert_safe_output_path(output_path)
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive or None")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")

    manifest_file = Path(manifest_path).resolve()
    validation = validate_dataset(manifest_file.parent)
    manifest_sha256 = validation["manifest_sha256"]
    rights = validate_rights_attestation(
        attestation_path,
        manifest_sha256,
        approval_path=approval_path,
        trust_config_path=trust_config_path,
        expected_trust_config_sha256=expected_trust_config_sha256,
    )
    if validation["rights_status"] != "PENDING_OWNER_ATTESTATION":
        raise RightsGateError("Unexpected rights state in generated corpus manifest.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if any(manifest.get("legacy_inputs_consumed", {}).values()):
        raise RightsGateError("Dataset manifest reports legacy data or checkpoint consumption.")

    train_data = CleanroomSystem1Dataset(manifest_file.parent, "train", validate_hashes=False)
    validation_data = CleanroomSystem1Dataset(manifest_file.parent, "validation", validate_hashes=False)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        num_workers=0,
        collate_fn=collate_cleanroom_batch,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cleanroom_batch,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model, initial_state_sha256 = instantiate_from_random_initialization(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min(learning_rate * 0.02, 1e-5)
    )

    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, epochs + 1):
        train_losses = _epoch(model, train_loader, optimizer)
        validation_losses = _epoch(model, validation_loader, None)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train": train_losses,
            "validation": validation_losses,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if validation_losses["total"] < best_validation_loss - early_stopping_min_delta:
            best_validation_loss = validation_losses["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                stopped_early = True
                break
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected.")
    model.load_state_dict(best_state, strict=True)

    sidecar_path = output.with_suffix(".json")
    report_path = output.with_suffix(".training.json")
    for path in (output, sidecar_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output artifact: {path}")
    model_source_paths = (
        REPOSITORY_ROOT / "avcs" / "perception" / "system1_v1.py",
        REPOSITORY_ROOT / "avcs" / "perception" / "goal_grounder.py",
        REPOSITORY_ROOT / "avcs" / "perception" / "visual_pilot.py",
        REPOSITORY_ROOT / "avcs" / "perception" / "word_recognizer.py",
        REPOSITORY_ROOT / "avcs" / "perception" / "human_eye.py",
    )
    training_metadata = {
        "lineage": "clean_random_initialization_v1",
        "model_version": "avcs-system1-clean-v1.0.0-candidate.1",
        "output_checkpoint_name": output.name,
        "initialization": {
            "method": "AVCSSystem1V1 constructor with documented PyTorch module default initializers",
            "seed": seed,
            "initial_state_sha256": initial_state_sha256,
            "checkpoint_loads_permitted_during_construction": False,
            "all_parameters_trainable_at_start": True,
            "loaded_checkpoint_paths": [],
            "loaded_optimizer_state_paths": [],
            "temporal_engine": "non-learned deterministic component; not claimed as trained",
        },
        "rights_attestation": rights,
        "dataset": {
            "manifest_path": str(manifest_file),
            "manifest_sha256": manifest_sha256,
            "dataset_version": manifest["dataset_version"],
            "split_counts": validation["split_counts"],
            "layout_families": validation["layout_families"],
        },
        "configuration": {
            "epochs": epochs,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "optimizer_amsgrad": False,
            "scheduler": "CosineAnnealingLR",
            "scheduler_eta_min": min(learning_rate * 0.02, 1e-5),
            "device": "cpu",
            "cpu_threads": torch.get_num_threads(),
            "seed": seed,
            "torch_deterministic_algorithms": True,
            "dataloader_shuffle": True,
            "dataloader_num_workers": 0,
            "selection_split": "validation",
            "final_test_split_used_during_training": False,
        },
        "supervision": {
            "trained_modules": [
                "FPN proposal backbone and category head",
                "multi-scale interactive-region, box, and element-kind heads",
                "query byte encoder, grounder heatmap, and target-presence head",
                "pixel word recognizer over generated known words and <none> samples",
            ],
            "losses": ["balanced_affordance_cross_entropy", "positive_box_smooth_l1", "element_kind_cross_entropy", "category_cross_entropy", "grounding_heatmap_cross_entropy", "target_presence_binary_cross_entropy", "unknown_word_cross_entropy"],
            "loss_parameters": {
                "positive_affordance_pixel_weight": 0.75,
                "negative_affordance_pixel_weight": 0.25,
                "negative_grounding_heatmap_weight": 0.1,
                "grounding_target_gaussian_sigma_cells": 1.5,
                "box_smooth_l1_beta": 1.0,
                "proposal_scale_assignment": {
                    "p2_max_longest_side_exclusive_px": 64,
                    "p3_max_longest_side_inclusive_px": 96,
                    "p4_min_longest_side_exclusive_px": 96,
                },
            },
            "untrained_components": ["non-learned temporal motion engine"],
            "gradient_audit": "Every trained module prefix required finite non-zero gradient on the first training batch.",
        },
        "source": {
            **_source_revision(),
            "trainer_sha256": _sha256(Path(__file__)),
            "model_and_dependency_sources": [
                {
                    "relative_path": source.relative_to(REPOSITORY_ROOT).as_posix(),
                    "sha256": _sha256(source),
                }
                for source in model_source_paths
            ],
            "generator_sources": manifest["generator_sources"],
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
        },
        "exact_command": command or shlex.join(sys.argv) if sys.argv[0].endswith("train_system1_clean_v1.py") else (
            command or "programmatic invocation of train_clean_candidate()"
        ),
        "duration_seconds": round(elapsed, 6),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "history": history,
    }
    checkpoint_sha256 = model.save_checkpoint(
        str(output),
        training_metadata=training_metadata,
        checkpoint_name=output.name,
    )
    report = {
        "checkpoint_path": output.relative_to(REPOSITORY_ROOT).as_posix()
        if REPOSITORY_ROOT in output.parents
        else str(output),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sidecar_path": str(sidecar_path),
        "training_metadata": training_metadata,
    }
    _write_json_exclusive(report_path, report)
    return {
        "checkpoint_path": str(output),
        "checkpoint_sha256": checkpoint_sha256,
        "sidecar_path": str(sidecar_path),
        "training_report_path": str(report_path),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--trust-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    args = parser.parse_args()
    result = train_clean_candidate(
        args.manifest,
        args.attestation,
        args.output,
        approval_path=args.approval,
        trust_config_path=args.trust_config,
        expected_trust_config_sha256=os.environ.get("AVCS_RIGHTS_TRUST_CONFIG_SHA256"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        command=shlex.join(sys.argv),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
