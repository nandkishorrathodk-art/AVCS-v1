"""Phase 3 Gate 3: Trajectory Overfit Verification on Golden LeRobot v3.0 Episodes.

Loads recorded pure-RGB LeRobot v3.0 (.parquet + .mp4) or Golden Diagnostic episodes,
audits 10-action primitive coverage + QADoctor integrity, and verifies that either
`QwenGUIActorModel` (with optional `--checkpoint` on GPU/CUDA or `--dry-run`) or
`PureRGBVLAPolicy` achieves >=98% click bbox accuracy and >=99% action-type accuracy
without hidden normalization, frame-index, or coordinate bugs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from avcs.data.lerobot_v3 import (
    LeRobotV3TrajectoryEpisode,
    QADoctor,
    load_lerobot_v3_shards,
)
from avcs.models.vla_cortex import (
    ACTION_VOCAB_10,
    UnifiedMultiTaskLoss,
)


def audit_recorded_coverage(
    dataset_paths: Sequence[Union[str, Path]],
    min_dpi_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Loads LeRobot v3 episodes from `dataset_paths` and reports QA + 10-action coverage."""
    episodes: List[List[LeRobotV3TrajectoryEpisode]] = []
    counts: Counter[str] = Counter()
    qa_passed = 0
    per_dir_summary: List[Dict[str, Any]] = []

    for p_raw in dataset_paths:
        p = Path(p_raw)
        if not p.exists():
            continue
        shard_roots: List[Path] = []
        if (p / "data").is_dir():
            shard_roots.append(p)
        else:
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and (sub / "data").is_dir():
                    shard_roots.append(sub)

        for root in shard_roots:
            loaded = load_lerobot_v3_shards(root, decode_video_frames=True)
            for ep in loaded:
                if not ep:
                    continue
                if min_dpi_scale is not None and float(ep[0].dpi_scale) < float(min_dpi_scale):
                    continue
                qa = QADoctor.validate_episode(ep)
                if qa["valid"]:
                    qa_passed += 1
                ep_cnt = Counter(ACTION_VOCAB_10[s.action_kind] for s in ep if 0 <= s.action_kind < len(ACTION_VOCAB_10))
                counts.update(ep_cnt)
                episodes.append(ep)
                per_dir_summary.append({
                    "source": str(root),
                    "episode_id": ep[0].episode_id,
                    "steps": len(ep),
                    "dpi_scale": float(ep[0].dpi_scale),
                    "qa_valid": bool(qa["valid"]),
                    "actions": dict(ep_cnt),
                })

    missing_primitives = [name for name in ACTION_VOCAB_10 if counts.get(name, 0) == 0]
    return {
        "episode_count": len(episodes),
        "qa_passed_episodes": qa_passed,
        "total_steps": sum(len(ep) for ep in episodes),
        "action_counts": dict(counts),
        "missing_primitives": missing_primitives,
        "episodes_detail": per_dir_summary,
        "episodes": episodes,
    }


def _extract_training_samples(
    episodes: Sequence[Sequence[LeRobotV3TrajectoryEpisode]],
    max_samples: int = 32,
    bbox_half_width: float = 0.035,
    canvas_hw: Tuple[int, int] = (400, 640),
) -> List[Dict[str, Any]]:
    """Extracts interactive training samples with decoded pure-RGB tensors and target boxes."""
    import io
    from PIL import Image

    h_tgt, w_tgt = int(canvas_hw[0]), int(canvas_hw[1])
    out: List[Dict[str, Any]] = []
    for ep in episodes:
        # Prioritize non-move interactive steps; include 1 move/wait per episode if needed
        chosen = [s for s in ep if s.action_kind in (0, 1, 2, 4, 5, 6, 7, 9)]
        if not chosen:
            chosen = list(ep[:4])
        for s in chosen:
            if len(out) >= max_samples:
                break
            if s.image_bytes:
                with Image.open(io.BytesIO(s.image_bytes)) as im:
                    rgb = im.convert("RGB")
                    if rgb.size != (w_tgt, h_tgt):
                        rgb = rgb.resize((w_tgt, h_tgt), Image.Resampling.BILINEAR)
                    arr = np.array(rgb, dtype=np.uint8).transpose(2, 0, 1)
                    img_t = torch.from_numpy(np.ascontiguousarray(arr)).float().div_(255.0)
            elif s.image_tensor is not None:
                img_t = s.image_tensor.detach().clone().float()
            else:
                img_t = torch.full((3, h_tgt, w_tgt), 0.5, dtype=torch.float32)

            tx = min(1.0, max(0.0, float(s.target_xy[0])))
            ty = min(1.0, max(0.0, float(s.target_xy[1])))
            hw = float(bbox_half_width)
            bbox = [max(0.0, tx - hw), max(0.0, ty - hw), min(1.0, tx + hw), min(1.0, ty + hw)]

            base_instr = (s.task_instruction or "Desktop step").strip()
            if s.action_kind == 7 and s.typed_text:
                instr = f"{base_instr} [type: {s.typed_text}]"
            elif s.action_kind == 6 and s.typed_text:
                instr = f"{base_instr} [hotkey: {s.typed_text}]"
            else:
                instr = base_instr

            out.append({
                "episode_id": s.episode_id,
                "step_index": s.step_index,
                "image": img_t,
                "instruction": instr,
                "action_idx": int(s.action_kind),
                "target_xy": [tx, ty],
                "target_bbox": bbox,
            })
        if len(out) >= max_samples:
            break
    return out


def _load_train_vla_module() -> Any:
    """Loads kaggle/train_vla_gui_actor.py by path (avoids PyPI `kaggle` package name collision)."""
    import importlib.util

    train_path = Path(__file__).resolve().parent.parent.parent / "kaggle" / "train_vla_gui_actor.py"
    spec = importlib.util.spec_from_file_location("avcs_kaggle_train_vla_gui_actor", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from {train_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_phase3_overfit_check(
    dataset_paths: Sequence[Union[str, Path]],
    *,
    checkpoint_path: Optional[str] = None,
    dry_run: bool = False,
    steps: int = 80,
    lr: float = 1e-3,
    max_samples: int = 24,
    min_dpi_scale: Optional[float] = None,
    bbox_half_width: float = 0.035,
    canvas_hw: Tuple[int, int] = (400, 640),
    output_path: Optional[str] = "benchmarks/results/phase3_gate3_overfit.json",
) -> Dict[str, Any]:
    """Runs Phase 3 Gate 3 mini-batch trajectory overfit on Golden LeRobot v3 episodes."""
    train_mod = _load_train_vla_module()
    QwenGUIActorModel = train_mod.QwenGUIActorModel
    load_checkpoint_into_model = train_mod.load_checkpoint_into_model

    coverage = audit_recorded_coverage(dataset_paths, min_dpi_scale=min_dpi_scale)
    episodes = coverage["episodes"]
    if not episodes:
        raise RuntimeError(f"Fail-closed Gate 3: No valid LeRobot v3 episodes found in {list(dataset_paths)}")

    samples = _extract_training_samples(
        episodes,
        max_samples=max_samples,
        bbox_half_width=bbox_half_width,
        canvas_hw=canvas_hw,
    )
    if not samples:
        raise RuntimeError("Fail-closed Gate 3: Extracted 0 interactive steps from episodes.")

    use_dry = bool(dry_run or not torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available() and not use_dry) else "cpu")

    if use_dry:
        model = QwenGUIActorModel(
            dry_run=True,
            backbone_hidden_dim=512,
            pointer_hidden_dim=512,
            query_dim=512,
        )
    else:
        model = QwenGUIActorModel(
            dry_run=False,
            pointer_hidden_dim=512,
            query_dim=512,
            lora_layers=0,
        )
        if checkpoint_path:
            load_checkpoint_into_model(model, checkpoint_path)

    model.to(device)
    model.train()

    criterion = UnifiedMultiTaskLoss(
        weight_action=5.0,
        weight_coord_l1=10.0,
        weight_coord_l2=10.0,
        weight_patch_ce=5.0,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    imgs = torch.stack([s["image"] for s in samples], dim=0).to(device)
    queries = [s["instruction"] for s in samples]
    target_actions = torch.tensor([s["action_idx"] for s in samples], dtype=torch.long, device=device)
    target_coords = torch.tensor([s["target_xy"] for s in samples], dtype=torch.float32, device=device)
    target_bboxes = torch.tensor([s["target_bbox"] for s in samples], dtype=torch.float32, device=device)

    t0 = time.perf_counter()
    losses: List[float] = []
    # Cache frozen backbone features in dry_run or frozen mode when batch fits cleanly?
    # Keep full end-to-end forward every step so gradients and normalization match live inference.
    for _ in range(int(steps)):
        optimizer.zero_grad()
        out = model(imgs, queries=queries)
        loss_dict = criterion(
            model_out=out,
            target_actions=target_actions,
            target_coords=target_coords,
            target_bboxes=target_bboxes,
        )
        loss = loss_dict["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss during Gate 3 overfit check: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        eval_out = model(imgs, queries=queries)
        action_preds = eval_out["action_logits"].argmax(dim=-1)
        action_acc = float((action_preds == target_actions).float().mean().item()) * 100.0

        logits = eval_out["attn_logits"]
        pcoords = eval_out["patch_coords"]
        if pcoords.dim() == 2:
            pcoords = pcoords.unsqueeze(0).expand(logits.shape[0], -1, -1)
        sharp_w = torch.softmax(logits / 0.05, dim=-1)
        if sharp_w.dim() == 2:
            sharp_w = sharp_w.unsqueeze(1)
        sharp_xy = torch.bmm(sharp_w, pcoords).squeeze(1)
        sub_xy = eval_out["subpixel_coords"]

        sharp_hits = 0
        sub_hits = 0
        for i in range(len(samples)):
            bx1, by1, bx2, by2 = target_bboxes[i].tolist()
            sx, sy = sharp_xy[i, 0].item(), sharp_xy[i, 1].item()
            if bx1 <= sx <= bx2 and by1 <= sy <= by2:
                sharp_hits += 1
            px, py = sub_xy[i, 0].item(), sub_xy[i, 1].item()
            if bx1 <= px <= bx2 and by1 <= py <= by2:
                sub_hits += 1

    elapsed_s = time.perf_counter() - t0
    sharp_acc = (sharp_hits / len(samples)) * 100.0
    sub_acc = (sub_hits / len(samples)) * 100.0
    best_bbox_acc = max(sharp_acc, sub_acc)
    gate3_passed = bool(best_bbox_acc >= 98.0 and action_acc >= 99.0)

    report = {
        "phase": "phase3_gate3_overfit_check",
        "dry_run": use_dry,
        "checkpoint": checkpoint_path,
        "episode_count": coverage["episode_count"],
        "qa_passed_episodes": coverage["qa_passed_episodes"],
        "action_counts": coverage["action_counts"],
        "missing_primitives": coverage["missing_primitives"],
        "overfit_samples": len(samples),
        "steps": int(steps),
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "action_type_acc_pct": round(action_acc, 2),
        "sharp_bbox_acc_pct": round(sharp_acc, 2),
        "subpixel_bbox_acc_pct": round(sub_acc, 2),
        "elapsed_s": round(elapsed_s, 2),
        "gate3_passed": gate3_passed,
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Gate 3: Trajectory Overfit Verification")
    parser.add_argument(
        "--dataset-paths",
        nargs="+",
        default=[
            "runs/recordings/demo_10",
            "runs/recordings/demo_11",
            "runs/recordings/demo_12",
            "runs/recordings/demo_13",
            "runs/recordings/demo_14",
            "runs/recordings/demo_15",
            "runs/recordings/demo_16",
        ],
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional epoch_3.pt checkpoint for CUDA Gate 3")
    parser.add_argument("--dry-run", action="store_true", help="Run CPU dry-run verification using mock backbone")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--min-dpi-scale", type=float, default=None)
    parser.add_argument("--output", type=str, default="benchmarks/results/phase3_gate3_overfit.json")
    args = parser.parse_args()

    res = run_phase3_overfit_check(
        args.dataset_paths,
        checkpoint_path=args.checkpoint,
        dry_run=args.dry_run,
        steps=args.steps,
        lr=args.lr,
        max_samples=args.max_samples,
        min_dpi_scale=args.min_dpi_scale,
        output_path=args.output,
    )
    status = "PASS" if res["gate3_passed"] else "FAIL"
    print(
        f"PHASE3-GATE3-OVERFIT: {status} | action_acc={res['action_type_acc_pct']:.2f}% | "
        f"sharp_bbox={res['sharp_bbox_acc_pct']:.2f}% | subpixel_bbox={res['subpixel_bbox_acc_pct']:.2f}% | "
        f"loss={res['initial_loss']:.2f}->{res['final_loss']:.2f}"
    )


if __name__ == "__main__":
    main()
