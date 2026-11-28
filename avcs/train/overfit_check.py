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
    max_samples: int = 24,
    bbox_half_width: float = 0.035,
    canvas_hw: Tuple[int, int] = (400, 640),
    balance_classes: bool = True,
) -> List[Dict[str, Any]]:
    """Extracts stratified pure-RGB training samples across all present primitives."""
    import io
    from PIL import Image

    h_tgt, w_tgt = int(canvas_hw[0]), int(canvas_hw[1])

    def _pack_step(s: LeRobotV3TrajectoryEpisode) -> Dict[str, Any]:
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

        return {
            "episode_id": s.episode_id,
            "step_index": s.step_index,
            "image": img_t,
            "instruction": instr,
            "action_idx": int(s.action_kind),
            "target_xy": [tx, ty],
            "target_bbox": bbox,
        }

    if not balance_classes:
        out: List[Dict[str, Any]] = []
        for ep in episodes:
            chosen = [s for s in ep if s.action_kind in (0, 1, 2, 4, 5, 6, 7, 9)] or list(ep[:4])
            for s in chosen:
                if len(out) >= max_samples:
                    break
                out.append(_pack_step(s))
            if len(out) >= max_samples:
                break
        return out

    # Stratified round-robin selection across ALL 10 primitives so rare ops
    # (double_click, right_click, drag_start, stop, scroll) are guaranteed
    # representation while mouse_move/wait cannot crowd them out.
    by_kind: Dict[int, List[LeRobotV3TrajectoryEpisode]] = {}
    for ep in episodes:
        for s in ep:
            k = int(s.action_kind)
            if 0 <= k < len(ACTION_VOCAB_10):
                by_kind.setdefault(k, []).append(s)

    priority_order = [1, 2, 4, 9, 5, 6, 7, 0, 8, 3]
    selected: List[LeRobotV3TrajectoryEpisode] = []
    cursors = {k: 0 for k in by_kind}
    idle_cap = max(2, max_samples // 6)
    idle_counts = {3: 0, 8: 0}

    while len(selected) < max_samples:
        progressed = False
        for k in priority_order:
            pool = by_kind.get(k)
            if not pool or cursors[k] >= len(pool):
                continue
            if k in idle_counts and idle_counts[k] >= idle_cap:
                # Only allow more idle frames if non-idle pools are all exhausted
                has_non_idle = any(
                    ok not in idle_counts and cursors[ok] < len(by_kind[ok])
                    for ok in by_kind
                )
                if has_non_idle:
                    continue
            selected.append(pool[cursors[k]])
            cursors[k] += 1
            if k in idle_counts:
                idle_counts[k] += 1
            progressed = True
            if len(selected) >= max_samples:
                break
        if not progressed:
            break

    return [_pack_step(s) for s in selected]


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
    class_balanced: bool = True,
    micro_batch_size: Optional[int] = None,
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
        balance_classes=class_balanced,
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

    # Compute inverse-frequency class weights so rare actions (double_click,
    # right_click, drag_start, stop, scroll) are weighted equally with mouse_move.
    sampled_counts: Counter[int] = Counter(int(s["action_idx"]) for s in samples)
    action_class_weights: Optional[torch.Tensor] = None
    if class_balanced and len(sampled_counts) > 1:
        num_classes = len(ACTION_VOCAB_10)
        weights = torch.ones(num_classes, dtype=torch.float32, device=device)
        total_n = float(len(samples))
        n_present = float(len(sampled_counts))
        for cls_idx, cnt in sampled_counts.items():
            raw_w = total_n / (n_present * float(max(1, cnt)))
            weights[cls_idx] = min(10.0, max(0.25, raw_w))
        present_mean = float(sum(weights[c].item() for c in sampled_counts) / n_present)
        if present_mean > 0:
            for c in sampled_counts:
                weights[c] = weights[c] / present_mean
        action_class_weights = weights

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

    n_samples = len(samples)
    if micro_batch_size is not None and micro_batch_size > 0:
        mb = min(n_samples, int(micro_batch_size))
    elif use_dry:
        mb = n_samples
    else:
        has_lora = bool(getattr(model, "lora_config", None))
        mb = 1 if has_lora else 2

    t0 = time.perf_counter()
    losses: List[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad()
        step_loss_acc = 0.0
        n_chunks = max(1, math.ceil(n_samples / mb))
        for s_idx in range(0, n_samples, mb):
            e_idx = min(n_samples, s_idx + mb)
            chunk_weight = float(e_idx - s_idx) / float(n_samples)
            out = model(imgs[s_idx:e_idx], queries=queries[s_idx:e_idx])
            loss_dict = criterion(
                model_out=out,
                target_actions=target_actions[s_idx:e_idx],
                target_coords=target_coords[s_idx:e_idx],
                target_bboxes=target_bboxes[s_idx:e_idx],
                action_class_weights=action_class_weights,
            )
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss during Gate 3 overfit check: {loss.item()}")
            (loss * chunk_weight).backward()
            step_loss_acc += float(loss.item()) * chunk_weight
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(step_loss_acc)

    model.eval()
    all_preds: List[torch.Tensor] = []
    all_sharp_xy: List[torch.Tensor] = []
    all_sub_xy: List[torch.Tensor] = []
    with torch.no_grad():
        for s_idx in range(0, n_samples, mb):
            e_idx = min(n_samples, s_idx + mb)
            eval_out = model(imgs[s_idx:e_idx], queries=queries[s_idx:e_idx])
            all_preds.append(eval_out["action_logits"].argmax(dim=-1))

            logits = eval_out["attn_logits"]
            pcoords = eval_out["patch_coords"]
            if pcoords.dim() == 2:
                pcoords = pcoords.unsqueeze(0).expand(logits.shape[0], -1, -1)
            sharp_w = torch.softmax(logits / 0.05, dim=-1)
            if sharp_w.dim() == 2:
                sharp_w = sharp_w.unsqueeze(1)
            all_sharp_xy.append(torch.bmm(sharp_w, pcoords).squeeze(1))
            all_sub_xy.append(eval_out["subpixel_coords"])

        action_preds = torch.cat(all_preds, dim=0)
        sharp_xy = torch.cat(all_sharp_xy, dim=0)
        sub_xy = torch.cat(all_sub_xy, dim=0)
        action_acc = float((action_preds == target_actions).float().mean().item()) * 100.0

        sharp_hits = 0
        sub_hits = 0
        per_prim_total: Counter[int] = Counter()
        per_prim_ok: Counter[int] = Counter()
        for i in range(n_samples):
            tgt_cls = int(target_actions[i].item())
            pred_cls = int(action_preds[i].item())
            per_prim_total[tgt_cls] += 1
            if pred_cls == tgt_cls:
                per_prim_ok[tgt_cls] += 1

            bx1, by1, bx2, by2 = target_bboxes[i].tolist()
            sx, sy = sharp_xy[i, 0].item(), sharp_xy[i, 1].item()
            if bx1 <= sx <= bx2 and by1 <= sy <= by2:
                sharp_hits += 1
            px, py = sub_xy[i, 0].item(), sub_xy[i, 1].item()
            if bx1 <= px <= bx2 and by1 <= py <= by2:
                sub_hits += 1

    per_primitive_recall: Dict[str, Dict[str, Any]] = {}
    for cls_idx in sorted(per_prim_total):
        name = ACTION_VOCAB_10[cls_idx] if 0 <= cls_idx < len(ACTION_VOCAB_10) else str(cls_idx)
        tot = per_prim_total[cls_idx]
        ok = per_prim_ok[cls_idx]
        per_primitive_recall[name] = {
            "correct": ok,
            "total": tot,
            "recall_pct": round((ok / max(1, tot)) * 100.0, 2),
        }
    macro_recall = (
        sum(v["recall_pct"] for v in per_primitive_recall.values()) / max(1, len(per_primitive_recall))
    )

    elapsed_s = time.perf_counter() - t0
    sharp_acc = (sharp_hits / n_samples) * 100.0
    sub_acc = (sub_hits / n_samples) * 100.0
    best_bbox_acc = max(sharp_acc, sub_acc)
    gate3_passed = bool(best_bbox_acc >= 98.0 and action_acc >= 99.0)

    sampled_action_counts = {
        ACTION_VOCAB_10[k]: int(v) for k, v in sorted(sampled_counts.items()) if 0 <= k < len(ACTION_VOCAB_10)
    }

    report = {
        "phase": "phase3_gate3_overfit_check",
        "dry_run": use_dry,
        "checkpoint": checkpoint_path,
        "class_balanced": bool(class_balanced),
        "micro_batch_size": mb,
        "episode_count": coverage["episode_count"],
        "qa_passed_episodes": coverage["qa_passed_episodes"],
        "action_counts": coverage["action_counts"],
        "sampled_action_counts": sampled_action_counts,
        "missing_primitives": coverage["missing_primitives"],
        "overfit_samples": n_samples,
        "steps": int(steps),
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "action_type_acc_pct": round(action_acc, 2),
        "macro_action_recall_pct": round(macro_recall, 2),
        "per_primitive_recall": per_primitive_recall,
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
            "runs/recordings/demo_17",
            "runs/recordings/demo_18",
            "runs/recordings/demo_19",
        ],
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional epoch_3.pt checkpoint for CUDA Gate 3")
    parser.add_argument("--dry-run", action="store_true", help="Run CPU dry-run verification using mock backbone")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--no-class-balanced", action="store_true", help="Disable stratified/class-weighted action loss")
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
        class_balanced=not args.no_class_balanced,
        micro_batch_size=args.micro_batch_size,
        output_path=args.output,
    )
    status = "PASS" if res["gate3_passed"] else "FAIL"
    print(
        f"PHASE3-GATE3-OVERFIT: {status} | action_acc={res['action_type_acc_pct']:.2f}% "
        f"(macro={res['macro_action_recall_pct']:.2f}%) | "
        f"sharp_bbox={res['sharp_bbox_acc_pct']:.2f}% | subpixel_bbox={res['subpixel_bbox_acc_pct']:.2f}% | "
        f"loss={res['initial_loss']:.2f}->{res['final_loss']:.2f}"
    )
    for name, info in res.get("per_primitive_recall", {}).items():
        print(f"  - {name:12s}: {info['correct']}/{info['total']} ({info['recall_pct']:.1f}%)")


if __name__ == "__main__":
    main()
