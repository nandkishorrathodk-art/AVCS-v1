"""Self-Contained Cloud GPU Training Script for Artificial Visual Cortex (AVCS) Phase 4.

Optimized for Kaggle (Dual T4 16GB Turing), Google Colab, RunPod, or AWS EC2 instances.
Trains the unified PureRGBVLAPolicy end-to-end on scaled behavioral cloning data mixtures
(Golden demonstration stream + OS-Atlas / ShowUI-desktop / AgentNet schemas + DAgger recovery).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# If running directly inside Kaggle without local workspace directory
if not os.path.exists("avcs") and os.path.exists("/kaggle"):
    import subprocess
    print("Setting up AVCS repository workspace inside Kaggle...")
    if not os.path.exists("/kaggle/working/AVCS-v1"):
        try:
            subprocess.run(["git", "clone", "https://github.com/nandkishorrathodk-art/AVCS-v1.git", "/kaggle/working/AVCS-v1"], check=False)
        except Exception as e:
            print(f"Git clone notice: {e}")
    if "/kaggle/working/AVCS-v1" not in sys.path and os.path.exists("/kaggle/working/AVCS-v1"):
        sys.path.insert(0, "/kaggle/working/AVCS-v1")
    if "/kaggle/working" not in sys.path:
        sys.path.insert(0, "/kaggle/working")

from avcs.models import (
    AVCSPhase4MultiTaskLoss,
    PureRGBVLAOutput,
    PureRGBVLAPolicy,
    VLACortexConfig,
)
from avcs.models.vla_cortex import (
    ACTION_VOCAB_10,
    LiteTrain,
)
from avcs.data import (
    Phase4DataMixtureDataset,
    generate_golden_diagnostic_dataset,
)
from avcs.data.lerobot_v3 import (
    create_phase4_train_val_dataloaders,
)


class VLAPolicyTrainingWrapper(nn.Module):
    """PyTorch nn.DataParallel-compatible training wrapper for PureRGBVLAPolicy.

    Accepts all-tensor inputs (images, cursor_xy, goal_tokens) and returns
    a dictionary of tensors which can be seamlessly scattered and gathered
    across dual Nvidia Turing T4 GPUs.
    Supports gradient checkpointing to reduce VRAM footprint on 16GB Turing T4.
    """

    def __init__(
        self,
        policy: PureRGBVLAPolicy,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        images: torch.Tensor,
        goals: Optional[torch.Tensor] = None,
        cursor_xy: Optional[torch.Tensor] = None,
        goal_tokens: Optional[torch.Tensor] = None,
        force_drag: bool = False,
    ) -> Dict[str, torch.Tensor]:
        actual_goals = goals if goals is not None else goal_tokens
        if self.gradient_checkpointing and self.training:
            def _custom_forward(imgs, gls, c_xy):
                return self.policy(
                    images=imgs,
                    goals=gls,
                    cursor_xy=c_xy,
                    force_drag=force_drag,
                )
            out = torch.utils.checkpoint.checkpoint(
                _custom_forward, images, actual_goals, cursor_xy, use_reentrant=False
            )
        else:
            out = self.policy(
                images=images,
                goals=actual_goals,
                cursor_xy=cursor_xy,
                force_drag=force_drag,
            )
        res = {
            "action_logits": out.action_logits,
            "target_coords": out.target_coords,
            "subpixel_coords": out.subpixel_coords,
            "scroll_delta": out.scroll_delta,
        }
        if hasattr(out, "context") and out.context is not None:
            res["context"] = out.context
        return res


def run_phase4_scaled_bc_training(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    save_dir: str = "runs/checkpoints",
    enable_flow_drag: bool = False,
    enable_scroll: bool = False,
    checkpoint_step_interval: int = 500,
    dry_run: bool = False,
    num_golden_episodes: int = 500,
    val_episodes: int = 50,
    gradient_checkpointing: bool = True,
    open_source_samples: Optional[List[Dict[str, Any]]] = None,
    dagger_samples: Optional[List[Dict[str, Any]]] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Production Phase 4 Scaled Behavioral Cloning training loop."""
    print("=" * 80)
    print("      AVCS PHASE 4 SCALED BEHAVIORAL CLONING & POST-TRAINING PIPELINE      ")
    print("=" * 80)

    # 1. Device Configuration & Hardware Scaling (Kaggle Dual-T4 Support)
    if device_override is not None:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if (torch.cuda.is_available() and not dry_run) else "cpu")

    gpu_count = torch.cuda.device_count() if (device.type == "cuda" and torch.cuda.is_available()) else 0
    is_multi_gpu = (device.type == "cuda" and gpu_count > 1)

    print(f"Hardware Compute Device: {device}")
    if device.type == "cuda":
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  • GPU {i}: {name} | VRAM: {vram_gb:.1f} GB")
        if gpu_count > 1:
            print(f"  • Dual-T4 GPU Scaling: Enabled (torch.nn.DataParallel across {gpu_count} GPUs)")
    else:
        print("  • Compute Engine: CPU Mode (Dry-run compatible)")

    os.makedirs(save_dir, exist_ok=True)

    # 2. Model Initialization & LiteTrain Architecture Recipe
    # Fast sub-350ms compact resolution matching native 16:10 aspect ratio
    target_h, target_w = (200, 320) if dry_run else (400, 640)
    cfg = VLACortexConfig(
        input_height=target_h,
        input_width=target_w,
        stem_channels=32,
        stage_channels=(32, 64, 128, 256),
        neck_channels=128,
        text_dim=128,
    )
    raw_policy = PureRGBVLAPolicy(cortex_config=cfg, target_token_budget=400)

    # Stage 1 LiteTrain: Freeze visual cortex backbone & deferred drag expert, train strictly pointer & discrete heads
    raw_policy.configure_lite_train(
        freeze_backbone=True,
        freeze_pruner=True,
        freeze_drag=not enable_flow_drag,
        freeze_aux_heads=not enable_scroll,
    )
    param_stats = LiteTrain.get_trainable_param_stats(raw_policy)
    print(f"Model Parameters: Total={param_stats['total_params']:,} | "
          f"Trainable={param_stats['trainable_params']:,} | "
          f"Frozen={param_stats['frozen_params']:,}")
    print(f"Stage 1 LiteTrain Recipe: Visual Backbone Frozen ({param_stats['frozen_params']:,} params).")
    print(f"Flow-Matching Drag Expert: {'Enabled' if enable_flow_drag else 'Deferred (enable_flow_drag=False)'}.")

    raw_policy.to(device)

    # Wrap with DataParallel for dual-T4 GPU scaling
    train_module = VLAPolicyTrainingWrapper(raw_policy, gradient_checkpointing=gradient_checkpointing)
    train_module.to(device)
    if is_multi_gpu:
        policy = nn.DataParallel(train_module)
    else:
        policy = train_module

    # 3. Data Mixture Assembly
    print("\nPreparing Phase 4 Data Mixture...")
    total_golden = 20 if dry_run else num_golden_episodes
    split_val = 5 if dry_run else val_episodes

    golden_eps = generate_golden_diagnostic_dataset(num_episodes=total_golden, steps_per_episode=5)
    train_loader, val_loader = create_phase4_train_val_dataloaders(
        golden_episodes=golden_eps,
        open_source_records=open_source_samples,
        dagger_interventions=dagger_samples,
        batch_size=4 if dry_run else batch_size,
        target_size=(target_h, target_w),
        val_split_episodes=split_val,
    )
    print(f"  • Training Samples:   {len(train_loader.dataset):,} items in {len(train_loader)} batches")
    print(f"  • Validation Samples: {len(val_loader.dataset):,} items in {len(val_loader)} batches (Golden Stream)")

    # 4. Multi-Task Loss, Optimizer & Hardware Precision Setup
    # PyTorch Native Scaled Dot Product Attention Configuration (Turing T4 CC 7.5)
    if device.type == "cuda":
        # Turing T4 cc 7.5: FlashAttention-2 is unsupported; enforce PyTorch native SDPA kernels
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    # Strict FP16 on Nvidia Turing T4 (BF16 is unsupported on T4)
    use_fp16 = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    loss_fn = AVCSPhase4MultiTaskLoss(
        weight_action=1.0,
        weight_coord_l1=15.0,
        weight_coord_l2=35.0,
        weight_scroll=2.0,
        weight_drag=5.0,
        enable_flow_drag=enable_flow_drag,
        enable_scroll=enable_scroll,
    )

    trainable_parameters = [p for p in raw_policy.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_parameters, lr=lr, weight_decay=1e-4)
    effective_epochs = 1 if dry_run else epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, effective_epochs))

    print(f"\nStarting {effective_epochs} epochs of behavioral cloning (Mixed Precision FP16: {use_fp16})...\n")

    best_val_loss = float("inf")
    global_step = 0
    training_metrics: List[Dict[str, Any]] = []
    t_start_training = time.perf_counter()

    for epoch in range(1, effective_epochs + 1):
        # --- TRAIN ---
        raw_policy.train()
        raw_policy.backbone.eval()  # Keep frozen backbone strictly in eval mode
        epoch_t0 = time.perf_counter()

        train_loss = 0.0
        train_act_loss = 0.0
        train_coord_loss = 0.0
        correct_actions = 0
        total_actions = 0
        coord_l1_errors: List[float] = []
        steps = 0

        for batch in train_loader:
            global_step += 1
            images = batch["image"].to(device)
            target_actions = batch["action_kind"].to(device)
            target_coords = batch["target_xy"].to(device)
            cursor_pre = batch["cursor_pre_xy"].to(device)
            instructions = batch["task_instruction"]
            target_scroll = batch["scroll_delta"].to(device)
            target_drag = batch["drag_trajectory"].to(device)

            # Tokenize instructions for multi-GPU scattering
            goal_tokens = raw_policy.backbone.query_encoder.tokenize(instructions, device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16 if use_fp16 else torch.float32, enabled=use_fp16):
                # Policy forward pass via policy (DataParallel on dual GPUs)
                model_out = policy(
                    images=images,
                    goals=goal_tokens,
                    cursor_xy=cursor_pre,
                    force_drag=enable_flow_drag,
                )

                context_val = model_out.get("context") if isinstance(model_out, dict) else getattr(model_out, "context", None)
                losses = loss_fn(
                    model_out=model_out,
                    target_actions=target_actions,
                    target_coords=target_coords,
                    target_scroll=target_scroll,
                    target_drag=target_drag,
                    raw_policy=raw_policy,
                    context=context_val,
                    cursor_pre_xy=cursor_pre,
                )
                loss = losses["loss"]

            if use_fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()

            # Step metrics
            batch_loss = float(loss.item())
            train_loss += batch_loss
            train_act_loss += float(losses["loss_action"].item())
            train_coord_loss += float(losses["loss_coord"].item())

            pred_logits = model_out["action_logits"] if isinstance(model_out, dict) else model_out.action_logits
            pred_act = pred_logits.argmax(dim=-1)
            correct_actions += int((pred_act == target_actions).sum().item())
            total_actions += int(target_actions.size(0))

            pred_coords = model_out["target_coords"] if isinstance(model_out, dict) else model_out.target_coords
            l1_err = float(F.l1_loss(pred_coords, target_coords).item())
            coord_l1_errors.append(l1_err)
            steps += 1

            # Periodic checkpoint saving every 500 steps (Kaggle 12-hour session resilience)
            if global_step % checkpoint_step_interval == 0:
                ckpt_path = os.path.join(save_dir, f"avcs_vla_step_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": raw_policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                }, ckpt_path)
                print(f"  [CHECKPOINT] Periodic state saved to {ckpt_path} at step {global_step}")

            if dry_run and steps >= 2:
                break

        scheduler.step()
        train_time = time.perf_counter() - epoch_t0
        avg_train_loss = train_loss / max(steps, 1)
        avg_act_loss = train_act_loss / max(steps, 1)
        avg_coord_loss = train_coord_loss / max(steps, 1)
        action_acc = (correct_actions / max(total_actions, 1)) * 100.0
        mean_l1 = float(np.mean(coord_l1_errors)) if coord_l1_errors else 0.0

        # --- EVALUATION ON GOLDEN VALIDATION SUITE ---
        raw_policy.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_l1_errors: List[float] = []
        val_steps = 0

        with torch.no_grad():
            for val_batch in val_loader:
                v_images = val_batch["image"].to(device)
                v_actions = val_batch["action_kind"].to(device)
                v_coords = val_batch["target_xy"].to(device)
                v_cursors = val_batch["cursor_pre_xy"].to(device)
                v_instr = val_batch["task_instruction"]
                v_scroll = val_batch["scroll_delta"].to(device)
                v_drag = val_batch["drag_trajectory"].to(device)

                v_tokens = raw_policy.backbone.query_encoder.tokenize(v_instr, device)

                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16 if use_fp16 else torch.float32, enabled=use_fp16):
                    v_out = policy(
                        images=v_images,
                        goals=v_tokens,
                        cursor_xy=v_cursors,
                        force_drag=False,
                    )
                    v_context_val = v_out.get("context") if isinstance(v_out, dict) else getattr(v_out, "context", None)
                    v_losses = loss_fn(
                        model_out=v_out,
                        target_actions=v_actions,
                        target_coords=v_coords,
                        target_scroll=v_scroll,
                        target_drag=v_drag,
                        raw_policy=raw_policy,
                        context=v_context_val,
                        cursor_pre_xy=v_cursors,
                    )
                    val_loss += float(v_losses["loss"].item())

                v_act_logits = v_out["action_logits"] if isinstance(v_out, dict) else v_out.action_logits
                v_preds = v_act_logits.argmax(dim=-1)
                val_correct += int((v_preds == v_actions).sum().item())
                val_total += int(v_actions.size(0))
                v_tgt_coords = v_out["target_coords"] if isinstance(v_out, dict) else v_out.target_coords
                val_l1_errors.append(float(F.l1_loss(v_tgt_coords, v_coords).item()))
                val_steps += 1

                if dry_run and val_steps >= 2:
                    break

        avg_val_loss = val_loss / max(val_steps, 1)
        val_acc = (val_correct / max(val_total, 1)) * 100.0
        val_l1 = float(np.mean(val_l1_errors)) if val_l1_errors else 0.0

        print(
            f"Epoch {epoch:02d}/{effective_epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} (Act: {avg_act_loss:.3f}, Coord: {avg_coord_loss:.3f}) | "
            f"Train Acc: {action_acc:.1f}% | Coord L1: {mean_l1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
            f"Time: {train_time:.1f}s"
        )

        epoch_metric = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "train_action_acc": action_acc,
            "train_coord_l1": mean_l1,
            "val_loss": avg_val_loss,
            "val_action_acc": val_acc,
            "val_coord_l1": val_l1,
            "epoch_time_s": train_time,
        }
        training_metrics.append(epoch_metric)

        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(save_dir, "avcs_vla_best.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": raw_policy.state_dict(),
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "config": cfg,
            }, best_path)
            print(f"  [BEST MODEL] Checkpoint saved: {best_path} (Val Loss: {avg_val_loss:.4f})")

        if dry_run:
            break

    # Save final serving checkpoint & metrics
    final_path = os.path.join(save_dir, "avcs_vla_final.pt")
    torch.save(raw_policy.state_dict(), final_path)

    # Phase 4 Production Serving Graph Export (open-ace-small.pt)
    serving_graph_path = os.path.join(save_dir, "open-ace-small.pt")
    try:
        raw_policy.export_serving_graph(export_path=serving_graph_path)
    except Exception as e:
        print(f"Serving graph export notice: {e}")

    metrics_file = os.path.join(save_dir, "training_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)

    total_training_time = time.perf_counter() - t_start_training

    print("\n" + "=" * 80)
    print("TRAINING FINISHED! PHASE 4 ARTIFACTS EXPORTED SUCCESSFULLY.")
    print(f"  • Final Model Checkpoint: {final_path}")
    print(f"  • Best Model Checkpoint:  {os.path.join(save_dir, 'avcs_vla_best.pt')}")
    print(f"  • Serving Graph (TorchScript): {serving_graph_path}")
    print(f"  • Metrics Log:            {metrics_file}")
    print(f"  • Total Elapsed Time:     {total_training_time:.1f}s")
    print("=" * 80)

    # 5. Hardware Latency Budget Profile
    print("\nPROFILING TRAINED VLA POLICY ON THIS HARDWARE:")
    lat_stats = raw_policy.profile_latency(runs=10 if dry_run else 30, device=device)
    p50 = lat_stats["p50_ms"]
    p95 = lat_stats["p95_ms"]
    fps = 1000.0 / max(p50, 0.001)
    print(f"  • Hardware Latency: p50={p50:.2f} ms | p95={p95:.2f} ms")
    print(f"  • Throughput:       {fps:.1f} FPS")
    print(f"  • Under 350ms Budget: {lat_stats['under_budget']}")
    print("=" * 80)

    return {
        "final_checkpoint": final_path,
        "best_checkpoint": os.path.join(save_dir, "avcs_vla_best.pt"),
        "best_val_loss": best_val_loss,
        "training_metrics": training_metrics,
        "total_steps": global_step,
        "latency_stats": lat_stats,
        "param_stats": param_stats,
    }


def main() -> None:
    """CLI entrypoint for Cloud GPU Behavioral Cloning training."""
    parser = argparse.ArgumentParser(description="AVCS Phase 4 Scaled Behavioral Cloning Training Engine")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="runs/checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--enable-flow-drag", action="store_true", help="Enable Flow-Matching Drag Expert")
    parser.add_argument("--enable-scroll", action="store_true", help="Enable continuous scroll loss")
    parser.add_argument("--checkpoint-step-interval", type=int, default=500, help="Steps between periodic checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-step CPU verification without training")
    parser.add_argument("--golden-episodes", type=int, default=500, help="Number of golden diagnostic episodes")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing for 16GB Turing T4 memory efficiency")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", help="Disable gradient checkpointing")
    args = parser.parse_args()

    run_phase4_scaled_bc_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        enable_flow_drag=args.enable_flow_drag,
        enable_scroll=args.enable_scroll,
        checkpoint_step_interval=args.checkpoint_step_interval,
        dry_run=args.dry_run,
        num_golden_episodes=args.golden_episodes,
        gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
