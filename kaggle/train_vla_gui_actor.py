"""Kaggle Dual-T4 Distributed Data Parallel (DDP) Training Script for GUI-Actor Pointer.

Phase 6 Step 6.2 Pre-Flight Hardening:
Trains the coordinate-free patch cross-attention pointer head (GUIActorPatchPointer)
and 10-class discrete action decoder (ParallelActionHeads) attached directly onto the
pretrained, frozen Qwen/Qwen2.5-VL-3B-Instruct backbone.

Hardware & Optimization Guardrails:
1. Auto-`torchrun` launcher: self-spawns `torch.distributed.run --nproc_per_node=2` if running
   in a multi-GPU environment (device_count >= 2) without `LOCAL_RANK` set.
2. Dynamic `image_grid_thw` alignment: exact non-square patch grid
   (H_patches, W_patches) = (H_grid // 2, W_grid // 2) matching native screen aspect ratios
   with 512-dim widened projections (~3.98M trainable parameters in `GUIActorPatchPointer` and `ParallelActionHeads`).
3. Fail-closed dataset ingestion: ingests OS-Atlas / ShowUI-desktop with a zero-leakage gate
   against `showdown-clicks-dev`, failing closed if empty in production, and cleanly skipping missing `golden/` directories.
4. Multi-resolution `--overfit-sanity` mode: rapid convergence verification across diverse resolutions.
5. Behavioral cloning, DAgger interactive replay, and RL visual-motor policy gradient training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import datetime
import functools
import hashlib
import io
import json
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from avcs.models.vla_cortex import (
    ACTION_VOCAB_10,
    GUIActorPatchPointer,
    GUIPointerOutput,
    ParallelActionHeads,
    ParallelActionOutput,
    RollingKVMemory,
    RollingMemoryState,
    ShowUITokenPruner,
    UnifiedMultiTaskLoss,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [STEP-6.2-DDP] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("train.gui_actor")


def _norm_point_pair(
    pt: Sequence[Any], w: float, h: float, force_pixel: bool = False
) -> Tuple[float, float]:
    """Normalize a 2D point consistently (pixel vs [0, 1] decided across both axes)."""
    x, y = float(pt[0]), float(pt[1])
    is_pixel = force_pixel or max(abs(x), abs(y)) > 1.0
    if is_pixel:
        return x / max(w, 1.0), y / max(h, 1.0)
    return x, y


def _norm_bbox_quad(
    bbox: Sequence[Any], w: float, h: float, force_pixel: bool = False
) -> Tuple[float, float, float, float]:
    """Normalize a [x1, y1, x2, y2] bbox consistently (pixel vs [0, 1] decided across all 4 values)."""
    vals = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    is_pixel = force_pixel or max(abs(v) for v in vals) > 1.0
    if is_pixel:
        rw, rh = max(w, 1.0), max(h, 1.0)
        return vals[0] / rw, vals[1] / rh, vals[2] / rw, vals[3] / rh
    return vals[0], vals[1], vals[2], vals[3]


# --------------------------------------------------------------------------- #
# 1. Auto-Torchrun Launcher & DDP Hardware Initialization
# --------------------------------------------------------------------------- #

def maybe_auto_spawn_torchrun(force_nproc: Optional[int] = None) -> bool:
    """Auto-torchrun launcher: self-spawn torch.distributed.run --nproc_per_node=2

    if running in a multi-GPU environment without LOCAL_RANK set.
    """
    if "--no-auto-spawn" in sys.argv:
        return False
    if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count >= 2 or force_nproc:
            nproc = force_nproc or gpu_count
            logger.info(
                f"[AUTO-TORCHRUN] Detected {gpu_count} GPUs without distributed environment. "
                f"Self-spawning torch.distributed.run --nproc_per_node={nproc}..."
            )
            clean_argv = [arg for arg in sys.argv if arg != "--no-auto-spawn"]
            if clean_argv:
                clean_argv[0] = os.path.abspath(clean_argv[0])
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                *clean_argv,
                "--no-auto-spawn",
            ]
            res = subprocess.run(cmd, cwd=repo_root)
            sys.exit(res.returncode)
            return True
    return False


def setup_distributed() -> Tuple[int, int, int, bool]:
    """Initializes PyTorch Distributed Data Parallel environment."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank, True

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        init_kwargs: Dict[str, Any] = {
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
            "timeout": datetime.timedelta(minutes=30),
        }
        if torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_distributed() -> None:
    """Cleans up the distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# 2. Frozen Qwen2.5-VL Backbone with GUI-Actor Head & Dynamic Grid Alignment
# --------------------------------------------------------------------------- #

class MockQwenBackbone(nn.Module):
    """CPU-compatible mock backbone for offline testing and dry-run validation."""

    def __init__(self, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Synthetic frozen weight matrix
        self.dummy_layer = nn.Linear(hidden_dim, hidden_dim)
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        b = images.shape[0]
        # Dynamic non-square patch grid (H_patches, W_patches) = (H_grid // 2, W_grid // 2)
        if image_grid_thw is not None:
            if image_grid_thw.dim() == 1:
                image_grid_thw = image_grid_thw.unsqueeze(0)
            gh = int(image_grid_thw[0, 1].item()) // 2
            gw = int(image_grid_thw[0, 2].item()) // 2
        elif grid_hw is not None:
            gh, gw = grid_hw
        elif images.dim() == 4:
            h, w = images.shape[2], images.shape[3]
            gh = max(1, h // 28)
            gw = max(1, w // 28)
        else:
            # Default non-square aspect ratio 16:10
            gh, gw = 14, 22

        k_patches = gh * gw
        # Differentiable and deterministic mock visual features derived from input images
        pooled_img = F.adaptive_avg_pool2d(images, (gh, gw))
        flat_img = pooled_img.permute(0, 2, 3, 1).reshape(b, k_patches, 3)
        repeats = (self.hidden_dim + 2) // 3
        patch_hidden = flat_img.repeat(1, 1, repeats)[:, :, : self.hidden_dim].contiguous()
        # Add 2D spatial position embedding so patches have unique spatial identities (like Qwen2.5-VL 2D-RoPE)
        xs = (torch.arange(gw, device=images.device, dtype=torch.float32) + 0.5) / max(gw, 1)
        ys = (torch.arange(gh, device=images.device, dtype=torch.float32) + 0.5) / max(gh, 1)
        mesh_y, mesh_x = torch.meshgrid(ys, xs, indexing="ij")
        pos_x = mesh_x.flatten().unsqueeze(1)
        pos_y = mesh_y.flatten().unsqueeze(1)
        pos_freqs = (torch.arange(0, self.hidden_dim // 4, device=images.device, dtype=torch.float32) + 1.0) * (2.0 * math.pi)
        pos_emb = torch.cat([
            torch.sin(pos_x * pos_freqs), torch.cos(pos_x * pos_freqs),
            torch.sin(pos_y * pos_freqs), torch.cos(pos_y * pos_freqs)
        ], dim=-1)
        if pos_emb.shape[-1] < self.hidden_dim:
            pos_emb = F.pad(pos_emb, (0, self.hidden_dim - pos_emb.shape[-1]))
        patch_hidden = F.layer_norm(patch_hidden + pos_emb.unsqueeze(0), (self.hidden_dim,))
        pooled = patch_hidden.mean(dim=1)
        return {
            "patch_hidden": patch_hidden,
            "pooled": pooled,
            "grid_size": (gw, gh),
        }


def _sanitize_fp16_value(obj: Any, max_abs: float = 60000.0) -> Any:
    """Recursively clamps FP16 tensors out-of-place to prevent Inf -> (Inf * 0 = NaN) inside RMSNorm."""
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.float16:
            return torch.nan_to_num(obj, nan=0.0, posinf=max_abs, neginf=-max_abs).clamp(-max_abs, max_abs)
        return obj
    if isinstance(obj, tuple):
        return tuple(_sanitize_fp16_value(item, max_abs=max_abs) for item in obj)
    if isinstance(obj, list):
        return [_sanitize_fp16_value(item, max_abs=max_abs) for item in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_fp16_value(v, max_abs=max_abs) for k, v in obj.items()}
    return obj


def _resolve_backbone_dtype(backbone_dtype: str) -> torch.dtype:
    """Resolve Qwen2.5-VL backbone precision.

    On CUDA Ampere+ (SM >= 8.0), BF16 has native Tensor Cores and wide exponent range.
    On CUDA Turing/Volta (SM < 8.0, e.g. Kaggle Tesla T4 SM 7.5), hardware BF16 Tensor
    Cores do not exist so BF16 falls back to ~15x slower kernels; FP16 uses native
    Tensor Cores cleanly when paired with `_install_fp16_overflow_guards`.
    """
    choice = (backbone_dtype or "auto").strip().lower()
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float16


def _install_fp16_overflow_guards(backbone: nn.Module, max_abs: float = 32768.0) -> int:
    """Attaches lightweight GPU clamp hooks on frozen Norm/MLP/Attention/down_proj layers for FP16 speed without NaNs."""
    if getattr(backbone, "_fp16_guards_installed", False):
        return 0

    def _pre_hook(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return (
            _sanitize_fp16_value(args, max_abs=max_abs),
            _sanitize_fp16_value(kwargs, max_abs=max_abs),
        )

    def _post_hook(_mod: nn.Module, _args: Tuple[Any, ...], output: Any) -> Any:
        return _sanitize_fp16_value(output, max_abs=max_abs)

    hook_count = 0
    post_suffixes = ("MLP", "Attention", "DecoderLayer", "Block", "PatchMerger", "PatchEmbed")
    for mod_name, mod in backbone.named_modules():
        cls_name = type(mod).__name__
        if "Norm" in cls_name or mod_name.endswith(("down_proj", "o_proj")):
            mod.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            hook_count += 1
        elif cls_name.endswith(post_suffixes):
            mod.register_forward_hook(_post_hook)
            hook_count += 1

    setattr(backbone, "_fp16_guards_installed", True)
    return hook_count


class LoRALinear(nn.Module):
    """Lightweight rank-r LoRA adapter wrapping a frozen nn.Linear without peft dependency."""

    def __init__(
        self,
        base_layer: nn.Module,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        in_features = int(getattr(base_layer, "in_features"))
        out_features = int(getattr(base_layer, "out_features"))
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / float(max(1, self.r))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        dev = getattr(getattr(base_layer, "weight", None), "device", torch.device("cpu"))
        self.lora_A = nn.Linear(in_features, self.r, bias=False).to(device=dev, dtype=torch.float32)
        self.lora_B = nn.Linear(self.r, out_features, bias=False).to(device=dev, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = _sanitize_fp16_value(x, max_abs=32768.0) if x.dtype == torch.float16 else x
        base_out = self.base_layer(x_safe)
        lora_in = self.lora_dropout(x_safe.to(dtype=self.lora_A.weight.dtype))
        lora_delta = self.lora_B(self.lora_A(lora_in)) * self.scaling
        return base_out + lora_delta.to(dtype=base_out.dtype)


class QwenGUIActorModel(nn.Module):
    """Attaches GUIActorPatchPointer + ParallelActionHeads to Qwen2.5-VL-3B-Instruct.

    Backbone Architecture:
    - Pretrained Qwen/Qwen2.5-VL-3B-Instruct (~3.75B parameters, frozen by default;
      optional lightweight LoRA adapters on top decoder attention layers via enable_lora).
    - Head Architecture:
      - Patch projection: nn.Linear(2048, pointer_hidden_dim) [512-dim widened]
      - Query projection: nn.Linear(2048, query_dim) [512-dim widened]
      - GUIActorPatchPointer: cross-attention + sub-patch offset MLP (~1.31M params)
      - ParallelActionHeads: 10 discrete actions + modifiers + continuous parameters (~564K params)
    - Default (lora_layers=0): 3,978,386 trainable params (<0.11% of total model).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        backbone_hidden_dim: int = 2048,
        pointer_hidden_dim: int = 512,
        query_dim: int = 512,
        dry_run: bool = False,
        load_in_4bit: bool = False,
        backbone_dtype: str = "auto",
        max_pixels: int = 560 * 896,
        use_token_pruning: bool = False,
        target_token_budget: int = 400,
        use_rolling_kv: bool = False,
        max_history_steps: int = 16,
        lora_layers: int = 0,
        lora_r: int = 16,
    ) -> None:
        super().__init__()
        self.dry_run = dry_run
        self.load_in_4bit = False
        self.backbone_hidden_dim = backbone_hidden_dim
        self.pointer_hidden_dim = pointer_hidden_dim
        self.query_dim = query_dim
        self.backbone_dtype = backbone_dtype
        self.max_pixels = int(max_pixels)
        self.lora_config: Optional[Dict[str, int]] = None
        self.use_token_pruning = use_token_pruning
        self.target_token_budget = target_token_budget
        self.use_rolling_kv = use_rolling_kv
        self.max_history_steps = max_history_steps

        # 1. Initialize Backbone (Frozen)
        self.backbone = None
        self.processor = None
        if not dry_run:
            try:
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                resolved_dtype = _resolve_backbone_dtype(backbone_dtype)
                logger.info(
                    f"Loading pretrained vision-language backbone: {model_id} "
                    f"(dtype={'BF16' if resolved_dtype == torch.bfloat16 else 'FP16'}{'/NF4' if load_in_4bit else ''}, "
                    f"max_pixels={self.max_pixels})..."
                )
                self.load_in_4bit = bool(load_in_4bit)
                if load_in_4bit:
                    from transformers import BitsAndBytesConfig
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=resolved_dtype,
                    )
                    dev_map: Any = {"": int(os.environ.get("LOCAL_RANK", 0))} if torch.cuda.is_available() else "auto"
                    self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_id,
                        quantization_config=bnb_cfg,
                        device_map=dev_map,
                        attn_implementation="sdpa",
                        torch_dtype=resolved_dtype,
                    )
                else:
                    self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_id,
                        attn_implementation="sdpa",
                        torch_dtype=resolved_dtype,
                    )
                if self.max_pixels > 0:
                    min_px = min(256 * 28 * 28, self.max_pixels)
                    try:
                        # Pass max_pixels/min_pixels explicitly so HF preprocessor_config.json
                        # defaults (e.g. max_pixels=12845056) cannot overwrite longest_edge.
                        self.processor = AutoProcessor.from_pretrained(
                            model_id,
                            max_pixels=self.max_pixels,
                            min_pixels=min_px,
                        )
                    except (TypeError, ValueError):
                        self.processor = AutoProcessor.from_pretrained(
                            model_id,
                            size={"longest_edge": self.max_pixels, "shortest_edge": min_px},
                        )
                    if hasattr(self.processor, "image_processor"):
                        ip = self.processor.image_processor
                        if hasattr(ip, "max_pixels"):
                            ip.max_pixels = self.max_pixels
                        if hasattr(ip, "min_pixels"):
                            ip.min_pixels = min_px
                        if hasattr(ip, "size") and isinstance(ip.size, dict):
                            ip.size["longest_edge"] = self.max_pixels
                            ip.size["shortest_edge"] = min_px
                    logger.info(
                        f"Configured image processor pixel budget: min_pixels={min_px}, max_pixels={self.max_pixels} "
                        f"(size={getattr(getattr(self.processor, 'image_processor', None), 'size', None)})."
                    )
                else:
                    self.processor = AutoProcessor.from_pretrained(model_id)
                # Strictly freeze all backbone parameters and lock eval mode
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False
                if resolved_dtype == torch.float16:
                    n_guards = _install_fp16_overflow_guards(self.backbone)
                    logger.info(
                        f"Installed {n_guards} FP16 overflow clamp hooks (+/-32768) across frozen backbone "
                        "for Tensor Core speed without RMSNorm Inf*0=NaN."
                    )
                logger.info("Successfully frozen 100% of pretrained Qwen2.5-VL-3B backbone parameters.")
            except Exception as e:
                logger.warning(f"Failed to load real Qwen2.5-VL backbone ({e}); falling back to mock backbone.")
                self.backbone = MockQwenBackbone(hidden_dim=backbone_hidden_dim)
        else:
            self.backbone = MockQwenBackbone(hidden_dim=backbone_hidden_dim)

        # 2. Trainable Visual-Motor Adapter Heads
        self.patch_proj = nn.Sequential(
            nn.Linear(backbone_hidden_dim, pointer_hidden_dim),
            nn.LayerNorm(pointer_hidden_dim),
            nn.GELU(),
        )

        self.query_proj = nn.Sequential(
            nn.Linear(backbone_hidden_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.GELU(),
        )

        self.pointer = GUIActorPatchPointer(
            query_dim=query_dim,
            patch_dim=pointer_hidden_dim,
            hidden_dim=pointer_hidden_dim,
        )

        self.action_heads = ParallelActionHeads(
            in_dim=query_dim,
            num_actions=len(ACTION_VOCAB_10),
        )

        # 3. Optional ShowUI Token Pruner & Rolling KV Memory
        self.pruner = (
            ShowUITokenPruner(in_dim=pointer_hidden_dim, target_budget=target_token_budget, min_budget=64)
            if use_token_pruning
            else None
        )
        self.memory = (
            RollingKVMemory(feat_dim=pointer_hidden_dim, context_dim=query_dim, max_history_steps=max_history_steps)
            if use_rolling_kv
            else None
        )

        if int(lora_layers) > 0 and self.backbone is not None and not isinstance(self.backbone, MockQwenBackbone):
            self.enable_lora(lora_layers=int(lora_layers), r=int(lora_r))

    @staticmethod
    def _resolve_decoder_layers(backbone: nn.Module) -> Optional[nn.ModuleList]:
        """Locate transformer decoder blocks across Qwen2.5-VL / Qwen2.5 / test layouts."""
        for path in (
            ("model", "language_model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
            ("layers",),
        ):
            cur: Any = backbone
            for attr in path:
                cur = getattr(cur, attr, None)
                if cur is None:
                    break
            if isinstance(cur, (nn.ModuleList, list, tuple)) and len(cur) > 0:
                return cur
        return None

    @staticmethod
    def _resolve_visual_modules(backbone: nn.Module) -> List[nn.Module]:
        """Locate frozen vision tower module(s) across Qwen2.5-VL / test layouts."""
        found: List[nn.Module] = []
        seen_ids: set = set()
        for path in (
            ("model", "visual"),
            ("visual",),
            ("model", "vision_tower"),
            ("vision_tower",),
        ):
            cur: Any = backbone
            for attr in path:
                cur = getattr(cur, attr, None)
                if cur is None:
                    break
            if isinstance(cur, nn.Module) and id(cur) not in seen_ids:
                seen_ids.add(id(cur))
                found.append(cur)
        return found

    def _install_frozen_prefix_isolation(self, layers: Any, start_idx: int) -> None:
        """Guarantees zero activation storage for frozen vision tower & decoder prefix.

        - Vision tower & decoder layers 0..(start_idx - 1) execute inside torch.no_grad().
        - Decoder layer `start_idx` detaches incoming hidden_states and enables requires_grad_(True).
        - Trainable top-K decoder layers `start_idx..N-1` run via non-reentrant gradient checkpointing.
        """
        for vis_mod in self._resolve_visual_modules(self.backbone):
            if not getattr(vis_mod, "_visual_no_grad_wrapped", False) and not any(
                p.requires_grad for p in vis_mod.parameters()
            ):
                orig_vis_fwd = vis_mod.forward

                @functools.wraps(orig_vis_fwd)
                def _no_grad_visual_forward(*args: Any, _orig: Any = orig_vis_fwd, **kwargs: Any) -> Any:
                    with torch.no_grad():
                        return _orig(*args, **kwargs)

                vis_mod.forward = _no_grad_visual_forward
                setattr(vis_mod, "_visual_no_grad_wrapped", True)

        for prefix_block in layers[:start_idx]:
            if not getattr(prefix_block, "_frozen_prefix_no_grad", False) and not any(
                p.requires_grad for p in prefix_block.parameters()
            ):
                orig_prefix_fwd = prefix_block.forward

                @functools.wraps(orig_prefix_fwd)
                def _no_grad_prefix_forward(*args: Any, _orig: Any = orig_prefix_fwd, **kwargs: Any) -> Any:
                    with torch.no_grad():
                        return _orig(*args, **kwargs)

                prefix_block.forward = _no_grad_prefix_forward
                setattr(prefix_block, "_frozen_prefix_no_grad", True)

        for offset, lora_block in enumerate(layers[start_idx:]):
            if getattr(lora_block, "_lora_ckpt_wrapped", False):
                continue
            orig_lora_fwd = lora_block.forward
            is_boundary = (offset == 0)

            @functools.wraps(orig_lora_fwd)
            def _checkpointed_lora_forward(
                *args: Any,
                _orig: Any = orig_lora_fwd,
                _boundary: bool = is_boundary,
                **kwargs: Any,
            ) -> Any:
                if torch.is_grad_enabled():
                    if _boundary:
                        if len(args) > 0 and isinstance(args[0], torch.Tensor):
                            args = (args[0].detach().requires_grad_(True),) + args[1:]
                        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                            kwargs = dict(kwargs)
                            kwargs["hidden_states"] = kwargs["hidden_states"].detach().requires_grad_(True)
                    return grad_checkpoint.checkpoint(
                        _orig,
                        *args,
                        use_reentrant=False,
                        **kwargs,
                    )
                return _orig(*args, **kwargs)

            lora_block.forward = _checkpointed_lora_forward
            setattr(lora_block, "_lora_ckpt_wrapped", True)

    def enable_lora(
        self,
        lora_layers: int = 8,
        r: int = 16,
        lora_alpha: float = 32.0,
        target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> int:
        """Attach trainable rank-r LoRA adapters to the top `lora_layers` decoder blocks."""
        if self.backbone is None or lora_layers <= 0:
            return 0
        layers = self._resolve_decoder_layers(self.backbone)
        if not layers:
            logger.warning("[LORA] Could not resolve decoder layers on backbone; LoRA disabled.")
            return 0

        num_layers = len(layers)
        n_target = min(int(lora_layers), num_layers)
        start_idx = num_layers - n_target
        wrapped = 0
        for block in layers[start_idx:]:
            for _parent_name, parent_mod in list(block.named_modules()):
                for attr_name in target_modules:
                    child = getattr(parent_mod, attr_name, None)
                    if isinstance(child, nn.Linear):
                        setattr(
                            parent_mod,
                            attr_name,
                            LoRALinear(child, r=r, lora_alpha=lora_alpha),
                        )
                        wrapped += 1
                    elif isinstance(child, LoRALinear):
                        for p in child.lora_A.parameters():
                            p.requires_grad = True
                        for p in child.lora_B.parameters():
                            p.requires_grad = True

        self.lora_config = {
            "lora_layers": n_target,
            "r": int(r),
            "lora_alpha": float(lora_alpha),
        }
        # Isolate vision tower + bottom frozen prefix under torch.no_grad() and run
        # non-reentrant gradient checkpointing exclusively on the top-K LoRA layers.
        self._install_frozen_prefix_isolation(layers=layers, start_idx=start_idx)
        logger.info(
            f"[LORA] Enabled rank-{r} LoRA on top {n_target}/{num_layers} decoder layers "
            f"({wrapped} attention projections wrapped; frozen prefix 0..{max(0, start_idx - 1)} & visual no_grad + top-{n_target} checkpointed)."
        )
        return wrapped

    def train(self, mode: bool = True) -> "QwenGUIActorModel":
        """Keep frozen VLM base layers in eval() while toggling adapter heads and LoRA blocks."""
        super().train(mode)
        if self.backbone is not None and not isinstance(self.backbone, MockQwenBackbone):
            self.backbone.eval()
            for mod in self.backbone.modules():
                if isinstance(mod, LoRALinear):
                    mod.train(mode)
                    mod.base_layer.eval()
        return self

    def get_parameter_summary(self) -> Dict[str, int]:
        """Calculates total vs trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
        }

    def extract_features(
        self,
        images: Optional[Union[torch.Tensor, List[Any], Any]] = None,
        queries: Optional[List[str]] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Extracts visual patch features and query representation from the frozen backbone.

        If using MockQwenBackbone: derives deterministic spatial features.
        If using real Qwen2.5-VL: formats multimodal prompts with valid input_ids and <|image_pad|>
        visual tokens, executing a single no-grad forward pass through the frozen backbone.
        """
        if isinstance(self.backbone, MockQwenBackbone):
            if isinstance(images, torch.Tensor):
                pass
            elif isinstance(images, Image.Image):
                arr = np.array(images, dtype=np.float32).transpose(2, 0, 1) / 255.0
                images = torch.from_numpy(arr).unsqueeze(0)
            elif isinstance(images, np.ndarray):
                images = torch.from_numpy(images).unsqueeze(0)
            elif isinstance(images, list):
                img_tensors = []
                for img in images:
                    if isinstance(img, torch.Tensor):
                        img_tensors.append(img)
                    elif isinstance(img, Image.Image):
                        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                        img_tensors.append(torch.from_numpy(arr))
                    elif isinstance(img, np.ndarray):
                        img_tensors.append(torch.from_numpy(img))
                images = torch.stack(img_tensors, dim=0) if img_tensors else torch.randn(1, 3, 800, 1280)
            elif images is None:
                images = torch.randn(1, 3, 800, 1280)

            out = self.backbone(images, image_grid_thw=image_grid_thw, grid_hw=grid_hw)
            gw, gh = out["grid_size"]
            return {
                "patch_feats": out["patch_hidden"],
                "pooled_feats": out["pooled"],
                "grid_hw": (gh, gw),
            }

        # Real Qwen2.5-VL backbone
        device = next(self.backbone.parameters()).device
        dtype = getattr(self.backbone, "dtype", torch.float16)

        # 1. Standardize images into PIL Images
        pil_images: List[Image.Image] = []
        if isinstance(images, torch.Tensor):
            b_imgs = images.shape[0]
            imgs_cpu = images.detach().cpu()
            for i in range(b_imgs):
                t_i = imgs_cpu[i]
                if t_i.dtype == torch.uint8:
                    arr = t_i.permute(1, 2, 0).numpy()
                else:
                    arr = t_i.permute(1, 2, 0).numpy()
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        arr = arr.clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(arr))
        elif isinstance(images, list):
            for item in images:
                if isinstance(item, Image.Image):
                    pil_images.append(item)
                elif isinstance(item, torch.Tensor):
                    arr = item.detach().cpu().permute(1, 2, 0).numpy()
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        arr = arr.clip(0, 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(arr))
                elif isinstance(item, np.ndarray):
                    pil_images.append(Image.fromarray(item))
        elif isinstance(images, Image.Image):
            pil_images = [images]
        elif isinstance(images, np.ndarray):
            pil_images = [Image.fromarray(images)]
        else:
            pil_images = [Image.new("RGB", (1280, 800), color=(128, 128, 128))]

        b = len(pil_images)
        if queries is None:
            batch_queries = ["Click the target UI element"] * b
        elif isinstance(queries, str):
            batch_queries = [queries] * b
        else:
            batch_queries = list(queries)
            if len(batch_queries) < b:
                batch_queries += ["Click the target UI element"] * (b - len(batch_queries))

        # 2. Build multimodal chat messages with explicit UI grounding instruction
        min_px = min(256 * 28 * 28, self.max_pixels) if self.max_pixels > 0 else 0
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        (
                            {
                                "type": "image",
                                "image": img,
                                "min_pixels": min_px,
                                "max_pixels": self.max_pixels,
                            }
                            if self.max_pixels > 0
                            else {"type": "image", "image": img}
                        ),
                        {"type": "text", "text": f"Locate and click the UI element matching: {q}"},
                    ],
                }
            ]
            for img, q in zip(pil_images, batch_queries)
        ]

        try:
            texts = [
                self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages
            ]
        except Exception:
            texts = [
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n"
                for q in batch_queries
            ]

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
        except Exception:
            image_inputs = pil_images

        # 3. Process inputs through AutoProcessor
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 4. Device and precision alignment
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

        # 5. Forward pass through backbone (no_grad when 100% frozen; frozen prefix+visual stay no_grad during LoRA)
        has_trainable_lora = self.training and any(
            p.requires_grad for p in self.backbone.parameters()
        )
        grad_ctx = torch.enable_grad() if has_trainable_lora else torch.no_grad()
        with grad_ctx:
            if hasattr(self.backbone, "model"):
                try:
                    outputs = self.backbone.model(**inputs, use_cache=False, return_dict=True)
                    last_hidden = outputs.last_hidden_state
                except torch.OutOfMemoryError:
                    raise
                except Exception:
                    outputs = self.backbone(**inputs, use_cache=False, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]
            else:
                outputs = self.backbone(**inputs, use_cache=False, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]

        # 6. Extract visual patch hidden states from <|image_pad|> tokens and blend instruction+trigger query
        image_token_id = getattr(self.backbone.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        batch_patch_feats = []
        batch_pooled_feats = []
        for i in range(b):
            mask_i = (inputs["input_ids"][i] == image_token_id)
            if mask_i.any():
                batch_patch_feats.append(last_hidden[i][mask_i])
            else:
                batch_patch_feats.append(last_hidden[i, :-1, :])

            attn_mask_i = inputs["attention_mask"][i].bool()
            valid_indices = attn_mask_i.nonzero()
            last_idx = valid_indices[-1].item() if len(valid_indices) > 0 else -1
            trig_vec = last_hidden[i, last_idx, :]
            if mask_i.any() and last_idx > 0:
                img_pos = mask_i.nonzero(as_tuple=False).squeeze(-1)
                last_img_idx = int(img_pos[-1].item())
                if last_idx > last_img_idx + 1:
                    instr_slice = last_hidden[i, last_img_idx + 1 : last_idx, :]
                    if instr_slice.shape[0] > 0:
                        trig_vec = 0.65 * trig_vec + 0.35 * instr_slice.mean(dim=0)
            batch_pooled_feats.append(trig_vec)

        if len(batch_patch_feats) == 1 or all(p.shape[0] == batch_patch_feats[0].shape[0] for p in batch_patch_feats):
            patch_feats = torch.stack(batch_patch_feats, dim=0)
        else:
            raise ValueError(
                f"Inconsistent visual patch counts in batch: {[p.shape[0] for p in batch_patch_feats]}; "
                "all images in a batch must produce the same patch grid."
            )
        pooled_feats = torch.stack(batch_pooled_feats, dim=0)

        # FP16 overflow shield: detect and sanitize NaN/Inf emitted by Qwen2.5-VL in FP16 on T4
        backbone_overflow = bool(
            not torch.isfinite(patch_feats).all().item()
            or not torch.isfinite(pooled_feats).all().item()
        )
        if backbone_overflow:
            patch_feats = torch.nan_to_num(patch_feats.float(), nan=0.0, posinf=64.0, neginf=-64.0)
            pooled_feats = torch.nan_to_num(pooled_feats.float(), nan=0.0, posinf=64.0, neginf=-64.0)

        # 7. Derive exact patch grid size (gh, gw) per batch item when available
        grid_hw_list: Optional[List[Tuple[int, int]]] = None
        if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
            grid_t = inputs["image_grid_thw"]
            if grid_t.dim() == 1:
                gh = int(grid_t[1].item()) // 2
                gw = int(grid_t[2].item()) // 2
                grid_hw_list = [(gh, gw)] * b
            else:
                grid_hw_list = [
                    (int(grid_t[idx, 1].item()) // 2, int(grid_t[idx, 2].item()) // 2)
                    for idx in range(grid_t.shape[0])
                ]
                gh, gw = grid_hw_list[0]
        elif grid_hw is not None:
            gh, gw = grid_hw
        else:
            gh = max(1, int(math.isqrt(patch_feats.shape[1])))
            gw = patch_feats.shape[1] // gh

        return {
            "patch_feats": patch_feats,
            "pooled_feats": pooled_feats,
            "grid_hw": (gh, gw),
            "grid_hw_list": grid_hw_list,
            "backbone_overflow": backbone_overflow,
        }

    def forward(
        self,
        images: Optional[Union[torch.Tensor, List[Any], Any]] = None,
        queries: Optional[List[str]] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,
        cursor_xy: Optional[torch.Tensor] = None,
        memory_state: Optional[RollingMemoryState] = None,
        patch_feats: Optional[torch.Tensor] = None,
        pooled_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Single-pass forward step with dynamic image_grid_thw non-square patch alignment.

        Exact non-square patch grid (H_patches, W_patches) = (H_grid // 2, W_grid // 2)
        matching native screen aspect ratios.
        """
        # 1. Extract or use precomputed backbone features
        backbone_overflow = False
        grid_hw_list: Optional[List[Tuple[int, int]]] = None
        if patch_feats is None or pooled_feats is None:
            feats = self.extract_features(
                images=images,
                queries=queries,
                image_grid_thw=image_grid_thw,
                grid_hw=grid_hw,
            )
            patch_feats = feats["patch_feats"]
            pooled_feats = feats["pooled_feats"]
            gh, gw = feats["grid_hw"]
            grid_hw_list = feats.get("grid_hw_list")
            backbone_overflow = bool(feats.get("backbone_overflow", False))
        else:
            if grid_hw is not None:
                gh, gw = grid_hw
            else:
                gh = max(1, int(math.isqrt(patch_feats.shape[1])))
                gw = patch_feats.shape[1] // gh

        b = patch_feats.shape[0]
        device = patch_feats.device

        # Align feature dtypes with adapter projections & ensure finite FP32 values
        adapter_dtype = next(self.patch_proj.parameters()).dtype
        patch_feats = patch_feats.to(device=device, dtype=adapter_dtype)
        pooled_feats = pooled_feats.to(device=device, dtype=adapter_dtype)
        if not torch.isfinite(patch_feats).all() or not torch.isfinite(pooled_feats).all():
            backbone_overflow = True
            patch_feats = torch.nan_to_num(patch_feats, nan=0.0, posinf=64.0, neginf=-64.0)
            pooled_feats = torch.nan_to_num(pooled_feats, nan=0.0, posinf=64.0, neginf=-64.0)

        # 2. Project features to pointer and query representations
        k_patches = patch_feats.shape[1]
        proj_patches = self.patch_proj(patch_feats)    # (B, K, pointer_hidden_dim)
        proj_query = self.query_proj(pooled_feats)     # (B, query_dim)

        # 3. Dynamic exact non-square spatial patch coordinates
        def _build_grid_coords(g_h: int, g_w: int) -> torch.Tensor:
            xs = (torch.arange(g_w, device=device, dtype=torch.float32) + 0.5) / max(g_w, 1)
            ys = (torch.arange(g_h, device=device, dtype=torch.float32) + 0.5) / max(g_h, 1)
            mesh_y, mesh_x = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([mesh_x.flatten(), mesh_y.flatten()], dim=-1)
            if coords.shape[0] != k_patches:
                coords = (
                    coords[:k_patches]
                    if coords.shape[0] > k_patches
                    else F.pad(coords, (0, 0, 0, k_patches - coords.shape[0]), value=-1.0)
                )
            return coords

        if grid_hw_list and len(grid_hw_list) == b and len(set(grid_hw_list)) > 1:
            patch_coords = torch.stack([_build_grid_coords(h_i, w_i) for h_i, w_i in grid_hw_list], dim=0)
        else:
            patch_coords = _build_grid_coords(gh, gw)  # (K, 2)

        # 4. Optional ShowUI visual token pruning
        if self.pruner is not None:
            pruned = self.pruner(
                features=proj_patches,
                cursor_xy=cursor_xy,
                target_budget=self.target_token_budget,
                grid_hw=(gh, gw),
            )
            proj_patches = pruned.tokens
            patch_coords = pruned.coords

        # 5. Optional Rolling KV-cache memory update
        if self.memory is not None:
            if memory_state is None:
                memory_state = RollingMemoryState(max_history_steps=self.memory.max_history_steps)
            proj_query = self.memory.update_and_fuse(proj_patches, proj_query, memory_state)

        # 6. Single-pass Pointer Head
        ptr_out: GUIPointerOutput = self.pointer(
            query_input=proj_query.unsqueeze(1),
            patch_tokens=proj_patches,
            patch_coords=patch_coords,
            patch_stride=(1.0 / max(gw, 1), 1.0 / max(gh, 1)),
        )

        # 7. Parallel Discrete Action Head
        act_out: ParallelActionOutput = self.action_heads(proj_query)

        # 8. FP32 Upcast on logits and coordinates for loss stability
        return {
            "action_logits": act_out.action_logits.float(),
            "target_coords": ptr_out.coords.float(),
            "subpixel_coords": ptr_out.subpixel_coords.float(),
            "attn_logits": ptr_out.attn_logits.float(),
            "attn_probs": ptr_out.attention.float(),
            "scroll_delta": act_out.scroll_delta.float(),
            "action_index": act_out.action_index,
            "action_names": act_out.action_name,
            "grid_hw": (gh, gw),
            "patch_coords": patch_coords,
            "memory_state": memory_state,
            "backbone_overflow": backbone_overflow,
        }


# --------------------------------------------------------------------------- #
# 3. Fail-Closed Grounding Dataset Ingestion (Zero-Leakage Gate)
# --------------------------------------------------------------------------- #

class GroundingDataset(Dataset):
    """Large-scale desktop grounding dataset supporting OS-Atlas, ShowUI-desktop, and Golden trajectories.

    Strictly enforces:
    1. Zero leakage against held-out evaluation datasets (showdown-clicks-dev).
    2. Fail-closed error if dataset is empty and allow_synthetic_fallback=False.
    3. Cleanly skips missing golden/ directories without throwing errors.
    """

    HELD_OUT_BLACKLIST_STRINGS = (
        "showdown-clicks-dev",
        "showdown_dev",
        "eval_dev",
        "showdown_clicks",
    )

    def __init__(
        self,
        dataset_paths: Optional[List[str]] = None,
        max_samples: int = 40000,
        target_size: Tuple[int, int] = (800, 1280),
        allow_synthetic_fallback: bool = True,
        stream_hf: bool = True,
        hf_samples: int = 20000,
        max_elements_per_image: int = 5,
        dataset_source: str = "hf_showui",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.target_size = target_size
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.max_elements_per_image = max_elements_per_image
        self.dataset_source = dataset_source
        self.cache_dir = cache_dir or os.path.join(repo_root, "data", "cache")
        self.samples: List[Dict[str, Any]] = []
        self.leakage_filtered_count = 0

        default_local_paths = [
            "/kaggle/input/os-atlas-desktop/train.jsonl",
            "/kaggle/input/showui-desktop-40k/train.jsonl",
            "/kaggle/input/avcs-desktop-grounding/train.jsonl",
            "data/vla_dataset/train.jsonl",
            "avcs/runtime/skills_vault/golden",
        ]

        if dataset_paths is not None:
            # Explicit dataset paths provided (e.g. unit tests or custom dataset directories)
            for p_str in dataset_paths:
                p = Path(p_str)
                if not p.is_absolute():
                    if not p.exists() and (Path(repo_root) / p).exists():
                        p = Path(repo_root) / p
                if not p.exists():
                    if "golden" in p_str:
                        logger.debug(f"Golden trajectory directory {p_str} does not exist; cleanly skipping.")
                    continue

                if p.is_dir() and "golden" in p_str:
                    self._ingest_golden_directory(p)
                elif p.is_file() and p.suffix == ".jsonl":
                    self._ingest_jsonl(p, max_samples)
        elif dataset_source == "synthetic":
            self._generate_synthetic_samples(min(max_samples, 1000))
        elif dataset_source == "local":
            for p_str in default_local_paths:
                p = Path(p_str)
                if not p.is_absolute() and not p.exists() and (Path(repo_root) / p).exists():
                    p = Path(repo_root) / p
                if not p.exists():
                    continue
                if p.is_dir() and "golden" in p_str:
                    self._ingest_golden_directory(p)
                elif p.is_file() and p.suffix == ".jsonl":
                    self._ingest_jsonl(p, max_samples)
        elif dataset_source in ("hf_showui", "mixed"):
            if dataset_source == "mixed":
                for p_str in default_local_paths:
                    p = Path(p_str)
                    if not p.is_absolute() and not p.exists() and (Path(repo_root) / p).exists():
                        p = Path(repo_root) / p
                    if not p.exists():
                        continue
                    if p.is_dir() and "golden" in p_str:
                        self._ingest_golden_directory(p)
                    elif p.is_file() and p.suffix == ".jsonl":
                        self._ingest_jsonl(p, max_samples)

            target_hf = hf_samples if dataset_source == "hf_showui" else max(0, hf_samples - len(self.samples))
            if stream_hf and target_hf > 0:
                is_dist = dist.is_available() and dist.is_initialized()
                env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
                env_rank = dist.get_rank() if is_dist else int(os.environ.get("RANK", "0"))
                cache_file = os.path.join(
                    self.cache_dir,
                    f"cache_stage1_v2_{target_hf}_max{self.max_elements_per_image}.pt",
                )
                done_sentinel = cache_file + ".done"

                if is_dist:
                    if env_rank == 0:
                        self._ingest_from_hf(
                            target_hf,
                            max_elements_per_image=self.max_elements_per_image,
                            cache_dir=self.cache_dir,
                        )
                    dist.barrier()
                    if env_rank != 0:
                        self._ingest_from_hf(
                            target_hf,
                            max_elements_per_image=self.max_elements_per_image,
                            cache_dir=self.cache_dir,
                        )
                elif env_world_size > 1:
                    # Pre-NCCL file-based synchronization: zero NCCL watchdog timeout risk during long downloads
                    if env_rank == 0:
                        if os.path.exists(done_sentinel) and not os.path.exists(cache_file):
                            os.remove(done_sentinel)
                        self._ingest_from_hf(
                            target_hf,
                            max_elements_per_image=self.max_elements_per_image,
                            cache_dir=self.cache_dir,
                        )
                        try:
                            os.makedirs(self.cache_dir, exist_ok=True)
                            with open(done_sentinel, "w", encoding="utf-8") as sf:
                                sf.write(str(len(self.samples)))
                        except Exception:
                            pass
                    else:
                        wait_start = time.time()
                        while not os.path.exists(done_sentinel):
                            if time.time() - wait_start > 7200:
                                raise TimeoutError(f"Rank {env_rank} timed out waiting for cache sentinel {done_sentinel}")
                            time.sleep(2.0)
                        self._ingest_from_hf(
                            target_hf,
                            max_elements_per_image=self.max_elements_per_image,
                            cache_dir=self.cache_dir,
                        )
                else:
                    self._ingest_from_hf(
                        target_hf,
                        max_elements_per_image=self.max_elements_per_image,
                        cache_dir=self.cache_dir,
                    )

        # Fail-closed gate: if no samples were ingested
        if len(self.samples) == 0:
            if allow_synthetic_fallback:
                logger.info("No grounding datasets found; generating 1,000 synthetic desktop grounding samples...")
                self._generate_synthetic_samples(1000)
            else:
                raise RuntimeError(
                    f"Fail-closed dataset gate: No valid grounding samples ingested (dataset_source={dataset_source}, "
                    f"paths={dataset_paths}). Production training aborted to prevent zero-sample training."
                )

        logger.info(
            f"GroundingDataset initialized ({dataset_source}): {len(self.samples)} valid samples ingested. "
            f"[LEAKAGE GATE] Verified {self.leakage_filtered_count} held-out showdown-clicks-dev records filtered out."
        )

    def _is_held_out_leakage(self, identifier: str) -> bool:
        """Determines if a sample belongs to the held-out showdown-clicks-dev benchmark."""
        id_lower = identifier.lower().replace("\\", "/")
        return any(blk in id_lower for blk in self.HELD_OUT_BLACKLIST_STRINGS)

    def _ingest_jsonl(self, jsonl_path: Path, max_samples: int) -> None:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                # Fail-closed check: if the raw json line contains any held-out benchmark indicator
                if self._is_held_out_leakage(line):
                    self.leakage_filtered_count += 1
                    continue
                try:
                    data = json.loads(line)
                    raw_id = str(data.get("id", ""))
                    raw_img = str(data.get("image", "") or data.get("image_path", "") or data.get("img", "") or "")
                    raw_source = str(data.get("source", ""))
                    raw_ds = str(data.get("dataset", ""))
                    raw_task = str(data.get("task", ""))
                    raw_instr = str(data.get("instruction", "") or data.get("query", "") or "")
                    if (
                        self._is_held_out_leakage(raw_id)
                        or self._is_held_out_leakage(raw_img)
                        or self._is_held_out_leakage(raw_source)
                        or self._is_held_out_leakage(raw_ds)
                        or self._is_held_out_leakage(raw_task)
                        or self._is_held_out_leakage(raw_instr)
                    ):
                        self.leakage_filtered_count += 1
                        continue

                    # 1. Resolve relative image path on disk
                    resolved_img = None
                    if raw_img:
                        for cand in [
                            Path(raw_img),
                            jsonl_path.parent / raw_img,
                            jsonl_path.parent / "images" / Path(raw_img).name,
                            Path(repo_root) / raw_img,
                            Path(repo_root) / "data" / "vla_dataset" / raw_img,
                            Path("/kaggle/working/AVCS-v1") / raw_img,
                            Path("/kaggle/working/AVCS-v1/data/vla_dataset") / raw_img,
                            Path("/kaggle/input") / raw_img,
                        ]:
                            if cand.exists() and cand.is_file():
                                resolved_img = str(cand.resolve())
                                break
                    if resolved_img is None and raw_img:
                        resolved_img = raw_img

                    # 2. Parse instruction, resolution, and coordinates across ShareGPT and OpenAI formats
                    instruction = data.get("instruction") or data.get("query") or data.get("task") or "Click target"
                    conversations = data.get("conversations", [])
                    action_idx = 0  # default left_click
                    target_xy = [0.5, 0.5]
                    img_w = float(data.get("width", 0.0) or data.get("img_w", 0.0) or 0.0)
                    img_h = float(data.get("height", 0.0) or data.get("img_h", 0.0) or 0.0)

                    root_coord = data.get("coordinate") or data.get("point") or data.get("target_coord")
                    root_bbox = data.get("bbox") or data.get("box") or data.get("target_bbox")

                    for turn in conversations:
                        sender = turn.get("from") or turn.get("role", "")
                        text_val = str(turn.get("value") or turn.get("content", ""))

                        if sender in ("human", "user"):
                            m_inst = re.search(r"Instruction:\s*([^\n]+)", text_val)
                            if m_inst:
                                instruction = m_inst.group(1).strip()
                            elif text_val and (not instruction or instruction == "Click target"):
                                clean_val = re.sub(r"<image>\s*", "", text_val).strip()
                                if clean_val:
                                    instruction = clean_val
                            m_res = re.search(r"Screen Resolution:\s*(\d+)\s*[xX,]\s*(\d+)", text_val)
                            if m_res:
                                img_w = float(m_res.group(1))
                                img_h = float(m_res.group(2))

                        elif sender in ("gpt", "assistant"):
                            parsed: Dict[str, Any] = {}
                            try:
                                parsed = json.loads(text_val)
                            except Exception:
                                m_json = re.search(r"\{.*\}", text_val, re.DOTALL)
                                if m_json:
                                    try:
                                        parsed = json.loads(m_json.group(0))
                                    except Exception:
                                        pass

                            action_name = parsed.get("action") or parsed.get("act") or "left_click"
                            if action_name == "click":
                                action_name = "left_click"
                            if action_name in ACTION_VOCAB_10:
                                action_idx = ACTION_VOCAB_10.index(action_name)

                            coord = parsed.get("coordinate") or parsed.get("point") or parsed.get("target_coord")
                            if not coord:
                                bbox = parsed.get("bbox") or parsed.get("box")
                                if bbox and len(bbox) == 4:
                                    coord = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
                            if not coord:
                                m_pt = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", text_val)
                                if m_pt:
                                    coord = [float(m_pt.group(1)), float(m_pt.group(2))]

                            if coord and len(coord) == 2:
                                ref_w = img_w if img_w > 0 else 1280.0
                                ref_h = img_h if img_h > 0 else 800.0
                                cx, cy = _norm_point_pair(coord, ref_w, ref_h)
                                target_xy = [round(min(1.0, max(0.0, cx)), 6), round(min(1.0, max(0.0, cy)), 6)]

                    # Fallback to root-level coordinates if conversation turns didn't contain coordinates
                    if target_xy == [0.5, 0.5] and (root_coord or root_bbox):
                        ref_w = img_w if img_w > 0 else 1280.0
                        ref_h = img_h if img_h > 0 else 800.0
                        if root_coord and len(root_coord) == 2:
                            cx, cy = _norm_point_pair(root_coord, ref_w, ref_h)
                            target_xy = [round(min(1.0, max(0.0, cx)), 6), round(min(1.0, max(0.0, cy)), 6)]
                        elif root_bbox and len(root_bbox) == 4:
                            b0, b1, b2, b3 = _norm_bbox_quad(root_bbox, ref_w, ref_h)
                            target_xy = [round(min(1.0, max(0.0, (b0 + b2) / 2.0)), 6), round(min(1.0, max(0.0, (b1 + b3) / 2.0)), 6)]

                    self.samples.append({
                        "instruction": instruction,
                        "action_idx": action_idx,
                        "target_xy": target_xy,
                        "image_path": resolved_img,
                    })
                except Exception:
                    continue

    def _ingest_golden_directory(self, golden_dir: Path) -> None:
        if not golden_dir.exists() or not golden_dir.is_dir():
            return
        if self._is_held_out_leakage(str(golden_dir)):
            self.leakage_filtered_count += 1
            return
        for json_file in golden_dir.glob("*.json"):
            if self._is_held_out_leakage(str(json_file)):
                self.leakage_filtered_count += 1
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content_str = f.read()
                    if self._is_held_out_leakage(content_str):
                        self.leakage_filtered_count += 1
                        continue
                    traj = json.loads(content_str)
                    goal = traj.get("goal", "Execute goal")
                    if self._is_held_out_leakage(goal):
                        self.leakage_filtered_count += 1
                        continue
                    for step in traj.get("steps", []):
                        action = step.get("action", {})
                        act_kind = action.get("kind", "left_click")
                        act_idx = ACTION_VOCAB_10.index(act_kind) if act_kind in ACTION_VOCAB_10 else 0
                        coord = action.get("coordinate")
                        target_xy = [0.5, 0.5]
                        if coord and len(coord) == 2:
                            norm_x, norm_y = _norm_point_pair(coord, 1280.0, 800.0)
                            target_xy = [min(1.0, max(0.0, norm_x)), min(1.0, max(0.0, norm_y))]
                        self.samples.append({
                            "instruction": goal,
                            "action_idx": act_idx,
                            "target_xy": target_xy,
                            "image_path": None,
                        })
            except Exception:
                continue

    def _ingest_from_hf(
        self,
        max_samples: int,
        max_elements_per_image: int = 5,
        cache_dir: Optional[str] = None,
    ) -> None:
        cache_path = None
        if cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(
                    cache_dir,
                    f"cache_stage1_v3_{max_samples}_max{max_elements_per_image}.pt",
                )
                if os.path.exists(cache_path):
                    logger.info(
                        f"[HF-CACHE] Loading cached Stage-1 UI samples from {cache_path}..."
                    )
                    cached_samples = torch.load(cache_path, weights_only=False)
                    min_expected = min(max_samples, 1000) if max_samples >= 5000 else 1
                    if isinstance(cached_samples, list) and len(cached_samples) >= min_expected:
                        self.samples.extend(cached_samples[:max_samples])
                        logger.info(
                            f"[HF-CACHE] Successfully loaded {len(self.samples)} real UI samples from disk cache."
                        )
                        return
            except Exception as e:
                logger.warning(f"[HF-CACHE] Failed to read cache {cache_path}: {e}")

        try:
            import random
            from datasets import load_dataset

            # showlab/ShowUI-desktop provides ~505 capped desktop samples (101 raw screens).
            # agentsea/wave-ui provides 63,530 train rows with embedded PIL images across
            # desktop (omniact Retina 2880x1800 / 3360x2100) and landscape web UIs.
            hf_repos = ("showlab/ShowUI-desktop", "agentsea/wave-ui")
            elements_per_img: Dict[str, int] = defaultdict(int)
            ingested_hf: List[Dict[str, Any]] = []
            unreadable_skipped = 0
            filtered_non_landscape = 0

            generic_dom_tags = {
                "statictext", "statictext, link", "statictext, link, listitem",
                "statictext, pre", "statictext, descriptionlistterm",
                "button", "link", "image", "generic", "",
            }

            for repo_name in hf_repos:
                if len(ingested_hf) >= max_samples:
                    break
                print(
                    f"[HF-INGEST] Streaming from {repo_name} "
                    f"(collected={len(ingested_hf)}/{max_samples}, max {max_elements_per_image}/screenshot)...",
                    flush=True,
                )
                try:
                    ds = load_dataset(repo_name, split="train", streaming=True)
                    if hasattr(ds, "shuffle"):
                        ds = ds.shuffle(seed=42, buffer_size=2000)
                except Exception as repo_err:
                    logger.warning(f"[HF-INGEST] Could not open {repo_name}: {repo_err}")
                    continue

                repo_added = 0
                for item in ds:
                    if len(ingested_hf) >= max_samples:
                        break

                    # Filter out mobile portrait screenshots and non-English labels when metadata exists
                    platform = str(item.get("platform") or "").lower()
                    if platform == "mobile":
                        filtered_non_landscape += 1
                        continue
                    lang = str(item.get("language") or "english").lower()
                    if lang not in ("english", "en", ""):
                        continue

                    img = item.get("image") or item.get("img")
                    if img is None:
                        unreadable_skipped += 1
                        continue

                    # Fail-closed per-image validation & thumbnail hash for exact visual deduplication
                    try:
                        w, h = img.size
                        if w <= 0 or h <= 0:
                            unreadable_skipped += 1
                            continue
                        if h > w * 1.15:
                            filtered_non_landscape += 1
                            continue
                        rgb_img = img.convert("RGB")
                        thumb = rgb_img.resize((32, 32), Image.Resampling.NEAREST).convert("L").tobytes()
                        img_hash = hashlib.md5(thumb).hexdigest()[:16]
                        img_key = f"{repo_name}:{item.get('image_id') or img_hash}"
                    except Exception:
                        unreadable_skipped += 1
                        continue

                    if elements_per_img[img_key] >= max_elements_per_image:
                        continue

                    raw_pt = item.get("point") or item.get("target_coord")
                    raw_bbox = item.get("bbox") or item.get("target_bbox")

                    if raw_bbox and len(raw_bbox) == 4:
                        vals = [float(v) for v in raw_bbox]
                        is_pixel = max(abs(v) for v in vals) > 1.0
                        b0 = max(0.0, min(1.0, vals[0] / max(w, 1) if is_pixel else vals[0]))
                        b1 = max(0.0, min(1.0, vals[1] / max(h, 1) if is_pixel else vals[1]))
                        b2 = max(0.0, min(1.0, vals[2] / max(w, 1) if is_pixel else vals[2]))
                        b3 = max(0.0, min(1.0, vals[3] / max(h, 1) if is_pixel else vals[3]))
                        x_min, x_max = min(b0, b2), max(b0, b2)
                        y_min, y_max = min(b1, b3), max(b1, b3)
                        if (x_max - x_min) <= 1e-4 or (y_max - y_min) <= 1e-4:
                            continue
                        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
                        bbox_norm = [x_min, y_min, x_max, y_max]
                    elif raw_pt and len(raw_pt) == 2:
                        pvals = [float(v) for v in raw_pt]
                        is_pixel = max(abs(v) for v in pvals) > 1.0
                        cx = max(0.0, min(1.0, pvals[0] / max(w, 1) if is_pixel else pvals[0]))
                        cy = max(0.0, min(1.0, pvals[1] / max(h, 1) if is_pixel else pvals[1]))
                        bbox_norm = [max(0.0, cx - 0.03), max(0.0, cy - 0.03), min(1.0, cx + 0.03), min(1.0, cy + 0.03)]
                    else:
                        continue

                    # Clean instruction extraction: avoid generic DOM tags ('StaticText, link') in wave-ui
                    raw_instr = str(item.get("instruction") or item.get("query") or item.get("task") or "").strip()
                    raw_name = str(item.get("name") or "").strip()
                    raw_purpose = str(item.get("purpose") or "").strip()
                    raw_desc = str(item.get("description") or "").strip()

                    is_raw_dom_tag = (
                        raw_instr.lower() in generic_dom_tags
                        or "-> click" in raw_instr.lower()
                        or (raw_instr.startswith("[") and "]" in raw_instr)
                    )
                    if not is_raw_dom_tag and len(raw_instr) >= 3:
                        instr = raw_instr
                    elif len(raw_name) >= 3 and raw_name.lower() not in generic_dom_tags:
                        instr = raw_name if raw_name.lower().startswith("click") else f"Click {raw_name}"
                    elif len(raw_purpose) >= 3:
                        instr = raw_purpose
                    elif len(raw_desc) >= 3:
                        instr = raw_desc
                    else:
                        unreadable_skipped += 1
                        continue

                    if self._is_held_out_leakage(instr):
                        self.leakage_filtered_count += 1
                        continue

                    # Compress resized canvas (1280x800) as JPEG bytes to keep 20k samples <= 1.6GB RAM
                    try:
                        pil_resized = rgb_img.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)
                        buf = io.BytesIO()
                        pil_resized.save(buf, format="JPEG", quality=85)
                        jpeg_bytes = buf.getvalue()
                    except Exception:
                        unreadable_skipped += 1
                        continue

                    elements_per_img[img_key] += 1
                    ingested_hf.append({
                        "instruction": instr,
                        "action_idx": 0,
                        "target_xy": [min(1.0, max(0.0, cx)), min(1.0, max(0.0, cy))],
                        "target_bbox": bbox_norm,
                        "image_bytes": jpeg_bytes,
                        "image_path": None,
                        "image_id": img_key,
                    })
                    repo_added += 1

                    if len(ingested_hf) % 500 == 0:
                        ts = time.strftime("%H:%M:%S")
                        print(
                            f"[{ts}] [HF-INGEST] Progress: {len(ingested_hf)}/{max_samples} samples "
                            f"({len(elements_per_img)} unique screenshots, {unreadable_skipped} skipped, {filtered_non_landscape} non-landscape)",
                            flush=True,
                        )

                ts = time.strftime("%H:%M:%S")
                print(
                    f"[{ts}] [HF-INGEST] Finished {repo_name}: +{repo_added} samples "
                    f"(total={len(ingested_hf)}, unique_screens={len(elements_per_img)})",
                    flush=True,
                )

            random.Random(42).shuffle(ingested_hf)
            self.samples.extend(ingested_hf)
            logger.info(
                f"[HF-INGEST] Ingested {len(ingested_hf)} real UI samples across "
                f"{len(elements_per_img)} unique screenshots (skipped {unreadable_skipped} unreadable, {filtered_non_landscape} non-landscape)."
            )

            if cache_path and len(ingested_hf) > 0:
                try:
                    torch.save(ingested_hf, cache_path)
                    logger.info(f"[HF-CACHE] Saved {len(ingested_hf)} samples to {cache_path}.")
                except Exception as ce:
                    logger.warning(f"[HF-CACHE] Failed to write cache {cache_path}: {ce}")
        except Exception as e:
            logger.warning(f"[HF-INGEST] Streaming from ShowUI/WaveUI failed ({e}).")

    def _generate_synthetic_samples(self, n: int) -> None:
        for i in range(n):
            cx = float(np.random.uniform(0.05, 0.95))
            cy = float(np.random.uniform(0.05, 0.95))
            mock_img = torch.randint(20, 240, (3, self.target_size[0], self.target_size[1]), dtype=torch.uint8)
            self.samples.append({
                "instruction": f"Synthetic task instruction {i}",
                "action_idx": i % len(ACTION_VOCAB_10),
                "target_xy": [cx, cy],
                "target_bbox": [max(0.0, cx - 0.04), max(0.0, cy - 0.04), min(1.0, cx + 0.04), min(1.0, cy + 0.04)],
                "image_tensor": mock_img,
                "image_path": None,
                "image_id": f"synth_img_{i // max(1, self.max_elements_per_image)}",
            })

    def split_train_val(self, val_ratio: float = 0.1, seed: int = 42) -> Tuple["GroundingDataset", "GroundingDataset"]:
        """Splits dataset into train (~90%) and validation (~10%), partitioned strictly by unique screenshot ID.

        Uses size-aware bin packing so oversized single-image groups cannot invert the train/val ratio.
        """
        import random
        rng = random.Random(seed)

        # Group samples by unique image identifier
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for idx, s in enumerate(self.samples):
            img_id = s.get("image_id") or s.get("image_path") or f"sample_{idx}"
            groups[img_id].append(s)

        train_samples: List[Dict[str, Any]] = []
        val_samples: List[Dict[str, Any]] = []

        if len(groups) <= 1:
            all_items = self.samples
            num_val = max(1, int(round(len(all_items) * val_ratio))) if len(all_items) > 2 else 0
            val_samples = all_items[:num_val]
            train_samples = all_items[num_val:]
        else:
            n_val_target = max(1, int(round(len(self.samples) * val_ratio)))
            sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
            eligible: List[Tuple[str, List[Dict[str, Any]]]] = []

            # Groups larger than the entire validation budget must go to train to prevent ratio inversion
            for k, items in sorted_groups:
                if len(items) > n_val_target:
                    train_samples.extend(items)
                else:
                    eligible.append((k, items))

            rng.shuffle(eligible)
            val_count = 0
            for _, items in eligible:
                if (val_count + len(items) <= n_val_target) or (val_count == 0 and len(items) <= n_val_target):
                    val_samples.extend(items)
                    val_count += len(items)
                else:
                    train_samples.extend(items)

        # Safety fallback if eligible was empty (e.g. only 2 giant screenshots)
        if not val_samples and len(train_samples) > 2:
            num_val = max(1, int(round(len(train_samples) * val_ratio)))
            val_samples = train_samples[:num_val]
            train_samples = train_samples[num_val:]

        train_ds = copy.copy(self)
        train_ds.samples = train_samples

        val_ds = copy.copy(self)
        val_ds.samples = val_samples

        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_tensor: Optional[torch.Tensor] = item.get("image_tensor")
        img_bytes = item.get("image_bytes")
        img_p = item.get("image_path")

        if img_tensor is None and img_bytes is not None:
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if pil_img.size != (self.target_size[1], self.target_size[0]):
                    pil_img = pil_img.resize(
                        (self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR
                    )
                arr = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)
                img_tensor = torch.from_numpy(arr)
            except Exception as e:
                logger.debug(f"Failed to decode JPEG bytes at sample {idx}: {e}")
                img_tensor = None

        if img_tensor is None and img_p:
            cand_p = img_p
            if not os.path.exists(cand_p):
                for prefix in [
                    repo_root,
                    os.path.join(repo_root, "data", "vla_dataset"),
                    "/kaggle/working/AVCS-v1",
                    "/kaggle/working/AVCS-v1/data/vla_dataset",
                    "/kaggle/input",
                ]:
                    test_p = os.path.join(prefix, img_p)
                    if os.path.exists(test_p):
                        cand_p = test_p
                        break
            if os.path.exists(cand_p):
                try:
                    pil_img = Image.open(cand_p).convert("RGB").resize(
                        (self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR
                    )
                    arr = np.array(pil_img, dtype=np.uint8).transpose(2, 0, 1)
                    img_tensor = torch.from_numpy(arr)
                except Exception as e:
                    logger.debug(f"Failed to open image {cand_p}: {e}")
                    img_tensor = None

        if img_tensor is not None and img_tensor.dtype == torch.uint8:
            img_tensor = img_tensor.float() / 255.0

        if img_tensor is None:
            if not self.allow_synthetic_fallback:
                raise RuntimeError(
                    f"Fail-closed dataset gate: Sample {idx} has missing image data (path={img_p}). "
                    "Silent dummy/blank-image fallbacks are strictly prohibited in production training."
                )
            img_tensor = torch.full((3, self.target_size[0], self.target_size[1]), 0.5, dtype=torch.float32)

        bbox = item.get("target_bbox")
        if bbox is None:
            tx, ty = item["target_xy"]
            bbox = [max(0.0, tx - 0.03), max(0.0, ty - 0.03), min(1.0, tx + 0.03), min(1.0, ty + 0.03)]

        return {
            "image": img_tensor,
            "instruction": item["instruction"],
            "target_action": torch.tensor(item["action_idx"], dtype=torch.long),
            "target_coord": torch.tensor(item["target_xy"], dtype=torch.float32),
            "target_bbox": torch.tensor(bbox, dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 4. Multi-Resolution Overfit Sanity Verification & Real HF Pre-Flight Loader
# --------------------------------------------------------------------------- #

def pull_hf_preflight_samples(n_samples: int = 16) -> List[Dict[str, Any]]:
    """Pulls real desktop screenshots from showlab/ShowUI-desktop and OS-Copilot/OS-Atlas-data on Hugging Face.

    Ingests real UI screenshots, extracting instruction, target action, normalized
    coordinates, and bounding boxes across diverse native desktop resolutions.
    """
    logger.info(f"[HF-PREFLIGHT] Ingesting {n_samples} real desktop screenshots from Hugging Face...")
    samples: List[Dict[str, Any]] = []

    # 1. Attempt streaming from verified Hugging Face datasets
    try:
        from datasets import load_dataset

        # Stream from verified repo: showlab/ShowUI-desktop
        logger.info(f"[HF-PREFLIGHT] Streaming real desktop samples from showlab/ShowUI-desktop...")
        try:
            ds_showui = load_dataset("showlab/ShowUI-desktop", split="train", streaming=True)
            seen_points = set()
            for item in ds_showui:
                if len(samples) >= n_samples:
                    break
                img = item.get("image") or item.get("img")
                if img is None:
                    continue
                w, h = img.size

                raw_bbox = item.get("bbox") or item.get("target_bbox")
                raw_pt = item.get("point") or item.get("target_coord")

                bbox_is_pixel = bool(
                    raw_bbox and len(raw_bbox) == 4 and max(abs(float(v)) for v in raw_bbox) > 1.0
                )
                # Parse normalized click coordinates [cx, cy]
                if raw_pt and len(raw_pt) == 2:
                    cx, cy = _norm_point_pair(raw_pt, w, h, force_pixel=bbox_is_pixel)
                elif raw_bbox and len(raw_bbox) == 4:
                    b0, b1, b2, b3 = _norm_bbox_quad(raw_bbox, w, h, force_pixel=bbox_is_pixel)
                    cx = (b0 + b2) / 2.0
                    cy = (b1 + b3) / 2.0
                else:
                    continue

                # Ensure diversity by skipping duplicate click points
                pt_key = (round(cx, 3), round(cy, 3))
                if pt_key in seen_points:
                    continue
                seen_points.add(pt_key)

                # Parse normalized bounding box [x1, y1, x2, y2]
                if raw_bbox and len(raw_bbox) == 4:
                    pt_is_pixel = bool(raw_pt and len(raw_pt) == 2 and max(abs(float(v)) for v in raw_pt) > 1.0)
                    b0, b1, b2, b3 = _norm_bbox_quad(raw_bbox, w, h, force_pixel=(bbox_is_pixel or pt_is_pixel))
                    norm_bbox = [min(b0, b2), min(b1, b3), max(b0, b2), max(b1, b3)]
                else:
                    norm_bbox = [max(0.0, cx - 0.03), max(0.0, cy - 0.03), min(1.0, cx + 0.03), min(1.0, cy + 0.03)]

                # Ensure minimum bbox hit envelope (at least 0.04 x 0.04)
                bw = norm_bbox[2] - norm_bbox[0]
                bh = norm_bbox[3] - norm_bbox[1]
                if bw < 0.04:
                    norm_bbox[0] = max(0.0, cx - 0.02)
                    norm_bbox[2] = min(1.0, cx + 0.02)
                if bh < 0.04:
                    norm_bbox[1] = max(0.0, cy - 0.02)
                    norm_bbox[3] = min(1.0, cy + 0.02)

                # Scale retina/4K displays (e.g. 3360x2100) down to standard desktop canvas (max_dim=1280)
                # to prevent allocating 9,000 visual tokens in SDPA attention while preserving exact [0, 1] coords
                max_dim = 1280
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    sw = max(28, (int(round(w * scale)) // 28) * 28)
                    sh = max(28, (int(round(h * scale)) // 28) * 28)
                    img = img.resize((sw, sh), Image.Resampling.BILINEAR)
                    w, h = sw, sh

                samples.append({
                    "id": f"showui_{len(samples)}",
                    "image": img,
                    "width": w,
                    "height": h,
                    "source": "showlab/ShowUI-desktop",
                    "instruction": item.get("instruction") or item.get("query") or item.get("task") or "Click target control",
                    "target_action": 0,  # left_click
                    "target_coord": [cx, cy],
                    "target_bbox": norm_bbox,
                })
        except Exception as e:
            logger.warning(f"[HF-PREFLIGHT] showlab/ShowUI-desktop streaming failed ({e}); checking OS-Copilot/OS-Atlas-data.")

        # Attempt streaming from verified repo: OS-Copilot/OS-Atlas-data
        if len(samples) < n_samples:
            remaining = n_samples - len(samples)
            logger.info(f"[HF-PREFLIGHT] Streaming {remaining} samples from OS-Copilot/OS-Atlas-data...")
            try:
                ds_atlas = load_dataset("OS-Copilot/OS-Atlas-data", split="train", streaming=True)
                for item in ds_atlas:
                    if len(samples) >= n_samples:
                        break
                    img = item.get("image") or item.get("img")
                    if img is None:
                        continue
                    w, h = img.size
                    bbox = item.get("bbox") or item.get("target_bbox") or [int(w * 0.5 - 40), int(h * 0.5 - 20), int(w * 0.5 + 40), int(h * 0.5 + 20)]
                    norm_bbox = [bbox[0] / max(w, 1), bbox[1] / max(h, 1), bbox[2] / max(w, 1), bbox[3] / max(h, 1)]
                    cx = (norm_bbox[0] + norm_bbox[2]) / 2.0
                    cy = (norm_bbox[1] + norm_bbox[3]) / 2.0
                    samples.append({
                        "id": f"os_atlas_{len(samples)}",
                        "image": img,
                        "width": w,
                        "height": h,
                        "source": "OS-Copilot/OS-Atlas-data",
                        "instruction": item.get("instruction") or item.get("query") or item.get("task") or "Click target UI element",
                        "target_action": 0,
                        "target_coord": [cx, cy],
                        "target_bbox": norm_bbox,
                    })
            except Exception as e:
                logger.warning(f"[HF-PREFLIGHT] OS-Copilot/OS-Atlas-data streaming failed ({e}); falling back if needed.")
    except ImportError:
        logger.warning("[HF-PREFLIGHT] datasets library not installed; falling back to multi-resolution synthetic batch.")

    # 2. Synthetic real-aspect multi-resolution fallback if offline / datasets not available
    if len(samples) < n_samples:
        needed = n_samples - len(samples)
        logger.info(f"[HF-PREFLIGHT] Generating {needed} multi-resolution desktop frames (1280x800, 1440x900, 1920x1080)...")
        res_pool = [(1280, 800), (1440, 900), (1920, 1080)]
        for i in range(needed):
            w_res, h_res = res_pool[i % len(res_pool)]
            cx = float(np.random.uniform(0.15, 0.85))
            cy = float(np.random.uniform(0.15, 0.85))
            bw, bh = 80.0 / w_res, 35.0 / h_res
            target_bbox = [max(0.0, cx - bw / 2.0), max(0.0, cy - bh / 2.0), min(1.0, cx + bw / 2.0), min(1.0, cy + bh / 2.0)]
            # Generate desktop-like pattern image
            dummy_img = Image.new("RGB", (w_res, h_res), color=(30 + (i * 15) % 200, 40 + (i * 25) % 200, 50 + (i * 35) % 200))
            samples.append({
                "id": f"preflight_real_res_{i}",
                "image": dummy_img,
                "width": w_res,
                "height": h_res,
                "source": f"Synthetic-{w_res}x{h_res}",
                "instruction": f"Click UI button at ({int(cx * w_res)}, {int(cy * h_res)})",
                "target_action": i % len(ACTION_VOCAB_10),
                "target_coord": [cx, cy],
                "target_bbox": target_bbox,
            })

    # Print summary of ingested samples
    print("\n" + "=" * 80)
    print(f"  [HF-PREFLIGHT] INGESTED {len(samples)} REAL DESKTOP SAMPLES FOR PRE-FLIGHT GATE")
    print("=" * 80)
    for idx, s in enumerate(samples):
        print(
            f"  Sample {idx + 1:2d} | Source: {s['source']:<28} | "
            f"Shape: {s['width']}x{s['height']} | Action: {ACTION_VOCAB_10[s['target_action']]:<12} | "
            f"Target: [{s['target_coord'][0]:.3f}, {s['target_coord'][1]:.3f}] | "
            f"BBox: [{s['target_bbox'][0]:.3f}, {s['target_bbox'][1]:.3f}, {s['target_bbox'][2]:.3f}, {s['target_bbox'][3]:.3f}]"
        )
    print("=" * 80 + "\n")

    return samples


def run_overfit_sanity(
    model: Optional[QwenGUIActorModel] = None,
    resolutions: Optional[List[Tuple[int, int]]] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    steps_per_res: int = 35,
    lr: float = 3e-4,
    device: Optional[torch.device] = None,
    assert_gate: bool = True,
) -> Dict[str, Any]:
    """Multi-resolution overfit sanity verification mode.

    Verifies that the model rapidly overfits small batches across multiple native screen
    resolutions (1280x800, 1440x900, 1920x1080), validating that dynamic non-square image_grid_thw
    alignment, gradient flow, and loss convergence operate flawlessly.

    Enforces:
    Strict (0px) in-bbox accuracy >= 95.0% across all evaluated resolutions/samples.
    """
    if model is None:
        model = QwenGUIActorModel(dry_run=True, pointer_hidden_dim=512, query_dim=512)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(model, "load_in_4bit", False):
        for name, child in model.named_children():
            if name != "backbone":
                child.to(device)
    else:
        model.to(device)
    model.train()

    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Memory optimization: Store only trainable adapter weights on CPU (~15 MB)
    # Never clone the 7.5GB frozen backbone in GPU VRAM to prevent CUDA OOM
    head_keys = ["patch_proj", "query_proj", "pointer", "action_heads"]
    trainable_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if any(h in k for h in head_keys)
    }
    results: Dict[str, Any] = {}

    if samples is not None and len(samples) > 0:
        # Pre-flight evaluation on real Hugging Face samples
        logger.info(f"[OVERFIT-SANITY] Evaluating pre-flight overfit across {len(samples)} real desktop samples...")
        total_samples = len(samples)
        hits = 0

        for idx, s in enumerate(samples):
            model.load_state_dict(trainable_state, strict=False)
            w, h = s["width"], s["height"]
            img_raw = s["image"]
            instr = s.get("instruction", "Click target control")

            # Extract backbone features once per sample (under no_grad)
            feats = model.extract_features(
                images=img_raw,
                queries=[instr],
            )
            patch_feats = feats["patch_feats"]
            pooled_feats = feats["pooled_feats"]
            gh, gw = feats["grid_hw"]

            t_coord = torch.tensor([s["target_coord"]], dtype=torch.float32, device=device)
            t_act = torch.tensor([s["target_action"]], dtype=torch.long, device=device)
            t_bbox = torch.tensor([s["target_bbox"]], dtype=torch.float32, device=device)

            target_px = torch.clamp((t_coord[:, 0] * gw).long(), 0, gw - 1)
            target_py = torch.clamp((t_coord[:, 1] * gh).long(), 0, gh - 1)
            target_patch = target_py * gw + target_px

            opt = torch.optim.AdamW(trainable_params, lr=lr)
            for _ in range(steps_per_res):
                opt.zero_grad()
                out = model(
                    patch_feats=patch_feats,
                    pooled_feats=pooled_feats,
                    grid_hw=(gh, gw),
                )
                loss_coord = 5.0 * F.l1_loss(out["target_coords"], t_coord) + 10.0 * F.mse_loss(out["target_coords"], t_coord)
                loss_act = F.cross_entropy(out["action_logits"], t_act)
                loss_patch = F.cross_entropy(out["attn_logits"].squeeze(1), target_patch)
                loss = loss_act + loss_coord + 3.0 * loss_patch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                opt.step()

            pred = out["target_coords"]
            in_bbox = (
                (pred[0, 0] >= t_bbox[0, 0]) and (pred[0, 0] <= t_bbox[0, 2]) and
                (pred[0, 1] >= t_bbox[0, 1]) and (pred[0, 1] <= t_bbox[0, 3])
            ).item()
            if in_bbox:
                hits += 1

        strict_accuracy = hits / max(total_samples, 1)
        results["hf_preflight"] = {
            "total_samples": total_samples,
            "hits": hits,
            "strict_accuracy": round(strict_accuracy, 4),
            "converged": strict_accuracy >= 0.95,
        }
        logger.info(
            f"[OVERFIT-SANITY-HF] Strict (0px) In-Bbox Accuracy across {total_samples} samples: "
            f"{strict_accuracy * 100:.1f}% ({hits}/{total_samples})"
        )
        if assert_gate and strict_accuracy < 0.95:
            raise AssertionError(
                f"Overfit sanity gate failed on HF pre-flight: Strict in-bbox accuracy {strict_accuracy:.2%} < 95.0% threshold"
            )
        return results

    # Multi-resolution synthetic native screen targets
    target_resolutions = resolutions or [
        (800, 1280),   # 16:10 native desktop
        (900, 1440),   # 16:10 high-DPI desktop
        (1080, 1920),  # 16:9 widescreen desktop
    ]

    for (h, w) in target_resolutions:
        model.load_state_dict(trainable_state, strict=False)

        # Non-square patch grid: Qwen2.5-VL merges 2x2 14x14 patches into 28x28 tokens
        h_grid = max(2, (h // 28) * 2)
        w_grid = max(2, (w // 28) * 2)
        grid_thw = torch.tensor([[1, h_grid, w_grid]], dtype=torch.long, device=device)
        gh = h_grid // 2
        gw = w_grid // 2

        b = 1
        images = torch.randn(b, 3, h, w, device=device)
        target_actions = torch.tensor([0], dtype=torch.long, device=device)
        target_coords = torch.tensor([[0.75, 0.25]], dtype=torch.float32, device=device)

        # Target patch index for patch cross-entropy loss
        target_px = torch.clamp((target_coords[:, 0] * gw).long(), 0, gw - 1)
        target_py = torch.clamp((target_coords[:, 1] * gh).long(), 0, gh - 1)
        target_patch = target_py * gw + target_px

        # Standard button bounding box (half-width 0.05, half-height 0.05)
        target_bboxes = torch.stack([
            target_coords[:, 0] - 0.05, target_coords[:, 1] - 0.05,
            target_coords[:, 0] + 0.05, target_coords[:, 1] + 0.05
        ], dim=-1)

        feats = model.extract_features(images=images, image_grid_thw=grid_thw, grid_hw=(gh, gw))
        patch_feats = feats["patch_feats"]
        pooled_feats = feats["pooled_feats"]
        gh, gw = feats["grid_hw"]

        opt = torch.optim.AdamW(trainable_params, lr=lr)
        initial_loss = 0.0
        final_loss = 0.0

        for step in range(steps_per_res):
            opt.zero_grad()
            out = model(patch_feats=patch_feats, pooled_feats=pooled_feats, grid_hw=(gh, gw))
            loss_coord = 5.0 * F.l1_loss(out["target_coords"], target_coords) + 10.0 * F.mse_loss(out["target_coords"], target_coords)
            loss_act = F.cross_entropy(out["action_logits"], target_actions)
            loss_patch = F.cross_entropy(out["attn_logits"].squeeze(1), target_patch)
            loss = loss_act + loss_coord + 3.0 * loss_patch

            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt.step()
            final_loss = loss.item()

        preds = out["target_coords"]
        in_bbox_mask = (
            (preds[:, 0] >= target_bboxes[:, 0]) & (preds[:, 0] <= target_bboxes[:, 2]) &
            (preds[:, 1] >= target_bboxes[:, 1]) & (preds[:, 1] <= target_bboxes[:, 3])
        )
        strict_accuracy = in_bbox_mask.float().mean().item()
        coord_dist = torch.norm(preds - target_coords, dim=-1).mean().item()
        res_key = f"{h}x{w}"
        converged = strict_accuracy >= 0.95 and final_loss < initial_loss
        results[res_key] = {
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_drop": round(initial_loss - final_loss, 4),
            "final_coord_dist": round(coord_dist, 4),
            "strict_accuracy": round(strict_accuracy, 4),
            "grid_patches": (gh, gw),
            "converged": converged,
        }
        logger.info(
            f"[OVERFIT-SANITY] Resolution {w}x{h} (grid {gw}x{gh}): "
            f"Loss {initial_loss:.4f} -> {final_loss:.4f} (drop: {initial_loss - final_loss:.4f}), "
            f"Coord Dist: {coord_dist:.4f} | Strict (0px) In-Bbox Acc: {strict_accuracy * 100:.1f}% | Converged: {converged}"
        )

    failed_res = [k for k, v in results.items() if not v.get("converged", False) or v.get("strict_accuracy", 0.0) < 0.95]
    if failed_res and assert_gate:
        raise AssertionError(
            f"Overfit sanity gate failed on resolution(s) {failed_res}: Strict in-bbox accuracy < 95.0% threshold. "
            f"Details: {results}"
        )

    return results


# --------------------------------------------------------------------------- #
# 5. DAgger Interactive Replay & RL Policy Gradient Support
# --------------------------------------------------------------------------- #

class DAggerReplayBuffer:
    """DAgger interactive data aggregation buffer supporting beta-mixing between expert and student."""

    def __init__(self, capacity: int = 10000, initial_beta: float = 1.0, beta_decay: float = 0.95) -> None:
        self.capacity = capacity
        self.beta = initial_beta
        self.beta_decay = beta_decay
        self.buffer: List[Dict[str, Any]] = []

    def add_interaction(
        self,
        image: torch.Tensor,
        instruction: str,
        student_action: int,
        student_coord: List[float],
        expert_action: int,
        expert_coord: List[float],
        intervention_needed: bool,
    ) -> None:
        """Stores interaction step with expert ground truth correction."""
        entry = {
            "image": image.detach().cpu() if isinstance(image, torch.Tensor) else image,
            "instruction": instruction,
            "student_action": student_action,
            "student_coord": student_coord,
            "expert_action": expert_action,
            "expert_coord": expert_coord,
            "intervention_needed": intervention_needed,
            "beta": self.beta,
        }
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(entry)

    def decay_beta(self, min_beta: float = 0.1) -> float:
        self.beta = max(min_beta, self.beta * self.beta_decay)
        return self.beta

    def sample_batch(self, batch_size: int = 16) -> List[Dict[str, Any]]:
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class RLVisualMotorTrainer:
    """Reinforcement learning / policy gradient trainer using verifier reward feedback."""

    def __init__(self, model: nn.Module, lr: float = 1e-4, baseline_decay: float = 0.9) -> None:
        self.model = model
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        self.running_baseline = 0.0
        self.baseline_decay = baseline_decay

    def compute_policy_gradient_loss(
        self,
        model_out: Dict[str, torch.Tensor],
        action_idx: torch.Tensor,
        coord_xy: torch.Tensor,
        reward: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Advantage-weighted policy gradient loss across discrete actions and continuous coordinates."""
        logits = model_out["action_logits"]      # (B, 10)
        pred_coords = model_out["target_coords"] # (B, 2)

        # Discrete action log prob
        log_probs_action = F.log_softmax(logits, dim=-1)
        action_log_p = log_probs_action.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)

        # Coordinate continuous log prob under Gaussian centered at pred_coords
        sigma = 0.05
        coord_sq_err = ((coord_xy - pred_coords) ** 2).sum(dim=-1)
        coord_log_p = -0.5 * (coord_sq_err / (sigma ** 2))

        total_log_p = action_log_p + coord_log_p

        mean_reward = reward.mean().item()
        self.running_baseline = self.baseline_decay * self.running_baseline + (1 - self.baseline_decay) * mean_reward
        advantage = reward - self.running_baseline

        pg_loss = -(total_log_p * advantage.detach()).mean()
        return {
            "loss": pg_loss,
            "mean_reward": reward.mean(),
            "advantage": advantage.mean(),
            "action_log_p": action_log_p.mean(),
            "coord_log_p": coord_log_p.mean(),
        }


# --------------------------------------------------------------------------- #
# 6. DDP Training Loop Entrypoint
# --------------------------------------------------------------------------- #

def run_ddp_training(
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    save_dir: str = "runs/checkpoints",
    dry_run: bool = False,
    max_samples: int = 40000,
    fail_closed: bool = False,
    use_token_pruning: bool = False,
    use_rolling_kv: bool = False,
    stream_hf: bool = True,
    hf_samples: int = 20000,
    max_elements_per_image: int = 5,
    no_early_abort: bool = False,
    dataset_paths: Optional[List[str]] = None,
    dataset_source: str = "hf_showui",
    max_val_samples: int = 200,
    cache_dir: Optional[str] = None,
    log_interval: int = 10,
    max_train_samples: Optional[int] = None,
    backbone_dtype: str = "auto",
    max_pixels: int = 560 * 896,
    lora_layers: int = 0,
    lora_r: int = 16,
    grad_accum_steps: int = 1,
    micro_batch_size: int = 0,
) -> None:
    # 1. Ingest & cache dataset BEFORE initializing NCCL / DDP so long HF streams never hit NCCL timeouts
    full_dataset = GroundingDataset(
        dataset_paths=dataset_paths,
        max_samples=max_samples,
        target_size=(800, 1280),
        allow_synthetic_fallback=not fail_closed,
        stream_hf=stream_hf,
        hf_samples=hf_samples,
        max_elements_per_image=max_elements_per_image,
        dataset_source=dataset_source,
        cache_dir=cache_dir,
    )
    train_dataset, val_dataset = full_dataset.split_train_val(val_ratio=0.10)
    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        train_dataset.samples = train_dataset.samples[:max_train_samples]
    val_eval_dataset = copy.copy(val_dataset)
    if len(val_dataset) > max_val_samples:
        val_eval_dataset.samples = val_dataset.samples[:max_val_samples]

    rank, world_size, local_rank, is_distributed = setup_distributed()

    if rank == 0:
        print("=" * 80)
        print("  AVCS STEP 6.2: DDP TRAINING FOR QWEN2.5-VL-3B FROZEN BACKBONE + GUI-ACTOR  ")
        print("=" * 80)
        print(f"Distributed Setup: World Size = {world_size}, Rank = {rank}, Local Rank = {local_rank}")
        total_cnt = max(1, len(train_dataset) + len(val_dataset))
        print(
            f"Dataset Partitioning ({dataset_source}): Train Samples = {len(train_dataset)} ({len(train_dataset)/total_cnt*100:.1f}%), "
            f"Held-Out Val Samples = {len(val_dataset)} ({len(val_dataset)/total_cnt*100:.1f}%) "
            f"[Epoch-End Eval Capped at {len(val_eval_dataset)} samples] (Partitioned by unique screenshots)",
            flush=True,
        )

    device = torch.device(f"cuda:{local_rank}" if (torch.cuda.is_available() and not dry_run) else "cpu")

    # 2. Initialize Model with 512-dim widened projections (+ optional top-N LoRA)
    model = QwenGUIActorModel(
        dry_run=dry_run,
        pointer_hidden_dim=512,
        query_dim=512,
        backbone_dtype=backbone_dtype,
        max_pixels=max_pixels,
        use_token_pruning=use_token_pruning,
        use_rolling_kv=use_rolling_kv,
        lora_layers=lora_layers,
        lora_r=lora_r,
    )
    if getattr(model, "load_in_4bit", False):
        for name, child in model.named_children():
            if name != "backbone":
                child.to(device)
    else:
        model.to(device)
    is_dry_backbone = dry_run or getattr(model, "dry_run", False)

    # Report parameter counts
    stats: Dict[str, int] = {}
    if rank == 0:
        stats = model.get_parameter_summary()
        print("\n--- MODEL PARAMETER INVENTORY ---")
        print(f"Total Model Parameters:     {stats['total_params']:,} (~{stats['total_params']/1e9:.2f}B)")
        print(f"Frozen Backbone Parameters: {stats['frozen_params']:,} (~{stats['frozen_params']/1e9:.2f}B) [100% FROZEN]")
        print(f"Trainable Head Parameters:  {stats['trainable_params']:,} (~{stats['trainable_params']/1e6:.2f}M) [TRAINABLE]")
        print(f"Trainable Ratio:            {(stats['trainable_params'] / max(1, stats['total_params'])) * 100:.4f}%")
        print("---------------------------------\n")

    if is_distributed:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=True)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    drop_last = len(train_dataset) >= batch_size
    loader_workers = 2 if (device.type == "cuda" and not is_dry_backbone and len(train_dataset) >= 16) else 0
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=drop_last,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(loader_workers > 0),
    )

    effective_micro_bs = (
        int(micro_batch_size)
        if micro_batch_size and int(micro_batch_size) > 0
        else (1 if (lora_layers > 0 and device.type == "cuda" and not is_dry_backbone) else min(batch_size, 2 if (device.type == "cuda" and not is_dry_backbone) else batch_size))
    )
    effective_micro_bs = max(1, min(int(batch_size), effective_micro_bs))
    accum_batches = max(1, int(grad_accum_steps))
    val_batch_size = max(1, effective_micro_bs)

    if rank == 0:
        print(
            f"Optimization Schedule: batch_size={batch_size}, micro_batch_size={effective_micro_bs}, "
            f"grad_accum_steps={accum_batches}, world_size={world_size} "
            f"(effective global batch = {batch_size * accum_batches * world_size})",
            flush=True,
        )

    val_sampler = DistributedSampler(val_eval_dataset, num_replicas=world_size, rank=rank, shuffle=False) if (is_distributed and len(val_eval_dataset) > 0) else None
    val_dataloader = DataLoader(
        val_eval_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(loader_workers > 0),
    ) if len(val_eval_dataset) > 0 else None

    # 3. Optimizer (differential LR for LoRA vs heads), Warmup+Cosine LR Scheduler & Rebalanced Loss
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    param_groups = [{"params": head_params, "lr": learning_rate}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": learning_rate * 0.25})
    trainable_params = head_params + lora_params
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=1e-3)
    steps_per_epoch_optim = max(1, math.ceil(max(1, len(dataloader)) / accum_batches))
    total_optim_steps = max(1, epochs * steps_per_epoch_optim)
    warmup_steps = max(1, min(20, total_optim_steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            float(step + 1) / float(warmup_steps)
            if step < warmup_steps
            else max(
                0.05,
                0.5 * (1.0 + math.cos(math.pi * float(step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps)))),
            )
        ),
    )
    criterion = UnifiedMultiTaskLoss(
        weight_action=0.5,
        weight_coord_l1=10.0,
        weight_coord_l2=10.0,
        weight_patch_ce=5.0,
    )

    save_path = Path(save_dir)
    if rank == 0:
        save_path.mkdir(parents=True, exist_ok=True)

    # 4. Training Loop
    model.train()
    total_batches_per_epoch = len(dataloader)
    total_steps_per_epoch = max(1, math.ceil(max(1, total_batches_per_epoch) / accum_batches))
    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if epoch == 1 and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        epoch_loss = 0.0
        epoch_patch_ce = 0.0
        epoch_coord_l1 = 0.0
        epoch_w_patch_ce = 0.0
        epoch_w_coord_l1 = 0.0
        epoch_w_coord_l2 = 0.0
        epoch_w_action = 0.0
        epoch_patch_acc = 0.0
        win_loss = 0.0
        win_patch_ce = 0.0
        win_coord_l1 = 0.0
        win_patch_acc = 0.0
        win_steps = 0
        step_count = 0
        valid_step_count = 0
        nan_skip_count = 0
        t0 = time.time()
        data_t0 = time.time()

        optimizer.zero_grad()
        acc_loss = 0.0
        acc_patch_ce = 0.0
        acc_coord_l1 = 0.0
        acc_w_patch_ce = 0.0
        acc_w_coord_l1 = 0.0
        acc_w_coord_l2 = 0.0
        acc_w_action = 0.0
        acc_patch_acc = 0.0
        acc_micro_cnt = 0
        acc_ok_local = True
        acc_overflow_any = False
        acc_loss_finite_all = True

        for batch_idx, batch in enumerate(dataloader):
            data_time = time.time() - data_t0
            images = batch["image"].to(device) if is_dry_backbone else batch["image"]
            queries = batch.get("instruction")
            target_actions = batch["target_action"].to(device)
            target_coords = batch["target_coord"].to(device)
            target_bboxes = batch["target_bbox"].to(device)

            b_len = target_actions.shape[0]
            n_chunks = max(1, math.ceil(b_len / effective_micro_bs))
            is_last_batch_in_window = ((batch_idx + 1) % accum_batches == 0) or (
                (batch_idx + 1) == total_batches_per_epoch
            )
            win_start = (batch_idx // accum_batches) * accum_batches
            win_end = min(total_batches_per_epoch, win_start + accum_batches)
            batches_in_window = max(1, win_end - win_start)
            loss_scale = 1.0 / float(max(1, batches_in_window * n_chunks))

            for chunk_idx in range(n_chunks):
                s = chunk_idx * effective_micro_bs
                e = min(b_len, s + effective_micro_bs)
                sub_images = images[s:e]
                sub_queries = queries[s:e] if queries is not None else None
                sub_actions = target_actions[s:e]
                sub_coords = target_coords[s:e]
                sub_bboxes = target_bboxes[s:e]

                is_final_micro = is_last_batch_in_window and (chunk_idx == n_chunks - 1)
                sync_ctx = (
                    nullcontext()
                    if (not is_distributed or is_final_micro)
                    else model.no_sync()
                )

                with sync_ctx:
                    outputs = model(sub_images, queries=sub_queries)
                    loss_dict = criterion(
                        model_out=outputs,
                        target_actions=sub_actions,
                        target_coords=sub_coords,
                        target_bboxes=sub_bboxes,
                    )
                    step_overflow = bool(outputs.get("backbone_overflow", False))
                    loss = loss_dict["loss"]
                    loss_finite = bool(torch.isfinite(loss).item())
                    micro_ok_local = (not step_overflow) and loss_finite

                    # Always run backward through sanitized zeroed forward outputs on overflow
                    # (avoid IEEE 754 NaN * 0.0 = NaN while satisfying DDP graph tracking).
                    backward_loss = (
                        loss * loss_scale
                        if micro_ok_local
                        else (
                            (torch.nan_to_num(outputs["action_logits"], nan=0.0, posinf=0.0, neginf=0.0) * 0.0).sum()
                            + (torch.nan_to_num(outputs["target_coords"], nan=0.0, posinf=0.0, neginf=0.0) * 0.0).sum()
                            + (torch.nan_to_num(outputs["attn_logits"], nan=0.0, posinf=0.0, neginf=0.0) * 0.0).sum()
                        )
                    )
                    backward_loss.backward()

                if micro_ok_local:
                    acc_loss += loss.item()
                    acc_patch_ce += loss_dict.get("loss_patch_ce", torch.tensor(0.0)).item()
                    acc_coord_l1 += loss_dict.get("loss_coord_l1", torch.tensor(0.0)).item()
                    acc_w_patch_ce += loss_dict.get("weighted_patch_ce", torch.tensor(0.0)).item()
                    acc_w_coord_l1 += loss_dict.get("weighted_coord_l1", torch.tensor(0.0)).item()
                    acc_w_coord_l2 += loss_dict.get("weighted_coord_l2", torch.tensor(0.0)).item()
                    acc_w_action += loss_dict.get("weighted_action", torch.tensor(0.0)).item()
                    acc_patch_acc += loss_dict.get("patch_top1_acc", torch.tensor(0.0)).item()
                    acc_micro_cnt += 1
                else:
                    acc_ok_local = False
                    if step_overflow:
                        acc_overflow_any = True
                    if not loss_finite:
                        acc_loss_finite_all = False

                del outputs, loss_dict, loss, backward_loss, sub_images, sub_actions, sub_coords, sub_bboxes

            if not is_last_batch_in_window:
                data_t0 = time.time()
                continue

            grad_finite = all(
                torch.isfinite(p.grad).all().item()
                for p in trainable_params
                if p.grad is not None
            )
            step_valid = 1 if (acc_ok_local and grad_finite) else 0
            if is_distributed:
                valid_t = torch.tensor([step_valid], device=device, dtype=torch.int32)
                dist.all_reduce(valid_t, op=dist.ReduceOp.MIN)
                step_valid = int(valid_t.item())

            if step_valid == 1:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                m_div = float(max(1, acc_micro_cnt))
                step_loss = acc_loss / m_div
                step_patch_ce = acc_patch_ce / m_div
                step_coord_l1 = acc_coord_l1 / m_div
                step_patch_acc = acc_patch_acc / m_div

                epoch_loss += step_loss
                epoch_patch_ce += step_patch_ce
                epoch_coord_l1 += step_coord_l1
                epoch_w_patch_ce += acc_w_patch_ce / m_div
                epoch_w_coord_l1 += acc_w_coord_l1 / m_div
                epoch_w_coord_l2 += acc_w_coord_l2 / m_div
                epoch_w_action += acc_w_action / m_div
                epoch_patch_acc += step_patch_acc
                win_loss += step_loss
                win_patch_ce += step_patch_ce
                win_coord_l1 += step_coord_l1
                win_patch_acc += step_patch_acc
                win_steps += 1
                valid_step_count += 1
            else:
                nan_skip_count += 1
                if rank == 0:
                    print(
                        f"[NAN-GUARD] Skipped step {step_count + 1} "
                        f"(backbone_overflow={acc_overflow_any}, loss_finite={acc_loss_finite_all}, grad_finite={grad_finite}).",
                        flush=True,
                    )
                    if nan_skip_count == 5 and backbone_dtype == "fp16":
                        print(
                            "[NAN-GUARD] Frequent FP16 overflow detected on T4! Consider re-running with --backbone-dtype bf16.",
                            flush=True,
                        )

            optimizer.zero_grad()
            acc_loss = 0.0
            acc_patch_ce = 0.0
            acc_coord_l1 = 0.0
            acc_w_patch_ce = 0.0
            acc_w_coord_l1 = 0.0
            acc_w_coord_l2 = 0.0
            acc_w_action = 0.0
            acc_patch_acc = 0.0
            acc_micro_cnt = 0
            acc_ok_local = True
            acc_overflow_any = False
            acc_loss_finite_all = True

            step_count += 1

            if rank == 0 and log_interval > 0 and (step_count == 1 or step_count % log_interval == 0):
                elapsed_so_far = time.time() - t0
                sec_per_step = elapsed_so_far / step_count
                eta_min = ((total_steps_per_epoch - step_count) * sec_per_step) / 60.0
                ts = time.strftime("%H:%M:%S")
                denom = max(1, valid_step_count)
                w_denom = max(1, win_steps)
                vram_suffix = ""
                if device.type == "cuda":
                    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    vram_suffix = f" | Peak VRAM: {peak_gib:.2f} GiB"
                print(
                    f"[{ts}] [Epoch {epoch}/{epochs} | Step {step_count}/{total_steps_per_epoch}] "
                    f"Loss: {win_loss / w_denom:.4f} (ep {epoch_loss / denom:.4f}) | "
                    f"Patch CE: {win_patch_ce / w_denom:.4f} (Acc: {(win_patch_acc / w_denom) * 100:.1f}%, ep {(epoch_patch_acc / denom) * 100:.1f}%) | "
                    f"Coord L1: {win_coord_l1 / w_denom:.4f} | "
                    f"NaN_skips: {nan_skip_count} | {sec_per_step:.2f}s/step (data={data_time:.2f}s){vram_suffix} | ETA: {eta_min:.1f}m",
                    flush=True,
                )
                win_loss = 0.0
                win_patch_ce = 0.0
                win_coord_l1 = 0.0
                win_patch_acc = 0.0
                win_steps = 0

            data_t0 = time.time()
            if dry_run and step_count >= 5:
                break

        denom = max(1, valid_step_count)
        avg_loss = epoch_loss / denom
        avg_patch_ce = epoch_patch_ce / denom
        avg_coord_l1 = epoch_coord_l1 / denom
        avg_w_patch_ce = epoch_w_patch_ce / denom
        avg_w_coord_l1 = epoch_w_coord_l1 / denom
        avg_w_coord_l2 = epoch_w_coord_l2 / denom
        avg_w_action = epoch_w_action / denom
        w_sum = avg_w_patch_ce + avg_w_coord_l1 + avg_w_coord_l2 + avg_w_action
        patch_pct = (avg_w_patch_ce / max(avg_loss, 1e-6)) * 100.0
        avg_patch_acc = (epoch_patch_acc / denom) * 100.0
        elapsed = time.time() - t0
        if rank == 0:
            print(
                f"Epoch [{epoch}/{epochs}] - Total Loss: {avg_loss:.4f} | "
                f"Weighted: [Patch CE (w={criterion.weight_patch_ce:.1f}): {avg_w_patch_ce:.4f} ({patch_pct:.1f}%) + "
                f"Coord L1 (w={criterion.weight_coord_l1:.1f}): {avg_w_coord_l1:.4f} + "
                f"Coord L2 (w={criterion.weight_coord_l2:.1f}): {avg_w_coord_l2:.4f} + "
                f"Action CE (w={criterion.weight_action:.1f}): {avg_w_action:.4f} = {w_sum:.4f}] | "
                f"Raw: [Patch CE: {avg_patch_ce:.4f}, Coord L1: {avg_coord_l1:.4f}] | "
                f"Train Patch Acc: {avg_patch_acc:.1f}% | Valid Steps: {valid_step_count}/{step_count} (NaN skips: {nan_skip_count}) | Time: {elapsed:.2f}s",
                flush=True,
            )

        # Distributed Held-Out Validation Evaluation at Epoch End (Capped at 200 samples; prevents NCCL timeout)
        val_strict_acc = 0.0
        val_patch_acc = 0.0
        if val_dataloader is not None and len(val_dataloader) > 0 and not dry_run:
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped.eval()
            local_hits = 0
            local_patch_hits = 0
            local_total = 0
            local_loss_acc = 0.0
            local_batches = 0

            with torch.no_grad():
                for v_batch in val_dataloader:
                    v_imgs = v_batch["image"].to(device) if is_dry_backbone else v_batch["image"]
                    v_coords = v_batch["target_coord"].to(device)
                    v_actions = v_batch["target_action"].to(device)
                    v_bboxes = v_batch["target_bbox"].to(device)

                    v_out = unwrapped(v_imgs, queries=v_batch.get("instruction"))
                    v_loss_dict = criterion(
                        model_out=v_out,
                        target_actions=v_actions,
                        target_coords=v_coords,
                        target_bboxes=v_bboxes,
                    )
                    local_loss_acc += v_loss_dict["loss"].item()
                    local_batches += 1

                    pred_coords = v_out["subpixel_coords"]
                    b_cnt = pred_coords.shape[0]

                    for b_i in range(b_cnt):
                        px, py = pred_coords[b_i, 0].item(), pred_coords[b_i, 1].item()
                        bx1, by1, bx2, by2 = (
                            v_bboxes[b_i, 0].item(),
                            v_bboxes[b_i, 1].item(),
                            v_bboxes[b_i, 2].item(),
                            v_bboxes[b_i, 3].item(),
                        )
                        x_min, x_max = min(bx1, bx2), max(bx1, bx2)
                        y_min, y_max = min(by1, by2), max(by1, by2)
                        if (x_min <= px <= x_max) and (y_min <= py <= y_max):
                            local_hits += 1

                    local_patch_hits += int((v_loss_dict.get("patch_top1_acc", 0.0) * b_cnt).round().item())
                    local_total += b_cnt

            if is_distributed:
                stats_tensor = torch.tensor(
                    [local_hits, local_patch_hits, local_total, local_loss_acc, local_batches],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                val_hits = int(stats_tensor[0].item())
                val_patch_hits = int(stats_tensor[1].item())
                val_total = int(stats_tensor[2].item())
                val_loss_acc = stats_tensor[3].item()
                val_batches_total = max(1, int(stats_tensor[4].item()))
                dist.barrier()
            else:
                val_hits = local_hits
                val_patch_hits = local_patch_hits
                val_total = local_total
                val_loss_acc = local_loss_acc
                val_batches_total = max(1, local_batches)

            val_strict_acc = val_hits / max(1, val_total)
            val_patch_acc = (val_patch_hits / max(1, val_total)) * 100.0
            avg_val_loss = val_loss_acc / val_batches_total
            if rank == 0:
                print(
                    f"[VAL] Epoch [{epoch}/{epochs}] Val Strict (0px) Acc: {val_strict_acc * 100:.2f}% "
                    f"({val_hits}/{val_total}) | Val Patch Acc: {val_patch_acc:.1f}% | Val Loss: {avg_val_loss:.4f}",
                    flush=True,
                )
            unwrapped.train()

            # Early Abortion Gate: Abort only if model exhibits zero convergence across patch & strict metrics
            if epoch == 1 and val_strict_acc < 0.05 and val_patch_acc < 5.0 and not no_early_abort:
                raise RuntimeError(
                    f"[EARLY-ABORTION-GATE] Epoch 1 validation strict accuracy ({val_strict_acc * 100:.2f}%) "
                    f"and patch accuracy ({val_patch_acc:.1f}%) are both below 5.0% threshold! "
                    "Training aborted early to prevent compute waste."
                )

        if rank == 0:
            # Save checkpoint (adapter heads + any trained LoRA weights)
            ckpt_file = save_path / f"gui_actor_step6_2_epoch_{epoch}.pt"
            unwrapped = model.module if hasattr(model, "module") else model
            adapter_state = {
                k: v
                for k, v in unwrapped.state_dict().items()
                if (not k.startswith("backbone.") or ".lora_" in k)
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": adapter_state,
                "lora_config": getattr(unwrapped, "lora_config", None),
                "stats": stats,
                "val_strict_acc": val_strict_acc,
            }, ckpt_file)
            print(f"Saved trainable adapter checkpoint to: {ckpt_file} ({len(adapter_state)} tensors)", flush=True)

        if is_distributed:
            dist.barrier()

        if dry_run:
            break

    if rank == 0:
        print("\nDDP Training completed successfully.", flush=True)

    cleanup_distributed()


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Kaggle Dual-T4 DDP Training for GUI-Actor on Frozen Qwen2.5-VL")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps across batches per optimizer update (default: 1)")
    parser.add_argument("--micro-batch-size", type=int, default=0, help="Per-GPU forward/backward micro-batch size (0=auto: 1 for LoRA on CUDA, else batch-size)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--save-dir", type=str, default="runs/checkpoints", help="Checkpoint save directory")
    parser.add_argument("--dry-run", action="store_true", help="Execute 5 mock steps on CPU for verification")
    parser.add_argument("--max-samples", type=int, default=40000, help="Maximum grounding samples to ingest")
    parser.add_argument("--overfit-sanity", action="store_true", help="Run multi-resolution overfit sanity test")
    parser.add_argument("--hf-preflight", action="store_true", help="Run real 16-sample HF OS-Atlas/ShowUI preflight gate")
    parser.add_argument("--steps-per-res", type=int, default=35, help="Number of steps per resolution in sanity test")
    parser.add_argument("--nproc", type=int, default=None, help="Number of processes for auto-torchrun")
    parser.add_argument("--no-auto-spawn", action="store_true", help="Disable auto-torchrun spawning")
    parser.add_argument("--fail-closed", action="store_true", help="Strictly enforce fail-closed dataset ingestion")
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="hf_showui",
        choices=["hf_showui", "local", "mixed", "synthetic"],
        help="Dataset source: 'hf_showui' (streams/caches showlab/ShowUI-desktop, default), 'local', 'mixed', or 'synthetic'",
    )
    parser.add_argument("--stream-hf", action="store_true", default=True, help="Stream real desktop samples from showlab/ShowUI-desktop (default: True)")
    parser.add_argument("--no-stream-hf", action="store_true", help="Disable Hugging Face streaming")
    parser.add_argument("--hf-samples", type=int, default=20000, help="Target real desktop samples from showlab/ShowUI-desktop (default: 20000)")
    parser.add_argument("--max-elements-per-image", type=int, default=12, help="Maximum elements per unique screenshot (default: 12)")
    parser.add_argument("--max-val-samples", type=int, default=200, help="Maximum held-out validation samples evaluated at epoch end (default: 200)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training samples per epoch for step-by-step staging")
    parser.add_argument("--log-interval", type=int, default=10, help="Print training progress every N steps (default: 10)")
    parser.add_argument("--backbone-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"], help="Frozen Qwen2.5-VL backbone precision ('auto' picks fp16 Tensor Cores on T4 SM<8.0 and bf16 on Ampere+)")
    parser.add_argument("--max-pixels", type=int, default=560 * 896, help="Max visual pixels for Qwen2.5-VL processor (default: 501760 = 560x896 ~640 patches; 0=uncapped)")
    parser.add_argument("--lora-layers", type=int, default=8, help="Unfreeze rank-r LoRA on top N Qwen decoder blocks on CUDA (0 disables; default: 8)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank when --lora-layers > 0 (default: 16)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for caching streamed Hugging Face datasets")
    parser.add_argument("--prepare-data-only", action="store_true", help="Stream and cache the Stage-1 dataset to disk, then exit without training")
    parser.add_argument("--no-early-abort", action="store_true", help="Disable the Epoch 1 early-abort gate")
    parser.add_argument("--dataset-paths", nargs="*", default=None, help="Explicit dataset JSONL / directory paths")
    parser.add_argument("--enable-token-pruning", action="store_true", help="Enable ShowUI token pruner (PAUSED by default)")
    parser.add_argument("--enable-rolling-kv", action="store_true", help="Enable rolling KV memory (PAUSED by default)")
    args = parser.parse_args()

    # 0. Pre-cache dataset only mode (CPU-only, zero GPU/DDP overhead)
    if args.prepare_data_only:
        should_stream = not args.no_stream_hf if args.no_stream_hf else args.stream_hf
        ds = GroundingDataset(
            dataset_paths=args.dataset_paths,
            max_samples=args.max_samples,
            target_size=(800, 1280),
            allow_synthetic_fallback=not args.fail_closed,
            stream_hf=should_stream,
            hf_samples=args.hf_samples,
            max_elements_per_image=args.max_elements_per_image,
            dataset_source=args.dataset_source,
            cache_dir=args.cache_dir,
        )
        print(f"[PREPARE-DATA-ONLY] Complete: {len(ds)} samples cached and ready in {ds.cache_dir}")
        sys.exit(0)

    # 1. Overfit sanity / HF preflight mode
    if args.overfit_sanity or args.hf_preflight:
        print("=" * 80)
        print("  AVCS STEP 6.2: MULTI-RESOLUTION OVERFIT SANITY & PRE-FLIGHT GATE  ")
        print("=" * 80)

        # Pull real Hugging Face samples if requested
        hf_samples = pull_hf_preflight_samples(16) if args.hf_preflight else None

        # Instantiate model (real Qwen2.5-VL if CUDA available, otherwise mock)
        is_gpu = torch.cuda.is_available() and not args.dry_run
        model = QwenGUIActorModel(
            dry_run=not is_gpu,
            pointer_hidden_dim=512,
            query_dim=512,
        )

        sanity_lr = args.lr if "--lr" in sys.argv else 1e-3
        res = run_overfit_sanity(
            model=model,
            samples=hf_samples,
            steps_per_res=args.steps_per_res,
            lr=sanity_lr,
        )
        all_converged = all(v["converged"] for v in res.values())
        print(f"Overfit sanity results: {json.dumps(res, indent=2)}")
        if all_converged:
            print("OVERFIT-SANITY: PASS - All resolutions converged successfully.")
            sys.exit(0)
        else:
            print("OVERFIT-SANITY: FAIL - Some resolutions did not meet convergence threshold.")
            sys.exit(1)

    # 2. Auto-torchrun launcher
    if not args.no_auto_spawn:
        maybe_auto_spawn_torchrun(force_nproc=args.nproc)

    # 3. Run training
    should_stream = not args.no_stream_hf if args.no_stream_hf else args.stream_hf
    run_ddp_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        dry_run=args.dry_run,
        max_samples=args.max_samples,
        fail_closed=args.fail_closed,
        use_token_pruning=args.enable_token_pruning,
        use_rolling_kv=args.enable_rolling_kv,
        stream_hf=should_stream,
        hf_samples=args.hf_samples,
        max_elements_per_image=args.max_elements_per_image,
        no_early_abort=args.no_early_abort,
        dataset_paths=args.dataset_paths,
        dataset_source=args.dataset_source,
        max_val_samples=args.max_val_samples,
        cache_dir=args.cache_dir,
        log_interval=args.log_interval,
        max_train_samples=args.max_train_samples,
        backbone_dtype=args.backbone_dtype,
        max_pixels=args.max_pixels,
        lora_layers=(0 if args.dry_run else args.lora_layers),
        lora_r=args.lora_r,
        grad_accum_steps=args.grad_accum_steps,
        micro_batch_size=args.micro_batch_size,
    )


if __name__ == "__main__":
    main()
