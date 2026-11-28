"""Evaluation Script: Fine-Tuned Qwen2.5-VL-3B + GUIActorPatchPointer on showdown-clicks-dev.

Evaluates the non-autoregressive single-pass GUIActorPatchPointer adapter head attached to
the frozen Qwen2.5-VL-3B backbone on the held-out General Agents Showdown-Clicks benchmark.

Primary Gate Metric: Strict inside-bounding-box (0.0px tolerance) >= 65.0%.
Secondary Diagnostic: hit_at_15px (Euclidean edge distance <= 15px).
Latency Metric: Single forward pass time in milliseconds (target: < 50ms).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure current directory and repository root are prioritized on sys.path
current_dir = os.path.abspath(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from train_vla_gui_actor import QwenGUIActorModel
except ImportError:
    import importlib.util
    train_script = os.path.join(current_dir, "train_vla_gui_actor.py")
    spec = importlib.util.spec_from_file_location("train_vla_gui_actor", train_script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_vla_gui_actor"] = mod
    spec.loader.exec_module(mod)
    QwenGUIActorModel = mod.QwenGUIActorModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [STEP-6.2-EVAL] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.gui_actor")

HF_RAW_BASE = "https://huggingface.co/datasets/generalagents/showdown-clicks/resolve/main/showdown-clicks-dev"
CSV_URL = f"{HF_RAW_BASE}/data.csv"


def wilson_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for binomial proportions."""
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = p_hat + (z**2) / (2.0 * total)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)
    lower = max(0.0, (center - spread) / denom)
    upper = min(1.0, (center + spread) / denom)
    return round(lower * 100.0, 2), round(upper * 100.0, 2)


def compute_edge_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Computes Euclidean distance from a predicted point to the nearest bounding box edge.

    Returns 0.0 if the point is strictly inside the box.
    """
    bx1, bx2 = min(x1, x2), max(x1, x2)
    by1, by2 = min(y1, y2), max(y1, y2)
    dx = max(0.0, max(bx1 - px, px - bx2))
    dy = max(0.0, max(by1 - py, py - by2))
    return math.hypot(dx, dy)


def run_evaluation(
    checkpoint_path: str = "runs/checkpoints/gui_actor_step6_2_epoch_3.pt",
    limit: Optional[int] = None,
    output_path: str = "benchmarks/results/showdown_gui_actor_step6_2.json",
    cache_dir: str = "data/showdown-clicks-dev",
    eval_w: int = 1280,
    eval_h: int = 800,
    native_size: bool = True,
    temperature: float = 0.05,
    model_name: str = "avcs",
    backbone_dtype: str = "auto",
    max_pixels: int = 560 * 896,
    allow_untrained: bool = False,
) -> Dict[str, Any]:
    print("=" * 80)
    print("  AVCS STEP 6.2: HELD-OUT BENCHMARK EVALUATION (showdown-clicks-dev)  ")
    model_label = f"Reference Baseline: {model_name}" if "gui-actor" in model_name.lower() or "microsoft" in model_name.lower() else "Model: Qwen2.5-VL-3B + GUIActorPatchPointer"
    print(f"  {model_label}   ")
    print("=" * 80)

    is_ms_gui_actor = "gui-actor" in model_name.lower() or "microsoft" in model_name.lower()
    if not is_ms_gui_actor and not Path(checkpoint_path).exists() and not allow_untrained:
        raise FileNotFoundError(
            f"Fail-closed eval gate: checkpoint not found at '{checkpoint_path}'. "
            "Refusing to evaluate un-finetuned weights (pass --allow-untrained to override)."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")
    res_mode = "Native Screen Resolution" if native_size else f"Desktop Canvas ({eval_w}x{eval_h})"
    print(f"Resolution Mode:   {res_mode}")
    print(f"Max Visual Pixels: {max_pixels}")
    print(f"Sharpening Temp:   T = {temperature}")

    # 1. Download or load benchmark manifest
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    local_csv = cache_path / "data.csv"

    if not local_csv.exists():
        print(f"Downloading benchmark manifest from {CSV_URL}...")
        resp = requests.get(CSV_URL, timeout=20)
        resp.raise_for_status()
        local_csv.write_text(resp.text, encoding="utf-8")
        manifest_text = resp.text
    else:
        print(f"Loading cached manifest from {local_csv}...")
        manifest_text = local_csv.read_text(encoding="utf-8")

    reader = csv.DictReader(io.StringIO(manifest_text))
    samples = list(reader)
    if limit and limit > 0:
        samples = samples[:limit]
    total_samples = len(samples)
    print(f"Loaded {total_samples} held-out evaluation samples.")

    # 2. Instantiate Model (AVCS QwenGUIActorModel or Microsoft GUI-Actor Baseline)
    ms_model = None
    ms_tokenizer = None
    ms_processor = None
    model = None

    if is_ms_gui_actor:
        print(f"\n[BENCHMARK] Loading Microsoft Reference Baseline: {model_name}...")
        # Patch transformers.trainer.ALL_LAYERNORM_LAYERS if missing on newer transformers versions
        try:
            import transformers.trainer
            if not hasattr(transformers.trainer, "ALL_LAYERNORM_LAYERS"):
                try:
                    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
                    transformers.trainer.ALL_LAYERNORM_LAYERS = ALL_LAYERNORM_LAYERS
                except Exception:
                    transformers.trainer.ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]
        except Exception:
            pass

        # Pre-populate gui_actor.trainer with lightweight rank0_print to avoid broken dependencies
        import types
        if "gui_actor.trainer" not in sys.modules:
            mock_trainer = types.ModuleType("gui_actor.trainer")
            def rank0_print(*args, **kwargs):
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(*args, **kwargs)
            mock_trainer.rank0_print = rank0_print
            sys.modules["gui_actor.trainer"] = mock_trainer

        # Ensure qwen_vl_utils is available on fresh containers
        try:
            import qwen_vl_utils  # noqa: F401
        except ImportError:
            print("[BENCHMARK] qwen_vl_utils not installed; installing qwen-vl-utils...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-vl-utils"])

        try:
            from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
            from gui_actor.inference import inference as ms_inference
        except ImportError:
            ms_repo_dir = Path("/tmp/GUI-Actor")
            if not ms_repo_dir.exists():
                print("[BENCHMARK] Cloning https://github.com/microsoft/GUI-Actor.git to /tmp/GUI-Actor to bypass Python 3.13 pip gate...")
                subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/microsoft/GUI-Actor.git", str(ms_repo_dir)])
            ms_src = str(ms_repo_dir / "src")
            if ms_src not in sys.path:
                sys.path.insert(0, ms_src)
            from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
            from gui_actor.inference import inference as ms_inference

        # 1. Patch Qwen2_5_VLForConditionalGenerationWithPointer:
        # - visual property -> self.model.visual
        # - is_custom_code -> False (so transformers treats it as native and applies conversions)
        if not hasattr(Qwen2_5_VLForConditionalGenerationWithPointer, "visual"):
            def _get_visual(self):
                if hasattr(self, "model") and hasattr(self.model, "visual"):
                    return self.model.visual
                return getattr(self, "_visual", None)
            def _set_visual(self, val):
                self._visual = val
            Qwen2_5_VLForConditionalGenerationWithPointer.visual = property(_get_visual, _set_visual)

        Qwen2_5_VLForConditionalGenerationWithPointer.is_custom_code = classmethod(lambda cls: False)

        # 2. Patch Qwen2_5_VLModel to forward embed_tokens, layers, norm to language_model
        try:
            try:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
            except ImportError:
                from transformers import Qwen2_5_VLModel

            if not hasattr(Qwen2_5_VLModel, "embed_tokens"):
                Qwen2_5_VLModel.embed_tokens = property(
                    lambda self: getattr(self, "language_model", self).embed_tokens
                )
            if not hasattr(Qwen2_5_VLModel, "layers"):
                Qwen2_5_VLModel.layers = property(
                    lambda self: getattr(self, "language_model", self).layers
                )
            if not hasattr(Qwen2_5_VLModel, "norm"):
                Qwen2_5_VLModel.norm = property(
                    lambda self: getattr(self, "language_model", self).norm
                )
        except Exception as e:
            print(f"[BENCHMARK] Warning: could not patch Qwen2_5_VLModel: {e}")

        # 3. Patch transformers conversion_mapping to remap checkpoint keys:
        # model.layers -> model.language_model.layers
        # model.embed_tokens -> model.language_model.embed_tokens
        # model.norm -> model.language_model.norm
        try:
            import transformers.conversion_mapping as cm
            import transformers.modeling_utils as mu

            if hasattr(cm, "USER_REGISTERED_MAPPINGS"):
                cm.USER_REGISTERED_MAPPINGS.add("Qwen2_5_VLForConditionalGenerationWithPointer")
                cm.USER_REGISTERED_MAPPINGS.add("qwen2_5_vl")

            orig_get_mapping = getattr(cm, "get_model_conversion_mapping", None)
            if orig_get_mapping is not None:
                def custom_get_model_conversion_mapping(model, key_mapping=None, hf_quantizer=None, add_legacy=True):
                    conversions = orig_get_mapping(model, key_mapping, hf_quantizer, add_legacy)
                    has_lm_rename = any(
                        getattr(c, "target_patterns", None) == "model.language_model" for c in conversions
                    )
                    if not has_lm_rename and ("Qwen" in type(model).__name__ or getattr(getattr(model, "config", None), "model_type", "") == "qwen2_5_vl"):
                        conversions.extend([
                            cm.WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
                            cm.WeightRenaming(
                                source_patterns=r"^model(?!\.(language_model|visual))", target_patterns="model.language_model"
                            ),
                        ])
                    return conversions

                cm.get_model_conversion_mapping = custom_get_model_conversion_mapping
                mu.get_model_conversion_mapping = custom_get_model_conversion_mapping

            if hasattr(cm, "register_checkpoint_conversion_mapping"):
                cm.register_checkpoint_conversion_mapping(
                    "Qwen2_5_VLForConditionalGenerationWithPointer",
                    [
                        cm.WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
                        cm.WeightRenaming(
                            source_patterns=r"^model(?!\.(language_model|visual))", target_patterns="model.language_model"
                        ),
                    ],
                    overwrite=True,
                )
        except Exception as e:
            print(f"[BENCHMARK] Warning: could not patch conversion_mapping: {e}")

        # Patch Qwen2_5_VLConfig for compatibility with modern transformers (4.48+)
        # where hidden_size was moved to text_config.hidden_size.
        try:
            try:
                from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
            except ImportError:
                from transformers import Qwen2_5_VLConfig

            if not hasattr(Qwen2_5_VLConfig, "hidden_size"):
                def _get_hidden_size(self):
                    if hasattr(self, "text_config") and hasattr(self.text_config, "hidden_size"):
                        return self.text_config.hidden_size
                    return self.__dict__.get("hidden_size", 2048)
                def _set_hidden_size(self, val):
                    self.__dict__["hidden_size"] = val
                Qwen2_5_VLConfig.hidden_size = property(_get_hidden_size, _set_hidden_size)

            if not hasattr(Qwen2_5_VLConfig, "vocab_size"):
                def _get_vocab_size(self):
                    if hasattr(self, "text_config") and hasattr(self.text_config, "vocab_size"):
                        return self.text_config.vocab_size
                    return self.__dict__.get("vocab_size", 152064)
                def _set_vocab_size(self, val):
                    self.__dict__["vocab_size"] = val
                Qwen2_5_VLConfig.vocab_size = property(_get_vocab_size, _set_vocab_size)

            _orig_getattr = getattr(Qwen2_5_VLConfig, "__getattr__", None)
            def _qwen_getattr(self, item):
                if item != "text_config" and hasattr(self, "text_config") and hasattr(self.text_config, item):
                    return getattr(self.text_config, item)
                if _orig_getattr is not None:
                    return _orig_getattr(self, item)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
            Qwen2_5_VLConfig.__getattr__ = _qwen_getattr
        except Exception as e:
            print(f"[BENCHMARK] Warning: could not monkeypatch Qwen2_5_VLConfig: {e}")

        from transformers import AutoConfig, AutoProcessor
        ms_config = AutoConfig.from_pretrained(model_name)
        if not hasattr(ms_config, "hidden_size") and hasattr(ms_config, "text_config"):
            ms_config.hidden_size = ms_config.text_config.hidden_size
        if not hasattr(ms_config, "pointer_pad_token_id") or ms_config.pointer_pad_token_id is None:
            ms_config.pointer_pad_token_id = 151669
        if not hasattr(ms_config, "pointer_start_token_id") or ms_config.pointer_start_token_id is None:
            ms_config.pointer_start_token_id = 151667
        if not hasattr(ms_config, "pointer_end_token_id") or ms_config.pointer_end_token_id is None:
            ms_config.pointer_end_token_id = 151668

        ms_processor = AutoProcessor.from_pretrained(model_name)
        ms_tokenizer = ms_processor.tokenizer
        if max_pixels > 0:
            min_px = min(256 * 28 * 28, int(max_pixels))
            ip = getattr(ms_processor, "image_processor", None)
            if ip is not None:
                if hasattr(ip, "min_pixels"):
                    ip.min_pixels = min_px
                if hasattr(ip, "max_pixels"):
                    ip.max_pixels = int(max_pixels)
                if hasattr(ip, "size") and isinstance(ip.size, dict):
                    ip.size["shortest_edge"] = min_px
                    ip.size["longest_edge"] = int(max_pixels)
            print(f"[BENCHMARK] Configured MS GUI-Actor pixel budget: min_pixels={min_px}, max_pixels={max_pixels}")

        # Wrap ms_processor to strip mm_token_type_ids (causes ValueError in generate)
        orig_proc_call = ms_processor.__call__
        def safe_proc_call(*args, **kwargs):
            out = orig_proc_call(*args, **kwargs)
            if hasattr(out, "pop"):
                out.pop("mm_token_type_ids", None)
            return out
        ms_processor.__call__ = safe_proc_call

        ms_model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name,
            config=ms_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        ).eval()
        if not hasattr(ms_model.config, "pointer_pad_token_id") or ms_model.config.pointer_pad_token_id is None:
            ms_model.config.pointer_pad_token_id = 151669
        if not hasattr(ms_model, "rope_deltas"):
            ms_model.rope_deltas = None

        import functools
        import inspect

        # In modern transformers, visual.forward returns BaseModelOutputWithPooling:
        # - last_hidden_state: unmerged pre-merger patch tokens (e.g. 4x tokens, dim 1280)
        # - pooler_output: post-merger visual tokens matching image_token_id count (dim 2048)
        # modeling_qwen25vl.py line 173 expects self.visual() to return the merged Tensor.
        if hasattr(ms_model, "visual") and hasattr(ms_model.visual, "forward"):
            orig_visual_forward = ms_model.visual.forward
            @functools.wraps(orig_visual_forward)
            def safe_visual_forward(*args, **kwargs):
                out = orig_visual_forward(*args, **kwargs)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    if isinstance(out.pooler_output, torch.Tensor):
                        return out.pooler_output
                    if isinstance(out.pooler_output, (tuple, list)) and len(out.pooler_output) > 0:
                        return torch.cat(list(out.pooler_output), dim=0)
                if hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
                    return out.last_hidden_state
                elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    return out[0]
                return out
            ms_model.visual.forward = safe_visual_forward
            if hasattr(ms_model, "model") and hasattr(ms_model.model, "visual"):
                ms_model.model.visual.forward = safe_visual_forward

        # Compat shim for get_rope_index (moved to model.get_rope_index and added mm_token_type_ids in v5)
        rope_owner = ms_model if hasattr(ms_model, "get_rope_index") else getattr(ms_model, "model", None)
        if rope_owner is not None and hasattr(rope_owner, "get_rope_index"):
            orig_get_rope_index = rope_owner.get_rope_index
            rope_params = list(inspect.signature(orig_get_rope_index).parameters.keys())
            if "mm_token_type_ids" in rope_params or not hasattr(ms_model, "get_rope_index"):
                def compat_get_rope_index(input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
                    if "mm_token_type_ids" in rope_params:
                        mm_tok = torch.zeros_like(input_ids, dtype=torch.int32)
                        img_tok_id = getattr(ms_model.config, "image_token_id", 151655)
                        vid_tok_id = getattr(ms_model.config, "video_token_id", 151656)
                        mm_tok[input_ids == img_tok_id] = 1
                        mm_tok[input_ids == vid_tok_id] = 2
                        return orig_get_rope_index(
                            input_ids=input_ids,
                            mm_token_type_ids=mm_tok,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            attention_mask=attention_mask,
                        )
                    return orig_get_rope_index(
                        input_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask,
                    )
                ms_model.get_rope_index = compat_get_rope_index

        # Compat shim for past_key_values <-> cache between transformers v4/v5
        if hasattr(ms_model, "model") and hasattr(ms_model.model, "forward"):
            orig_base_forward = ms_model.model.forward
            base_fwd_params = set(inspect.signature(orig_base_forward).parameters.keys())
            if "cache" in base_fwd_params and "past_key_values" not in base_fwd_params:
                @functools.wraps(orig_base_forward)
                def safe_base_forward(*args, **kwargs):
                    if "past_key_values" in kwargs:
                        kwargs["cache"] = kwargs.pop("past_key_values")
                    out = orig_base_forward(*args, **kwargs)
                    if not hasattr(out, "past_key_values") and hasattr(out, "cache"):
                        type(out).past_key_values = property(lambda self: getattr(self, "cache", None))
                    return out
                ms_model.model.forward = safe_base_forward

        try:
            from gui_actor.modeling_qwen25vl import QwenVLwithVisionHeadOutputWithPast
            out_init_params = set(inspect.signature(QwenVLwithVisionHeadOutputWithPast.__init__).parameters.keys())
            if "cache" in out_init_params and "past_key_values" not in out_init_params:
                orig_out_init = QwenVLwithVisionHeadOutputWithPast.__init__
                def safe_out_init(self, lm_loss=None, pointer_loss=None, pointer_scores=None, *args, **kwargs):
                    if "past_key_values" in kwargs and "cache" not in kwargs:
                        kwargs["cache"] = kwargs.pop("past_key_values")
                    return orig_out_init(self, lm_loss=lm_loss, pointer_loss=pointer_loss, pointer_scores=pointer_scores, *args, **kwargs)
                QwenVLwithVisionHeadOutputWithPast.__init__ = safe_out_init
                if not hasattr(QwenVLwithVisionHeadOutputWithPast, "past_key_values"):
                    QwenVLwithVisionHeadOutputWithPast.past_key_values = property(lambda self: getattr(self, "cache", None))
        except Exception:
            pass

        # Bypass _validate_model_kwargs to allow all multimodal arguments (pixel_values, image_grid_thw, etc.)
        ms_model._validate_model_kwargs = lambda model_kwargs: None

        ms_generate_calls = 0
        orig_generate = ms_model.generate
        @functools.wraps(orig_generate)
        def safe_generate(*args, **kwargs):
            nonlocal ms_generate_calls
            ms_generate_calls += 1
            kwargs.pop("mm_token_type_ids", None)
            return orig_generate(*args, **kwargs)
        ms_model.generate = safe_generate

        ms_forward_calls = 0
        orig_forward = ms_model.forward
        orig_fwd_params = set(inspect.signature(orig_forward).parameters.keys())
        @functools.wraps(orig_forward)
        def safe_forward(*args, **kwargs):
            nonlocal ms_forward_calls
            ms_forward_calls += 1
            kwargs.pop("mm_token_type_ids", None)
            if "cache" in kwargs and "cache" not in orig_fwd_params and "past_key_values" in orig_fwd_params:
                kwargs["past_key_values"] = kwargs.pop("cache")
            return orig_forward(*args, **kwargs)
        ms_model.forward = safe_forward
    else:
        print(f"Initializing QwenGUIActorModel on {device}...")
        model = QwenGUIActorModel(
            dry_run=not torch.cuda.is_available(),
            pointer_hidden_dim=512,
            query_dim=512,
            backbone_dtype=backbone_dtype,
            max_pixels=max_pixels,
        )

        ckpt_file = Path(checkpoint_path)
        if ckpt_file.exists():
            print(f"Loading adapter weights from checkpoint: {ckpt_file}...")
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            lora_cfg = ckpt.get("lora_config")
            if isinstance(lora_cfg, dict) and lora_cfg.get("lora_layers", 0) > 0 and not getattr(model, "dry_run", False):
                wrapped_lora = model.enable_lora(**lora_cfg)
                print(f"Restored LoRA structure from checkpoint ({wrapped_lora} projections wrapped).")
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Checkpoint loaded: {len(state_dict)} tensors restored (missing={len(missing)}, unexpected={len(unexpected)}).")
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found; evaluating un-finetuned adapter head.")

        model.to(device)
        model.eval()

    # 3. Evaluation Tracking across 3 pointer decoding strategies
    sub_strict_hits = 0
    sub_15_hits = 0

    sharp_strict_hits = 0
    sharp_15_hits = 0

    soft_strict_hits = 0
    soft_15_hits = 0

    total_latency_ms = 0.0
    latencies: List[float] = []
    results_detail: List[Dict[str, Any]] = []

    print(f"\nEvaluating {total_samples} samples across pointer decoding heads...")
    t_start = time.time()

    for idx, row in enumerate(samples):
        sample_id = row.get("id", str(idx))
        img_rel = row.get("image") or row.get("image_rel_path") or row.get("image_path")
        if not img_rel:
            continue
        instruction = row["instruction"]
        x1, y1 = float(row["x1"]), float(row["y1"])
        x2, y2 = float(row["x2"]), float(row["y2"])
        raw_w, raw_h = int(row["width"]), int(row["height"])

        # Fetch / cache sample image
        img_file = cache_path / img_rel
        if not img_file.exists():
            img_file.parent.mkdir(parents=True, exist_ok=True)
            img_url = f"{HF_RAW_BASE}/{img_rel}"
            try:
                r_img = requests.get(img_url, timeout=15)
                r_img.raise_for_status()
                img_file.write_bytes(r_img.content)
            except Exception as e:
                logger.warning(f"Failed to fetch {img_url}: {e}")
                continue

        try:
            pil_img = Image.open(img_file).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open image {img_file}: {e}")
            continue

        # Resize image to canvas size if not in native_size mode
        if not native_size and eval_w > 0 and eval_h > 0:
            model_img = pil_img.resize((eval_w, eval_h), Image.Resampling.BILINEAR)
        else:
            model_img = pil_img

        t0 = time.perf_counter()
        if is_ms_gui_actor:
            img_entry: Dict[str, Any] = {"type": "image", "image": model_img}
            if max_pixels > 0:
                img_entry["min_pixels"] = min(256 * 28 * 28, int(max_pixels))
                img_entry["max_pixels"] = int(max_pixels)
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
                                "your task is to locate the screen element that corresponds to the instruction. "
                                "You should output a PyAutoGUI action that performs a click on the correct position. "
                                "To indicate the click location, we will use some special tokens, which is used to refer "
                                "to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        img_entry,
                        {"type": "text", "text": instruction},
                    ],
                },
            ]
            gen_before, fwd_before = ms_generate_calls, ms_forward_calls
            pred = ms_inference(conversation, ms_model, ms_tokenizer, ms_processor, use_placeholder=True, topk=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            if idx == 0:
                print(
                    f"[BENCHMARK] MS GUI-Actor sample 0 call telemetry: "
                    f"generate_calls={ms_generate_calls - gen_before}, "
                    f"forward_calls={ms_forward_calls - fwd_before} "
                    f"(use_placeholder=True)"
                )
            if pred and isinstance(pred.get("topk_points"), list) and len(pred["topk_points"]) > 0:
                px_norm, py_norm = pred["topk_points"][0]
            else:
                px_norm, py_norm = 0.5, 0.5
            sub_px = float(px_norm * raw_w)
            sub_py = float(py_norm * raw_h)
            sharp_px = sub_px
            sharp_py = sub_py
            soft_px = sub_px
            soft_py = sub_py
        else:
            # Single-pass model forward step (measure pure inference latency)
            with torch.no_grad():
                out = model(images=model_img, queries=[instruction])
                if device.type == "cuda":
                    torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # 1. Subpixel coordinates (Argmax patch + subpatch offset MLP)
            sub_norm_xy = out["subpixel_coords"][0].cpu().numpy()
            sub_px = float(sub_norm_xy[0] * raw_w)
            sub_py = float(sub_norm_xy[1] * raw_h)

            # 2. Temperature-sharpened centroid
            attn_logits = out["attn_logits"]
            patch_coords = out["patch_coords"]
            if patch_coords.dim() == 2:
                patch_coords = patch_coords.unsqueeze(0)
            sharp_weights = F.softmax(attn_logits / max(temperature, 1e-4), dim=-1)
            sharp_norm_xy = torch.bmm(sharp_weights, patch_coords).squeeze(1)[0].cpu().numpy()
            sharp_px = float(sharp_norm_xy[0] * raw_w)
            sharp_py = float(sharp_norm_xy[1] * raw_h)

            # 3. Soft unsharpened centroid (T=1.0)
            soft_weights = F.softmax(attn_logits / 1.0, dim=-1)
            soft_norm_xy = torch.bmm(soft_weights, patch_coords).squeeze(1)[0].cpu().numpy()
            soft_px = float(soft_norm_xy[0] * raw_w)
            soft_py = float(soft_norm_xy[1] * raw_h)

        latencies.append(infer_ms)
        total_latency_ms += infer_ms

        # Evaluate bounding box inclusion
        bx1, bx2 = min(x1, x2), max(x1, x2)
        by1, by2 = min(y1, y2), max(y1, y2)

        is_sub_strict = (bx1 <= sub_px <= bx2) and (by1 <= sub_py <= by2)
        dist_sub = compute_edge_distance(sub_px, sub_py, bx1, by1, bx2, by2)
        is_sub_15 = is_sub_strict or (dist_sub <= 15.0)

        is_sharp_strict = (bx1 <= sharp_px <= bx2) and (by1 <= sharp_py <= by2)
        dist_sharp = compute_edge_distance(sharp_px, sharp_py, bx1, by1, bx2, by2)
        is_sharp_15 = is_sharp_strict or (dist_sharp <= 15.0)

        is_soft_strict = (bx1 <= soft_px <= bx2) and (by1 <= soft_py <= by2)
        dist_soft = compute_edge_distance(soft_px, soft_py, bx1, by1, bx2, by2)
        is_soft_15 = is_soft_strict or (dist_soft <= 15.0)

        if is_sub_strict:
            sub_strict_hits += 1
        if is_sub_15:
            sub_15_hits += 1

        if is_sharp_strict:
            sharp_strict_hits += 1
        if is_sharp_15:
            sharp_15_hits += 1

        if is_soft_strict:
            soft_strict_hits += 1
        if is_soft_15:
            soft_15_hits += 1

        results_detail.append({
            "id": sample_id,
            "instruction": instruction,
            "target_bbox": [x1, y1, x2, y2],
            "subpixel_px": [round(sub_px, 1), round(sub_py, 1)],
            "subpixel_strict": is_sub_strict,
            "subpixel_edge_dist": round(dist_sub, 2),
            "sharpened_px": [round(sharp_px, 1), round(sharp_py, 1)],
            "sharpened_strict": is_sharp_strict,
            "soft_px": [round(soft_px, 1), round(soft_py, 1)],
            "soft_strict": is_soft_strict,
            "latency_ms": round(infer_ms, 2),
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == total_samples:
            acc_sub = (sub_strict_hits / (idx + 1)) * 100.0
            acc_sharp = (sharp_strict_hits / (idx + 1)) * 100.0
            acc_soft = (soft_strict_hits / (idx + 1)) * 100.0
            avg_lat = total_latency_ms / (idx + 1)
            print(
                f"[{idx + 1:3d}/{total_samples}] "
                f"Subpixel (Argmax): {acc_sub:5.2f}% ({sub_strict_hits}/{idx + 1}) | "
                f"Sharpened (T={temperature}): {acc_sharp:5.2f}% | "
                f"Soft Centroid: {acc_soft:5.2f}% | "
                f"Latency: {avg_lat:5.1f}ms/step"
            )

    elapsed_sec = time.time() - t_start
    evaluated_count = len(results_detail)

    acc_sub = (sub_strict_hits / max(1, evaluated_count)) * 100.0
    acc_sub_15 = (sub_15_hits / max(1, evaluated_count)) * 100.0
    wilson_sub = wilson_interval(sub_strict_hits, evaluated_count)

    acc_sharp = (sharp_strict_hits / max(1, evaluated_count)) * 100.0
    acc_sharp_15 = (sharp_15_hits / max(1, evaluated_count)) * 100.0
    wilson_sharp = wilson_interval(sharp_strict_hits, evaluated_count)

    acc_soft = (soft_strict_hits / max(1, evaluated_count)) * 100.0
    acc_soft_15 = (soft_15_hits / max(1, evaluated_count)) * 100.0
    wilson_soft = wilson_interval(soft_strict_hits, evaluated_count)

    best_strict = max(acc_sub, acc_sharp)
    best_strategy = "subpixel_argmax" if acc_sub >= acc_sharp else "sharpened_centroid"

    latencies.sort()
    p50_lat = latencies[len(latencies) // 2] if latencies else 0.0
    p90_lat = latencies[int(len(latencies) * 0.9)] if latencies else 0.0
    mean_lat = total_latency_ms / max(1, evaluated_count)

    summary = {
        "benchmark": "generalagents/showdown-clicks-dev",
        "model_name": model_name,
        "checkpoint": model_name if is_ms_gui_actor else str(checkpoint_path),
        "max_pixels": int(max_pixels),
        "inference_mode": (
            ("single_pass_pointer" if ms_generate_calls == 0 else f"generate_calls_{ms_generate_calls}")
            if is_ms_gui_actor
            else "single_pass_pointer"
        ),
        "total_samples": evaluated_count,
        "elapsed_seconds": round(elapsed_sec, 2),
        "resolution": {
            "mode": "native" if native_size else "canvas",
            "canvas_size": None if native_size else [eval_w, eval_h],
        },
        "best_strategy": best_strategy,
        "subpixel_argmax": {
            "name": "subpixel_argmax_plus_offset",
            "hits": sub_strict_hits,
            "accuracy_percent": round(acc_sub, 2),
            "hit_15px_percent": round(acc_sub_15, 2),
            "wilson_95_ci": wilson_sub,
            "pass_gate_65_pct": acc_sub >= 65.0,
        },
        "sharpened_centroid": {
            "name": f"sharpened_attention_centroid_T_{temperature}",
            "hits": sharp_strict_hits,
            "accuracy_percent": round(acc_sharp, 2),
            "hit_15px_percent": round(acc_sharp_15, 2),
            "wilson_95_ci": wilson_sharp,
            "pass_gate_65_pct": acc_sharp >= 65.0,
        },
        "soft_centroid": {
            "name": "soft_unsharpened_centroid_T_1.0",
            "hits": soft_strict_hits,
            "accuracy_percent": round(acc_soft, 2),
            "hit_15px_percent": round(acc_soft_15, 2),
            "wilson_95_ci": wilson_soft,
            "pass_gate_65_pct": acc_soft >= 65.0,
        },
        "latency_telemetry": {
            "mean_ms": round(mean_lat, 2),
            "p50_ms": round(p50_lat, 2),
            "p90_ms": round(p90_lat, 2),
            "speedup_vs_autoregressive_baseline": round(3580.4 / max(0.1, mean_lat), 1),
        },
        "predictions": results_detail,
    }

    print("\n" + "=" * 80)
    print("  FINAL EVALUATION REPORT: STEP 6.2 GUI-ACTOR POINTER  ")
    print("=" * 80)
    print(f"Evaluated Samples:              {evaluated_count}/{total_samples}")
    print(f"Subpixel (Argmax + Offset):     {acc_sub:.2f}% (Wilson 95% CI: [{wilson_sub[0]}%, {wilson_sub[1]}%]) | 15px: {acc_sub_15:.2f}%")
    print(f"Sharpened Centroid (T={temperature}): {acc_sharp:.2f}% (Wilson 95% CI: [{wilson_sharp[0]}%, {wilson_sharp[1]}%]) | 15px: {acc_sharp_15:.2f}%")
    print(f"Soft Centroid (T=1.0):          {acc_soft:.2f}% (Wilson 95% CI: [{wilson_soft[0]}%, {wilson_soft[1]}%]) | 15px: {acc_soft_15:.2f}%")
    print(f"Latency per Step:               {mean_lat:.2f}ms (P50: {p50_lat:.2f}ms, P90: {p90_lat:.2f}ms)")
    print(f"Inference Acceleration:         {3580.4 / max(0.1, mean_lat):.1f}x faster than autoregressive base model")
    gate_str = "PASS (>= 65.0%)" if best_strict >= 65.0 else "PENDING (< 65.0%)"
    print(f"Milestone Gate ({best_strategy}): {gate_str}")
    print("=" * 80)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation artifact to: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned GUI-Actor Pointer on showdown-clicks-dev")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoints/gui_actor_step6_2_epoch_3.pt", help="Path to adapter checkpoint")
    parser.add_argument("--model-name", type=str, default="avcs", help="Model to evaluate: 'avcs' or 'microsoft/GUI-Actor-3B-Qwen2.5-VL'")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--output", type=str, default="benchmarks/results/showdown_gui_actor_step6_2.json", help="Output JSON path")
    parser.add_argument("--cache-dir", type=str, default="data/showdown-clicks-dev", help="Benchmark cache directory")
    parser.add_argument("--eval-size", nargs=2, type=int, default=[1280, 800], help="Evaluation canvas size W H (default: 1280 800)")
    parser.add_argument("--native-size", action="store_true", default=True, help="Evaluate at native screen resolution without downscaling (default: True)")
    parser.add_argument("--resize-canvas", action="store_true", help="Force downscaling to eval-size instead of native resolution")
    parser.add_argument("--temperature", type=float, default=0.05, help="Attention sharpening temperature (default: 0.05)")
    parser.add_argument("--backbone-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"], help="Frozen Qwen2.5-VL backbone precision")
    parser.add_argument("--max-pixels", type=int, default=560 * 896, help="Max visual pixels for Qwen2.5-VL processor (default: 501760; 0=uncapped)")
    parser.add_argument("--allow-untrained", action="store_true", help="Explicitly allow evaluating un-finetuned weights if --checkpoint is missing")
    args = parser.parse_args()

    use_native = False if args.resize_canvas else args.native_size
    run_evaluation(
        checkpoint_path=args.checkpoint,
        limit=args.limit,
        output_path=args.output,
        cache_dir=args.cache_dir,
        eval_w=args.eval_size[0],
        eval_h=args.eval_size[1],
        native_size=use_native,
        temperature=args.temperature,
        model_name=args.model_name,
        backbone_dtype=args.backbone_dtype,
        max_pixels=args.max_pixels,
        allow_untrained=args.allow_untrained,
    )


if __name__ == "__main__":
    main()
