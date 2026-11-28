"""True 0-generate() single-pass GUI-Actor-3B backbone + ParallelActionHeads.

Replaces microsoft/GUI-Actor's `model.generate(max_new_tokens=1)` boilerplate with
a direct single-pass `backbone.model(...)` + `multi_patch_pointer_head` forward
call (skipping the 152k-vocab `lm_head` projection and generation loop) while
preserving mathematical parity with `GUI-Actor-3B-Qwen2.5-VL` and attaching AVCS's
10-primitive `ParallelActionHeads` onto the `<|pointer_pad|>` hidden state.
"""
from __future__ import annotations

from collections import deque
import functools
import inspect
import logging
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from avcs.models.vla_cortex import ACTION_VOCAB_10, ParallelActionHeads

logger = logging.getLogger("avcs.models.gui_actor_singlepass")

MS_GUI_ACTOR_SYSTEM_PROMPT = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
    "your task is to locate the screen element that corresponds to the instruction. "
    "You should output a PyAutoGUI action that performs a click on the correct position. "
    "To indicate the click location, we will use some special tokens, which is used to refer "
    "to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
)

MS_GUI_ACTOR_CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

MS_POINTER_ASSISTANT_PREFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)


def ms_region_point_from_attn(
    attn_scores: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float = 0.3,
) -> Tuple[float, float]:
    """Connected-component peak region center matching GUI-Actor's inference decoding.

    Args:
        attn_scores: Attention probabilities shaped (K,) or (1, K).
        n_width: Patch grid width.
        n_height: Patch grid height.
        activation_threshold: Fraction of max attention defining active patches.
    """
    scores_1d = attn_scores.reshape(-1).detach().float().cpu()
    if scores_1d.numel() == 0 or n_width <= 0 or n_height <= 0:
        return (0.5, 0.5)

    max_score = float(scores_1d.max().item())
    if not math.isfinite(max_score) or max_score <= 0.0:
        idx = int(torch.argmax(scores_1d).item())
        return ((idx % n_width + 0.5) / n_width, (idx // n_width + 0.5) / n_height)

    thresh = max_score * float(activation_threshold)
    active_idx = torch.nonzero(scores_1d > thresh, as_tuple=False).squeeze(-1).tolist()
    if not active_idx:
        idx = int(torch.argmax(scores_1d).item())
        return ((idx % n_width + 0.5) / n_width, (idx // n_width + 0.5) / n_height)

    active_map: Dict[Tuple[int, int], float] = {}
    for idx in active_idx:
        y = idx // n_width
        x = idx % n_width
        if y < n_height and x < n_width:
            active_map[(y, x)] = float(scores_1d[idx].item())

    visited = set()
    best_avg = -1.0
    best_pt = (0.5, 0.5)

    for start_yx in active_map:
        if start_yx in visited:
            continue
        visited.add(start_yx)
        q = deque([start_yx])
        region: List[Tuple[int, int, float]] = []
        while q:
            cy, cx = q.popleft()
            region.append((cy, cx, active_map[(cy, cx)]))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (cy + dy, cx + dx)
                if nb in active_map and nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        avg_score = sum(item[2] for item in region) / float(len(region))
        if avg_score > best_avg:
            total_w = sum(item[2] for item in region)
            if total_w > 0.0:
                wx = sum(((item[1] + 0.5) / n_width) * item[2] for item in region) / total_w
                wy = sum(((item[0] + 0.5) / n_height) * item[2] for item in region) / total_w
                best_pt = (float(wx), float(wy))
                best_avg = avg_score

    return (
        min(1.0, max(0.0, best_pt[0])),
        min(1.0, max(0.0, best_pt[1])),
    )


def resolve_backbone_dtype(backbone_dtype: str = "auto") -> torch.dtype:
    choice = (backbone_dtype or "auto").strip().lower()
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float16


def _sanitize_fp16(x: torch.Tensor, max_abs: float = 32768.0) -> torch.Tensor:
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=max_abs, neginf=-max_abs).clamp(-max_abs, max_abs)
    return x


def preserve_aspect_resize(
    rgb_img: Image.Image,
    max_w: int = 1280,
    max_h: int = 800,
) -> Image.Image:
    """Proportionally downscales an image only if larger than (max_w, max_h), preserving aspect ratio."""
    w, h = rgb_img.size
    if w <= 0 or h <= 0:
        return rgb_img
    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
    if scale >= 0.999:
        return rgb_img
    new_w = max(28, int(round(w * scale)))
    new_h = max(28, int(round(h * scale)))
    return rgb_img.resize((new_w, new_h), Image.Resampling.BILINEAR)


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
        x_safe = _sanitize_fp16(x, max_abs=32768.0) if x.dtype == torch.float16 else x
        base_out = self.base_layer(x_safe)
        lora_in = self.lora_dropout(x_safe.to(dtype=self.lora_A.weight.dtype))
        lora_delta = self.lora_B(self.lora_A(lora_in)) * self.scaling
        return base_out + lora_delta.to(dtype=base_out.dtype)


def _resolve_decoder_layers(backbone: nn.Module) -> Optional[nn.ModuleList]:
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


def _resolve_visual_modules(backbone: nn.Module) -> List[nn.Module]:
    found: List[nn.Module] = []
    seen_ids: set = set()
    for path in (("model", "visual"), ("visual",), ("model", "vision_tower"), ("vision_tower",)):
        cur: Any = backbone
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if isinstance(cur, nn.Module) and id(cur) not in seen_ids:
            seen_ids.add(id(cur))
            found.append(cur)
    return found


def _apply_transformers_compat_shims(Qwen2_5_VLForConditionalGenerationWithPointer: Any) -> None:
    """Patches transformers v4/v5 attribute & conversion differences for GUI-Actor."""
    if not hasattr(Qwen2_5_VLForConditionalGenerationWithPointer, "visual"):
        def _get_visual(self):
            if hasattr(self, "model") and hasattr(self.model, "visual"):
                return self.model.visual
            return getattr(self, "_visual", None)

        def _set_visual(self, val):
            self._visual = val

        Qwen2_5_VLForConditionalGenerationWithPointer.visual = property(_get_visual, _set_visual)

    Qwen2_5_VLForConditionalGenerationWithPointer.is_custom_code = classmethod(lambda cls: False)

    try:
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
        except ImportError:
            from transformers import Qwen2_5_VLModel

        if not hasattr(Qwen2_5_VLModel, "embed_tokens"):
            Qwen2_5_VLModel.embed_tokens = property(lambda self: getattr(self, "language_model", self).embed_tokens)
        if not hasattr(Qwen2_5_VLModel, "layers"):
            Qwen2_5_VLModel.layers = property(lambda self: getattr(self, "language_model", self).layers)
        if not hasattr(Qwen2_5_VLModel, "norm"):
            Qwen2_5_VLModel.norm = property(lambda self: getattr(self, "language_model", self).norm)
    except Exception:
        pass

    try:
        import transformers.conversion_mapping as cm
        import transformers.modeling_utils as mu

        if hasattr(cm, "USER_REGISTERED_MAPPINGS"):
            cm.USER_REGISTERED_MAPPINGS.add("Qwen2_5_VLForConditionalGenerationWithPointer")
            cm.USER_REGISTERED_MAPPINGS.add("qwen2_5_vl")

        orig_get_mapping = getattr(cm, "get_model_conversion_mapping", None)
        if orig_get_mapping is not None and not getattr(orig_get_mapping, "_avcs_patched", False):
            def custom_get_model_conversion_mapping(model, key_mapping=None, hf_quantizer=None, add_legacy=True):
                conversions = orig_get_mapping(model, key_mapping, hf_quantizer, add_legacy)
                has_lm_rename = any(getattr(c, "target_patterns", None) == "model.language_model" for c in conversions)
                if not has_lm_rename and (
                    "Qwen" in type(model).__name__
                    or getattr(getattr(model, "config", None), "model_type", "") == "qwen2_5_vl"
                ):
                    conversions.extend([
                        cm.WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
                        cm.WeightRenaming(
                            source_patterns=r"^model(?!\.(language_model|visual))",
                            target_patterns="model.language_model",
                        ),
                    ])
                return conversions

            custom_get_model_conversion_mapping._avcs_patched = True  # type: ignore[attr-defined]
            cm.get_model_conversion_mapping = custom_get_model_conversion_mapping
            mu.get_model_conversion_mapping = custom_get_model_conversion_mapping

        if hasattr(cm, "register_checkpoint_conversion_mapping"):
            cm.register_checkpoint_conversion_mapping(
                "Qwen2_5_VLForConditionalGenerationWithPointer",
                [
                    cm.WeightRenaming(source_patterns=r"^visual", target_patterns="model.visual"),
                    cm.WeightRenaming(
                        source_patterns=r"^model(?!\.(language_model|visual))",
                        target_patterns="model.language_model",
                    ),
                ],
                overwrite=True,
            )
    except Exception:
        pass

    try:
        try:
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
        except ImportError:
            from transformers import Qwen2_5_VLConfig

        if not hasattr(Qwen2_5_VLConfig, "hidden_size"):
            Qwen2_5_VLConfig.hidden_size = property(
                lambda self: self.text_config.hidden_size if hasattr(self, "text_config") else self.__dict__.get("hidden_size", 2048),
                lambda self, val: self.__dict__.__setitem__("hidden_size", val),
            )
        if not hasattr(Qwen2_5_VLConfig, "vocab_size"):
            Qwen2_5_VLConfig.vocab_size = property(
                lambda self: self.text_config.vocab_size if hasattr(self, "text_config") else self.__dict__.get("vocab_size", 152064),
                lambda self, val: self.__dict__.__setitem__("vocab_size", val),
            )
        _orig_getattr = getattr(Qwen2_5_VLConfig, "__getattr__", None)
        if not getattr(_orig_getattr, "_avcs_patched", False):
            def _qwen_getattr(self, item):
                if item != "text_config" and hasattr(self, "text_config") and hasattr(self.text_config, item):
                    return getattr(self.text_config, item)
                if _orig_getattr is not None:
                    return _orig_getattr(self, item)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

            _qwen_getattr._avcs_patched = True  # type: ignore[attr-defined]
            Qwen2_5_VLConfig.__getattr__ = _qwen_getattr
    except Exception:
        pass


def _attach_instance_compat_shims(ms_model: Any) -> None:
    """Normalizes BaseModelOutputWithPooling / rope / cache attributes on an instance."""
    if hasattr(ms_model, "visual") and hasattr(ms_model.visual, "forward"):
        orig_visual_forward = ms_model.visual.forward
        if not getattr(orig_visual_forward, "_avcs_wrapped", False):
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
                if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                    return out[0]
                return out

            safe_visual_forward._avcs_wrapped = True  # type: ignore[attr-defined]
            ms_model.visual.forward = safe_visual_forward
            if hasattr(ms_model, "model") and hasattr(ms_model.model, "visual"):
                ms_model.model.visual.forward = safe_visual_forward

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


def load_ms_gui_actor_backbone(
    model_name: str = "microsoft/GUI-Actor-3B-Qwen2.5-VL",
    max_pixels: int = 1003520,
    backbone_dtype: str = "auto",
) -> Tuple[Any, Any]:
    """Loads `microsoft/GUI-Actor-3B-Qwen2.5-VL` model and pixel-bounded AutoProcessor."""
    import types

    try:
        import transformers.trainer
        if not hasattr(transformers.trainer, "ALL_LAYERNORM_LAYERS"):
            try:
                from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
                transformers.trainer.ALL_LAYERNORM_LAYERS = ALL_LAYERNORM_LAYERS
            except Exception:
                transformers.trainer.ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
    except Exception:
        pass

    if "gui_actor.trainer" not in sys.modules:
        mock_trainer = types.ModuleType("gui_actor.trainer")

        def rank0_print(*args: Any, **kwargs: Any) -> None:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(*args, **kwargs)

        mock_trainer.rank0_print = rank0_print  # type: ignore[attr-defined]
        sys.modules["gui_actor.trainer"] = mock_trainer

    try:
        from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
    except ImportError:
        local_ref = Path(__file__).resolve().parent.parent.parent / "runs" / "_ref_gui_actor"
        ms_repo_dir = local_ref if (local_ref / "src" / "gui_actor").is_dir() else Path("/tmp/GUI-Actor")
        if not ms_repo_dir.exists():
            logger.info("Cloning https://github.com/microsoft/GUI-Actor.git to /tmp/GUI-Actor...")
            subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/microsoft/GUI-Actor.git", str(ms_repo_dir)])
        ms_src = str(ms_repo_dir / "src")
        if ms_src not in sys.path:
            sys.path.insert(0, ms_src)
        from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer

    _apply_transformers_compat_shims(Qwen2_5_VLForConditionalGenerationWithPointer)

    from transformers import AutoConfig, AutoProcessor
    ms_config = AutoConfig.from_pretrained(model_name)
    if not hasattr(ms_config, "hidden_size") and hasattr(ms_config, "text_config"):
        ms_config.hidden_size = ms_config.text_config.hidden_size
    if not getattr(ms_config, "pointer_pad_token_id", None):
        ms_config.pointer_pad_token_id = 151669
    if not getattr(ms_config, "pointer_start_token_id", None):
        ms_config.pointer_start_token_id = 151667
    if not getattr(ms_config, "pointer_end_token_id", None):
        ms_config.pointer_end_token_id = 151668

    ms_processor = AutoProcessor.from_pretrained(model_name)
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

    dtype = resolve_backbone_dtype(backbone_dtype) if torch.cuda.is_available() else torch.float32
    ms_model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_name,
        config=ms_config,
        torch_dtype=dtype,
    )
    if not getattr(ms_model.config, "pointer_pad_token_id", None):
        ms_model.config.pointer_pad_token_id = 151669
    if not hasattr(ms_model, "rope_deltas"):
        ms_model.rope_deltas = None

    _attach_instance_compat_shims(ms_model)
    return ms_model, ms_processor


def _pil_from_image_item(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, torch.Tensor):
        t = item.detach().cpu()
        arr = t.permute(1, 2, 0).numpy()
        if t.dtype != torch.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(item, np.ndarray):
        arr = item
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return Image.new("RGB", (1280, 800), color=(128, 128, 128))


def _standardize_pil_batch(images: Any) -> List[Image.Image]:
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:
            return [_pil_from_image_item(images)]
        return [_pil_from_image_item(images[i]) for i in range(images.shape[0])]
    if isinstance(images, (list, tuple)):
        return [_pil_from_image_item(im) for im in images]
    return [_pil_from_image_item(images)]


class _DryVisionHead(nn.Module):
    """Lightweight mock pointer head matching VisionHead_MultiPatch forward contract."""

    def __init__(self, d_model: int = 2048, hidden_dim: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj_enc = nn.Linear(d_model, hidden_dim)
        self.proj_dec = nn.Linear(d_model, hidden_dim)

    def forward(
        self,
        hidden_state_enc: torch.Tensor,
        hidden_state_dec: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        enc = self.proj_enc(hidden_state_enc)
        dec = self.proj_dec(hidden_state_dec)
        logits = torch.matmul(dec, enc.transpose(0, 1)) / math.sqrt(enc.shape[-1])
        attn = F.softmax(logits, dim=-1)
        return attn, None


class MSGUIActorSinglePassVLA(nn.Module):
    """Single-pass (0-generate) GUI-Actor-3B + AVCS ParallelActionHeads VLA.

    Produces pointer probabilities & connected-region coordinates without calling
    `model.generate()` or `lm_head`, and feeds the `<|pointer_pad|>` decoder
    state into `ParallelActionHeads` for all 10 Windows OS primitives.
    """

    def __init__(
        self,
        model_name: str = "microsoft/GUI-Actor-3B-Qwen2.5-VL",
        max_pixels: int = 1003520,
        backbone_dtype: str = "auto",
        query_dim: int = 512,
        backbone_hidden_dim: int = 2048,
        lora_layers: int = 0,
        lora_r: int = 16,
        train_pointer_head: bool = True,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_pixels = int(max_pixels)
        self.backbone_dtype = backbone_dtype
        self.query_dim = int(query_dim)
        self.backbone_hidden_dim = int(backbone_hidden_dim)
        self.dry_run = bool(dry_run)
        self.lora_config: Optional[Dict[str, Any]] = None
        self.generate_calls = 0

        if dry_run:
            from avcs.models.vla_cortex import ShowUITokenPruner  # noqa: F401
            self.backbone = nn.Module()
            self.backbone.multi_patch_pointer_head = _DryVisionHead(backbone_hidden_dim, min(256, query_dim))
            self.dry_img_proj = nn.Linear(3, backbone_hidden_dim)
            self.dry_q_emb = nn.Embedding(4096, backbone_hidden_dim)
            self.processor = None
        else:
            self.backbone, self.processor = load_ms_gui_actor_backbone(
                model_name=model_name,
                max_pixels=self.max_pixels,
                backbone_dtype=backbone_dtype,
            )
            self.backbone_hidden_dim = int(getattr(self.backbone.config, "hidden_size", backbone_hidden_dim))
            for p in self.backbone.parameters():
                p.requires_grad = False
            if train_pointer_head and hasattr(self.backbone, "multi_patch_pointer_head"):
                for p in self.backbone.multi_patch_pointer_head.parameters():
                    p.requires_grad = True
            self.backbone.eval()

        self.query_proj = nn.Sequential(
            nn.Linear(self.backbone_hidden_dim, self.query_dim),
            nn.LayerNorm(self.query_dim),
            nn.GELU(),
        )
        self.action_heads = ParallelActionHeads(
            in_dim=self.query_dim,
            num_actions=len(ACTION_VOCAB_10),
        )

        if int(lora_layers) > 0 and not dry_run:
            self.enable_lora(lora_layers=int(lora_layers), r=int(lora_r))

    def enable_lora(
        self,
        lora_layers: int = 8,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ) -> int:
        if self.dry_run or lora_layers <= 0:
            return 0

        layers = _resolve_decoder_layers(self.backbone)
        if not layers:
            return 0

        for vis in _resolve_visual_modules(self.backbone):
            for p in vis.parameters():
                p.requires_grad = False

        n_total = len(layers)
        start_idx = max(0, n_total - int(lora_layers))
        wrapped = 0
        target_names = ("q_proj", "k_proj", "v_proj", "o_proj")
        for layer_idx in range(start_idx, n_total):
            blk = layers[layer_idx]
            attn = getattr(blk, "self_attn", getattr(blk, "attn", None))
            if attn is None:
                continue
            for proj_name in target_names:
                proj = getattr(attn, proj_name, None)
                if isinstance(proj, nn.Linear):
                    setattr(
                        attn,
                        proj_name,
                        LoRALinear(proj, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    )
                    wrapped += 1
        self.lora_config = {
            "lora_layers": int(lora_layers),
            "r": int(r),
            "lora_alpha": float(lora_alpha),
            "lora_dropout": float(lora_dropout),
        }
        if wrapped > 0 and torch.cuda.is_available():
            import torch.utils.checkpoint as grad_checkpoint

            for offset, lora_block in enumerate(layers[start_idx:]):
                if getattr(lora_block, "_lora_ckpt_wrapped", False):
                    continue
                orig_lora_fwd = lora_block.forward
                is_boundary = offset == 0

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
                        return grad_checkpoint.checkpoint(_orig, *args, use_reentrant=False, **kwargs)
                    return _orig(*args, **kwargs)

                lora_block.forward = _checkpointed_lora_forward  # type: ignore[method-assign]
                setattr(lora_block, "_lora_ckpt_wrapped", True)
        return wrapped

    def train(self, mode: bool = True) -> "MSGUIActorSinglePassVLA":
        super().train(mode)
        if not self.dry_run and self.backbone is not None:
            self.backbone.eval()
            for mod in self.backbone.modules():
                if isinstance(mod, LoRALinear):
                    mod.train(mode)
                    mod.base_layer.eval()
            if hasattr(self.backbone, "multi_patch_pointer_head"):
                self.backbone.multi_patch_pointer_head.train(mode)
        return self

    def get_parameter_summary(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
        }

    def _forward_dry(self, images: Any, queries: Sequence[str]) -> Dict[str, Any]:
        device = next(self.query_proj.parameters()).device
        pil_list = _standardize_pil_batch(images)
        tensors = []
        for im in pil_list:
            resized = im.resize((448, 280), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensors.append(torch.from_numpy(arr))
        img_t = torch.stack(tensors, dim=0).to(device)
        b = img_t.shape[0]
        gh, gw = 10, 16
        pooled = F.adaptive_avg_pool2d(img_t, (gh, gw)).permute(0, 2, 3, 1).reshape(b, gh * gw, 3)
        xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) / gw
        ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) / gh
        my, mx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([mx.flatten(), my.flatten()], dim=-1)  # (K, 2)

        patch_feats = self.dry_img_proj(pooled)
        # Add deterministic spatial basis so linear attention can distinguish patches
        pos = torch.cat([torch.sin(coords[:, :1] * math.pi), torch.cos(coords[:, :1] * math.pi),
                         torch.sin(coords[:, 1:] * math.pi), torch.cos(coords[:, 1:] * math.pi)], dim=-1)
        patch_feats = patch_feats + F.pad(pos, (0, self.backbone_hidden_dim - 4)).unsqueeze(0)

        q_vecs = []
        for q in queries[:b]:
            toks = [1 + (abs(hash(w)) % 4095) for w in (q or "click").lower().split()[:8]] or [1]
            tok_t = torch.tensor(toks, dtype=torch.long, device=device)
            q_vecs.append(self.dry_q_emb(tok_t).mean(dim=0))
        while len(q_vecs) < b:
            q_vecs.append(self.dry_q_emb(torch.tensor([1], device=device)).squeeze(0))
        pointer_hidden = torch.stack(q_vecs, dim=0)

        attn_list = []
        sub_list = []
        soft_list = []
        for i in range(b):
            attn_i, _ = self.backbone.multi_patch_pointer_head(patch_feats[i], pointer_hidden[i : i + 1])
            attn_list.append(attn_i.float())
            soft_list.append(torch.matmul(attn_i.float(), coords).squeeze(0))
            rx, ry = ms_region_point_from_attn(attn_i, gw, gh)
            sub_list.append(torch.tensor([rx, ry], dtype=torch.float32, device=device))

        attn_probs = torch.stack(attn_list, dim=0)
        attn_logits = torch.log(attn_probs.clamp_min(1e-8))
        target_coords = torch.stack(soft_list, dim=0)
        subpixel_coords = torch.stack(sub_list, dim=0)

        proj_q = self.query_proj(pointer_hidden.float())
        act_out = self.action_heads(proj_q)
        return {
            "action_logits": act_out.action_logits.float(),
            "target_coords": target_coords.float(),
            "subpixel_coords": subpixel_coords.float(),
            "attn_logits": attn_logits.float(),
            "attn_probs": attn_probs.float(),
            "scroll_delta": act_out.scroll_delta.float(),
            "action_index": act_out.action_index,
            "action_names": act_out.action_name,
            "grid_hw": (gh, gw),
            "patch_coords": coords,
            "backbone_overflow": False,
            "generate_calls": 0,
        }

    def forward(
        self,
        images: Any,
        queries: Optional[Union[str, Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        if isinstance(queries, str):
            batch_queries = [queries]
        elif queries is None:
            batch_queries = ["Click the target UI element"]
        else:
            batch_queries = list(queries)

        if self.dry_run:
            return self._forward_dry(images, batch_queries)

        pil_images = _standardize_pil_batch(images)
        b = len(pil_images)
        while len(batch_queries) < b:
            batch_queries.append("Click the target UI element")
        batch_queries = batch_queries[:b]

        from qwen_vl_utils import process_vision_info

        device = next(self.query_proj.parameters()).device
        texts: List[str] = []
        image_inputs_all: List[Any] = []
        for img, instr in zip(pil_images, batch_queries):
            img_entry: Dict[str, Any] = {"type": "image", "image": img}
            if self.max_pixels > 0:
                img_entry["min_pixels"] = min(256 * 28 * 28, self.max_pixels)
                img_entry["max_pixels"] = self.max_pixels
            conv = [
                {"role": "system", "content": [{"type": "text", "text": MS_GUI_ACTOR_SYSTEM_PROMPT}]},
                {"role": "user", "content": [img_entry, {"type": "text", "text": instr}]},
            ]
            txt = self.processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=MS_GUI_ACTOR_CHAT_TEMPLATE,
            )
            texts.append(txt + MS_POINTER_ASSISTANT_PREFIX)
            imgs_i, _ = process_vision_info(conv)
            if imgs_i:
                image_inputs_all.extend(imgs_i)

        inputs = self.processor(
            text=texts,
            images=image_inputs_all if image_inputs_all else pil_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        has_backbone_grad = self.training and any(
            p.requires_grad for n, p in self.backbone.named_parameters() if "multi_patch_pointer_head" not in n
        )

        # Build inputs_embeds & visual scatter without invoking lm_head or generate().
        with torch.enable_grad() if has_backbone_grad else torch.no_grad():
            inputs_embeds = self.backbone.model.embed_tokens(input_ids)
            if pixel_values is not None:
                vis_dtype = getattr(self.backbone.visual, "dtype", inputs_embeds.dtype)
                pixel_values = pixel_values.to(dtype=vis_dtype)
                image_embeds_all = self.backbone.visual(pixel_values, grid_thw=image_grid_thw)
                img_tok_id = getattr(self.backbone.config, "image_token_id", 151655)
                image_mask = (input_ids == img_tok_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask, image_embeds_all.to(inputs_embeds.device, inputs_embeds.dtype)
                )
            position_ids, rope_deltas = self.backbone.get_rope_index(
                input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
            )
            self.backbone.rope_deltas = rope_deltas
            base_out = self.backbone.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            last_hidden = base_out.last_hidden_state

        overflow = bool(
            not torch.isfinite(inputs_embeds).all().item()
            or not torch.isfinite(last_hidden).all().item()
        )
        if overflow:
            inputs_embeds = torch.nan_to_num(inputs_embeds.float(), nan=0.0, posinf=64.0, neginf=-64.0).to(dtype=last_hidden.dtype)
            last_hidden = torch.nan_to_num(last_hidden.float(), nan=0.0, posinf=64.0, neginf=-64.0).to(dtype=last_hidden.dtype)

        img_tok_id = getattr(self.backbone.config, "image_token_id", 151655)
        ptr_pad_id = getattr(self.backbone.config, "pointer_pad_token_id", 151669)
        merge_size = int(getattr(getattr(self.backbone, "visual", None), "spatial_merge_size", 2))

        attn_list: List[torch.Tensor] = []
        sub_list: List[torch.Tensor] = []
        soft_list: List[torch.Tensor] = []
        ptr_vecs: List[torch.Tensor] = []
        coords_list: List[torch.Tensor] = []
        grid_hw_last = (1, 1)

        for i in range(b):
            vis_idx = torch.nonzero(input_ids[i] == img_tok_id, as_tuple=False).squeeze(-1)
            ptr_idx = torch.nonzero(input_ids[i] == ptr_pad_id, as_tuple=False).squeeze(-1)
            if ptr_idx.numel() == 0:
                ptr_idx = torch.tensor([input_ids.shape[1] - 1], device=device)

            vis_emb_i = inputs_embeds[i][vis_idx]
            ptr_hid_i = last_hidden[i][ptr_idx[:1]]
            ptr_vecs.append(ptr_hid_i.squeeze(0))

            attn_i, _ = self.backbone.multi_patch_pointer_head(vis_emb_i, ptr_hid_i)
            attn_f = attn_i.float()

            if image_grid_thw is not None:
                thw = image_grid_thw[0] if image_grid_thw.dim() == 1 else image_grid_thw[i]
                gh = max(1, int(thw[1].item()) // merge_size)
                gw = max(1, int(thw[2].item()) // merge_size)
            else:
                gh = max(1, int(math.isqrt(vis_emb_i.shape[0])))
                gw = max(1, vis_emb_i.shape[0] // gh)
            grid_hw_last = (gh, gw)

            xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) / gw
            ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) / gh
            my, mx = torch.meshgrid(ys, xs, indexing="ij")
            coords_i = torch.stack([mx.flatten(), my.flatten()], dim=-1)
            if coords_i.shape[0] != attn_f.shape[-1]:
                k_len = attn_f.shape[-1]
                coords_i = (
                    coords_i[:k_len]
                    if coords_i.shape[0] > k_len
                    else F.pad(coords_i, (0, 0, 0, k_len - coords_i.shape[0]), value=-1.0)
                )

            coords_list.append(coords_i)
            attn_list.append(attn_f)
            soft_list.append(torch.matmul(attn_f, coords_i).squeeze(0))
            rx, ry = ms_region_point_from_attn(attn_f, gw, gh)
            sub_list.append(torch.tensor([rx, ry], dtype=torch.float32, device=device))

        if b == 1 or all(a.shape[-1] == attn_list[0].shape[-1] for a in attn_list):
            attn_probs = torch.stack(attn_list, dim=0)
            patch_coords = coords_list[0] if b == 1 else torch.stack(coords_list, dim=0)
        else:
            max_k = max(a.shape[-1] for a in attn_list)
            attn_probs = torch.stack([F.pad(a, (0, max_k - a.shape[-1]), value=0.0) for a in attn_list], dim=0)
            patch_coords = torch.stack(
                [F.pad(c, (0, 0, 0, max_k - c.shape[0]), value=-1.0) for c in coords_list], dim=0
            )

        attn_logits = torch.log(attn_probs.clamp_min(1e-8))
        target_coords = torch.stack(soft_list, dim=0)
        subpixel_coords = torch.stack(sub_list, dim=0)

        ptr_stack = torch.stack(ptr_vecs, dim=0).float()
        proj_q = self.query_proj(ptr_stack)
        act_out = self.action_heads(proj_q)

        return {
            "action_logits": act_out.action_logits.float(),
            "target_coords": target_coords.float(),
            "subpixel_coords": subpixel_coords.float(),
            "attn_logits": attn_logits.float(),
            "attn_probs": attn_probs.float(),
            "scroll_delta": act_out.scroll_delta.float(),
            "action_index": act_out.action_index,
            "action_names": act_out.action_name,
            "grid_hw": grid_hw_last,
            "patch_coords": patch_coords,
            "backbone_overflow": bool(overflow or not torch.isfinite(ptr_stack).all().item()),
            "generate_calls": 0,
        }


__all__ = [
    "LoRALinear",
    "MSGUIActorSinglePassVLA",
    "load_ms_gui_actor_backbone",
    "ms_region_point_from_attn",
    "preserve_aspect_resize",
    "resolve_backbone_dtype",
]
