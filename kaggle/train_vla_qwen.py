"""Kaggle Dual-T4 QLoRA Fine-Tuning Pipeline for Qwen2.5-VL-3B.

Step 5.3 of Phase 5:
Fine-tunes Qwen2.5-VL-3B-Instruct on the curated AVCS-VLA desktop dataset.

Hardware Guardrails (Turing T4 Specific):
- Precision: fp16=True, bf16=False (T4 lacks native bf16 hardware ALUs).
- Attention: SDPA (PyTorch scaled dot-product attention).
- Quantization: 4-bit NormalFloat4 (NF4) with double quantization via bitsandbytes.
- LoRA Rank: r=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"].
- Checkpoint Interval: Save every 200 steps (guards against Kaggle 9-hour limit).
- Token Budget: max_pixels=1024000 (1280x800) to maintain high T4 training throughput.
- Trainer Guardrail: remove_unused_columns=False to prevent collator KeyError: 'conversations'.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

try:
    from PIL import Image
except ImportError:
    Image = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [STEP-5.3-TRAIN] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train.vla_qwen")


class VLAChatMLDataset(Dataset):
    """PyTorch Dataset for ChatML format VLA episodes."""

    def __init__(
        self,
        jsonl_path: str,
        base_dir: Optional[str] = None,
        max_pixels: int = 1280 * 800,
    ) -> None:
        self.p_jsonl = Path(jsonl_path)
        self.base_dir = Path(base_dir) if base_dir else self.p_jsonl.parent
        self.max_pixels = max_pixels

        self.records: List[Dict[str, Any]] = []
        if self.p_jsonl.exists():
            with open(self.p_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
        logger.info(f"Loaded {len(self.records)} records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_rel = rec.get("image", "")
        img_path = self.base_dir / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            # Fallback canvas if image not locally present
            img = Image.new("RGB", (1280, 800), color=(240, 240, 245))

        convs = rec.get("conversations", [])
        return {
            "id": rec.get("id", f"sample_{idx}"),
            "image": img,
            "conversations": convs,
        }


class VLADataCollator:
    """Collates multimodal chat conversations for Qwen2.5-VL with prompt loss masking."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        formatted_convs_list = []
        images = []

        for item in batch:
            convs = item["conversations"]
            images.append(item["image"])

            # Ensure canonical Qwen2.5-VL multimodal structure
            formatted_conv = []
            for msg in convs:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    clean_text = content.replace("<image>\n", "").replace("<image>", "").strip()
                    formatted_conv.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": clean_text},
                        ],
                    })
                else:
                    formatted_conv.append({
                        "role": "assistant",
                        "content": content,
                    })
            formatted_convs_list.append(formatted_conv)

        # Apply chat template
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in formatted_convs_list
        ]

        # Batch encode with dynamic vision token padding
        batch_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Construct causal LM labels with prompt masking
        labels = batch_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # Mask <|image_pad|> tokens
        image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_pad_id is not None:
            labels[labels == image_pad_id] = -100

        # Mask prompt tokens so loss is computed strictly on assistant response
        assistant_prefix_ids = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        for i in range(len(batch_inputs["input_ids"])):
            input_ids_i = batch_inputs["input_ids"][i].tolist()
            for j in range(len(input_ids_i) - len(assistant_prefix_ids) + 1):
                if input_ids_i[j:j + len(assistant_prefix_ids)] == assistant_prefix_ids:
                    labels[i, : j + len(assistant_prefix_ids)] = -100
                    break

        batch_inputs["labels"] = labels
        return batch_inputs


def create_qlora_model(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> Tuple[Any, Any]:
    """Loads 4-bit quantized base model and attaches PEFT LoRA adapter."""
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger.info(f"Initializing 4-bit NF4 Quantization for {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Turing T4 FP16
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 800,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    # Critical: Incompatible with gradient checkpointing
    model.config.use_cache = False

    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def train(
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str = "runs/vla_qwen_qlora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    save_steps: int = 200,
) -> None:
    """Executes the QLoRA fine-tuning loop under strict Kaggle Dual-T4 constraints."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model, processor = create_qlora_model()

    train_ds = VLAChatMLDataset(train_jsonl)
    val_ds = VLAChatMLDataset(val_jsonl)
    data_collator = VLADataCollator(processor)

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        fp16=True,                       # Mandatory for Nvidia Turing T4
        bf16=False,                      # Strictly False on T4 (no native hardware ALUs)
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,     # Critical: Prevents Trainer from stripping 'conversations'/'image'
        gradient_checkpointing=True,     # Activates memory-efficient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Critical: Fixes DDP hook firing twice
        ddp_find_unused_parameters=False, # Accelerates DDP graph execution
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    logger.info("Starting Trainer.train() on Dual Tesla T4s...")
    trainer.train()

    # Save final LoRA adapter and processor
    final_adapter_dir = out_path / "final_adapter"
    logger.info(f"Saving final trained adapter to {final_adapter_dir}...")
    trainer.model.save_pretrained(str(final_adapter_dir))
    processor.save_pretrained(str(final_adapter_dir))
    logger.info("Step 5.3 Training Complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Dual-T4 QLoRA Fine-Tuning Pipeline.")
    parser.add_argument("--train-jsonl", type=str, default="data/vla_dataset/train.jsonl", help="Train dataset JSONL")
    parser.add_argument("--val-jsonl", type=str, default="data/vla_dataset/val.jsonl", help="Val dataset JSONL")
    parser.add_argument("--output-dir", type=str, default="runs/vla_qwen_qlora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--dry-run", action="store_true", help="Perform configuration dry run without model load")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Performing Step 5.3 dry run...")
        train_ds = VLAChatMLDataset(args.train_jsonl)
        val_ds = VLAChatMLDataset(args.val_jsonl)
        assert len(train_ds) > 0, "Train dataset must not be empty"
        assert len(val_ds) > 0, "Validation dataset must not be empty"
        logger.info(f"Dry run PASSED: {len(train_ds)} train samples, {len(val_ds)} val samples verified.")
        return

    train(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
    )


if __name__ == "__main__":
    main()
