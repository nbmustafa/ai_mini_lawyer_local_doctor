"""
train.py — QLoRA fine-tuning of Mistral-7B-Instruct for legal document review.

Key design decisions:
  - 4-bit NF4 quantisation (BitsAndBytes) keeps VRAM under 16 GB on a single A100/A10G.
  - LoRA adapters (rank=16, alpha=32) update <1 % of parameters.
  - Gradient checkpointing + bf16 halves peak memory vs fp32.
  - HuggingFace Trainer + SFTTrainer for clean, reproducible training loops.
  - Weights & Biases integration for experiment tracking.

Usage:
  python train.py --config configs/train_config.yaml
"""

import os
import argparse
import yaml
import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb


# ─────────────────────────────────────────────
# Configuration dataclasses
# ─────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        metadata={"help": "Base model to fine-tune."},
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Enable flash-attention 2 (requires compatible GPU)."},
    )


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="data/legal_instruct.jsonl",
        metadata={"help": "Path to JSONL dataset with 'instruction' and 'output' keys."},
    )
    max_seq_length: int = field(default=2048)
    val_split_ratio: float = field(default=0.05)


@dataclass
class LoraArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


# ─────────────────────────────────────────────
# Prompt template (Mistral instruction format)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior legal analyst AI. You review contracts, identify key clauses, "
    "flag risks, and produce structured summaries for enterprise legal teams. "
    "Be concise, precise, and always cite the relevant clause number."
)

def format_prompt(example: dict) -> str:
    """Convert a dataset row into a Mistral-chat instruction string."""
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{example['instruction']} [/INST] {example['output']} </s>"
    )


# ─────────────────────────────────────────────
# BitsAndBytes 4-bit quantisation config
# ─────────────────────────────────────────────

def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # Normal Float 4 — best for LLMs
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # Nested quantisation saves ~0.4 GB
    )


# ─────────────────────────────────────────────
# Model + tokeniser loading
# ─────────────────────────────────────────────

def load_model_and_tokenizer(model_args: ModelArguments):
    bnb_config = build_bnb_config()

    attn_impl = "flash_attention_2" if model_args.use_flash_attention_2 else "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )

    # Required before attaching LoRA to a quantised model
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.config.use_cache = False  # Incompatible with gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=False,
        padding_side="right",  # Required for SFT loss masking
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ─────────────────────────────────────────────
# LoRA adapter
# ─────────────────────────────────────────────

def attach_lora(model, lora_args: LoraArguments):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

def load_and_prepare_dataset(
    data_args: DataArguments,
    tokenizer,
) -> DatasetDict:
    raw = load_dataset("json", data_files=data_args.dataset_path, split="train")
    raw = raw.map(lambda x: {"text": format_prompt(x)}, remove_columns=raw.column_names)

    split = raw.train_test_split(test_size=data_args.val_split_ratio, seed=42)
    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    return split


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(config_path: Optional[str] = None):
    # Parse args from dataclasses + optional YAML override
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_args, data_args, lora_args, training_args = parser.parse_dict(cfg)
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # W&B
    wandb.init(
        project="slm-legal-review",
        name=training_args.run_name or "mistral7b-lora-legal",
        config={
            "model": model_args.model_name_or_path,
            "lora_r": lora_args.lora_r,
            "lora_alpha": lora_args.lora_alpha,
            "max_seq_length": data_args.max_seq_length,
        },
    )

    # Build components
    model, tokenizer = load_model_and_tokenizer(model_args)
    model = attach_lora(model, lora_args)
    dataset = load_and_prepare_dataset(data_args, tokenizer)

    # Response-only loss masking — only compute loss on assistant turn
    response_template = " [/INST]"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=data_args.max_seq_length,
        data_collator=collator,
        packing=False,
    )

    # Train
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    wandb.finish()
    print(f"\n✅  Adapter saved to: {training_args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    train(config_path=args.config)
