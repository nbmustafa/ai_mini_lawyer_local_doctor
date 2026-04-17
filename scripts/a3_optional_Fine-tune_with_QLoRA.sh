#!/bin/bash
set -e

# Set W&B API key (optional but recommended)
export WANDB_API_KEY=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

# Single GPU (A100 80 GB)
python train.py --config configs/train_config.yaml

# Multi-GPU (2× A10G via FSDP)
accelerate launch --config_file configs/accelerate_fsdp.yaml \
    train.py --config configs/train_config.yaml
