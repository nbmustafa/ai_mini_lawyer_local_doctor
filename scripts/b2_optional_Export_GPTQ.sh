#!/bin/bash
set -e

python merge_and_export.py \
  --adapter_dir  outputs/mistral7b-legal-lora \
  --base_model   mistralai/Mistral-7B-Instruct-v0.3 \
  --hub_repo     your-username/mistral-7b-legal-review \
  --export_gptq \
  --gptq_bits    4
