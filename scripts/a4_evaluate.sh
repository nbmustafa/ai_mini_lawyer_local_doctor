#!/bin/bash
set -e

python evaluate.py \
  --adapter_dir outputs/mistral7b-legal-lora \
  --test_data   data/test_set.jsonl \
  --tasks       all

# Expected results:
# ROUGE-L:       0.48–0.52
# JSON validity: 95–98 %
# BERTScore F1:  0.85–0.89
