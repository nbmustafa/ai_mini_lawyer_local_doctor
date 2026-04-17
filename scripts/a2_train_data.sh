#!/bin/bash
set -e

# Put your raw .txt or .pdf contracts in data/raw/
mkdir -p data/raw

# Generate instruction pairs
python dataset_prep.py \
  --input_dir data/raw \
  --output    data/legal_instruct.jsonl \
  --tasks     contract_summary clause_extraction risk_flagging gdpr_compliance

# Verify output
head -n 2 data/legal_instruct.jsonl | python -m json.tool
