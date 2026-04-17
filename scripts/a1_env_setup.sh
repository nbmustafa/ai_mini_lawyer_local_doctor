#!/bin/bash
set -e

# Clone repo
# git clone https://github.com/nbmustafa/slm-legal.git && cd slm-legal

# cd to the repository
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install training dependencies
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl bitsandbytes accelerate datasets wandb rouge-score pdfplumber pyyaml
