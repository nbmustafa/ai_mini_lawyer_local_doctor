#!/bin/bash
set -e

# Install HuggingFace CLI
pip install huggingface-hub

# Log in (generates ~/.huggingface/token)
hf auth login
# Paste your token from: https://huggingface.co/settings/tokens
# Required scopes: write

# Create repository
hf repo create cogmazi_mistral-7b-legal-review --type model