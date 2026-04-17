#!/bin/bash
set -e
source venv/bin/activate
echo "🚀 Starting ${app_name}..."
python -m src.app.gradio_app
