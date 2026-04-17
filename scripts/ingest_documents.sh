#!/bin/bash
set -e
source venv/bin/activate  # Windows: call venv\Scripts\activate
python -m src.rag.ingestion
