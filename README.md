# Local Doctor / Mini-Lawyer – Sovereign SLM (2026)

Fully offline, privacy-first medical/legal assistant built on Phi-4-mini or Mistral-7B + RAG.

## Quick Start
1. `pip install -r requirements.txt`
2. Fine-tune (once) → `scripts/fine_tune.sh`
3. Convert to GGUF + import to Ollama as `local-doctor` (or `mini-lawyer`)
4. `./scripts/ingest_documents.sh`
5. `./scripts/start_app.sh`

All data stays on your machine. Air-gapped Docker ready.

## Models (Ollama names)
- Medical: `local-doctor` (Phi-4-mini-instruct fine-tune)
- Legal:  `mini-lawyer`   (Mistral-7B-dragon fine-tune)

Config: `config/config.yaml`

```text
local-doctor-slm/                  # (or local-lawyer-slm — rename as needed)
├── .env.example                   # Template for environment variables
├── .gitignore
├── Dockerfile
├── docker-compose.yml             # Optional: for air-gapped container
├── pyproject.toml                 # or requirements.txt + poetry/uv
├── README.md                      # Full setup + usage guide
├── LICENSE                        # MIT or your choice
│
├── config/                        # All configurable settings
│   ├── config.yaml                # Main config (model paths, RAG settings, etc.)
│   └── prompts.yaml               # System prompts, RAG templates
│
├── data/                          # NEVER commit real patient/legal data
│   ├── raw/                       # Original PDFs, docs, de-identified notes
│   ├── processed/                 # Chunked documents (after ingestion)
│   ├── vector_db/                 # Chroma (or LanceDB/Qdrant) persistent storage
│   └── fine_tune_datasets/        # JSON/Parquet instruction datasets (5K–20K rows)
│
├── notebooks/                     # One-off experiments (Colab-style)
│   └── 01_fine_tune_phi4.ipynb    # The exact Unsloth script from the blueprint
│
├── src/                           # All production Python code
│   ├── __init__.py
│   │
│   ├── core/                      # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py              # Loads config.yaml + .env
│   │   ├── logging.py
│   │   └── utils.py               # chunking, de-identification helpers
│   │
│   ├── fine_tune/                 # Step 2 — Fine-tuning (run once)
│   │   ├── __init__.py
│   │   ├── train.py               # Production version of the Unsloth script
│   │   └── dataset_formatter.py   # format_instruction + synthetic data helpers
│   │
│   ├── rag/                       # Step 3 — RAG (External Brain) ← SEPARATE MODULE
│   │   ├── __init__.py
│   │   ├── embeddings.py          # Jina v4 + BGE-M3 hybrid
│   │   ├── vector_store.py        # Chroma client + add/query logic
│   │   ├── chunking.py            # Semantic + recursive chunking (10-20% overlap)
│   │   ├── retriever.py           # Hybrid retrieval + Self-RAG critique
│   │   └── ingestion.py           # Pipeline: raw → chunks → vector_db
│   │
│   ├── inference/                 # Model loading & generation
│   │   ├── __init__.py
│   │   ├── model.py               # Load fine-tuned model (Ollama or HF + Unsloth)
│   │   └── generator.py           # combine RAG context + model inference
│   │
│   ├── evaluation/                # Step 4
│   │   ├── __init__.py
│   │   └── evaluator.py           # MedQA/LegalBench + RAGAS/DeepEval
│   │
│   └── app/                       # Step 5 — Frontend + API
│       ├── __init__.py
│       ├── gradio_app.py          # Main chat UI (recommended for local)
│       ├── streamlit_app.py       # Alternative UI
│       └── fastapi_server.py      # Optional headless API (for integration)
│
├── scripts/                       # One-click automation
│   ├── ingest_documents.sh        # Run RAG ingestion pipeline
│   ├── fine_tune.sh               # Run src/fine_tune/train.py
│   ├── build_gguf.sh              # llama.cpp conversion
│   ├── start_app.sh               # Launch Gradio + Ollama
│   └── docker_build.sh
│
└── tests/                         # Prod-grade testing
    ├── test_rag.py
    ├── test_inference.py
    └── test_end_to_end.py
```
