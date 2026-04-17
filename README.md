# Local Doctor / Mini-Lawyer – Sovereign SLM (2026)

Fully offline, privacy-first medical/legal assistant built on Phi-4-mini or Mistral-7B + RAG.

## Quick Start
1. `pip install -r requirements.txt`
2. Fine-tune (once) → `scripts/fine_tune.sh`
3. Convert to GGUF + import to Ollama as `local-doctor` (or `mini-lawyer`)
4. `./scripts/ingest_documents.sh`
5. `./scripts/start_app.sh`


Specialized Small Language Models (SLMs) The Trend: As of March 2026, developers are moving away from 70B+ models toward smaller, specialized 1B–7B parameter models (like Mistral-7B) to reduce inference costs and latency. Model to Build: Fine-tune a specialized model (e.g., Llama-3-8B or Mistral-7B) for legal document review, medical record summarization, or structured data extraction. Why it's used heavily: Enterprise clients want accuracy without the cost of massive LLMs.

Let me build this out comprehensively — architecture diagram first, then all the code files.Now let me build all the files:Here's everything you need, structured as a production-grade SLM project.

---

```mermaid
graph TD
    %% Node Definitions
    Title([Mistral-7B legal-review SLM — end-to-end pipeline])
    
    Corpus[Legal corpus<br/><small>contracts, case law, GDPR</small>]
    Prep[Dataset prep<br/><small>instruction pairs, chunking</small>]
    Base[Mistral-7B base<br/><small>HuggingFace pretrained</small>]
    
    FineTune[QLoRA fine-tuning rank=16, α=32, 4-bit NF4<br/><small>PEFT adapter training · gradient checkpointing · bf16</small>]
    
    Eval[Evaluation<br/><small>ROUGE, BERTScore, F1</small>]
    Merge[Merge + quantize<br/><small>GGUF / GPTQ / AWQ</small>]
    ModelCard[Model card<br/><small>README + metrics</small>]
    
    HFHub[HuggingFace Hub<br/><small>model repo · versioning · Space demo</small>]
    
    K8s[Kubernetes cluster<br/><small>Deployment · HPA · GPU nodepool<br/>vLLM inference server<br/>Prometheus · Grafana observability</small>]
    HFSpace[HuggingFace Space<br/><small>Gradio demo UI · public endpoint<br/>A10G GPU · Docker SDK space<br/>persistent storage · secrets</small>]

    %% Connections
    Title --- Corpus
    Corpus --> Prep
    Prep --> FineTune
    Corpus --> FineTune
    Base --> FineTune
    
    FineTune --> Eval
    FineTune --> Merge
    FineTune --> ModelCard
    
    Merge --> HFHub
    ModelCard --> HFHub
    
    HFHub --> K8s
    HFHub --> HFSpace

    %% Styling
    classDef title fill:#483D8B,stroke:#333,stroke-width:1px,color:#fff;
    classDef green fill:#004d40,stroke:#333,stroke-width:1px,color:#fff;
    classDef blue fill:#0d47a1,stroke:#333,stroke-width:1px,color:#fff;
    classDef red fill:#7f2214,stroke:#333,stroke-width:1px,color:#fff;
    classDef orange fill:#8B4513,stroke:#333,stroke-width:1px,color:#fff;
    classDef purple fill:#4527a0,stroke:#333,stroke-width:1px,color:#fff;
    classDef mergeGreen fill:#2e7d32,stroke:#333,stroke-width:1px,color:#fff;

    class Title title;
    class Corpus,Prep green;
    class Base,K8s,ModelCard blue;
    class FineTune orange; 
    class Eval,HFSpace red;
    class HFHub purple;
    class Merge mergeGreen;

```

## What was built

**The model:** Mistral-7B-Instruct fine-tuned for legal document review using QLoRA — trains on a single A100-80GB in ~6 hours, updating less than 1% of parameters via LoRA adapters. Four supported tasks: contract summary, clause extraction, risk flagging, and GDPR compliance — all returning structured JSON.

**The files:**

| File | What it does |
|---|---|
| `train.py` | QLoRA fine-tuning with 4-bit NF4 quantisation, response-only loss masking, W&B logging |
| `configs/train_config.yaml` | All hyperparameters (LR, LoRA rank, batch size, epochs) in one place |
| `dataset_prep.py` | Converts raw PDFs/TXTs to instruction-following JSONL pairs with text chunking and deduplication |
| `merge_and_export.py` | Merges LoRA adapter into base weights, optionally exports GPTQ or GGUF |
| `inference_server.py` | FastAPI + vLLM inference server with Prometheus metrics, health/readiness probes |
| `Dockerfile` | Production image on CUDA 12.4, non-root user, model cache via PVC |
| `k8s/deployment.yaml` | GPU-scheduled Kubernetes Deployment with rolling updates, init container for model download |
| `k8s/manifests.yaml` | Service, Ingress, HPA (scales 2–8 pods), PVC, ConfigMap, Secret |
| `hf_space/app.py` | Gradio demo with task selector, example contracts, and JSON output formatting |
| `MODEL_CARD.md` | HuggingFace README with benchmark table, quickstart, and citation |
| `DEPLOYMENT_GUIDE.md` | Full step-by-step instructions (open this file first) |

## Key engineering decisions

**QLoRA over full fine-tuning:** Reduces VRAM from ~120GB to ~16GB with negligible accuracy loss on domain-specific tasks.

**vLLM for serving:** Continuous batching gives 3–5× better throughput than naive `model.generate()`, critical for enterprise multi-user loads.

**Response-only loss masking:** The training loss is computed only on the assistant's answer, not the prompt — prevents the model from memorising boilerplate instruction templates.

**HPA on request rate:** The autoscaler in `manifests.yaml` scales pods based on `slm_requests_total` (a Prometheus custom metric) rather than CPU, which is the right signal for an LLM serving workload where CPU stays low but queue depth grows under load.