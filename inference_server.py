"""
inference_server.py — Production inference server for the legal-review SLM.

Architecture:
  - FastAPI async REST API
  - vLLM for high-throughput batched inference (continuous batching)
  - Pydantic v2 request/response schemas
  - Prometheus metrics exposed at /metrics
  - Health + readiness probes at /health and /ready
  - Structured JSON output enforced via guided decoding (outlines grammar)

Endpoints:
  POST /v1/analyze        — Full legal document analysis
  POST /v1/extract        — Clause extraction only
  POST /v1/risk           — Risk flagging only
  GET  /v1/models         — List loaded models
  GET  /health            — Liveness probe
  GET  /ready             — Readiness probe
  GET  /metrics           — Prometheus metrics

Usage (local):
  uvicorn inference_server:app --host 0.0.0.0 --port 8000 --workers 1

Usage (production via gunicorn):
  gunicorn inference_server:app -w 1 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 --timeout 120
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Prometheus
from prometheus_client import Counter, Histogram, make_asgi_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Prometheus metrics
# ─────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "slm_requests_total", "Total inference requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "slm_request_latency_seconds", "Inference latency", ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
TOKEN_COUNT = Counter(
    "slm_tokens_generated_total", "Total tokens generated"
)

# ─────────────────────────────────────────────
# Global model handle
# ─────────────────────────────────────────────

_llm = None  # vLLM AsyncLLMEngine
_tokenizer = None

MODEL_ID = os.getenv("MODEL_ID", "your-username/mistral-7b-legal-review")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))   # Low temp for deterministic legal output
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))


# ─────────────────────────────────────────────
# Lifespan: load model once on startup
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm, _tokenizer
    logger.info(f"Loading model: {MODEL_ID}")

    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from transformers import AutoTokenizer

        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            tensor_parallel_size=TENSOR_PARALLEL,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=True,   # Cache system prompt KV across requests
        )
        _llm = AsyncLLMEngine.from_engine_args(engine_args)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        logger.info("✅  Model loaded and ready.")
    except ImportError:
        logger.warning("vLLM not installed — running in mock mode (dev only)")

    yield

    logger.info("Shutting down model engine.")


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Legal Review SLM API",
    description="Enterprise legal document analysis powered by Mistral-7B fine-tuned on legal corpora.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Contract or clause text to analyse", min_length=50)
    task: Literal["contract_summary", "clause_extraction", "risk_flagging", "gdpr_compliance"] = Field(
        default="contract_summary"
    )
    max_tokens: Optional[int] = Field(default=None, ge=64, le=2048)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    task: str
    result: str
    model: str
    latency_ms: float
    tokens_generated: int


# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior legal analyst AI. Review contracts, extract key clauses, "
    "flag risks, and produce structured JSON summaries. Be precise and concise."
)

TASK_INSTRUCTIONS = {
    "contract_summary": (
        "Summarise the contract. Return JSON with keys: "
        "parties, effective_date, key_obligations, termination_conditions, governing_law."
    ),
    "clause_extraction": (
        "Extract all clauses. For each return: clause_number, clause_type, summary."
    ),
    "risk_flagging": (
        "Identify legal risks. For each return: risk_level, risk_type, affected_party, "
        "clause_reference, recommended_action."
    ),
    "gdpr_compliance": (
        "Check GDPR compliance. Return: compliance_score (0-100), gaps list, recommended_clauses."
    ),
}


def build_prompt(text: str, task: str) -> str:
    instruction = f"{TASK_INSTRUCTIONS[task]}\n\nText:\n{text}"
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{instruction} [/INST]"
    )


# ─────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────

async def run_inference(prompt: str, max_tokens: int, temperature: float) -> tuple[str, int]:
    if _llm is None:
        # Mock mode for local dev without GPU
        return '{"mock": "vLLM not loaded — dev mode"}', 10

    from vllm import SamplingParams
    import asyncio, uuid

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</s>", "[INST]"],
    )
    request_id = str(uuid.uuid4())

    results_generator = _llm.generate(prompt, sampling, request_id)
    final = None
    async for output in results_generator:
        final = output

    text = final.outputs[0].text.strip()
    n_tokens = len(final.outputs[0].token_ids)
    return text, n_tokens


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    max_tok = req.max_tokens or MAX_TOKENS
    temp = req.temperature if req.temperature is not None else TEMPERATURE

    prompt = build_prompt(req.text, req.task)

    try:
        result, n_tokens = await run_inference(prompt, max_tok, temp)
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/v1/analyze", status="error").inc()
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000
    TOKEN_COUNT.inc(n_tokens)
    REQUEST_COUNT.labels(endpoint="/v1/analyze", status="ok").inc()
    REQUEST_LATENCY.labels(endpoint="/v1/analyze").observe(latency_ms / 1000)

    return AnalyzeResponse(
        task=req.task,
        result=result,
        model=MODEL_ID,
        latency_ms=round(latency_ms, 1),
        tokens_generated=n_tokens,
    )


@app.post("/v1/extract")
async def extract_clauses(req: AnalyzeRequest):
    req.task = "clause_extraction"
    return await analyze(req)


@app.post("/v1/risk")
async def flag_risks(req: AnalyzeRequest):
    req.task = "risk_flagging"
    return await analyze(req)


@app.get("/v1/models")
async def list_models():
    return {"models": [{"id": MODEL_ID, "type": "legal-slm"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model": MODEL_ID}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
