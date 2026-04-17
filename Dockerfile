# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Legal Review SLM Inference Server
# Base: NVIDIA CUDA 12.4 + Python 3.11
# Target: Kubernetes pod with 1× A10G (24 GB) or 1× A100 (40/80 GB) GPU
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git curl wget ca-certificates \
        libglib2.0-0 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serve.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY inference_server.py .

# ── Non-root user for security ───────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

# Model is pulled from HuggingFace Hub at startup (cached to /model-cache)
# Mount a PVC at /model-cache in Kubernetes to persist across pod restarts
ENV MODEL_CACHE_DIR=/model-cache \
    TRANSFORMERS_CACHE=/model-cache \
    HF_HOME=/model-cache \
    PORT=8000

EXPOSE 8000

# Healthcheck — readiness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

CMD ["python", "-m", "uvicorn", "inference_server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
