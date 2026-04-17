# Step-by-Step Guide: Deploy on Kubernetes & Publish on HuggingFace

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Training & serving |
| CUDA | 12.4 | GPU compute |
| Docker | 24+ | Container builds |
| kubectl | 1.29+ | Kubernetes CLI |
| Helm | 3.14+ | K8s package manager |
| huggingface-cli | latest | Model publishing |

---

## Part A — Train the Model

### A1. Environment setup

```bash
# Clone repo
git clone https://github.com/your-org/slm-legal.git && cd slm-legal

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install training dependencies
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft trl bitsandbytes accelerate datasets \
            wandb rouge-score pdfplumber pyyaml
```

### A2. Prepare training data

```bash
# Put your raw .txt or .pdf contracts in data/raw/
mkdir -p data/raw

# Generate instruction pairs
python dataset_prep.py \
  --input_dir data/raw \
  --output    data/legal_instruct.jsonl \
  --tasks     contract_summary clause_extraction risk_flagging gdpr_compliance

# Verify output
head -n 2 data/legal_instruct.jsonl | python -m json.tool
```

Expected output: `48,000+ instruction pairs`

### A3. Fine-tune with QLoRA

```bash
# Set W&B API key (optional but recommended)
export WANDB_API_KEY=your_wandb_key

# Single GPU (A100 80 GB)
python train.py --config configs/train_config.yaml

# Multi-GPU (2× A10G via FSDP)
accelerate launch --config_file configs/accelerate_fsdp.yaml \
    train.py --config configs/train_config.yaml
```

Training takes ~6 hours on 1× A100-80GB.
Monitor at: https://wandb.ai/your-org/slm-legal-review

### A4. Evaluate the adapter

```bash
python evaluate.py \
  --adapter_dir outputs/mistral7b-legal-lora \
  --test_data   data/test_set.jsonl \
  --tasks       all

# Expected results:
# ROUGE-L:       0.48–0.52
# JSON validity: 95–98 %
# BERTScore F1:  0.85–0.89
```

---

## Part B — Merge & Quantise

### B1. Merge LoRA adapter into base weights

```bash
python merge_and_export.py \
  --adapter_dir  outputs/mistral7b-legal-lora \
  --base_model   mistralai/Mistral-7B-Instruct-v0.3 \
  --hub_repo     your-username/mistral-7b-legal-review \
  --push_to_hub          # uploads merged fp16 model to Hub
```

### B2. (Optional) Export GPTQ for faster vLLM serving

```bash
python merge_and_export.py \
  --adapter_dir  outputs/mistral7b-legal-lora \
  --base_model   mistralai/Mistral-7B-Instruct-v0.3 \
  --hub_repo     your-username/mistral-7b-legal-review \
  --export_gptq \
  --gptq_bits    4
```

---

## Part C — Publish on HuggingFace

### C1. Log in and create the model repo

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Log in (generates ~/.huggingface/token)
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens
# Required scopes: write

# Create repository
huggingface-cli repo create mistral-7b-legal-review --type model
```

### C2. Push the model

```bash
# The merge_and_export.py script handles this automatically,
# OR push manually:

from huggingface_hub import HfApi
api = HfApi()

# Push model weights
api.upload_folder(
    folder_path="outputs/mistral7b-legal-lora/merged",
    repo_id="your-username/mistral-7b-legal-review",
    repo_type="model",
)

# Push model card
api.upload_file(
    path_or_fileobj="MODEL_CARD.md",
    path_in_repo="README.md",
    repo_id="your-username/mistral-7b-legal-review",
)
```

### C3. Create a Gradio Space demo

```bash
# Create the Space
huggingface-cli repo create mistral-7b-legal-review-demo \
    --type space \
    --space_sdk gradio

# Clone the Space repo
git clone https://huggingface.co/spaces/your-username/mistral-7b-legal-review-demo
cd mistral-7b-legal-review-demo

# Copy the demo files
cp /path/to/slm-legal/hf_space/app.py .
cp /path/to/slm-legal/hf_space/requirements.txt .
```

### C4. Configure Space hardware and secrets

In the HuggingFace web UI:
1. Go to your Space → **Settings → Hardware**
2. Select **A10G (24 GB)** (billed ~$1/hr while running)
3. Go to **Settings → Secrets** and add:
   - `HF_TOKEN` = your HuggingFace token
   - `MODEL_ID` = `your-username/mistral-7b-legal-review`

### C5. Push and deploy

```bash
# From inside the cloned Space repo:
git add app.py requirements.txt
git commit -m "feat: add legal SLM demo"
git push

# Space builds automatically — watch logs at:
# https://huggingface.co/spaces/your-username/mistral-7b-legal-review-demo
```

Demo goes live at:
`https://huggingface.co/spaces/your-username/mistral-7b-legal-review-demo`

---

## Part D — Deploy on Kubernetes

### D1. Build and push the Docker image

```bash
# Build
docker build -t your-registry/legal-slm-server:1.0.0 .

# Push to your registry (Docker Hub / ECR / GCR / ACR)
docker push your-registry/legal-slm-server:1.0.0

# For AWS ECR:
aws ecr get-login-password | docker login --username AWS \
    --password-stdin 123456789.dkr.ecr.eu-west-1.amazonaws.com
docker tag  legal-slm-server:1.0.0 \
    123456789.dkr.ecr.eu-west-1.amazonaws.com/legal-slm-server:1.0.0
docker push 123456789.dkr.ecr.eu-west-1.amazonaws.com/legal-slm-server:1.0.0
```

### D2. Provision a GPU node pool

**GKE (Google Kubernetes Engine)**
```bash
gcloud container node-pools create gpu-pool \
  --cluster=your-cluster \
  --machine-type=a2-highgpu-1g \       # 1× A100 40GB
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --num-nodes=2 \
  --enable-autoscaling --min-nodes=1 --max-nodes=8 \
  --node-labels=accelerator=nvidia-a100

# Install NVIDIA device plugin
kubectl apply -f \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.0/deployments/static/nvidia-device-plugin.yml
```

**EKS (AWS)**
```bash
eksctl create nodegroup \
  --cluster=your-cluster \
  --name=gpu-ng \
  --node-type=g5.xlarge \             # 1× A10G 24GB
  --nodes=2 --nodes-min=1 --nodes-max=8 \
  --node-labels="accelerator=nvidia-a10g"
```

### D3. Create namespace and secrets

```bash
kubectl create namespace ai-inference

# HuggingFace token (for model download at startup)
kubectl create secret generic legal-slm-secrets \
  --from-literal=HF_TOKEN=hf_your_token_here \
  -n ai-inference

# Container registry credentials
kubectl create secret docker-registry registry-credentials \
  --docker-server=your-registry \
  --docker-username=your-user \
  --docker-password=your-password \
  -n ai-inference
```

### D4. Apply Kubernetes manifests

```bash
# Apply in order:
kubectl apply -f k8s/namespace.yaml        # Namespace (if not already created)
kubectl apply -f k8s/manifests.yaml        # PVC, ConfigMap, Secret template, Service, HPA, Ingress
kubectl apply -f k8s/deployment.yaml       # Deployment

# Verify pods are running
kubectl get pods -n ai-inference -w
# NAME                               READY   STATUS    RESTARTS   AGE
# legal-slm-server-abc12-xxx         1/1     Running   0          3m

# Check logs
kubectl logs -f deployment/legal-slm-server -n ai-inference
```

### D5. Verify the deployment

```bash
# Port-forward for local testing
kubectl port-forward svc/legal-slm-service 8080:80 -n ai-inference

# Test health
curl http://localhost:8080/health
# {"status": "ok"}

# Test readiness
curl http://localhost:8080/ready
# {"status": "ready", "model": "your-username/mistral-7b-legal-review"}

# Test inference
curl -X POST http://localhost:8080/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "task": "contract_summary",
    "text": "SERVICE AGREEMENT between Acme Corp and Beta Ltd. Payment: £240,000 in 4 instalments. Governing law: England and Wales."
  }'
```

### D6. Install observability stack

```bash
# Add Prometheus + Grafana via Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm upgrade --install kube-prometheus-stack \
  prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.enabled=true \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Install NVIDIA DCGM Exporter for GPU metrics
helm upgrade --install dcgm-exporter \
  gpu-helm-charts/dcgm-exporter \
  --namespace monitoring

# Access Grafana dashboard
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring
# Login: admin / prom-operator
```

Key metrics to monitor:
- `slm_requests_total` — request rate
- `slm_request_latency_seconds` — p50/p95/p99 latency
- `DCGM_FI_DEV_GPU_UTIL` — GPU utilisation (target: 70–85%)
- `DCGM_FI_DEV_FB_USED` — GPU memory used

### D7. Configure autoscaling with KEDA (optional but recommended)

KEDA enables scaling to zero during off-peak hours, saving GPU costs.

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace

# Apply ScaledObject (scales on Prometheus query)
kubectl apply -f k8s/keda-scaledobject.yaml
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| View pod logs | `kubectl logs -f deploy/legal-slm-server -n ai-inference` |
| Restart deployment | `kubectl rollout restart deploy/legal-slm-server -n ai-inference` |
| Scale manually | `kubectl scale deploy/legal-slm-server --replicas=4 -n ai-inference` |
| Check HPA status | `kubectl get hpa -n ai-inference` |
| Roll back to previous | `kubectl rollout undo deploy/legal-slm-server -n ai-inference` |
| Update model version | Update `MODEL_ID` in ConfigMap + restart pods |

---

## Cost Estimate (AWS)

| Component | Instance | $/hr | Monthly (24/7) |
|-----------|----------|------|----------------|
| Training (1-time) | p4d.24xlarge (8× A100) | $32.77 | ~$200 (6 hr) |
| Inference (min 2 pods) | g5.xlarge (1× A10G) × 2 | $1.006 × 2 | ~$1,460 |
| HF Space demo | A10G | $1.05 | ~$756 |
| Storage (EFS 30 GB) | — | — | ~$10 |

**Tip**: Use Spot Instances for training (70% saving) and Reserved Instances
for inference (40% saving on 1-year commitment).
