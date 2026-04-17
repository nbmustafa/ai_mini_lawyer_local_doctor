---
language:
  - en
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
  - legal
  - nlp
  - text-generation
  - peft
  - lora
  - fine-tuned
  - enterprise
datasets:
  - custom-legal-corpus
metrics:
  - rouge
  - bertscore
pipeline_tag: text-generation
library_name: transformers
---

# mistral-7b-legal-review

A Mistral-7B-Instruct model fine-tuned with QLoRA on a curated corpus of
contracts, legal opinions, and regulatory documents for enterprise legal
document review.

## Model Details

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Base model     | mistralai/Mistral-7B-Instruct-v0.3        |
| Parameters     | 7.24B (base) + 42M LoRA (adapter only)   |
| Quantisation   | NF4 4-bit (training), fp16 (merged)       |
| LoRA rank      | 16 · alpha 32 · dropout 0.05             |
| Context window | 4096 tokens                               |
| Training data  | 48,000 instruction pairs                  |
| Training time  | ~6 h on 1× A100-80GB                     |
| License        | Apache 2.0                                |

## Intended Use

This model is designed for **enterprise legal teams** who need:

- Structured contract summaries (JSON output)
- Clause extraction and classification
- Legal risk identification and scoring
- GDPR / data-processing clause compliance checks

## Performance

| Task                | ROUGE-L | JSON validity | BERTScore F1 |
|---------------------|---------|---------------|--------------|
| Contract summary    | 0.48    | 97.2 %        | 0.87         |
| Clause extraction   | 0.52    | 95.8 %        | 0.89         |
| Risk flagging       | 0.44    | 96.1 %        | 0.85         |
| GDPR compliance     | 0.51    | 98.0 %        | 0.88         |

Benchmarked on 500 held-out contracts not seen during training.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, json

model_id = "your-username/mistral-7b-legal-review"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",
)

SYSTEM = (
    "You are a senior legal analyst AI. Return structured JSON only."
)

contract = """
SERVICE AGREEMENT between Acme Corp ("Client") and Beta Ltd ("Provider").
Payment: £240,000 in four quarterly instalments.
Liability: capped at 12-month fees. Governing law: England and Wales.
"""

prompt = (
    f"<s>[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n\n"
    f"Summarise this contract as JSON with keys: parties, key_obligations, "
    f"governing_law.\n\nContract:\n{contract} [/INST]"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)

response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(json.loads(response))
```

## Prompt Format

This model uses the Mistral instruction format:

```
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_instruction}

Contract text:
{contract} [/INST] {assistant_response} </s>
```

Always set `temperature ≤ 0.15` for deterministic structured output.

## Limitations

- Not a substitute for qualified legal advice.
- Trained primarily on English-language contracts under English/US law.
- May hallucinate clause references — always cross-check against the source document.
- Performance degrades on documents > 3,500 tokens (truncation occurs).

## Training Procedure

Fine-tuned using the QLoRA technique:

1. Base model quantised to NF4 4-bit
2. LoRA adapters (r=16, α=32) attached to all attention and MLP projection layers
3. Response-only loss masking — loss computed only on assistant turns
4. 3 epochs over 48,000 instruction pairs with cosine LR schedule
5. Adapters merged back into fp16 weights for deployment

Training script and configs available at:
[github.com/your-org/slm-legal](https://github.com/your-org/slm-legal)

## Citation

```bibtex
@misc{slm-legal-review-2025,
  title  = {Mistral-7B Fine-tuned for Legal Document Review},
  author = {Your Name},
  year   = {2025},
  url    = {https://huggingface.co/your-username/mistral-7b-legal-review}
}
```
