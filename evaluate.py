"""
evaluate.py — Evaluate a fine-tuned legal-review model or LoRA adapter.

The evaluator is designed to match this repository's training prompt format and
supports two loading modes:
  1. A PEFT/LoRA adapter directory (contains adapter_config.json)
  2. A merged/full model directory

It computes:
  - ROUGE-1 / ROUGE-2 / ROUGE-L
  - JSON validity
  - BERTScore F1 (when `bert-score` is installed)

Usage:
  python evaluate.py \
    --adapter_dir outputs/mistral7b-legal-lora \
    --test_data data/test_set.jsonl \
    --tasks all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - handled at runtime
    PeftModel = None


SYSTEM_PROMPT = (
    "You are a senior legal analyst AI. You review contracts, identify key clauses, "
    "flag risks, and produce structured summaries for enterprise legal teams. "
    "Be concise, precise, and always cite the relevant clause number."
)

KNOWN_TASKS = {
    "contract_summary",
    "clause_extraction",
    "risk_flagging",
    "gdpr_compliance",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True, help="LoRA adapter directory or merged model directory")
    ap.add_argument(
        "--base_model",
        default=None,
        help="Base model for LoRA evaluation. If omitted, inferred from adapter_config.json when possible.",
    )
    ap.add_argument("--test_data", required=True, help="JSONL test set path")
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Tasks to evaluate: all or one/more of contract_summary clause_extraction risk_flagging gdpr_compliance",
    )
    ap.add_argument("--max_samples", type=int, default=None, help="Optional cap on evaluated examples")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_input_length", type=int, default=3072)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_path", default=None, help="Optional JSON file for per-example outputs and metrics")
    return ap.parse_args()


def normalize_task_filter(tasks: list[str]) -> set[str] | None:
    lowered = {task.strip().lower() for task in tasks}
    if "all" in lowered:
        return None
    unknown = lowered - KNOWN_TASKS
    if unknown:
        raise ValueError(f"Unknown task(s): {sorted(unknown)}")
    return lowered


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} in {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {lineno} in {path}")
            rows.append(row)
    return rows


def extract_instruction(row: dict[str, Any]) -> str:
    for key in ("instruction", "prompt", "input", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Row is missing an instruction-like field (instruction/prompt/input/question)")


def extract_reference(row: dict[str, Any]) -> str:
    for key in ("output", "reference", "answer", "target", "response"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
    raise ValueError("Row is missing a reference field (output/reference/answer/target/response)")


def format_prompt(instruction: str) -> str:
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{instruction} [/INST]"
    )


def infer_base_model(adapter_dir: Path, base_model_flag: str | None) -> str | None:
    if base_model_flag:
        return base_model_flag

    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("base_model_name_or_path")

    return None


def build_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(adapter_dir: str, base_model: str | None):
    model_path = Path(adapter_dir)
    is_adapter = (model_path / "adapter_config.json").exists()
    torch_dtype = build_torch_dtype()
    device_map = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if is_adapter:
        if PeftModel is None:
            raise RuntimeError("peft is required to evaluate a LoRA adapter. Install `peft` first.")
        if not base_model:
            raise ValueError(
                "Could not infer the base model for this adapter. Pass --base_model explicitly."
            )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            adapter_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )

    model.eval()
    return model, tokenizer


def batched(items: list[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def generate_predictions(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    predictions: list[str] = []
    use_sampling = temperature > 0

    for batch_prompts in batched(prompts, batch_size):
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        # When a model is loaded with an accelerate device map, keeping inputs on
        # CPU lets transformers dispatch them correctly across shards.
        if torch.cuda.is_available() and not hasattr(model, "hf_device_map"):
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

        generation_kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": use_sampling,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if use_sampling:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)

        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        for row_idx, input_len in enumerate(input_lengths):
            new_tokens = generated[row_idx][int(input_len):]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append(text)

    return predictions


def _tokenize_for_rouge(text: str) -> list[str]:
    return text.lower().split()


def _ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _f1_from_overlap(pred_total: int, ref_total: int, overlap: int) -> float:
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return (2 * precision * recall) / (precision + recall)


def rouge_n_f1(pred: str, ref: str, n: int) -> float:
    pred_tokens = _tokenize_for_rouge(pred)
    ref_tokens = _tokenize_for_rouge(ref)
    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)

    overlap = 0
    for gram, pred_count in pred_counts.items():
        overlap += min(pred_count, ref_counts.get(gram, 0))

    return _f1_from_overlap(sum(pred_counts.values()), sum(ref_counts.values()), overlap)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for tok_a in a:
        prev = 0
        for j, tok_b in enumerate(b, start=1):
            cur = dp[j]
            if tok_a == tok_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_tokens = _tokenize_for_rouge(pred)
    ref_tokens = _tokenize_for_rouge(ref)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    return _f1_from_overlap(len(pred_tokens), len(ref_tokens), lcs)


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge1 = sum(rouge_n_f1(pred, ref, 1) for pred, ref in zip(predictions, references)) / len(predictions)
    rouge2 = sum(rouge_n_f1(pred, ref, 2) for pred, ref in zip(predictions, references)) / len(predictions)
    rouge_l = sum(rouge_l_f1(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions)

    return {
        "rouge1": round(rouge1, 4),
        "rouge2": round(rouge2, 4),
        "rougeL": round(rouge_l, 4),
    }


def compute_json_validity(predictions: list[str]) -> float:
    if not predictions:
        return 0.0
    valid = 0
    for pred in predictions:
        try:
            json.loads(pred)
            valid += 1
        except json.JSONDecodeError:
            continue
    return round(valid / len(predictions), 4)


def compute_bertscore(predictions: list[str], references: list[str]) -> float | None:
    try:
        from bert_score import score as bertscore
    except ImportError:
        return None

    _, _, f1 = bertscore(
        predictions,
        references,
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
    )
    return round(float(f1.mean().item()), 4)


def prepare_examples(rows: list[dict[str, Any]], task_filter: set[str] | None, max_samples: int | None):
    examples: list[dict[str, Any]] = []
    for row in rows:
        row_task = str(row.get("task", "")).strip().lower() or None
        if task_filter is not None and row_task not in task_filter:
            continue

        examples.append(
            {
                "task": row_task,
                "instruction": extract_instruction(row),
                "reference": extract_reference(row),
            }
        )

        if max_samples is not None and len(examples) >= max_samples:
            break

    if not examples:
        raise ValueError("No evaluation examples matched the requested task filter.")

    return examples


def evaluate_subset(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = [format_prompt(example["instruction"]) for example in examples]
    references = [example["reference"] for example in examples]

    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    rouge = compute_rouge(predictions, references)
    json_validity = compute_json_validity(predictions)
    bertscore_f1 = compute_bertscore(predictions, references)

    metrics: dict[str, Any] = {
        "num_examples": len(examples),
        **rouge,
        "json_validity": json_validity,
        "bertscore_f1": bertscore_f1,
    }

    rows: list[dict[str, Any]] = []
    for example, prediction in zip(examples, predictions):
        rows.append(
            {
                "task": example["task"],
                "instruction": example["instruction"],
                "reference": example["reference"],
                "prediction": prediction,
            }
        )

    return metrics, rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {adapter_dir}")

    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    task_filter = normalize_task_filter(args.tasks)
    base_model = infer_base_model(adapter_dir, args.base_model)

    print(f"Loading evaluation data from: {test_data_path}")
    rows = read_jsonl(str(test_data_path))
    examples = prepare_examples(rows, task_filter, args.max_samples)

    print(f"Loading model from: {adapter_dir}")
    if base_model:
        print(f"Base model: {base_model}")
    model, tokenizer = load_model_and_tokenizer(str(adapter_dir), base_model)

    grouped: dict[str, list[dict[str, Any]]] = {"all": examples}
    tasks_present = sorted({example["task"] for example in examples if example["task"]})
    for task in tasks_present:
        grouped[task] = [example for example in examples if example["task"] == task]

    report: dict[str, Any] = {
        "model_path": str(adapter_dir),
        "base_model": base_model,
        "test_data": str(test_data_path),
        "results": {},
    }

    all_outputs: dict[str, list[dict[str, Any]]] = {}
    for group_name, group_examples in grouped.items():
        print(f"\nEvaluating: {group_name} ({len(group_examples)} examples)")
        metrics, output_rows = evaluate_subset(model, tokenizer, group_examples, args)
        report["results"][group_name] = metrics
        all_outputs[group_name] = output_rows

        print(f"  ROUGE-1:       {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2:       {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L:       {metrics['rougeL']:.4f}")
        print(f"  JSON validity: {metrics['json_validity']:.4f}")
        if metrics["bertscore_f1"] is None:
            print("  BERTScore F1:  unavailable (install `bert-score` to enable)")
        else:
            print(f"  BERTScore F1:  {metrics['bertscore_f1']:.4f}")

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**report, "predictions": all_outputs}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
