"""
merge_and_export.py — Merge LoRA adapter into base weights, then export to
GGUF (llama.cpp) and GPTQ (for fast vLLM serving).

Steps:
  1. Load base model in fp16
  2. Load LoRA adapter and merge with merge_and_unload()
  3. Push merged fp16 model to HuggingFace Hub
  4. (Optional) Convert to GGUF via llama.cpp convert script
  5. (Optional) Quantise to GPTQ via auto-gptq

Usage:
  python merge_and_export.py \
      --adapter_dir outputs/mistral7b-legal-lora \
      --base_model  mistralai/Mistral-7B-Instruct-v0.3 \
      --hub_repo    your-username/mistral-7b-legal-review \
      --export_gguf true
"""

import os
import argparse
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_push(
    adapter_dir: str,
    base_model: str,
    hub_repo: str,
    push_to_hub: bool = True,
):
    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    print(f"Applying adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("Merging LoRA weights into base…")
    model = model.merge_and_unload()
    model.eval()

    merged_dir = Path(adapter_dir) / "merged"
    merged_dir.mkdir(exist_ok=True)
    print(f"Saving merged model → {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub → {hub_repo}")
        model.push_to_hub(hub_repo, safe_serialization=True, private=False)
        tokenizer.push_to_hub(hub_repo)
        print("✅  Model live on Hub.")

    return str(merged_dir)


def export_gguf(merged_dir: str, llama_cpp_dir: str, quant_type: str = "Q4_K_M"):
    """
    Convert merged model to GGUF using llama.cpp's convert_hf_to_gguf.py.
    Requires llama.cpp cloned at llama_cpp_dir.
    """
    import subprocess

    gguf_out = Path(merged_dir) / f"model-{quant_type}.gguf"
    fp16_gguf = Path(merged_dir) / "model-f16.gguf"

    # Step 1: convert to fp16 GGUF
    subprocess.run([
        "python", f"{llama_cpp_dir}/convert_hf_to_gguf.py",
        merged_dir,
        "--outfile", str(fp16_gguf),
        "--outtype", "f16",
    ], check=True)

    # Step 2: quantise
    subprocess.run([
        f"{llama_cpp_dir}/build/bin/llama-quantize",
        str(fp16_gguf),
        str(gguf_out),
        quant_type,
    ], check=True)

    print(f"✅  GGUF saved: {gguf_out}")
    return str(gguf_out)


def export_gptq(merged_dir: str, bits: int = 4):
    """
    Quantise merged model to GPTQ using auto-gptq.
    Suitable for serving with vLLM (--quantization gptq).
    """
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(merged_dir)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=False,
    )

    # Small calibration set (128 samples from C4 works well)
    calib_data = load_dataset("c4", "en", split="train", streaming=True)
    calib_samples = [
        tokenizer(x["text"], return_tensors="pt", truncation=True, max_length=512)
        for x in list(calib_data.take(128))
    ]

    model = AutoGPTQForCausalLM.from_pretrained(merged_dir, quantize_config)
    model.quantize(calib_samples)

    gptq_dir = Path(merged_dir) / f"gptq-{bits}bit"
    model.save_quantized(gptq_dir, use_safetensors=True)
    tokenizer.save_pretrained(gptq_dir)
    print(f"✅  GPTQ model saved: {gptq_dir}")
    return str(gptq_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir",  required=True)
    ap.add_argument("--base_model",   default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--hub_repo",     required=True, help="your-username/model-name")
    ap.add_argument("--push_to_hub",  action="store_true", default=True)
    ap.add_argument("--export_gguf",  action="store_true")
    ap.add_argument("--llama_cpp_dir",default="./llama.cpp")
    ap.add_argument("--export_gptq",  action="store_true")
    ap.add_argument("--gptq_bits",    type=int, default=4)
    args = ap.parse_args()

    merged = merge_and_push(
        args.adapter_dir, args.base_model, args.hub_repo, args.push_to_hub
    )

    if args.export_gguf:
        export_gguf(merged, args.llama_cpp_dir)

    if args.export_gptq:
        export_gptq(merged, args.gptq_bits)
