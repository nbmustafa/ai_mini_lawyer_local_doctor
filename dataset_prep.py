"""
dataset_prep.py — Build instruction-following pairs for legal document review.

Supported tasks:
  1. contract_summary      — summarise a contract section into structured JSON
  2. clause_extraction     — extract and label clauses (payment, liability, IP, etc.)
  3. risk_flagging         — identify high-risk or non-standard language
  4. gdpr_compliance       — check data-processing clauses against GDPR obligations

Input  : raw .txt or .pdf legal documents in data/raw/
Output : data/legal_instruct.jsonl  ({"instruction": ..., "output": ...})

Usage:
  python dataset_prep.py --input_dir data/raw --output data/legal_instruct.jsonl
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Iterator


# ─────────────────────────────────────────────
# Task templates
# ─────────────────────────────────────────────

TASK_TEMPLATES = {
    "contract_summary": {
        "instruction_tmpl": (
            "Summarise the following contract section. "
            "Return a JSON object with keys: "
            "parties, effective_date, key_obligations, termination_conditions, governing_law.\n\n"
            "Contract text:\n{text}"
        ),
        "description": "Structured contract summary",
    },
    "clause_extraction": {
        "instruction_tmpl": (
            "Extract and categorise all clauses from the contract below. "
            "For each clause return: clause_number, clause_type "
            "(one of: payment, liability, ip_ownership, confidentiality, "
            "termination, dispute_resolution, force_majeure, other), "
            "and a one-sentence summary.\n\n"
            "Contract:\n{text}"
        ),
        "description": "Named clause extraction",
    },
    "risk_flagging": {
        "instruction_tmpl": (
            "Review the following contract clause and identify any legal risks "
            "or non-standard language. For each risk provide: "
            "risk_level (HIGH/MEDIUM/LOW), risk_type, affected_party, "
            "clause_reference, and recommended_action.\n\n"
            "Clause:\n{text}"
        ),
        "description": "Risk identification and scoring",
    },
    "gdpr_compliance": {
        "instruction_tmpl": (
            "Analyse the following data-processing clause for GDPR compliance. "
            "Check for: lawful basis, data subject rights, retention period, "
            "sub-processor obligations, and cross-border transfer safeguards. "
            "Return a compliance_score (0–100) and a list of gaps.\n\n"
            "Clause:\n{text}"
        ),
        "description": "GDPR compliance check",
    },
}


# ─────────────────────────────────────────────
# Chunk raw text into ~512-token windows
# (approximate via word count, tokeniser not needed at prep time)
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_words: int = 380, overlap_words: int = 40) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_words - overlap_words
    return [c for c in chunks if len(c.split()) > 60]  # Drop tiny tail chunks


# ─────────────────────────────────────────────
# Read raw documents
# ─────────────────────────────────────────────

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except ImportError:
        raise RuntimeError("Install pdfplumber: pip install pdfplumber")


def iter_documents(input_dir: str) -> Iterator[tuple[str, str]]:
    """Yield (filename, raw_text) for every .txt and .pdf in input_dir."""
    root = Path(input_dir)
    for p in sorted(root.rglob("*")):
        if p.suffix == ".txt":
            yield p.name, read_txt(p)
        elif p.suffix == ".pdf":
            yield p.name, read_pdf(p)


# ─────────────────────────────────────────────
# Synthetic output generator
# (Replace with real annotated outputs in production)
# ─────────────────────────────────────────────

SYNTHETIC_OUTPUTS = {
    "contract_summary": json.dumps({
        "parties": ["Acme Corp", "Beta Ltd"],
        "effective_date": "2024-01-01",
        "key_obligations": [
            "Acme delivers software by 2024-06-30",
            "Beta pays $120,000 in four instalments",
        ],
        "termination_conditions": "Either party may terminate with 30 days written notice",
        "governing_law": "England and Wales",
    }, indent=2),
    "clause_extraction": json.dumps([
        {"clause_number": "3.1", "clause_type": "payment",
         "summary": "Payment due within 30 days of invoice"},
        {"clause_number": "8.2", "clause_type": "liability",
         "summary": "Liability capped at total contract value"},
        {"clause_number": "12.4", "clause_type": "ip_ownership",
         "summary": "All deliverables vest in client upon full payment"},
    ], indent=2),
    "risk_flagging": json.dumps([
        {"risk_level": "HIGH",
         "risk_type": "Uncapped indemnification",
         "affected_party": "Service Provider",
         "clause_reference": "11.3",
         "recommended_action": "Negotiate mutual cap equal to 12-month fees"},
    ], indent=2),
    "gdpr_compliance": json.dumps({
        "compliance_score": 62,
        "gaps": [
            "No explicit lawful basis stated (Art. 6)",
            "Retention period not defined (Art. 5(1)(e))",
            "Sub-processor list not provided (Art. 28(3))",
        ],
        "recommended_clauses": [
            "Add Art. 6(1)(b) as lawful basis",
            "Define retention schedule in Schedule 2",
        ],
    }, indent=2),
}


# ─────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────

def build_dataset(input_dir: str, output_path: str, tasks: list[str] = None):
    if tasks is None:
        tasks = list(TASK_TEMPLATES.keys())

    seen_hashes = set()
    records = []

    for filename, raw_text in iter_documents(input_dir):
        chunks = chunk_text(raw_text)
        print(f"  {filename}: {len(chunks)} chunks")

        for chunk in chunks:
            # Deduplicate by content hash
            h = hashlib.md5(chunk.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            for task in tasks:
                tmpl = TASK_TEMPLATES[task]
                instruction = tmpl["instruction_tmpl"].format(text=chunk)
                # In production: replace with real human/GPT-4 annotated output
                output = SYNTHETIC_OUTPUTS[task]

                records.append({
                    "instruction": instruction,
                    "output": output,
                    "task": task,
                    "source_file": filename,
                })

    # Write JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✅  {len(records):,} instruction pairs written → {output_path}")
    print(f"   Tasks: {', '.join(tasks)}")
    print(f"   Unique chunks: {len(seen_hashes):,}")


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1/2/L for a batch of predictions vs references."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k] += scores[k].fmeasure
    n = len(predictions)
    return {k: round(v / n, 4) for k, v in agg.items()}


def compute_json_validity(predictions: list[str]) -> float:
    """Fraction of predictions that parse as valid JSON — key for structured tasks."""
    valid = 0
    for p in predictions:
        try:
            json.loads(p.strip())
            valid += 1
        except json.JSONDecodeError:
            pass
    return round(valid / len(predictions), 4) if predictions else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/raw", help="Directory of raw .txt/.pdf files")
    ap.add_argument("--output", default="data/legal_instruct.jsonl")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Subset of tasks to generate (default: all)")
    args = ap.parse_args()

    print(f"Building dataset from: {args.input_dir}")
    build_dataset(args.input_dir, args.output, args.tasks)
