"""
app.py — HuggingFace Space demo for the legal-review SLM.

Deploy this file in a HuggingFace Space (SDK: Gradio, Hardware: A10G or CPU upgrade).
The model is loaded once via the InferenceClient against the model repo.

Layout:
  - Left  : input panel (document text + task selector + parameters)
  - Right : output panel (structured result + latency badge)
  - Bottom: example showcase with real legal clause samples
"""

import os
import json
import time
import gradio as gr
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# Configuration (set via Space secrets)
# ─────────────────────────────────────────────

MODEL_ID = os.getenv("MODEL_ID", "your-username/mistral-7b-legal-review")
HF_TOKEN  = os.getenv("HF_TOKEN", None)   # Set as Space secret

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

SYSTEM_PROMPT = (
    "You are a senior legal analyst AI. Review contracts, extract key clauses, "
    "flag risks, and produce structured JSON summaries. Be precise and concise."
)

TASK_PROMPTS = {
    "📄  Contract Summary": (
        "Summarise the contract. Return JSON with keys: "
        "parties, effective_date, key_obligations, termination_conditions, governing_law."
    ),
    "🔍  Clause Extraction": (
        "Extract all clauses. For each return: clause_number, clause_type "
        "(payment/liability/ip_ownership/confidentiality/termination/other), summary."
    ),
    "⚠️  Risk Flagging": (
        "Identify legal risks. For each return: risk_level (HIGH/MEDIUM/LOW), "
        "risk_type, affected_party, clause_reference, recommended_action."
    ),
    "🔒  GDPR Compliance": (
        "Check GDPR compliance. Return: compliance_score (0-100), gaps, recommended_clauses."
    ),
}

EXAMPLES = [
    [
        "📄  Contract Summary",
        """SERVICE AGREEMENT dated 1 January 2025 between Acme Corp ("Client") and Beta Ltd ("Provider").
1. Services: Provider shall deliver a custom CRM platform by 30 June 2025.
2. Payment: Client shall pay £240,000 in four quarterly instalments.
3. Liability: Provider's total liability shall not exceed the fees paid in the preceding 12 months.
4. Termination: Either party may terminate with 60 days written notice without cause.
5. Governing Law: This agreement is governed by the laws of England and Wales.""",
        0.1,
        512,
    ],
    [
        "⚠️  Risk Flagging",
        """11.3 Indemnification: Provider shall indemnify, defend, and hold harmless Client and its affiliates,
officers, directors, employees, agents, successors and assigns from and against any and all claims,
damages, liabilities, costs and expenses (including reasonable legal fees) arising out of or related
to Provider's performance or failure to perform under this Agreement, with no limitation on the
aggregate amount of such indemnification.""",
        0.1,
        512,
    ],
    [
        "🔒  GDPR Compliance",
        """Data Processing Clause: Provider may process personal data of Client's customers as necessary
to deliver the Services. Provider agrees to implement appropriate technical measures. Data will be
retained for the duration of the contract. Provider may engage sub-processors at its discretion.
Data may be transferred to Provider's offices in the United States.""",
        0.1,
        512,
    ],
]


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def analyze_document(task_label: str, text: str, temperature: float, max_tokens: int):
    if not text.strip():
        return "⚠️  Please paste a contract or clause into the text box.", ""

    task_instruction = TASK_PROMPTS[task_label]
    prompt = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{task_instruction}\n\nText:\n{text.strip()} [/INST]"
    )

    t0 = time.perf_counter()
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            stop_sequences=["</s>", "[INST]"],
            do_sample=temperature > 0,
        )
        latency = (time.perf_counter() - t0) * 1000

        # Attempt to pretty-print if JSON
        output = response.strip()
        try:
            parsed = json.loads(output)
            output = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass  # Return raw text if not JSON

        stats = f"⏱ {latency:.0f} ms  |  Model: {MODEL_ID.split('/')[-1]}"
        return output, stats

    except Exception as e:
        return f"❌  Inference error: {str(e)}", ""


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

css = """
#title { text-align: center; font-size: 1.6rem; font-weight: 600; margin-bottom: 0.2rem; }
#subtitle { text-align: center; color: #666; margin-bottom: 1.5rem; }
#output-box textarea { font-family: monospace; font-size: 0.85rem; }
.stats-box { font-size: 0.8rem; color: #888; margin-top: 0.3rem; }
"""

with gr.Blocks(css=css, title="Legal Review SLM") as demo:

    gr.HTML("""
        <div id="title">⚖️  Legal Review SLM</div>
        <div id="subtitle">
          Mistral-7B fine-tuned for enterprise legal document analysis ·
          Clause extraction · Risk flagging · GDPR compliance
        </div>
    """)

    with gr.Row():
        # ── Input column ──
        with gr.Column(scale=1):
            task = gr.Dropdown(
                label="Analysis task",
                choices=list(TASK_PROMPTS.keys()),
                value="📄  Contract Summary",
            )
            text_input = gr.Textbox(
                label="Contract / clause text",
                placeholder="Paste your contract text here…",
                lines=18,
                max_lines=40,
            )
            with gr.Accordion("⚙️  Parameters", open=False):
                temperature = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature")
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")

            analyze_btn = gr.Button("🔍  Analyse document", variant="primary")

        # ── Output column ──
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Structured result (JSON)",
                lines=22,
                elem_id="output-box",
                interactive=False,
            )
            stats = gr.Textbox(
                label="", lines=1, interactive=False, elem_classes=["stats-box"]
            )

    # Examples
    gr.Examples(
        examples=EXAMPLES,
        inputs=[task, text_input, temperature, max_tokens],
        outputs=[output, stats],
        fn=analyze_document,
        cache_examples=True,
        label="📚  Example contracts (click to load)",
    )

    # Wire up
    analyze_btn.click(
        fn=analyze_document,
        inputs=[task, text_input, temperature, max_tokens],
        outputs=[output, stats],
    )

    gr.HTML("""
        <div style="text-align:center;font-size:0.75rem;color:#aaa;margin-top:1rem">
          Built with ❤️ using Mistral-7B + QLoRA + HuggingFace Transformers ·
          For enterprise use, deploy on your private Kubernetes cluster.
        </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
