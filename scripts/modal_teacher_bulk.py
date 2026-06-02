"""
W3 Step 3 — Bulk SFT generation via XMAiNframe-10.5b teacher on Modal A100.

Architettura:
  - Immagine ufficiale vllm/vllm-openai (dipendenze pre-testate, no conflitti)
  - vllm.LLM Python API diretta (no subprocess, no health-check polling)
  - Generazione in batch: tutti i prompt in una sola chiamata llm.chat()
  - Push checkpoint su HF Hub ogni PUSH_EVERY esempi
  - Ripresa automatica da HF Hub se il job viene interrotto

Costo stimato: ~500 esempi pilot ≈ $1.50 (~15 min); full 35k ≈ $10-12 (~5h).

Usage:
  python -m modal run scripts/modal_teacher_bulk.py               # full 35k
  python -m modal run scripts/modal_teacher_bulk.py --target 500  # pilot
"""

from __future__ import annotations

import logging
import os
import random

import modal

# ── Immagine: vllm ufficiale + nostri extra ───────────────────────────────────
image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.9.0")
    .pip_install("datasets", "huggingface-hub", "pandas", "pyarrow")
)

app = modal.App("qwen-cobol-teacher-bulk", image=image)

model_volume = modal.Volume.from_name("xmainframe-weights", create_if_missing=True)

TEACHER_MODEL     = "Fsoft-AIC/XMAiNframe-instruct-10.5b"
CORPUS_REPO       = "AlexThunder0/cobol-cpt-corpus"
SFT_REPO          = "AlexThunder0/cobol-sft-dataset"
MODEL_CACHE       = "/model-cache"
PUSH_EVERY        = 500
DEFAULT_TARGET    = 35_000
MIN_DIFFICULTY    = 0.15
MAX_SNIPPET_CHARS = 2_000

TASK_TEMPLATES: dict[str, str] = {
    "explain": (
        "Explain what the following COBOL program does, describing its inputs, "
        "outputs, and main logic:\n\n```cobol\n{source}\n```"
    ),
    "refactor": (
        "Refactor the following COBOL program to improve readability and structure "
        "while preserving its exact semantics:\n\n```cobol\n{source}\n```"
    ),
    "translate_to_java": (
        "Translate the following COBOL program to equivalent Java code:\n\n"
        "```cobol\n{source}\n```"
    ),
    "debug": (
        "The following COBOL program contains a bug. Identify the bug and provide "
        "a corrected version:\n\n```cobol\n{source}\n```"
    ),
}
BULK_TASKS = list(TASK_TEMPLATES.keys())


@app.function(
    gpu="A100-80GB",
    volumes={MODEL_CACHE: model_volume},
    timeout=4 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_teacher_bulk(target: int = DEFAULT_TARGET) -> dict:
    from datasets import Dataset, load_dataset
    from huggingface_hub import hf_hub_download
    from vllm import LLM, SamplingParams
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    token = os.environ["HF_TOKEN"]

    # ── 1. Riprendi da HF Hub ─────────────────────────────────────────────────
    accumulated: list[dict] = []
    done_keys: set[int] = set()
    try:
        existing = load_dataset(SFT_REPO, split="teacher_bulk", token=token)
        accumulated = [dict(r) for r in existing]
        done_keys = {
            hash(r["messages"][0]["content"])
            for r in accumulated if r.get("messages")
        }
        logger.info("Ripreso: %d esempi già presenti", len(accumulated))
    except Exception:
        logger.info("Nessun teacher_bulk su HF — parto da zero")

    if len(accumulated) >= target:
        logger.info("Target già raggiunto (%d) — esco", len(accumulated))
        return {"generated": len(accumulated), "status": "already_done"}

    # ── 2. Carica corpus ──────────────────────────────────────────────────────
    p = hf_hub_download(
        repo_id=CORPUS_REPO,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset", token=token,
    )
    df = pd.read_parquet(p)
    df = df[df["difficulty_score"] >= MIN_DIFFICULTY].copy()
    df["content"] = df["content"].str[:MAX_SNIPPET_CHARS]
    records = df.sample(frac=1, random_state=42).to_dict("records")
    logger.info("Corpus: %d snippet disponibili", len(records))

    # Build lista prompt (snippet × task, fino al target rimanente)
    needed = target - len(accumulated)
    task_cycle = BULK_TASKS * (needed // len(BULK_TASKS) + 2)
    random.seed(0)
    random.shuffle(task_cycle)

    prompts: list[dict] = []
    for record, task in zip(records * 10, task_cycle):
        if len(prompts) >= needed * 2:
            break
        user_content = TASK_TEMPLATES[task].format(source=record["content"])
        if hash(user_content) not in done_keys:
            prompts.append({
                "user_content": user_content,
                "task": task,
                "difficulty_score": float(record.get("difficulty_score", 0.3)),
            })
    logger.info("Prompt da generare: %d (needed: %d)", len(prompts), needed)

    # ── 3. Carica modello via Python API ──────────────────────────────────────
    logger.info("Caricamento %s …", TEACHER_MODEL)
    llm = LLM(
        model=TEACHER_MODEL,
        download_dir=MODEL_CACHE,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
    logger.info("Modello caricato.")

    # ── 4. Genera in batch con checkpoint push ────────────────────────────────
    def push(examples: list[dict]) -> None:
        ds = Dataset.from_list(examples)
        ds.push_to_hub(SFT_REPO, split="teacher_bulk", private=True, token=token)

    BATCH = 256  # processa 256 prompt alla volta (VRAM-friendly)
    new_since_push = 0

    for batch_start in range(0, len(prompts), BATCH):
        if len(accumulated) >= target:
            break

        batch = prompts[batch_start : batch_start + BATCH]
        conversations = [
            [{"role": "user", "content": p["user_content"]}]
            for p in batch
        ]

        try:
            outputs = llm.chat(conversations, sampling_params, use_tqdm=False)
        except Exception as e:
            logger.warning("Batch %d fallito: %s", batch_start, e)
            continue

        for p, out in zip(batch, outputs):
            answer = out.outputs[0].text.strip()
            if not answer:
                continue
            accumulated.append({
                "messages": [
                    {"role": "user",      "content": p["user_content"]},
                    {"role": "assistant", "content": answer},
                ],
                "source":           f"teacher_xmainframe_{p['task']}",
                "difficulty_score": p["difficulty_score"],
            })
            new_since_push += 1

        if new_since_push >= PUSH_EVERY:
            push(accumulated)
            logger.info("Checkpoint: %d/%d su HF Hub", len(accumulated), target)
            new_since_push = 0

        logger.info(
            "Batch %d/%d — totale: %d/%d",
            batch_start // BATCH + 1,
            (len(prompts) + BATCH - 1) // BATCH,
            len(accumulated), target,
        )

    push(accumulated)
    logger.info("Teacher bulk completato: %d esempi", len(accumulated))
    return {"generated": len(accumulated), "status": "done"}


@app.local_entrypoint()
def main(target: int = DEFAULT_TARGET):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Avvio teacher bulk (target={target:,}) su Modal A100-80GB …")
    result = run_teacher_bulk.remote(target=target)
    print(f"Completato: {result}")
