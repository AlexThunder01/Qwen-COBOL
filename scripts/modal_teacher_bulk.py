"""
W3 Step 3 — Bulk SFT generation via XMAiNframe-10.5b teacher on Modal A100.

Usa transformers direttamente (no vLLM) → zero dependency conflicts.
XMAiNframe-10.5b in bf16 = ~21GB → entra comodamente su A100-80GB.

Costo stimato: ~500 pilot ≈ $1.50 (~20 min); full 35k ≈ $12-15 (~6h).

Usage:
  python -m modal run scripts/modal_teacher_bulk.py               # full 35k
  python -m modal run scripts/modal_teacher_bulk.py --target 500  # pilot
"""

from __future__ import annotations

import logging
import os
import random

import modal

image = (
    modal.Image.from_registry("python:3.11-slim-trixie")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.51.1",
        "accelerate>=1.0.0",
        "datasets",
        "huggingface-hub",
        "pandas",
        "pyarrow",
        "sentencepiece",
        "protobuf",
    )
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
BATCH_SIZE        = 8

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


def _clean_output(text: str) -> str:
    """Rimuove artefatti BPE byte-level (stile GPT-2) e header del chat template."""
    import re
    text = re.sub(r'^\s*<\|assistant\|>\s*', '', text)  # header echoed
    text = text.replace('Ġ', ' ')   # Ġ → spazio
    text = text.replace('Ċ', '\n')  # Ċ → newline
    text = text.replace('ĉ', '\t')  # ĉ → tab
    return text.strip()


@app.function(
    gpu="A100-80GB",
    volumes={MODEL_CACHE: model_volume},
    timeout=6 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_teacher_bulk(target: int = DEFAULT_TARGET) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset, load_dataset
    from huggingface_hub import hf_hub_download
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
    logger.info("Corpus: %d snippet", len(records))

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
    logger.info("Prompt da generare: %d", len(prompts))

    # ── 3. Carica modello ─────────────────────────────────────────────────────
    logger.info("Caricamento %s …", TEACHER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL,
        cache_dir=MODEL_CACHE,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        cache_dir=MODEL_CACHE,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Modello caricato su %s", next(model.parameters()).device)

    # ── 4. Genera in batch con checkpoint push ────────────────────────────────
    def push(examples: list[dict]) -> None:
        Dataset.from_list(examples).push_to_hub(
            SFT_REPO, split="teacher_bulk", private=True, token=token
        )

    new_since_push = 0

    for batch_start in range(0, len(prompts), BATCH_SIZE):
        if len(accumulated) >= target:
            break

        batch = prompts[batch_start : batch_start + BATCH_SIZE]

        # Applica chat template
        try:
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p["user_content"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]
        except Exception:
            # Fallback: nessun chat template → prompt diretto
            texts = [p["user_content"] for p in batch]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Decodifica solo i token nuovi (dopo il prompt)
            new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            answers = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        except Exception as e:
            logger.warning("Batch %d fallito: %s", batch_start, e)
            continue

        for p, answer in zip(batch, answers):
            answer = _clean_output(answer)
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

        if (batch_start // BATCH_SIZE) % 10 == 0:
            logger.info(
                "Batch %d — totale: %d/%d",
                batch_start // BATCH_SIZE,
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
