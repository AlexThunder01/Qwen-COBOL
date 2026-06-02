"""
W3 Step 3 — Bulk SFT generation via XMAiNframe-10.5b teacher on Modal A100.

Architettura:
  - 1 container A100-80GB serve XMAiNframe-10.5b via vLLM (OpenAI-compatible API)
  - generate_bulk() chiama il server in parallelo (asyncio + httpx)
  - Push checkpoint su HF Hub ogni 500 esempi
  - Ripresa automatica: al restart scarica da HF quanto già fatto e salta

Costo stimato: ~35k esempi × ~0.5s/esempio = ~5h su A100 → ~$10-12 su Modal.

Usage:
  modal run scripts/modal_teacher_bulk.py
  modal run scripts/modal_teacher_bulk.py --target 5000   # pilot run ~$1.50
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path

import modal

# ── Immagine Docker ───────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry("python:3.11-slim-trixie")
    .pip_install(
        "vllm==0.9.0",
        "datasets",
        "huggingface-hub",
        "openai",
        "httpx",
        "pandas",
        "pyarrow",
    )
)

app = modal.App("qwen-cobol-teacher-bulk", image=image)

# Volume per cachare i pesi del modello (evita re-download a ogni run)
model_volume = modal.Volume.from_name("xmainframe-weights", create_if_missing=True)

TEACHER_MODEL  = "Fsoft-AIC/XMAiNframe-instruct-10.5b"
CORPUS_REPO    = "AlexThunder0/cobol-cpt-corpus"
SFT_REPO       = "AlexThunder0/cobol-sft-dataset"
MODEL_CACHE    = "/model-cache"
VLLM_PORT      = 8000
PUSH_EVERY     = 500        # checkpoint ogni 500 esempi (~10 min)
DEFAULT_TARGET = 35_000
MIN_DIFFICULTY = 0.15
MAX_SNIPPET_CHARS = 2_000

TASK_TEMPLATES: dict[str, str] = {
    "explain": (
        "Explain what the following COBOL program does, describing its inputs, outputs, "
        "and main logic:\n\n```cobol\n{source}\n```"
    ),
    "refactor": (
        "Refactor the following COBOL program to improve readability and structure "
        "while preserving its exact semantics:\n\n```cobol\n{source}\n```"
    ),
    "translate_to_java": (
        "Translate the following COBOL program to equivalent Java code:\n\n```cobol\n{source}\n```"
    ),
    "debug": (
        "The following COBOL program contains a bug. Identify the bug and provide "
        "a corrected version:\n\n```cobol\n{source}\n```"
    ),
}
BULK_TASKS = list(TASK_TEMPLATES.keys())


# ── Server vLLM ───────────────────────────────────────────────────────────────

@app.function(
    gpu=modal.gpu.A100(size="80GB"),
    volumes={MODEL_CACHE: model_volume},
    timeout=3 * 3600,  # max 3h
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_teacher_bulk(target: int = DEFAULT_TARGET) -> dict:
    import subprocess, httpx, os
    from datasets import load_dataset, Dataset
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
        done_keys = {hash(r["messages"][0]["content"]) for r in accumulated if r.get("messages")}
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

    # Build prompt list (ogni snippet × più task fino al target)
    prompts: list[dict] = []
    task_cycle = BULK_TASKS * (target // len(BULK_TASKS) + 2)
    random.seed(0)
    random.shuffle(task_cycle)
    for record, task in zip(records * 10, task_cycle):
        if len(prompts) >= target * 2:  # buffer largo
            break
        user_content = TASK_TEMPLATES[task].format(source=record["content"])
        key = hash(user_content)
        if key not in done_keys:
            prompts.append({
                "user_content": user_content,
                "key": key,
                "task": task,
                "difficulty_score": record.get("difficulty_score", 0.3),
            })
    logger.info("Prompt da generare: %d (target: %d)", len(prompts), target - len(accumulated))

    # ── 3. Avvia vLLM ─────────────────────────────────────────────────────────
    logger.info("Avvio vLLM server con %s …", TEACHER_MODEL)
    vllm_proc = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", TEACHER_MODEL,
        "--download-dir", MODEL_CACHE,
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.88",
        "--disable-log-requests",
    ], env={**os.environ, "HF_TOKEN": token, "HUGGING_FACE_HUB_TOKEN": token})

    # Aspetta che vLLM sia pronto
    base_url = f"http://localhost:{VLLM_PORT}"
    for attempt in range(60):
        time.sleep(5)
        try:
            r = httpx.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200:
                logger.info("vLLM pronto dopo %ds", (attempt + 1) * 5)
                break
        except Exception:
            pass
    else:
        vllm_proc.kill()
        raise RuntimeError("vLLM non è partito in 300s")

    # ── 4. Genera in batch asincroni ──────────────────────────────────────────
    async def call_vllm(session: httpx.AsyncClient, prompt: dict) -> dict | None:
        try:
            resp = await session.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": TEACHER_MODEL,
                    "messages": [{"role": "user", "content": prompt["user_content"]}],
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
                timeout=60,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
            return {
                "messages": [
                    {"role": "user",      "content": prompt["user_content"]},
                    {"role": "assistant", "content": answer},
                ],
                "source": f"teacher_xmainframe_{prompt['task']}",
                "difficulty_score": prompt["difficulty_score"],
            }
        except Exception as e:
            logger.debug("vLLM call failed: %s", e)
            return None

    async def generate_all(prompts_to_do: list[dict]) -> None:
        nonlocal accumulated
        CONCURRENCY = 16  # richieste parallele al vLLM
        sem = asyncio.Semaphore(CONCURRENCY)
        new_since_push = 0

        async def bounded(p):
            async with sem:
                return await call_vllm(session, p)

        async with httpx.AsyncClient() as session:
            tasks = [bounded(p) for p in prompts_to_do]
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                if len(accumulated) >= target:
                    break
                result = await coro
                if result:
                    accumulated.append(result)
                    new_since_push += 1

                if new_since_push >= PUSH_EVERY:
                    _push(accumulated, token)
                    logger.info("Checkpoint: %d/%d esempi su HF Hub", len(accumulated), target)
                    new_since_push = 0

                if (i + 1) % 200 == 0:
                    logger.info("Progresso: %d/%d generati", len(accumulated), target)

        # push finale
        _push(accumulated, token)

    def _push(examples: list[dict], tok: str) -> None:
        ds = Dataset.from_list(examples)
        ds.push_to_hub(SFT_REPO, split="teacher_bulk", private=True, token=tok)

    remaining = [p for p in prompts if len(accumulated) < target]
    asyncio.run(generate_all(remaining))

    vllm_proc.kill()
    logger.info("Teacher bulk completato: %d esempi", len(accumulated))
    return {"generated": len(accumulated), "status": "done"}


# ── Entry point locale ────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(target: int = DEFAULT_TARGET):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Avvio teacher bulk (target={target:,}) su Modal A100-80GB …")
    result = run_teacher_bulk.remote(target=target)
    print(f"Completato: {result}")
