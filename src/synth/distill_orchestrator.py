"""
Orchestrate the full synthetic data generation pipeline (W3).

Routing logic:
  - Hard tasks (translation REDEFINES, GO TO modernization) → Gemini gold (4-6k)
  - Remaining bulk → teacher vLLM (30-40k)
  - DPO pairs → dpo_pairs.py

Output: HF Hub private datasets cobol-sft-dataset + cobol-dpo-dataset.

Run order:
  1. Kaggle (CPU): python -m src.synth.distill_orchestrator --step mainframebench
  2. Kaggle (CPU): python -m src.synth.distill_orchestrator --step gemini
  3. Modal A100:   python -m src.synth.distill_orchestrator --step teacher
  4. Any:          python -m src.synth.distill_orchestrator --step dpo
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
import pandas as pd

from src.synth.teacher_gemini import configure as configure_gemini, generate_gold_example, TASK_TEMPLATES as GEMINI_TASK_TEMPLATES
from src.synth.teacher_vllm import generate_batch, TASK_TEMPLATES as VLLM_TEMPLATES
from src.synth.teacher_alibaba import generate_example as alibaba_generate, TEACHER_MODELS as ALIBABA_MODELS, BULK_TASKS as ALIBABA_TASKS
from src.synth.dpo_pairs import generate_dpo_pair
from src.synth.curriculum_sampler import build_curriculum
from src.synth.import_mainframebench import run_import as import_mainframebench

logger = logging.getLogger(__name__)

CORPUS_REPO = "AlexThunder0/cobol-cpt-corpus"
SFT_REPO    = "AlexThunder0/cobol-sft-dataset"
DPO_REPO    = "AlexThunder0/cobol-dpo-dataset"

GEMINI_TASKS  = ["translation_redefines", "modernize_goto", "explain_copybook", "debug"]
BULK_TASKS    = ["explain", "refactor", "translate_to_java", "debug"]

GEMINI_TARGET = 5_000
BULK_TARGET   = 35_000
DPO_TARGET    = 4_000

# Only take snippets with difficulty >= 0.15 for generation (skip trivial stubs)
MIN_DIFFICULTY = 0.15
# Max chars per snippet sent to teacher (keep prompts under ~8k tokens)
MAX_SNIPPET_CHARS = 2_000


def _load_corpus_df() -> pd.DataFrame:
    """Download corpus parquet from HF and return as DataFrame."""
    token = os.environ.get("HF_TOKEN")
    p = hf_hub_download(
        repo_id=CORPUS_REPO,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=token,
    )
    df = pd.read_parquet(p)
    df = df[df["difficulty_score"] >= MIN_DIFFICULTY].copy()
    df["content"] = df["content"].str[:MAX_SNIPPET_CHARS]
    logger.info("Corpus loaded: %d snippets (difficulty >= %.2f)", len(df), MIN_DIFFICULTY)
    return df


def _push_sft(examples: list[dict], split: str) -> None:
    """Push a list of SFT dicts to HF Hub as a new split (overwrites if exists)."""
    if not examples:
        logger.warning("No examples to push for split %s — skipping", split)
        return
    ds = Dataset.from_list(examples)
    ds = build_curriculum(ds)
    token = os.environ.get("HF_TOKEN")
    ds.push_to_hub(SFT_REPO, split=split, private=True, token=token)
    logger.info("Pushed %d examples → %s (split: %s)", len(ds), SFT_REPO, split)


# ── Step 1: MainframeBench (CPU, Kaggle) ─────────────────────────────────────

def step_mainframebench() -> None:
    """Import MainframeBench ~7k curated examples. CPU only, ~5 min."""
    import_mainframebench()


# ── Step 2: Gemini gold (CPU, Kaggle, 1500 req/day rate-limit) ───────────────

# Push a HF ogni ~15 min (10 RPM × 15 min = 150 richieste)
_GEMINI_PUSH_EVERY = 150


def step_gemini() -> None:
    """
    Generate high-quality SFT examples via Gemini 2.5 Flash free tier.
    - Riprende automaticamente da HF Hub (nessun dato perso tra sessioni Kaggle)
    - Push checkpoint ogni ~150 esempi (~15 min) per sopravvivere ai crash
    - Rate limit: ~1500 req/giorno → 3-4 giorni per 5k esempi
    Needs: GEMINI_API_KEY, HF_TOKEN env vars.
    """
    configure_gemini()
    token = os.environ.get("HF_TOKEN")

    # ── Riprendi da HF Hub se esiste già un gemini_gold split ────────────────
    accumulated: list[dict] = []
    done_keys: set[int] = set()
    try:
        existing = load_dataset(SFT_REPO, split="gemini_gold", token=token)
        accumulated = [dict(row) for row in existing]
        done_keys = {hash(row["messages"][0]["content"]) for row in accumulated if row.get("messages")}
        logger.info("Ripreso da HF Hub: %d esempi già generati", len(accumulated))
    except Exception:
        logger.info("Nessun gemini_gold su HF Hub — parto da zero")

    if len(accumulated) >= GEMINI_TARGET:
        logger.info("Target già raggiunto (%d/%d) — niente da fare", len(accumulated), GEMINI_TARGET)
        return

    # ── Carica corpus e prepara lista snippet × task ──────────────────────────
    df = _load_corpus_df()
    hard = df[df["difficulty_score"] >= 0.5].sample(frac=1, random_state=42)
    snippets = hard["content"].tolist()

    task_cycle = GEMINI_TASKS * (GEMINI_TARGET // len(GEMINI_TASKS) + 1)
    random.seed(42)
    random.shuffle(task_cycle)

    new_since_push = 0
    for i, (task, snippet) in enumerate(zip(task_cycle, snippets)):
        if len(accumulated) >= GEMINI_TARGET:
            break

        # Salta se già generato in una sessione precedente
        user_prompt = GEMINI_TASK_TEMPLATES[task].format(source=snippet)
        if hash(user_prompt) in done_keys:
            continue

        ex = generate_gold_example(task, snippet)
        if ex:
            accumulated.append(ex)
            done_keys.add(hash(user_prompt))
            new_since_push += 1

        # Checkpoint push ogni ~15 min
        if new_since_push >= _GEMINI_PUSH_EVERY:
            _push_sft(accumulated, split="gemini_gold")
            logger.info("Checkpoint: %d/%d esempi su HF Hub", len(accumulated), GEMINI_TARGET)
            new_since_push = 0

        if (i + 1) % 50 == 0:
            logger.info("Gemini: %d/%d (snippet %d)", len(accumulated), GEMINI_TARGET, i + 1)

    # Push finale
    _push_sft(accumulated, split="gemini_gold")
    logger.info("Gemini gold completato: %d esempi", len(accumulated))


# ── Step 3: Bulk teacher via vLLM (Modal A100, overnight) ────────────────────

def step_teacher(teacher_model: str) -> None:
    """
    Generate bulk SFT examples using a self-hosted teacher via vLLM.
    Runs against a vLLM server at localhost:8000 (launched on Modal A100).
    Needs: vLLM server running with teacher_model.
    """
    df = _load_corpus_df()
    snippets = df.sample(frac=1, random_state=0).to_dict("records")

    # Build prompt list: each snippet × 1-2 tasks, up to BULK_TARGET
    prompts: list[dict] = []
    task_cycle = BULK_TASKS * (BULK_TARGET // len(BULK_TASKS) + 1)
    for record, task in zip(snippets * 10, task_cycle):
        if len(prompts) >= BULK_TARGET:
            break
        prompts.append({
            "task_type": task,
            "source": record["content"],
            "difficulty_score": record.get("difficulty_score", 0.3),
        })

    logger.info("Teacher bulk: generating %d examples with %s …", len(prompts), teacher_model)

    # Process in batches of 500 to allow partial saves on interruption
    BATCH = 500
    all_examples: list[dict] = []
    for start in range(0, len(prompts), BATCH):
        batch = prompts[start : start + BATCH]
        results = generate_batch(batch, teacher_model=teacher_model)
        good = [r for r in results if r is not None]
        all_examples.extend(good)
        logger.info("Teacher: %d/%d done (%d ok in batch)", start + len(batch), len(prompts), len(good))

    _push_sft(all_examples, split="teacher_bulk")
    logger.info("Teacher bulk done: %d examples", len(all_examples))


# ── Step 4: DPO pairs ────────────────────────────────────────────────────────

def step_dpo(teacher_model: str) -> None:
    """
    Generate DPO pairs: chosen (teacher) vs rejected (baseline Qwen-instruct).
    Needs: vLLM server running.
    """
    token = os.environ.get("HF_TOKEN")
    try:
        sft_ds = load_dataset(SFT_REPO, split="teacher_bulk", token=token)
    except Exception:
        logger.warning("teacher_bulk split not found — using gemini_gold for DPO")
        sft_ds = load_dataset(SFT_REPO, split="gemini_gold", token=token)

    # Sample hard examples for DPO (most informative for preference learning)
    hard = sft_ds.filter(lambda x: x.get("difficulty_score", 0) >= 0.5)
    sample = hard.shuffle(seed=7).select(range(min(DPO_TARGET * 2, len(hard))))

    pairs: list[dict] = []
    for row in sample:
        if len(pairs) >= DPO_TARGET:
            break
        pair = generate_dpo_pair(row, teacher_model=teacher_model)
        if pair:
            pairs.append(pair)
        if len(pairs) % 200 == 0 and pairs:
            logger.info("DPO: %d/%d pairs generated", len(pairs), DPO_TARGET)

    if pairs:
        ds = Dataset.from_list(pairs)
        ds.push_to_hub(DPO_REPO, split="train", private=True, token=token)
        logger.info("DPO done: %d pairs → %s", len(pairs), DPO_REPO)


# ── Step: Alibaba DashScope teacher (CPU, Kaggle) ─────────────────────────────

ALIBABA_PER_MODEL_TARGET = 1_000  # ~1k esempi per modello (1M token budget)
_ALIBABA_PUSH_EVERY = 150         # push ogni 150 nuovi esempi (~15 min)


def step_alibaba() -> None:
    """
    Genera esempi SFT gold via Alibaba DashScope (OpenAI-compatible, CPU).
    Cascata di modelli: qwen3-coder-plus → 235b-thinking → max → qwq-plus.
    Resume automatico da HF Hub tra sessioni. Push ogni 150 esempi.
    Needs: ALIBABA_API env var.
    """
    token = os.environ.get("HF_TOKEN")
    df = _load_corpus_df()
    hard = df[df["difficulty_score"] >= 0.5].sample(frac=1, random_state=7)
    snippets = hard.to_dict("records")

    # Riprendi da HF Hub
    accumulated: list[dict] = []
    done_keys: set[int] = set()
    try:
        existing = load_dataset(SFT_REPO, split="alibaba_gold", token=token)
        accumulated = [dict(r) for r in existing]
        done_keys = {hash(r["messages"][0]["content"]) for r in accumulated if r.get("messages")}
        logger.info("Ripreso: %d esempi già generati", len(accumulated))
    except Exception:
        logger.info("Nessun alibaba_gold su HF — parto da zero")

    total_target = ALIBABA_PER_MODEL_TARGET * len(ALIBABA_MODELS)
    if len(accumulated) >= total_target:
        logger.info("Target già raggiunto (%d) — esco", len(accumulated))
        return

    task_cycle = ALIBABA_TASKS * (total_target // len(ALIBABA_TASKS) + 2)
    random.seed(7)
    random.shuffle(task_cycle)

    new_since_push = 0

    for model in ALIBABA_MODELS:
        model_examples = [r for r in accumulated if model.replace("-", "_") in r.get("source", "")]
        if len(model_examples) >= ALIBABA_PER_MODEL_TARGET:
            logger.info("Modello %s già completo (%d esempi) — skip", model, len(model_examples))
            continue

        logger.info("=== Teacher: %s ===", model)
        model_count = len(model_examples)
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10  # se 10 di fila falliscono → quota esaurita

        for record, task in zip(snippets * 5, task_cycle):
            if model_count >= ALIBABA_PER_MODEL_TARGET:
                break
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Modello %s: %d fallimenti consecutivi — quota esaurita, passo al prossimo",
                    model, consecutive_failures,
                )
                break

            user_content = f"{task}::{record['content'][:200]}"
            if hash(user_content) in done_keys:
                continue

            ex = alibaba_generate(
                task=task,
                source=record["content"],
                model=model,
                difficulty_score=record.get("difficulty_score", 0.5),
                use_thinking=(task in {"debug", "translation_redefines", "modernize_goto"}),
            )
            if ex:
                accumulated.append(ex)
                done_keys.add(hash(user_content))
                model_count += 1
                new_since_push += 1
                consecutive_failures = 0  # reset al primo successo
            else:
                consecutive_failures += 1

            if new_since_push >= _ALIBABA_PUSH_EVERY:
                _push_sft(accumulated, split="alibaba_gold")
                logger.info("Checkpoint: %d esempi totali su HF Hub", len(accumulated))
                new_since_push = 0

        logger.info("Modello %s completato: %d esempi", model, model_count)

    _push_sft(accumulated, split="alibaba_gold")
    logger.info("Alibaba gold completato: %d esempi totali", len(accumulated))


# ── Entry point ───────────────────────────────────────────────────────────────

def run_orchestration(step: str, teacher_model: str) -> None:
    steps = {
        "mainframebench": step_mainframebench,
        "gemini": step_gemini,
        "alibaba": step_alibaba,
        "teacher": lambda: step_teacher(teacher_model),
        "dpo": lambda: step_dpo(teacher_model),
    }
    if step not in steps:
        raise ValueError(f"Unknown step: {step}. Choose from: {list(steps)}")
    steps[step]()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, choices=["mainframebench", "gemini", "alibaba", "teacher", "dpo"])
    p.add_argument("--teacher", default="Fsoft-AIC/XMAiNframe-instruct-10.5b")
    args = p.parse_args()
    run_orchestration(args.step, args.teacher)
