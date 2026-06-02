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

from src.synth.teacher_gemini import configure as configure_gemini, generate_gold_example
from src.synth.teacher_vllm import generate_batch, TASK_TEMPLATES as VLLM_TEMPLATES
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

def step_gemini() -> None:
    """
    Generate high-quality SFT examples via Gemini 2.5 Flash free tier.
    Rate-limited to 1500 req/day → runs over 3-4 days.
    Resumes automatically via SQLite cache (no duplicate calls on restart).
    Needs: GEMINI_API_KEY env var.
    """
    configure_gemini()
    df = _load_corpus_df()

    # Prefer high-difficulty snippets for Gemini gold
    hard = df[df["difficulty_score"] >= 0.5].sample(frac=1, random_state=42)
    snippets = hard["content"].tolist()

    examples: list[dict] = []
    task_cycle = GEMINI_TASKS * (GEMINI_TARGET // len(GEMINI_TASKS) + 1)
    random.seed(42)
    random.shuffle(task_cycle)

    for i, (task, snippet) in enumerate(zip(task_cycle, snippets)):
        if len(examples) >= GEMINI_TARGET:
            break
        ex = generate_gold_example(task, snippet)
        if ex:
            examples.append(ex)
        if (i + 1) % 100 == 0:
            logger.info("Gemini: %d/%d generated (processed %d snippets)", len(examples), GEMINI_TARGET, i + 1)

    _push_sft(examples, split="gemini_gold")
    logger.info("Gemini gold done: %d examples", len(examples))


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


# ── Entry point ───────────────────────────────────────────────────────────────

def run_orchestration(step: str, teacher_model: str) -> None:
    steps = {
        "mainframebench": step_mainframebench,
        "gemini": step_gemini,
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
    p.add_argument("--step", required=True, choices=["mainframebench", "gemini", "teacher", "dpo"])
    p.add_argument("--teacher", default="Fsoft-AIC/XMAiNframe-instruct-10.5b")
    args = p.parse_args()
    run_orchestration(args.step, args.teacher)
