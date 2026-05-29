"""
Orchestrate the full synthetic data generation pipeline (W3).

Routing logic:
  - Hard tasks (translation REDEFINES, GO TO modernization) → Gemini gold (4-6k)
  - Remaining bulk → teacher vLLM (30-40k)
  - DPO pairs → dpo_pairs.py

Output: HF Hub private datasets cobol-sft-dataset + cobol-dpo-dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset

from src.synth.teacher_gemini import configure as configure_gemini, generate_gold_example
from src.synth.teacher_vllm import generate_batch
from src.synth.dpo_pairs import generate_dpo_pair
from src.synth.curriculum_sampler import build_curriculum

logger = logging.getLogger(__name__)

CORPUS_REPO = "YOUR_HF_USERNAME/cobol-cpt-corpus"
SFT_REPO = "YOUR_HF_USERNAME/cobol-sft-dataset"
DPO_REPO = "YOUR_HF_USERNAME/cobol-dpo-dataset"

# Task routing by difficulty threshold
GEMINI_TASKS = ["translation_redefines", "modernize_goto", "explain_copybook", "debug"]
BULK_TASKS = ["explain", "refactor", "translate_to_java", "translate_to_cobol", "generate"]

GEMINI_TARGET = 5_000
BULK_TARGET = 35_000
DPO_TARGET = 4_000


def run_orchestration(teacher_model: str) -> None:
    configure_gemini()

    logger.info("Step 1/3: Gemini gold examples (target %d) …", GEMINI_TARGET)
    # TODO W3: implement iteration over hard snippets from corpus

    logger.info("Step 2/3: Bulk teacher generation (target %d) …", BULK_TARGET)
    # TODO W3: load corpus, build prompt list, call generate_batch in batches of 500

    logger.info("Step 3/3: DPO pair generation (target %d) …", DPO_TARGET)
    # TODO W3: call generate_dpo_pair on chosen SFT examples

    logger.info("Orchestration complete.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="Fsoft-AIC/XMAiNframe-instruct-10.5b")
    args = p.parse_args()
    run_orchestration(args.teacher)
