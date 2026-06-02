"""
Import MainframeBench (Fsoft-AIC, MIT) directly into SFT format.

Splits:
  - COBOL_code_summarization  → 2,523 examples  (code → summary)
  - QA                        → 2,598 examples  (question answering)
  - MCQ                       → 1,931 examples  (multiple choice)

Output: HF Hub private dataset `AlexThunder0/cobol-sft-dataset` (appended).
Format: ChatML with optional <thinking> prefix for complex examples.

Run on Kaggle or local — no GPU needed.
"""

from __future__ import annotations

import logging

from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi

from src.synth.thinking_traces import maybe_add_thinking_prefix

logger = logging.getLogger(__name__)

HUB_SFT_REPO = "AlexThunder0/cobol-sft-dataset"
MAINFRAMEBENCH_REPO = "Fsoft-AIC/MainframeBench"


def _format_summarization(row: dict) -> dict:
    user = f"Explain the following COBOL program:\n\n```cobol\n{row['cobol_code']}\n```"
    assistant = maybe_add_thinking_prefix(row["summary"], task_type="explanation")
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], "source": "mainframebench_summarization", "difficulty_score": 0.3}


def _format_qa(row: dict) -> dict:
    user = row["question"]
    assistant = maybe_add_thinking_prefix(row["answer"], task_type="qa")
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], "source": "mainframebench_qa", "difficulty_score": 0.2}


def _format_mcq(row: dict) -> dict:
    choices = "\n".join(f"{k}. {v}" for k, v in row.get("choices", {}).items())
    user = f"{row['question']}\n\n{choices}"
    assistant = str(row.get("answer", ""))
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], "source": "mainframebench_mcq", "difficulty_score": 0.1}


def run_import() -> None:
    logger.info("Loading MainframeBench COBOL_code_summarization …")
    summ = load_dataset(MAINFRAMEBENCH_REPO, "COBOL_code_summarization", split="train")
    summ = summ.map(_format_summarization, remove_columns=summ.column_names)

    logger.info("Loading MainframeBench QA …")
    qa = load_dataset(MAINFRAMEBENCH_REPO, "QA", split="train")
    qa = qa.map(_format_qa, remove_columns=qa.column_names)

    logger.info("Loading MainframeBench MCQ …")
    mcq = load_dataset(MAINFRAMEBENCH_REPO, "MCQ", split="train")
    mcq = mcq.map(_format_mcq, remove_columns=mcq.column_names)

    combined = concatenate_datasets([summ, qa, mcq])
    logger.info("Total MainframeBench examples: %d", len(combined))

    combined.push_to_hub(HUB_SFT_REPO, split="mainframebench", private=True)
    logger.info("Pushed to %s (split: mainframebench)", HUB_SFT_REPO)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_import()
