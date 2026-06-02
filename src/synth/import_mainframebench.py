"""
Import MainframeBench (Fsoft-AIC, MIT) directly into SFT format.

Colonne reali (verificate via datasets-server API 2026-06-02):
  COBOL_code_summarization:   Unnamed:0, prompt, source, summary
  question_answering:         Unnamed:0, prompt, question, answer
  multiple_choice_question:   Unnamed:0, prompt, question, A, B, C, D, answer

Output: HF Hub private dataset `AlexThunder0/cobol-sft-dataset` split=mainframebench.
"""

from __future__ import annotations

import logging
import os

from datasets import concatenate_datasets, load_dataset

from src.synth.thinking_traces import maybe_add_thinking_prefix

logger = logging.getLogger(__name__)

HUB_SFT_REPO = "AlexThunder0/cobol-sft-dataset"
MAINFRAMEBENCH_REPO = "Fsoft-AIC/MainframeBench"


def _format_summarization(row: dict) -> dict:
    user = f"Explain the following COBOL program:\n\n```cobol\n{row['prompt']}\n```"
    assistant = maybe_add_thinking_prefix(row["summary"], task_type="explanation")
    return {
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        "source": "mainframebench_summarization",
        "difficulty_score": 0.3,
    }


def _format_qa(row: dict) -> dict:
    assistant = maybe_add_thinking_prefix(row["answer"], task_type="qa")
    return {
        "messages": [{"role": "user", "content": row["question"]}, {"role": "assistant", "content": assistant}],
        "source": "mainframebench_qa",
        "difficulty_score": 0.2,
    }


def _format_mcq(row: dict) -> dict:
    choices = f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
    user = f"{row['question']}\n\n{choices}"
    return {
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": str(row["answer"])}],
        "source": "mainframebench_mcq",
        "difficulty_score": 0.1,
    }


def run_import() -> None:
    token = os.environ.get("HF_TOKEN")

    logger.info("Loading MainframeBench COBOL_code_summarization …")
    summ = load_dataset(MAINFRAMEBENCH_REPO, "COBOL_code_summarization", split="train", token=token)
    summ = summ.map(_format_summarization, remove_columns=summ.column_names)
    logger.info("summarization: %d examples", len(summ))

    logger.info("Loading MainframeBench question_answering …")
    qa = load_dataset(MAINFRAMEBENCH_REPO, "question_answering", split="train", token=token)
    qa = qa.map(_format_qa, remove_columns=qa.column_names)
    logger.info("qa: %d examples", len(qa))

    logger.info("Loading MainframeBench multiple_choice_question …")
    mcq = load_dataset(MAINFRAMEBENCH_REPO, "multiple_choice_question", split="train", token=token)
    mcq = mcq.map(_format_mcq, remove_columns=mcq.column_names)
    logger.info("mcq: %d examples", len(mcq))

    combined = concatenate_datasets([summ, qa, mcq])
    logger.info("Total MainframeBench examples: %d", len(combined))

    combined.push_to_hub(HUB_SFT_REPO, split="mainframebench", private=True, token=token)
    logger.info("Pushed to %s (split: mainframebench)", HUB_SFT_REPO)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_import()
