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

from src.synth.thinking_traces import maybe_add_thinking_prefix

logger = logging.getLogger(__name__)

HUB_SFT_REPO = "AlexThunder0/cobol-sft-dataset"
MAINFRAMEBENCH_REPO = "Fsoft-AIC/MainframeBench"

# Colonne possibili per il codice COBOL (il dataset usa nomi diversi per split)
_CODE_COLS = ("cobol_code", "code", "source_code", "cobol", "program", "source")
# Colonne possibili per il summary/risposta
_SUMMARY_COLS = ("summary", "description", "explanation", "answer", "output")
# Colonne possibili per la domanda
_QUESTION_COLS = ("question", "query", "input", "prompt")


def _pick(row: dict, candidates: tuple[str, ...], fallback: str = "") -> str:
    for k in candidates:
        if k in row and row[k]:
            return str(row[k])
    # ultimo tentativo: primo valore stringa non vuoto
    for v in row.values():
        if isinstance(v, str) and v.strip():
            return v
    return fallback


def _format_summarization(row: dict) -> dict:
    code = _pick(row, _CODE_COLS)
    summary = _pick(row, _SUMMARY_COLS)
    user = f"Explain the following COBOL program:\n\n```cobol\n{code}\n```"
    assistant = maybe_add_thinking_prefix(summary, task_type="explanation")
    return {
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        "source": "mainframebench_summarization",
        "difficulty_score": 0.3,
    }


def _format_qa(row: dict) -> dict:
    question = _pick(row, _QUESTION_COLS)
    answer = _pick(row, _SUMMARY_COLS)
    assistant = maybe_add_thinking_prefix(answer, task_type="qa")
    return {
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": assistant}],
        "source": "mainframebench_qa",
        "difficulty_score": 0.2,
    }


def _format_mcq(row: dict) -> dict:
    question = _pick(row, _QUESTION_COLS)
    # choices può essere dict {"A": "...", "B": "..."} oppure list
    choices_raw = row.get("choices") or row.get("options") or {}
    if isinstance(choices_raw, dict):
        choices = "\n".join(f"{k}. {v}" for k, v in choices_raw.items())
    elif isinstance(choices_raw, list):
        choices = "\n".join(f"{chr(65+i)}. {v}" for i, v in enumerate(choices_raw))
    else:
        choices = str(choices_raw)
    user = f"{question}\n\n{choices}" if choices else question
    answer = str(row.get("answer") or row.get("label") or row.get("correct") or "")
    return {
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
        "source": "mainframebench_mcq",
        "difficulty_score": 0.1,
    }


def _log_columns(name: str, ds) -> None:
    logger.info("%s columns: %s", name, list(ds.column_names))
    if len(ds) > 0:
        logger.info("%s first row keys: %s", name, list(ds[0].keys()))


def run_import() -> None:
    logger.info("Loading MainframeBench COBOL_code_summarization …")
    summ = load_dataset(MAINFRAMEBENCH_REPO, "COBOL_code_summarization", split="train")
    _log_columns("summarization", summ)
    summ = summ.map(_format_summarization, remove_columns=summ.column_names)

    logger.info("Loading MainframeBench QA …")
    qa = load_dataset(MAINFRAMEBENCH_REPO, "QA", split="train")
    _log_columns("QA", qa)
    qa = qa.map(_format_qa, remove_columns=qa.column_names)

    logger.info("Loading MainframeBench MCQ …")
    mcq = load_dataset(MAINFRAMEBENCH_REPO, "MCQ", split="train")
    _log_columns("MCQ", mcq)
    mcq = mcq.map(_format_mcq, remove_columns=mcq.column_names)

    combined = concatenate_datasets([summ, qa, mcq])
    logger.info("Total MainframeBench examples: %d", len(combined))

    token = __import__("os").environ.get("HF_TOKEN")
    combined.push_to_hub(HUB_SFT_REPO, split="mainframebench", private=True, token=token)
    logger.info("Pushed to %s (split: mainframebench)", HUB_SFT_REPO)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_import()
