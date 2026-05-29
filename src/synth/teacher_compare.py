"""
W3 pilot: compare XMainframe-instruct-10.5b vs Qwen3.6-27B-Instruct vanilla
on 200 COBOL examples sampled from the validated corpus.

Produces a side-by-side CSV for manual inspection + automatic compiler validation rate.
Decision: whichever teacher has higher compilation rate AND passes manual quality check
becomes the bulk teacher for 30-40k generation.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

from datasets import load_dataset

from src.pipeline.validate import validate_batch
from src.synth.teacher_vllm import generate_batch

logger = logging.getLogger(__name__)

CORPUS_REPO = "YOUR_HF_USERNAME/cobol-cpt-corpus"
PILOT_SIZE = 200
OUTPUT_CSV = Path("data/teacher_compare_pilot.csv")

TEACHERS = [
    "Fsoft-AIC/XMAiNframe-instruct-10.5b",
    "Qwen/Qwen3.6-27B-Instruct",
]


def run_pilot() -> None:
    logger.info("Loading corpus sample …")
    ds = load_dataset(CORPUS_REPO, split="train", streaming=True)
    sample = [row for i, row in enumerate(ds) if i < PILOT_SIZE * 5 and row.get("compiles")]
    sample = random.sample(sample, min(PILOT_SIZE, len(sample)))

    rows: list[dict] = []
    for teacher in TEACHERS:
        logger.info("Generating with teacher: %s", teacher)
        prompts = [{"task_type": "explain", "source": r["content"], "difficulty_score": r.get("difficulty_score", 0.5)} for r in sample]
        results = generate_batch(prompts, teacher_model=teacher)
        validated = validate_batch([
            {"content": (r["messages"][1]["content"] if r else ""), "source": "generated"}
            for r in results
        ])
        for i, (ex, val) in enumerate(zip(results, validated)):
            rows.append({
                "idx": i,
                "teacher": teacher,
                "user": ex["messages"][0]["content"] if ex else "",
                "response": ex["messages"][1]["content"] if ex else "",
                "compiles": val.get("compiles", False),
            })

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "teacher", "user", "response", "compiles"])
        writer.writeheader()
        writer.writerows(rows)

    for teacher in TEACHERS:
        teacher_rows = [r for r in rows if r["teacher"] == teacher]
        compile_rate = sum(1 for r in teacher_rows if r["compiles"]) / len(teacher_rows)
        logger.info("Teacher %s: compile rate %.1f%%", teacher, compile_rate * 100)

    logger.info("Pilot CSV saved to %s — inspect manually before W3 bulk generation", OUTPUT_CSV)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pilot()
