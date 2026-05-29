"""
COBOL-JavaTrans runner — secondary eval (143 bidirectional COBOL↔Java pairs).
Source: https://github.com/COBOL-Coder (verify repo for dataset availability)

Metrics: Pass@1 for translation correctness (test execution).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"

COBOL_TO_JAVA_PROMPT = "Translate the following COBOL program to Java:\n\n```cobol\n{source}\n```"
JAVA_TO_COBOL_PROMPT = "Translate the following Java code to COBOL:\n\n```java\n{source}\n```"


def run_eval(args: argparse.Namespace) -> dict:
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    pairs_file = Path(args.pairs_file)

    with open(pairs_file) as f:
        pairs = json.load(f)

    results = []
    for pair in pairs:
        direction = pair.get("direction", "cobol_to_java")
        prompt_template = COBOL_TO_JAVA_PROMPT if direction == "cobol_to_java" else JAVA_TO_COBOL_PROMPT
        user_content = prompt_template.format(source=pair["source"])

        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=2048,
            temperature=0.0,
        )
        translation = response.choices[0].message.content or ""

        # TODO W7: implement test execution for translated code
        passed = False  # placeholder

        results.append({
            "task_id": pair.get("task_id"),
            "direction": direction,
            "translation": translation,
            "passed": passed,
        })

    summary = {
        "model": args.model,
        "n_pairs": len(pairs),
        "pass_at_1": sum(1 for r in results if r["passed"]) / len(pairs) if pairs else 0.0,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("COBOL-JavaTrans pass@1: %.2f", summary["pass_at_1"])
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="COBOL-JavaTrans runner")
    p.add_argument("--model", required=True)
    p.add_argument("--pairs-file", required=True, help="Path to translation pairs JSON")
    p.add_argument("--output", default="results/javatrans_results.json")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
