"""
COBOLEval runner — primary eval benchmark (146 HumanEval-style problems).
Source: https://github.com/BloopAI/COBOLEval

Metrics reported:
  - Compile rate (% of generated programs that compile with GnuCOBOL)
  - Pass@1 (% of problems where the first generation passes all test cases)

Usage:
    python -m src.eval.cobolceval_runner \
        --model YOUR_HF_USERNAME/qwen-cobol-27b-dpo \
        --coboleval-dir /path/to/COBOLEval \
        --output results/coboleval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

from src.eval.compile_check import compile_rate

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"

GENERATION_PROMPT = (
    "Complete the following COBOL program. Only output the COBOL code, no explanations.\n\n{prompt}"
)


def load_problems(coboleval_dir: Path) -> list[dict]:
    problems_file = coboleval_dir / "problems.json"
    if not problems_file.exists():
        raise FileNotFoundError(f"COBOLEval problems not found at {problems_file}")
    with open(problems_file) as f:
        return json.load(f)


def generate_solution(client: OpenAI, model: str, prompt: str, thinking_budget: int = 0) -> str:
    messages = [{"role": "user", "content": GENERATION_PROMPT.format(prompt=prompt)}]
    extra = {}
    if thinking_budget > 0:
        extra["extra_body"] = {"chat_template_kwargs": {"thinking": True}, "thinking": {"budget_tokens": thinking_budget}}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        **extra,
    )
    return response.choices[0].message.content or ""


def run_tests(solution_code: str, test_runner: Path, timeout: int = 30) -> bool:
    """Write solution to temp file and run the COBOLEval test harness."""
    with tempfile.NamedTemporaryFile(suffix=".cob", mode="w", delete=False) as f:
        f.write(solution_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", str(test_runner), tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_eval(args: argparse.Namespace) -> dict:
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    coboleval_dir = Path(args.coboleval_dir)
    problems = load_problems(coboleval_dir)
    test_runner = coboleval_dir / "run_tests.py"

    results = []
    solutions = []

    for prob in problems:
        solution = generate_solution(
            client, args.model, prob["prompt"], thinking_budget=args.thinking_budget
        )
        solutions.append(solution)
        passed = run_tests(solution, test_runner) if test_runner.exists() else False
        results.append({"task_id": prob["task_id"], "solution": solution, "passed": passed})
        logger.info("Task %s: %s", prob["task_id"], "PASS" if passed else "FAIL")

    n_pass = sum(1 for r in results if r["passed"])
    c_rate = compile_rate(solutions)

    summary = {
        "model": args.model,
        "n_problems": len(problems),
        "compile_rate": c_rate,
        "pass_at_1": n_pass / len(problems) if problems else 0.0,
        "n_pass": n_pass,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("COBOLEval — compile: %.1f%%, pass@1: %.2f", c_rate * 100, summary["pass_at_1"])
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="COBOLEval runner")
    p.add_argument("--model", required=True, help="HF model ID or vLLM model name")
    p.add_argument("--coboleval-dir", required=True, help="Path to cloned COBOLEval repo")
    p.add_argument("--output", default="results/coboleval_results.json")
    p.add_argument("--thinking-budget", type=int, default=0, help="Thinking tokens (0=disabled)")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
