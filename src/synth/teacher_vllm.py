"""
Bulk SFT data generation using a self-hosted teacher model via vLLM.

Teacher candidates (decision made in W3 after 200-example pilot):
  - XMainframe-instruct-10.5b  (Fsoft-AIC/XMAiNframe-instruct-10.5b)
  - Qwen3.6-27B-Instruct vanilla

XMainframe-10.5b is preferred if its COBOL quality ≥ Qwen vanilla on the pilot.
It fits comfortably on L40S 48GB in FP8 (<10GB).

Target: 30–40k examples, distributed across overnight/idle Studio sessions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import OpenAI  # vLLM exposes an OpenAI-compatible API

from src.synth.thinking_traces import maybe_add_thinking_prefix

logger = logging.getLogger(__name__)

# vLLM server started via src/deploy/vllm_serve.sh
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"  # vLLM local server doesn't need a key

# Swap this to Qwen3.6-27B-Instruct if XMainframe loses the W3 pilot comparison
DEFAULT_TEACHER = "Fsoft-AIC/XMAiNframe-instruct-10.5b"

TASK_TEMPLATES: dict[str, str] = {
    "explain": "Explain what the following COBOL program does:\n\n```cobol\n{source}\n```",
    "refactor": (
        "Refactor the following COBOL program to improve readability and structure, "
        "while preserving semantics:\n\n```cobol\n{source}\n```"
    ),
    "translate_to_java": (
        "Translate the following COBOL program to equivalent Java code:\n\n```cobol\n{source}\n```"
    ),
    "translate_to_cobol": (
        "Translate the following Java code to equivalent COBOL:\n\n```java\n{source}\n```"
    ),
    "generate": (
        "Write a COBOL program that: {description}"
    ),
    "debug": (
        "The following COBOL program contains a bug. Identify and fix it:\n\n```cobol\n{source}\n```"
    ),
}


def generate_batch(
    prompts: list[dict],
    teacher_model: str = DEFAULT_TEACHER,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> list[dict | None]:
    """Generate SFT examples for a list of prompt dicts.

    Each prompt dict: {"task_type": str, "source": str, "difficulty_score": float}
    Returns a list of SFT example dicts (None for failed generations).
    """
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    results: list[dict | None] = []

    for prompt in prompts:
        task_type = prompt["task_type"]
        template = TASK_TEMPLATES.get(task_type)
        if template is None:
            results.append(None)
            continue

        user_content = template.format(**{k: v for k, v in prompt.items() if k != "task_type"})

        try:
            response = client.chat.completions.create(
                model=teacher_model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            answer = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("vLLM call failed (%s) — skipping", exc)
            results.append(None)
            continue

        answer = maybe_add_thinking_prefix(answer, task_type=task_type)
        results.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
            "source": f"teacher_{teacher_model.split('/')[-1]}_{task_type}",
            "difficulty_score": prompt.get("difficulty_score", 0.5),
        })

    return results
