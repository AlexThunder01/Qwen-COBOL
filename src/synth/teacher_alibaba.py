"""
Generazione SFT gold via Alibaba DashScope (OpenAI-compatible).

Strategia teacher a cascata:
  1. qwen3-coder-plus       — code specialist, bulk generation (no thinking)
  2. qwen3-235b-a22b-thinking-2507 — reasoning traces per task difficili
  3. qwen3-max              — fallback
  4. qwq-plus               — reasoning specialist, debug/REDEFINES

Quota: 1M token INPUT + 1M token OUTPUT per modello.
Stima: ~1000 esempi per modello (no thinking) → ~3000-4000 esempi totali.
API: https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (OpenAI-compatible)

Gira su Kaggle CPU (niente GPU) o local.
"""

from __future__ import annotations

import logging
import os
import random
import time

from openai import OpenAI

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Cascata: dal più capace al fallback
TEACHER_MODELS = [
    "qwen3-coder-plus",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-max",
    "qwq-plus",
]

# Task che beneficiano del thinking (usa reasoning traces → esempi gold)
THINKING_TASKS = {"debug", "translation_redefines", "modernize_goto"}

TASK_TEMPLATES: dict[str, str] = {
    "explain": (
        "Explain what the following COBOL program does, describing its inputs, "
        "outputs, and main logic:\n\n```cobol\n{source}\n```"
    ),
    "refactor": (
        "Refactor the following COBOL program to improve readability and structure "
        "while preserving its exact semantics:\n\n```cobol\n{source}\n```"
    ),
    "debug": (
        "The following COBOL program contains a bug. Identify the bug and provide "
        "a corrected version:\n\n```cobol\n{source}\n```"
    ),
    "translation_redefines": (
        "Translate the following COBOL program to Java. Pay careful attention to "
        "REDEFINES clauses and data structure mapping.\n\n```cobol\n{source}\n```"
    ),
    "modernize_goto": (
        "Refactor the following COBOL program to eliminate all GO TO statements "
        "using structured PERFORM loops and EVALUATE blocks.\n\n"
        "```cobol\n{source}\n```"
    ),
}
BULK_TASKS = list(TASK_TEMPLATES.keys())

RATE_LIMIT_SLEEP = 1.0  # DashScope free tier: conservativo


def _get_client() -> OpenAI:
    api_key = os.environ.get("ALIBABA_API")
    if not api_key:
        raise RuntimeError("ALIBABA_API environment variable not set")
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)


def generate_example(
    task: str,
    source: str,
    model: str,
    difficulty_score: float = 0.5,
    use_thinking: bool = False,
) -> dict | None:
    """Chiama il modello Alibaba e restituisce un esempio SFT formattato."""
    template = TASK_TEMPLATES.get(task)
    if not template:
        return None

    user_content = template.format(source=source)
    client = _get_client()

    # Per i modelli thinking, possiamo scegliere se abilitare il reasoning
    extra = {}
    if use_thinking and task in THINKING_TASKS:
        extra = {"extra_body": {"enable_thinking": True}}
    else:
        # Disabilita thinking per velocità/costo
        if "thinking" in model or "qwq" in model:
            extra = {"extra_body": {"enable_thinking": False}}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1024,
            temperature=0.5,
            **extra,
        )
        answer = resp.choices[0].message.content or ""
        time.sleep(RATE_LIMIT_SLEEP)
    except Exception as e:
        logger.warning("DashScope call failed (%s, %s): %s", model, task, e)
        return None

    if not answer.strip():
        return None

    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": answer.strip()},
        ],
        "source":           f"alibaba_{model.replace('-', '_')}_{task}",
        "difficulty_score": float(difficulty_score),
    }
