"""
Generate "gold" SFT examples using Gemini 2.5 Flash (Google AI Studio free tier).

Quota: 1500 requests/day × 4–5 days = 6000 examples.
Uses a local SQLite cache to avoid re-requesting on resume.

Targets: translation with REDEFINES, modernization of GO TO spaghetti,
complex copybook scenarios — tasks where teacher self-hosted struggles.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import google.generativeai as genai
from sqlitedict import SqliteDict

from src.synth.thinking_traces import wrap_with_thinking

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"
CACHE_PATH = Path("data/gemini_cache.sqlite")
OUTPUT_REPO = "AlexThunder0/cobol-sft-dataset"
RATE_LIMIT_SLEEP = 0.1  # seconds between requests (stay well under 1500/day)


SYSTEM_PROMPT = """\
You are an expert COBOL developer and educator. For each task, first reason step-by-step
inside <thinking>...</thinking> tags, then provide your final answer.
Keep COBOL code syntactically valid for GnuCOBOL 3.2+ in free format.
"""

TASK_TEMPLATES: dict[str, str] = {
    "translation_redefines": (
        "Translate the following COBOL program to Java. Pay careful attention to REDEFINES "
        "clauses and data structure mapping.\n\n```cobol\n{source}\n```"
    ),
    "modernize_goto": (
        "Refactor the following COBOL program to eliminate all GO TO statements using structured "
        "PERFORM loops and EVALUATE blocks.\n\n```cobol\n{source}\n```"
    ),
    "explain_copybook": (
        "Explain what this COBOL program does, including the purpose of each copybook section.\n\n"
        "```cobol\n{source}\n```"
    ),
    "debug": (
        "The following COBOL program has a bug. Identify the bug and provide a corrected version.\n\n"
        "```cobol\n{source}\n```"
    ),
}


def generate_gold_example(task_type: str, source: str) -> dict | None:
    """Call Gemini and return a formatted SFT example dict, or None on failure."""
    template = TASK_TEMPLATES.get(task_type)
    if template is None:
        raise ValueError(f"Unknown task type: {task_type}")

    user_prompt = template.format(source=source)
    cache_key = f"{task_type}::{hash(user_prompt)}"

    with SqliteDict(str(CACHE_PATH)) as cache:
        if cache_key in cache:
            return cache[cache_key]

    try:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        response = model.generate_content(user_prompt)
        raw_text = response.text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini call failed (%s) — skipping", exc)
        return None

    # Extract <thinking> block if present
    thinking, answer = _split_thinking(raw_text)
    if thinking:
        assistant_content = wrap_with_thinking(thinking, answer)
    else:
        assistant_content = raw_text

    example = {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": f"gemini_{task_type}",
        "difficulty_score": 0.75,  # gold examples are hard by design
    }

    with SqliteDict(str(CACHE_PATH)) as cache:
        cache[cache_key] = example
        cache.commit()

    time.sleep(RATE_LIMIT_SLEEP)
    return example


def _split_thinking(text: str) -> tuple[str, str]:
    """Return (thinking_content, answer) or ('', full_text) if no thinking block."""
    start = text.find("<thinking>")
    end = text.find("</thinking>")
    if start == -1 or end == -1:
        return "", text

    thinking = text[start + len("<thinking>"):end].strip()
    answer = text[end + len("</thinking>"):].strip()
    return thinking, answer


def configure() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
