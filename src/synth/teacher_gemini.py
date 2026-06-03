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

from sqlitedict import SqliteDict

from src.synth.thinking_traces import wrap_with_thinking

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-1.5-flash"  # 2.5-flash free tier = 20 RPD; 1.5-flash = 1500 RPD
CACHE_PATH = Path("data/gemini_cache.sqlite")
OUTPUT_REPO = "AlexThunder0/cobol-sft-dataset"
RATE_LIMIT_SLEEP = 4.5  # 15 RPM free tier → min 4s tra richieste (margine sicurezza)


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


def _get_client():
    """Return a google.genai client (new SDK) or fall back to generativeai (old SDK)."""
    try:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return ("new", client)
    except ImportError:
        import google.generativeai as genai_old
        genai_old.configure(api_key=os.environ["GEMINI_API_KEY"])
        return ("old", genai_old)


def _call_gemini(client_info, user_prompt: str) -> str:
    """Call Gemini and return raw text. Works with both old and new SDK."""
    sdk, client = client_info
    if sdk == "new":
        from google.genai import types
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return response.text
    else:
        model = client.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        response = model.generate_content(user_prompt)
        return response.text


def generate_gold_example(task_type: str, source: str) -> dict | None:
    """Call Gemini and return a formatted SFT example dict, or None on failure."""
    template = TASK_TEMPLATES.get(task_type)
    if template is None:
        raise ValueError(f"Unknown task type: {task_type}")

    user_prompt = template.format(source=source)
    cache_key = f"{task_type}::{hash(user_prompt)}"

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteDict(str(CACHE_PATH)) as cache:
        if cache_key in cache:
            return cache[cache_key]

    client_info = _get_client()
    try:
        raw_text = _call_gemini(client_info, user_prompt)
    except Exception as exc:
        logger.warning("Gemini call failed (%s) — skipping", exc)
        return None

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
        "difficulty_score": 0.75,
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
