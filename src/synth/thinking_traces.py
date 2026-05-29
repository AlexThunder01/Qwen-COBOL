"""
Utilities for adding <thinking>...</thinking> traces to SFT examples.

Qwen3.6 hybrid mode: the model can produce an explicit reasoning trace
before its final answer. Including this in SFT examples teaches the model
to reason before answering — a key edge over COBOL-Coder (SFT-only, no traces).

For MainframeBench and simple tasks, thinking is kept minimal or omitted.
For Gemini-generated gold examples, Gemini returns traces natively.
For teacher-generated bulk examples, a lightweight template is prepended.
"""

from __future__ import annotations

# Tasks where thinking traces add most value
_THINKING_TASKS = frozenset({"translation", "refactoring", "modernization", "debugging"})

# Minimal template for tasks where the teacher didn't produce a trace
_TRACE_TEMPLATE = (
    "<thinking>\n"
    "Let me analyze the COBOL program structure before answering.\n"
    "</thinking>\n\n"
)


def maybe_add_thinking_prefix(answer: str, task_type: str = "general") -> str:
    """Prepend a thinking trace if the task benefits from it and the answer lacks one."""
    if answer.lstrip().startswith("<thinking>"):
        return answer  # already has a trace

    if task_type in _THINKING_TASKS:
        return _TRACE_TEMPLATE + answer

    return answer


def wrap_with_thinking(thinking: str, answer: str) -> str:
    """Combine an explicit thinking block with the final answer."""
    return f"<thinking>\n{thinking.strip()}\n</thinking>\n\n{answer.strip()}"
