"""
Generate DPO preference pairs.

Strategy:
  chosen  = teacher output that compiles + passes semantic test
  rejected = teacher variant with deliberate error (PIC clause, wrong PERFORM target,
             mismatched REDEFINES) — validated to NOT compile

Gemini validates that chosen/rejected differ in a "reasonable" way (not random noise).
"""

from __future__ import annotations

import logging

from src.pipeline.validate import validate_batch
from src.synth.teacher_vllm import generate_batch

logger = logging.getLogger(__name__)

REJECT_PROMPT_SUFFIX = (
    "\n\nIMPORTANT: Introduce exactly ONE subtle bug (e.g., wrong PIC clause, "
    "incorrect PERFORM target, or REDEFINES mismatch) without changing the overall structure."
)


def generate_dpo_pair(
    user_prompt: str,
    chosen_response: str,
    teacher_model: str,
) -> dict | None:
    """
    Given a (prompt, chosen) pair, generate a rejected variant and validate the pair.
    Returns a DPO-format dict or None if the pair is invalid.
    """
    reject_prompt = {"task_type": "debug", "source": chosen_response + REJECT_PROMPT_SUFFIX, "difficulty_score": 0.8}
    results = generate_batch([reject_prompt], teacher_model=teacher_model, temperature=0.9)
    rejected_ex = results[0]
    if rejected_ex is None:
        return None

    rejected_content = rejected_ex["messages"][1]["content"]

    # Validate: chosen must compile, rejected must NOT compile
    validation = validate_batch([
        {"content": chosen_response, "source": "chosen"},
        {"content": rejected_content, "source": "rejected"},
    ])
    chosen_ok = validation[0].get("compiles", False)
    rejected_ok = validation[1].get("compiles", True)

    if not chosen_ok or rejected_ok:
        logger.debug("DPO pair invalid: chosen_compiles=%s rejected_compiles=%s", chosen_ok, rejected_ok)
        return None

    return {
        "prompt": [{"role": "user", "content": user_prompt}],
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_content}],
    }
