"""
Batch GnuCOBOL compilation check.
Reuses validate.py logic but adds metrics reporting.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.pipeline.validate import validate_batch

logger = logging.getLogger(__name__)


def compile_rate(snippets: Sequence[str]) -> float:
    """Return fraction of snippets that compile with GnuCOBOL."""
    records = [{"content": s, "source": "eval"} for s in snippets]
    validated = validate_batch(list(records))
    n_ok = sum(1 for r in validated if r.get("compiles"))
    rate = n_ok / len(validated) if validated else 0.0
    logger.info("Compile rate: %d/%d (%.1f%%)", n_ok, len(validated), rate * 100)
    return rate
