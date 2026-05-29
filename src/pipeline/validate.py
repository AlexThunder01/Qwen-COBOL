"""
GnuCOBOL batch compiler validation.

Runs `cobc -fsyntax-only` on each snippet in a temp file.
Uses multiprocessing.Pool for parallelism (target: 32 CPUs on Kaggle).

NIST test suite files are assumed valid and skipped to save time.
"""

from __future__ import annotations

import logging
import multiprocessing
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_COBC_CMD = ["cobc", "-fsyntax-only", "-free"]  # -free for free-format; adjust if fixed-format
_TIMEOUT_SEC = 10


def _validate_single(record: dict) -> dict:
    """Add 'compiles' bool to record. Does not raise."""
    if record.get("source") == "nist":
        return {**record, "compiles": True}

    content = record.get("content", "")
    try:
        with tempfile.NamedTemporaryFile(suffix=".cob", mode="w", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        result = subprocess.run(
            [*_COBC_CMD, tmp_path],
            capture_output=True,
            timeout=_TIMEOUT_SEC,
        )
        compiles = result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.debug("cobc timeout on record from %s", record.get("source"))
        compiles = False
    except Exception as exc:  # noqa: BLE001
        logger.debug("cobc error: %s", exc)
        compiles = False
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {**record, "compiles": compiles}


def validate_batch(records: list[dict], workers: int | None = None) -> list[dict]:
    """Validate a batch of records in parallel. Returns all records with 'compiles' set."""
    if workers is None:
        workers = min(multiprocessing.cpu_count(), 32)

    with multiprocessing.Pool(workers) as pool:
        validated = pool.map(_validate_single, records)

    n_ok = sum(1 for r in validated if r.get("compiles"))
    logger.info("validate_batch: %d/%d compile", n_ok, len(validated))
    return validated
