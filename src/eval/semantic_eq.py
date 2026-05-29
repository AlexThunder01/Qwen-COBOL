"""
Semantic equivalence check for COBOL translation tasks.
When both original and translated programs compile, run both and compare stdout.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def programs_equivalent(source_a: str, source_b: str, timeout: int = 10) -> bool | None:
    """
    Compile and run both programs; return True if stdout matches.
    Returns None if either doesn't compile (can't determine equivalence).
    """
    out_a = _run_cobol(source_a, timeout)
    out_b = _run_cobol(source_b, timeout)
    if out_a is None or out_b is None:
        return None
    return out_a.strip() == out_b.strip()


def _run_cobol(source: str, timeout: int) -> str | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "prog.cob"
        exe_path = Path(tmpdir) / "prog"
        src_path.write_text(source)

        compile_result = subprocess.run(
            ["cobc", "-x", str(src_path), "-o", str(exe_path)],
            capture_output=True,
            timeout=timeout,
        )
        if compile_result.returncode != 0:
            return None

        run_result = subprocess.run(
            [str(exe_path)],
            capture_output=True,
            timeout=timeout,
        )
        return run_result.stdout.decode(errors="replace")
