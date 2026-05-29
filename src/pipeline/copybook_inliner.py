"""
Inject COPY member (copybook) content into WORKING-STORAGE SECTION of a COBOL program.

When a program contains `COPY <name>`, we attempt to inline the copybook text
so the model sees a self-contained snippet. Missing copybooks are left as-is.
"""

from __future__ import annotations

import re
from pathlib import Path


_COPY_RE = re.compile(r"^\s+COPY\s+([A-Z0-9\-]+)\s*\.", re.MULTILINE | re.IGNORECASE)


def inline_copybooks(source: str, copybook_dirs: list[Path]) -> str:
    """Replace COPY statements with the copybook content, if found."""

    def _replace(match: re.Match) -> str:
        name = match.group(1).upper()
        for d in copybook_dirs:
            for ext in ("", ".cpy", ".CPY", ".cob", ".COB"):
                candidate = d / f"{name}{ext}"
                if candidate.exists():
                    return f"      * -- BEGIN COPY {name} --\n{candidate.read_text(errors='replace')}\n      * -- END COPY {name} --"
        # Copybook not found — keep original COPY statement
        return match.group(0)

    return _COPY_RE.sub(_replace, source)
