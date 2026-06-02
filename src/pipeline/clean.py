"""
COBOL source normalization:
- Strip fixed-format column markers (cols 1-6 sequence number, col 7 indicator, cols 73-80 identification)
- Convert to free-format or normalized fixed-format for training
- Minimal dialect normalization for GnuCOBOL compatibility
"""

from __future__ import annotations

import re


# COBOL fixed format: col 1-6 sequence, col 7 indicator (*/-/D/space), cols 8-72 content, 73-80 ident
_FIXED_LINE_RE = re.compile(r"^.{0,6}(.)(.{0,65}).*$")

# Lines that are pure comments (col 7 = '*' or '/')
_COMMENT_INDICATOR = frozenset({"*", "/"})

# Minimum non-empty content length after cleaning to keep a record
MIN_CONTENT_BYTES = 50
MAX_CONTENT_BYTES = 16_384 * 4  # ~64K chars; longer programs get chunked at training time


# Marker che identificano COBOL valido (incl. copybook con sole SELECT/level).
_COBOL_MARKERS = re.compile(
    r"\b(IDENTIFICATION|DATA|PROCEDURE|ENVIRONMENT)\s+DIVISION\b"
    r"|\bPIC(TURE)?\s+[X9SVAP(]"
    r"|\b(PERFORM|MOVE|WORKING-STORAGE|COMPUTE|DISPLAY|GOBACK|STOP\s+RUN|EVALUATE)\b"
    r"|\b(SELECT\s+\w+\s+ASSIGN|FILE-CONTROL|ASSIGN\s+TO|ORGANIZATION\s+IS)\b",
    re.IGNORECASE,
)
_MARKUP_TAG = re.compile(r"<[a-zA-Z/!]")


def _is_cobol(content: str) -> bool:
    """True se il record è COBOL plausibile (non script/markup/dati misclassificati)."""
    if not _COBOL_MARKERS.search(content):
        return False
    if len(_MARKUP_TAG.findall(content)) > 20:  # XML/HTML travestito
        return False
    head = content[:2000]
    if head and sum(1 for ch in head if ord(ch) > 126) / len(head) > 0.40:
        return False  # encoding rotto/binario (soglia alta per tenere commenti CJK)
    return True


def clean_record(raw: dict) -> dict | None:
    """Normalize a raw COBOL source record. Returns None if the record should be dropped."""
    content = raw.get("content", "")
    if not isinstance(content, str):
        return None

    content = _strip_columns(content)
    content = _normalize_whitespace(content)

    if len(content.encode()) < MIN_CONTENT_BYTES:
        return None
    if not _is_cobol(content):
        return None

    # Cap dimensione: tronca i file giganti per limitare la dominanza di singoli
    # codebase (es. ORCA) e per il chunking a training time.
    if len(content) > MAX_CONTENT_BYTES:
        content = content[:MAX_CONTENT_BYTES]

    return {**raw, "content": content}


def _strip_columns(source: str) -> str:
    """Remove fixed-format sequence (cols 1-6) and identification (cols 73-80) fields."""
    lines = source.splitlines()
    cleaned: list[str] = []

    for line in lines:
        if len(line) < 7:
            # Short line — likely already free-format or blank
            cleaned.append(line)
            continue

        indicator = line[6]
        content = line[7:72].rstrip() if len(line) > 7 else ""

        if indicator in _COMMENT_INDICATOR:
            # Preserve comments — useful signal for code explanation tasks
            cleaned.append(f"      {indicator} {content}")
        else:
            cleaned.append(f"       {content}")

    return "\n".join(cleaned)


def _normalize_whitespace(source: str) -> str:
    """Collapse runs of blank lines to at most two, strip trailing spaces."""
    lines = source.splitlines()
    result: list[str] = []
    blank_run = 0

    for line in lines:
        stripped = line.rstrip()
        if stripped == "":
            blank_run += 1
            if blank_run <= 2:
                result.append("")
        else:
            blank_run = 0
            result.append(stripped)

    return "\n".join(result).strip()
