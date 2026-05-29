"""
COBOL AST parser using tree-sitter-language-pack.

Fallback: if tree-sitter COBOL grammar is unavailable or crashes on a dialect,
returns None so the record is kept as raw text (still usable for CPT).

W1 task: smoke-test this on 10 real COBOL files with different dialects.

tree-sitter 0.25.x API notes:
  - Parser.parse(str) → Tree
  - Tree.root_node()  → Node  (method call, not property)
  - Node.kind         → str   (replaces Node.type)
  - Node.child(i)     → Node  (replaces Node.children list)
  - Node.child_count  → int
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from tree_sitter_language_pack import get_parser

    _PARSER = get_parser("cobol")
    TREE_SITTER_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    logger.warning("tree-sitter COBOL grammar unavailable (%s) — parse will return None", exc)
    TREE_SITTER_AVAILABLE = False
    _PARSER = None


def parse_cobol(source: str) -> dict[str, Any] | None:
    """Return a minimal AST summary or None if parsing fails."""
    if not TREE_SITTER_AVAILABLE or _PARSER is None:
        return None

    try:
        tree = _PARSER.parse(source)
        root = tree.root_node()  # method call in tree-sitter 0.25.x
        return {
            "has_errors": root.has_error(),
            "node_count": root.child_count(),
            "divisions": _collect_divisions(root),
            "paragraphs": _collect_paragraphs(root),
            "calls": _collect_calls(root),
            "redefines_count": _count_node_type(root, "redefines_clause"),
            "occurs_count": _count_node_type(root, "occurs_clause"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("parse_cobol failed: %s", exc)
        return None


def _collect_divisions(root: Any) -> list[str]:
    divisions = []
    for i in range(root.child_count()):
        child = root.child(i)
        if child and "division" in child.kind():
            divisions.append(child.kind())
    return divisions


def _collect_paragraphs(root: Any) -> list[str]:
    paragraphs: list[str] = []
    _walk(root, "paragraph", lambda n: paragraphs.append(
        n.child_by_field_name("name").to_sexp() if n.child_by_field_name("name") else "?"
    ))
    return paragraphs[:100]


def _collect_calls(root: Any) -> list[str]:
    calls: list[str] = []
    _walk(root, "call_statement", lambda n: calls.append(
        n.child_by_field_name("program").to_sexp() if n.child_by_field_name("program") else "?"
    ))
    return calls[:50]


def _count_node_type(root: Any, node_type: str) -> int:
    count = [0]
    _walk(root, node_type, lambda _: count.__setitem__(0, count[0] + 1))
    return count[0]


def _walk(node: Any, target_kind: str, callback: Any) -> None:
    if node.kind() == target_kind:
        callback(node)
    for i in range(node.child_count()):
        child = node.child(i)
        if child:
            _walk(child, target_kind, callback)
