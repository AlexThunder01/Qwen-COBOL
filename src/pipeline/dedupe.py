"""
Near-duplicate detection using MinHash LSH (datasketch).
Jaccard similarity threshold 0.7 — configurable.

Skip deduplication if using `the-stack-v2-dedup` as primary source (already deduped).
"""

from __future__ import annotations

import re
from datasketch import MinHash, MinHashLSH


_TOKENIZE_RE = re.compile(r"[A-Z0-9\-_]+")


class MinHashDeduper:
    def __init__(self, threshold: float = 0.7, num_perm: int = 128) -> None:
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._num_perm = num_perm
        self._counter = 0

    def _make_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self._num_perm)
        for token in _TOKENIZE_RE.findall(text.upper()):
            m.update(token.encode())
        return m

    def is_duplicate(self, text: str) -> bool:
        m = self._make_minhash(text)
        return bool(self._lsh.query(m))

    def add(self, text: str) -> None:
        key = str(self._counter)
        self._counter += 1
        m = self._make_minhash(text)
        self._lsh.insert(key, m)
