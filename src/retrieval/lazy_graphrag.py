"""
LazyGraphRAG — community summarization on demand.
Inspired by Microsoft LazyGraphRAG (2024): only summarize graph communities
that are relevant to the query, rather than pre-computing all summaries.

Pipeline:
  1. Vector search → top_k candidate snippets
  2. Graph traversal → expand to related programs/paragraphs (BFS depth=2)
  3. Lazy summarization → LLM summarizes only the retrieved community
  4. Return context string for injection into the user prompt
"""

from __future__ import annotations

import logging
from typing import Any

from src.retrieval.vector_index import CobolVectorIndex
from src.retrieval.graph_builder import CobolGraphBuilder

logger = logging.getLogger(__name__)


class LazyGraphRAG:
    def __init__(
        self,
        vector_index: CobolVectorIndex,
        graph: CobolGraphBuilder,
        summarizer_fn: Any,  # callable(text: str) -> str — points to vLLM client
        top_k_vector: int = 10,
        top_k_graph: int = 5,
        max_community_size: int = 50,
    ) -> None:
        self._vec = vector_index
        self._graph = graph
        self._summarize = summarizer_fn
        self._top_k_vector = top_k_vector
        self._top_k_graph = top_k_graph
        self._max_community_size = max_community_size

    def retrieve(self, query: str) -> str:
        # Step 1: vector search
        hits = self._vec.search(query, top_k=self._top_k_vector)
        program_names = [h["snippet_id"] for h in hits if "snippet_id" in h]

        # Step 2: graph expansion
        community = self._expand_graph(program_names)

        # Step 3: lazy summarization of the community
        community_text = "\n\n---\n\n".join(
            f"Program: {p['name']}\n{p.get('snippet', '')}" for p in community
        )
        if not community_text.strip():
            return ""

        summary = self._summarize(
            f"Summarize the following COBOL programs for the query: '{query}'\n\n{community_text}"
        )
        return summary

    def _expand_graph(self, program_names: list[str]) -> list[dict]:
        expanded: dict[str, dict] = {}
        with self._graph._driver.session() as session:
            for name in program_names:
                result = session.run(
                    """
                    MATCH (p:Program {name: $name})
                    OPTIONAL MATCH (p)-[:CALLS|PERFORMS*1..2]-(related:Program)
                    RETURN p, collect(related)[..$limit] AS related
                    """,
                    name=name, limit=self._top_k_graph,
                )
                for record in result:
                    node = record["p"]
                    expanded[node["name"]] = dict(node)
                    for rel in record["related"] or []:
                        if rel and len(expanded) < self._max_community_size:
                            expanded[rel["name"]] = dict(rel)

        return list(expanded.values())
