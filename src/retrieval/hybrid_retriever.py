"""
Hybrid retriever: combines vector search + graph traversal + lazy summarization.
Entry point for inference-time context injection.
"""

from __future__ import annotations

from src.retrieval.lazy_graphrag import LazyGraphRAG
from src.retrieval.vector_index import CobolVectorIndex
from src.retrieval.graph_builder import CobolGraphBuilder


def build_retriever(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    summarizer_fn=None,
) -> LazyGraphRAG:
    vector_index = CobolVectorIndex(host=qdrant_host, port=qdrant_port)
    graph = CobolGraphBuilder(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    return LazyGraphRAG(vector_index=vector_index, graph=graph, summarizer_fn=summarizer_fn)
