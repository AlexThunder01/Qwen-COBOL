"""
Qdrant vector index for COBOL snippets.
Embedding model: BAAI/bge-m3 (multilingual, free, runs locally).

Requires Qdrant running via docker/qdrant/docker-compose.yml.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "cobol_snippets"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # bge-m3 output dim


class CobolVectorIndex:
    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        self._client = QdrantClient(host=host, port=port)
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", COLLECTION_NAME)

    def index_snippet(self, snippet_id: str, text: str, metadata: dict | None = None) -> None:
        vector = self._model.encode(text, normalize_embeddings=True).tolist()
        point = PointStruct(
            id=abs(hash(snippet_id)) % (2**53),
            vector=vector,
            payload={"text": text[:2000], "snippet_id": snippet_id, **(metadata or {})},
        )
        self._client.upsert(collection_name=COLLECTION_NAME, points=[point])

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        vector = self._model.encode(query, normalize_embeddings=True).tolist()
        results = self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        return [{"score": r.score, **r.payload} for r in results]
