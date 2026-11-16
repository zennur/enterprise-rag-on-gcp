"""Vector store contract and the in-process store used for local runs and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_service.models import Chunk, ScoredChunk
from rag_service.textutil import cosine


class VectorStore(Protocol):
    def generation(self) -> int: ...

    async def replace_document(
        self, document_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None: ...

    async def query(self, vector: list[float], top_k: int) -> list[ScoredChunk]: ...

    async def document_hash(self, document_id: str) -> str | None: ...

    async def ping(self) -> bool: ...

    async def ensure_ready(self) -> None: ...

    async def close(self) -> None: ...


@dataclass
class _Row:
    chunk: Chunk
    vector: list[float]


class MemoryVectorStore:
    """Brute-force cosine search. Fine for local corpora, not for millions of chunks."""

    def __init__(self) -> None:
        self._rows: dict[str, _Row] = {}
        self._generation = 0

    def generation(self) -> int:
        return self._generation

    async def replace_document(
        self, document_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        stale = [
            chunk_id for chunk_id, row in self._rows.items() if row.chunk.document_id == document_id
        ]
        for chunk_id in stale:
            del self._rows[chunk_id]
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._rows[chunk.chunk_id] = _Row(chunk=chunk, vector=list(vector))
        self._generation += 1

    async def query(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=row.chunk, score=cosine(vector, row.vector))
            for row in self._rows.values()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    async def document_hash(self, document_id: str) -> str | None:
        for row in self._rows.values():
            if row.chunk.document_id == document_id:
                return row.chunk.document_hash
        return None

    async def ping(self) -> bool:
        return True

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None
