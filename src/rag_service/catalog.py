"""Chunk text catalog used beside Vertex AI Vector Search.

Vector Search returns neighbor ids, not passages. The catalog is the source of
truth for text and for the document hash that makes redelivery idempotent.
"""

from __future__ import annotations

from typing import Protocol

from rag_service.models import Chunk


class ChunkCatalog(Protocol):
    async def ids_for_document(self, document_id: str) -> list[str]: ...

    async def replace(self, document_id: str, chunks: list[Chunk]) -> None: ...

    async def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]: ...

    async def document_hash(self, document_id: str) -> str | None: ...

    async def ping(self) -> bool: ...

    async def ensure_ready(self) -> None: ...

    async def close(self) -> None: ...


class MemoryChunkCatalog:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    async def ids_for_document(self, document_id: str) -> list[str]:
        return [
            chunk_id for chunk_id, chunk in self._chunks.items() if chunk.document_id == document_id
        ]

    async def replace(self, document_id: str, chunks: list[Chunk]) -> None:
        stale = await self.ids_for_document(document_id)
        for chunk_id in stale:
            del self._chunks[chunk_id]
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    async def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        return {
            chunk_id: self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks
        }

    async def document_hash(self, document_id: str) -> str | None:
        for chunk in self._chunks.values():
            if chunk.document_id == document_id:
                return chunk.document_hash
        return None

    async def ping(self) -> bool:
        return True

    async def ensure_ready(self) -> None:
        return None

    async def close(self) -> None:
        return None
