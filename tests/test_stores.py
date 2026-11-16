from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
from tests.conftest import sample_chunk

from rag_service.catalog import MemoryChunkCatalog
from rag_service.models import Chunk
from rag_service.pgvector import PgChunkCatalog, PgVectorStore, vector_schema
from rag_service.textutil import hashing_embed
from rag_service.vertex_search import VertexVectorSearchStore


class RecordingExecutor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.many: list[tuple[str, list[tuple[object, ...]]]] = []
        self.fetch_rows: list[dict[str, Any]] = []

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[RecordingExecutor]:
        yield self

    async def execute(self, sql: str, *args: object) -> None:
        self.statements.append((sql, args))

    async def executemany(self, sql: str, args_seq: Sequence[tuple[object, ...]]) -> None:
        self.many.append((sql, list(args_seq)))

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        self.statements.append((sql, args))
        return list(self.fetch_rows)

    async def close(self) -> None:
        return None


def test_vector_schema_uses_cosine_hnsw() -> None:
    statements = " ".join(vector_schema(128))
    assert "vector(128)" in statements
    assert "vector_cosine_ops" in statements


@pytest.mark.asyncio
async def test_pgvector_replace_and_row_mapping() -> None:
    executor = RecordingExecutor()
    store = PgVectorStore(executor, 8)
    await store.ensure_ready()
    chunk = sample_chunk()
    await store.replace_document(chunk.document_id, [chunk], [[0.1] * 8])
    assert any(sql.startswith("DELETE FROM chunks") for sql, _args in executor.statements)
    assert executor.many
    assert "ON CONFLICT" in executor.many[0][0]
    executor.fetch_rows = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_hash": chunk.document_hash,
            "title": chunk.title,
            "content": chunk.text,
            "chunk_index": 0,
            "metadata": {"source": "sample"},
            "score": 0.42,
        }
    ]
    hits = await store.query([0.1] * 8, 3)
    assert hits[0].score == pytest.approx(0.42)
    assert hits[0].chunk.text == chunk.text
    assert hits[0].chunk.metadata["source"] == "sample"
    executor.fetch_rows = [{"document_hash": "abc"}]
    assert await store.document_hash(chunk.document_id) == "abc"


@pytest.mark.asyncio
async def test_pg_catalog_deletes_then_inserts() -> None:
    executor = RecordingExecutor()
    catalog = PgChunkCatalog(executor)
    chunk = sample_chunk("access")
    await catalog.replace(chunk.document_id, [chunk])
    assert any("DELETE FROM chunk_text" in sql for sql, _args in executor.statements)
    assert "chunk_text" in executor.many[0][0]


class FakeNeighbors:
    def __init__(self) -> None:
        self.points: dict[str, tuple[list[float], str]] = {}
        self.deleted: list[str] = []

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.points[chunk.chunk_id] = (vector, chunk.document_id)

    async def delete(self, chunk_ids: list[str]) -> None:
        self.deleted.extend(chunk_ids)
        for chunk_id in chunk_ids:
            self.points.pop(chunk_id, None)

    async def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        from rag_service.textutil import cosine

        scored = [
            (chunk_id, cosine(vector, stored)) for chunk_id, (stored, _doc) in self.points.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


@pytest.mark.asyncio
async def test_vertex_store_hydrates_from_the_catalog_and_skips_missing_ids() -> None:
    neighbors = FakeNeighbors()
    catalog = MemoryChunkCatalog()
    store = VertexVectorSearchStore(neighbors, catalog)
    kept = sample_chunk("expenses", "The mealcap is seventyfive dollars.")
    dropped = sample_chunk("old", "old policy text")
    await store.replace_document("expenses", [kept], [hashing_embed(kept.text, 16)])
    await neighbors.upsert([dropped], [hashing_embed(dropped.text, 16)])
    hits = await store.query(hashing_embed("mealcap seventyfive", 16), 4)
    assert [hit.chunk.document_id for hit in hits] == ["expenses"]
    await store.replace_document("expenses", [], [])
    assert kept.chunk_id in neighbors.deleted
    assert await store.document_hash("expenses") is None
