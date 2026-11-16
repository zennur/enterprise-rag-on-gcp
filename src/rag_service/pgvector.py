"""AlloyDB / PostgreSQL pgvector store, and a text catalog for Vector Search.

The runtime user can create the extension and tables on startup. That is a
convenience for v1. A locked-down database should run the same statements from
a migration role and keep the service account on DML only.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from rag_service.catalog import ChunkCatalog
from rag_service.models import Chunk, ScoredChunk

_INSERT_VECTOR = """
INSERT INTO chunks (
    chunk_id, document_id, document_hash, title, content, chunk_index, metadata, embedding
) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::vector)
ON CONFLICT (chunk_id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    document_hash = EXCLUDED.document_hash,
    title = EXCLUDED.title,
    content = EXCLUDED.content,
    chunk_index = EXCLUDED.chunk_index,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding
"""

_INSERT_TEXT = """
INSERT INTO chunk_text (
    chunk_id, document_id, document_hash, title, content, chunk_index, metadata
) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
ON CONFLICT (chunk_id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    document_hash = EXCLUDED.document_hash,
    title = EXCLUDED.title,
    content = EXCLUDED.content,
    chunk_index = EXCLUDED.chunk_index,
    metadata = EXCLUDED.metadata
"""

_QUERY = """
SELECT chunk_id, document_id, document_hash, title, content, chunk_index, metadata,
       1 - (embedding <=> $1::vector) AS score
FROM chunks
ORDER BY embedding <=> $1::vector
LIMIT $2
"""


class PgConnection(Protocol):
    async def execute(self, sql: str, *args: object) -> None: ...

    async def executemany(self, sql: str, args_seq: Sequence[tuple[object, ...]]) -> None: ...

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]: ...


class PgExecutor(PgConnection, Protocol):
    def transaction(self) -> AbstractAsyncContextManager[PgConnection]: ...

    async def close(self) -> None: ...


def vector_schema(dimensions: int) -> list[str]:
    _check_dimensions(dimensions)
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding vector({dimensions}) NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id)",
        (
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        ),
    ]


def catalog_schema() -> list[str]:
    return [
        """
        CREATE TABLE IF NOT EXISTS chunk_text (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """,
        "CREATE INDEX IF NOT EXISTS chunk_text_document_id_idx ON chunk_text (document_id)",
    ]


class PgVectorStore:
    """Cosine search in AlloyDB or any Postgres with pgvector.

    ``generation`` is local to this process. Cache staleness across instances
    is bounded by the response-cache TTL, not by this counter.
    """

    def __init__(self, executor: PgExecutor, dimensions: int) -> None:
        self._executor = executor
        self._dimensions = dimensions
        self._generation = 0

    def generation(self) -> int:
        return self._generation

    async def replace_document(
        self, document_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        rows = [_vector_row(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)]
        async with self._executor.transaction() as connection:
            await connection.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
            if rows:
                await connection.executemany(_INSERT_VECTOR, rows)
        self._generation += 1

    async def query(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        rows = await self._executor.fetch(_QUERY, _vector_literal(vector), top_k)
        return [ScoredChunk(chunk=_chunk_from_row(row), score=float(row["score"])) for row in rows]

    async def document_hash(self, document_id: str) -> str | None:
        rows = await self._executor.fetch(
            "SELECT document_hash FROM chunks WHERE document_id = $1 LIMIT 1",
            document_id,
        )
        if not rows:
            return None
        return str(rows[0]["document_hash"])

    async def ping(self) -> bool:
        rows = await self._executor.fetch("SELECT 1 AS ok")
        return bool(rows)

    async def ensure_ready(self) -> None:
        for statement in vector_schema(self._dimensions):
            await self._executor.execute(statement)

    async def close(self) -> None:
        await self._executor.close()


class PgChunkCatalog(ChunkCatalog):
    def __init__(self, executor: PgExecutor) -> None:
        self._executor = executor

    async def ids_for_document(self, document_id: str) -> list[str]:
        rows = await self._executor.fetch(
            "SELECT chunk_id FROM chunk_text WHERE document_id = $1",
            document_id,
        )
        return [str(row["chunk_id"]) for row in rows]

    async def replace(self, document_id: str, chunks: list[Chunk]) -> None:
        rows = [_text_row(chunk) for chunk in chunks]
        async with self._executor.transaction() as connection:
            await connection.execute("DELETE FROM chunk_text WHERE document_id = $1", document_id)
            if rows:
                await connection.executemany(_INSERT_TEXT, rows)

    async def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        rows = await self._executor.fetch(
            "SELECT chunk_id, document_id, document_hash, title, content, chunk_index, metadata "
            "FROM chunk_text WHERE chunk_id = ANY($1::text[])",
            chunk_ids,
        )
        return {str(row["chunk_id"]): _chunk_from_row(row) for row in rows}

    async def document_hash(self, document_id: str) -> str | None:
        rows = await self._executor.fetch(
            "SELECT document_hash FROM chunk_text WHERE document_id = $1 LIMIT 1",
            document_id,
        )
        if not rows:
            return None
        return str(rows[0]["document_hash"])

    async def ping(self) -> bool:
        rows = await self._executor.fetch("SELECT 1 AS ok")
        return bool(rows)

    async def ensure_ready(self) -> None:
        for statement in catalog_schema():
            await self._executor.execute(statement)

    async def close(self) -> None:
        await self._executor.close()


class AsyncpgExecutor:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None

    async def _pool_or_create(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as exc:
                raise RuntimeError("Install the gcp extra to use pgvector") from exc
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
        return self._pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[PgConnection]:
        pool = await self._pool_or_create()
        async with pool.acquire() as connection, connection.transaction():
            yield _ConnectionExecutor(connection)

    async def execute(self, sql: str, *args: object) -> None:
        async with self.transaction() as connection:
            await connection.execute(sql, *args)

    async def executemany(self, sql: str, args_seq: Sequence[tuple[object, ...]]) -> None:
        async with self.transaction() as connection:
            await connection.executemany(sql, args_seq)

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        pool = await self._pool_or_create()
        async with pool.acquire() as connection:
            rows = await connection.fetch(sql, *args)
        return [dict(row) for row in rows]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


class _ConnectionExecutor:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def execute(self, sql: str, *args: object) -> None:
        await self._connection.execute(sql, *args)

    async def executemany(self, sql: str, args_seq: Sequence[tuple[object, ...]]) -> None:
        if args_seq:
            await self._connection.executemany(sql, list(args_seq))

    async def fetch(self, sql: str, *args: object) -> list[dict[str, Any]]:
        rows = await self._connection.fetch(sql, *args)
        return [dict(row) for row in rows]


def _check_dimensions(dimensions: int) -> None:
    if dimensions < 8 or dimensions > 3072:
        raise ValueError("embedding dimensions are out of range")


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(format(value, ".8f") for value in values) + "]"


def _metadata(value: object) -> dict[str, str]:
    if isinstance(value, str):
        loaded = json.loads(value)
        value = loaded
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    return {}


def _chunk_from_row(row: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        document_hash=str(row["document_hash"]),
        title=str(row["title"]),
        text=str(row["content"]),
        index=int(row["chunk_index"]),
        metadata=_metadata(row.get("metadata")),
    )


def _vector_row(chunk: Chunk, vector: list[float]) -> tuple[object, ...]:
    return (
        chunk.chunk_id,
        chunk.document_id,
        chunk.document_hash,
        chunk.title,
        chunk.text,
        chunk.index,
        json.dumps(chunk.metadata, sort_keys=True),
        _vector_literal(vector),
    )


def _text_row(chunk: Chunk) -> tuple[object, ...]:
    return (
        chunk.chunk_id,
        chunk.document_id,
        chunk.document_hash,
        chunk.title,
        chunk.text,
        chunk.index,
        json.dumps(chunk.metadata, sort_keys=True),
    )
