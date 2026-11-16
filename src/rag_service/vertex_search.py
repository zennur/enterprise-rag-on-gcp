"""Vertex AI Vector Search adapter.

The index is an approximate nearest-neighbor tier. Passage text lives in a
catalog because a neighbor response is an id and a distance. Streaming updates
are briefly inconsistent, so a neighbor whose id is missing from the catalog
is skipped rather than failing the query.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import structlog

from rag_service.catalog import ChunkCatalog
from rag_service.config import Settings
from rag_service.models import Chunk, ScoredChunk
from rag_service.resilience import ProviderError, TransientError, call_with_retry, is_transient
from rag_service.textutil import distance_to_score

logger = structlog.get_logger(__name__)


class NeighborIndex(Protocol):
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    async def delete(self, chunk_ids: list[str]) -> None: ...

    async def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]: ...


class VertexVectorSearchStore:
    def __init__(self, index: NeighborIndex, catalog: ChunkCatalog) -> None:
        self._index = index
        self._catalog = catalog
        self._generation = 0

    def generation(self) -> int:
        return self._generation

    async def replace_document(
        self, document_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        # The catalog hash is the commit point. Index writes happen first so a
        # crash retries the document instead of skipping a half-updated index.
        old_ids = set(await self._catalog.ids_for_document(document_id))
        if chunks:
            await self._index.upsert(chunks, vectors)
        stale = [chunk_id for chunk_id in old_ids if chunk_id not in {c.chunk_id for c in chunks}]
        if stale:
            await self._index.delete(stale)
        await self._catalog.replace(document_id, chunks)
        self._generation += 1

    async def query(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        neighbors = await self._index.search(vector, top_k)
        found = await self._catalog.get_many([chunk_id for chunk_id, _score in neighbors])
        scored: list[ScoredChunk] = []
        for chunk_id, score in neighbors:
            chunk = found.get(chunk_id)
            if chunk is None:
                continue
            scored.append(ScoredChunk(chunk=chunk, score=score))
        return scored

    async def document_hash(self, document_id: str) -> str | None:
        return await self._catalog.document_hash(document_id)

    async def ping(self) -> bool:
        return await self._catalog.ping()

    async def ensure_ready(self) -> None:
        await self._catalog.ensure_ready()

    async def close(self) -> None:
        await self._catalog.close()


class VertexNeighborIndex:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        await self._run(lambda: self._upsert_sync(chunks, vectors))

    async def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        await self._run(lambda: self._delete_sync(chunk_ids))

    async def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        result = await self._run(lambda: self._search_sync(vector, top_k))
        return [(str(chunk_id), float(score)) for chunk_id, score in result]

    async def _run(self, func: Any) -> Any:
        async def attempt() -> Any:
            return await asyncio.to_thread(func)

        try:
            return await call_with_retry(
                attempt,
                attempts=self._settings.provider_attempts,
                timeout_s=self._settings.provider_timeout_s,
                backoff_s=self._settings.provider_backoff_s,
            )
        except TimeoutError as exc:
            raise ProviderError("vector search timed out") from exc

    def _upsert_sync(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        index = self._index()
        datapoints = []
        types = self._types()
        for chunk, vector in zip(chunks, vectors, strict=True):
            datapoints.append(
                types.IndexDatapoint(
                    datapoint_id=chunk.chunk_id,
                    feature_vector=vector,
                    restricts=[
                        types.IndexDatapoint.Restriction(
                            namespace="document_id",
                            allow_list=[chunk.document_id],
                        )
                    ],
                )
            )
        self._call(lambda: index.upsert_datapoints(datapoints=datapoints))

    def _delete_sync(self, chunk_ids: list[str]) -> None:
        index = self._index()
        self._call(lambda: index.remove_datapoints(datapoint_ids=chunk_ids))

    def _search_sync(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        endpoint = self._endpoint()
        responses = self._call(
            lambda: endpoint.find_neighbors(
                deployed_index_id=self._settings.vector_search_deployed_index_id,
                queries=[vector],
                num_neighbors=top_k,
            )
        )
        neighbors = responses[0] if responses else []
        scored: list[tuple[str, float]] = []
        for neighbor in neighbors:
            distance = float(getattr(neighbor, "distance", 0.0) or 0.0)
            scored.append(
                (
                    str(neighbor.id),
                    distance_to_score(distance, self._settings.vector_search_distance),
                )
            )
        return scored

    def _index(self) -> Any:
        aiplatform = self._sdk()
        return aiplatform.MatchingEngineIndex(index_name=self._settings.vector_search_index)

    def _endpoint(self) -> Any:
        aiplatform = self._sdk()
        return aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self._settings.vector_search_endpoint
        )

    def _sdk(self) -> Any:
        try:
            from google.cloud import aiplatform
        except ImportError as exc:
            raise ProviderError("Install the gcp extra to use Vertex AI Vector Search") from exc
        aiplatform.init(project=self._settings.gcp_project, location=self._settings.gcp_location)
        return aiplatform

    def _types(self) -> Any:
        try:
            from google.cloud import aiplatform
        except ImportError as exc:
            raise ProviderError("Install the gcp extra to use Vertex AI Vector Search") from exc
        return aiplatform.compat.types.index_v1beta1

    def _call(self, func: Any) -> Any:
        try:
            return func()
        except Exception as exc:
            if is_transient(exc):
                raise TransientError(str(exc)) from exc
            raise
