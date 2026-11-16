"""Composition root. Tests pass fakes instead of reading GCP credentials."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from rag_service.agent import RagAgent
from rag_service.cache import ResponseCache
from rag_service.catalog import MemoryChunkCatalog
from rag_service.config import Settings, configuration_error
from rag_service.ingest import IngestService
from rag_service.messaging import DocumentPublisher, InlinePublisher, PubSubPublisher
from rag_service.pgvector import AsyncpgExecutor, PgChunkCatalog, PgVectorStore
from rag_service.providers import (
    EmbeddingProvider,
    GenerationProvider,
    build_embedder,
    build_generator,
)
from rag_service.resilience import InflightGate, TokenBucketLimiter
from rag_service.vectorstore import MemoryVectorStore, VectorStore
from rag_service.vertex_search import VertexNeighborIndex, VertexVectorSearchStore

logger = structlog.get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    embedder: EmbeddingProvider
    generator: GenerationProvider
    store: VectorStore
    publisher: DocumentPublisher
    cache: ResponseCache
    limiter: TokenBucketLimiter
    gate: InflightGate
    agent: RagAgent
    ingest: IngestService
    ready_reason: str = "starting"

    async def startup(self) -> None:
        problem = configuration_error(self.settings)
        if problem:
            self.ready_reason = problem
            logger.error("configuration_invalid", reason=problem)
            return
        if self.settings.vector_store == "vertex" and not self.settings.database_url:
            logger.warning(
                "vertex_catalog_is_process_local",
                detail="Chunk text will not survive instance replacement without DATABASE_URL.",
            )
        try:
            await self.store.ensure_ready()
        except Exception:
            self.ready_reason = "vector store is not ready"
            logger.exception("vector_store_not_ready")
            return
        self.ready_reason = "ok"

    async def ready(self) -> tuple[bool, str]:
        if self.ready_reason != "ok":
            return False, self.ready_reason
        try:
            ok = await self.store.ping()
        except Exception:
            logger.exception("ready_ping_failed")
            return False, "vector store ping failed"
        if not ok:
            return False, "vector store ping failed"
        return True, "ok"

    async def shutdown(self) -> None:
        await self.store.close()


def build_container(
    settings: Settings,
    *,
    embedder: EmbeddingProvider | None = None,
    generator: GenerationProvider | None = None,
    store: VectorStore | None = None,
    publisher: DocumentPublisher | None = None,
) -> Container:
    selected_embedder = embedder if embedder is not None else build_embedder(settings)
    selected_generator = generator if generator is not None else build_generator(settings)
    selected_store = store if store is not None else _build_store(settings)
    ingest = IngestService(settings, selected_embedder, selected_store)
    if publisher is not None:
        selected_publisher = publisher
    elif settings.ingest_transport == "inline":
        selected_publisher = InlinePublisher(ingest.handle_batch)
    else:
        selected_publisher = PubSubPublisher(settings.pubsub_topic)
    cache = ResponseCache(settings)
    return Container(
        settings=settings,
        embedder=selected_embedder,
        generator=selected_generator,
        store=selected_store,
        publisher=selected_publisher,
        cache=cache,
        limiter=TokenBucketLimiter(settings.rate_limit_rps, settings.rate_limit_burst),
        gate=InflightGate(settings.max_inflight),
        agent=RagAgent(
            settings,
            selected_embedder,
            selected_generator,
            selected_store,
            cache,
        ),
        ingest=ingest,
    )


def _build_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return MemoryVectorStore()
    if settings.vector_store == "pgvector":
        return PgVectorStore(
            AsyncpgExecutor(settings.database_url), settings.embedding_dimensionality
        )
    catalog: MemoryChunkCatalog | PgChunkCatalog
    if settings.database_url:
        catalog = PgChunkCatalog(AsyncpgExecutor(settings.database_url))
    else:
        catalog = MemoryChunkCatalog()
    return VertexVectorSearchStore(VertexNeighborIndex(settings), catalog)
