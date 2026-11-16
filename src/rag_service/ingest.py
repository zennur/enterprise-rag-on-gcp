"""Chunk, embed in batches, and replace a document as one idempotent unit."""

from __future__ import annotations

import structlog

from rag_service.batching import batched
from rag_service.chunking import chunk_document
from rag_service.config import Settings
from rag_service.models import Chunk, Document, IngestStats, document_hash
from rag_service.providers import EmbeddingProvider
from rag_service.resilience import ProviderError
from rag_service.vectorstore import VectorStore

logger = structlog.get_logger(__name__)


class IngestService:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider,
        store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store

    async def handle_batch(self, documents: list[Document]) -> IngestStats:
        # A batch can mention one id twice. Keep the last copy so the embed
        # call happens once and the stored hash matches the final text.
        collapsed: dict[str, Document] = {}
        for document in documents:
            collapsed[document.document_id] = document

        pending: list[tuple[Document, list[Chunk]]] = []
        for document in collapsed.values():
            digest = document_hash(document)
            current = await self._store.document_hash(document.document_id)
            if current == digest:
                continue
            pending.append((document, chunk_document(document, self._settings)))

        flat = [chunk for _document, chunks in pending for chunk in chunks]
        vectors: list[list[float]] = []
        for batch in batched(flat, self._settings.embed_batch_size):
            embedded = await self._embedder.embed([chunk.text for chunk in batch], task="document")
            if len(embedded) != len(batch):
                raise ProviderError("embedding count mismatch")
            vectors.extend(embedded)

        offset = 0
        for document, chunks in pending:
            piece = vectors[offset : offset + len(chunks)]
            offset += len(chunks)
            await self._store.replace_document(document.document_id, chunks, piece)

        logger.info(
            "documents_indexed",
            received=len(documents),
            embedded_documents=len(pending),
            chunks=len(flat),
        )
        return IngestStats(
            received=len(documents),
            embedded_documents=len(pending),
            chunks=len(flat),
        )
