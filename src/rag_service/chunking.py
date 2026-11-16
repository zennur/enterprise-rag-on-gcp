"""Deterministic character windows with a whitespace bias and overlap."""

from __future__ import annotations

import hashlib

from rag_service.config import Settings
from rag_service.models import Chunk, Document, document_hash


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be smaller than size")
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + size, length)
        if end < length:
            split_at = normalized.rfind(" ", start, end)
            if split_at > start + size // 2:
                end = split_at
        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return chunks


def chunk_document(document: Document, settings: Settings) -> list[Chunk]:
    digest = document_hash(document)
    pieces = chunk_text(document.text, settings.chunk_size, settings.chunk_overlap)
    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        content_hash = hashlib.sha256(piece.encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{index}:{content_hash}",
                document_id=document.document_id,
                document_hash=digest,
                title=document.title,
                text=piece,
                index=index,
                metadata=dict(document.metadata),
            )
        )
    return chunks
