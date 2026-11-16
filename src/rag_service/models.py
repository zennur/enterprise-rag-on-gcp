"""Domain records shared by the API, the worker, and the vector stores."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

_DOC_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(pattern=_DOC_ID.pattern)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=200_000)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text is empty")
        return value

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 32:
            raise ValueError("too many metadata keys")
        for key, item in value.items():
            if not key or len(key) > 64 or len(item) > 200:
                raise ValueError("metadata entry is too large")
        return value


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_hash: str
    title: str
    text: str
    index: int
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    score: float
    excerpt: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    cached: bool = False
    rewritten: bool = False


class IngestStats(BaseModel):
    received: int
    embedded_documents: int
    chunks: int


def document_hash(document: Document) -> str:
    payload = json.dumps(
        {"metadata": document.metadata, "text": document.text, "title": document.title},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_jsonl_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                documents.append(Document.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid document on line {line_number}") from exc
    return documents
