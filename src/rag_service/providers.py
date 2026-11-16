"""Embedding and generation providers.

The fake provider is a signed hashing embedder plus an extractive answerer.
It makes tests, CI, and a local demo deterministic without GCP credentials.
Vertex AI is the production implementation and is imported only when selected.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import structlog

from rag_service.config import Settings
from rag_service.prompts import (
    ABSTAIN_TEXT,
    REWRITE_SYSTEM,
    parse_answer_prompt,
    strip_citations,
)
from rag_service.resilience import ProviderError, TransientError, call_with_retry, is_transient
from rag_service.textutil import content_tokens, hashing_embed

logger = structlog.get_logger(__name__)


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str], *, task: str) -> list[list[float]]: ...


class GenerationProvider(Protocol):
    async def generate(self, prompt: str, *, system: str) -> str: ...


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    async def embed(self, texts: list[str], *, task: str) -> list[list[float]]:
        del task
        return [hashing_embed(text, self._dimensions) for text in texts]


class FakeGenerationProvider:
    """Copies context sentences that overlap the question.

    This is not a language model. It exists so the agent loop, citations, and
    the lexical groundedness check can run without a network call.
    """

    async def generate(self, prompt: str, *, system: str) -> str:
        if system == REWRITE_SYSTEM:
            words = content_tokens(prompt)
            return " ".join(words) if words else prompt.strip()
        question, chunks = parse_answer_prompt(prompt)
        if not chunks:
            return ABSTAIN_TEXT
        question_tokens = set(content_tokens(question))
        scored: list[tuple[int, str, str]] = []
        for chunk_id, text in chunks:
            for sentence in _sentences(text):
                overlap = sum(1 for token in content_tokens(sentence) if token in question_tokens)
                scored.append((overlap, chunk_id, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = [item for item in scored if item[0] > 0][:2] or scored[:1]
        lines = [sentence for _, _, sentence in chosen]
        body = ". ".join(lines).strip()
        if body and not body.endswith("."):
            body += "."
        cites = " ".join(f"[{chunk_id}]" for _, chunk_id, _ in chosen)
        if cites:
            return f"{strip_citations(body).strip()} {cites}"
        return strip_citations(body).strip()


class VertexEmbeddingProvider:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client

    async def embed(self, texts: list[str], *, task: str) -> list[list[float]]:
        if not texts:
            return []
        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"

        async def attempt() -> list[list[float]]:
            return await asyncio.to_thread(self._embed_sync, texts, task_type)

        vectors = await call_with_retry(
            attempt,
            attempts=self._settings.provider_attempts,
            timeout_s=self._settings.provider_timeout_s,
            backoff_s=self._settings.provider_backoff_s,
        )
        if not isinstance(vectors, list):
            raise ProviderError("embedding provider returned an unexpected payload")
        return vectors

    def _embed_sync(self, texts: list[str], task_type: str) -> list[list[float]]:
        client = self._require_client()
        try:
            response = client.models.embed_content(
                model=self._settings.embedding_model,
                contents=texts,
                config={
                    "output_dimensionality": self._settings.embedding_dimensionality,
                    "task_type": task_type,
                },
            )
        except Exception as exc:
            if is_transient(exc):
                raise TransientError(str(exc)) from exc
            raise
        self._log_embed_usage(response, len(texts), task_type)
        embeddings = list(getattr(response, "embeddings", None) or [])
        vectors: list[list[float]] = []
        for item in embeddings:
            values = [float(value) for value in list(getattr(item, "values", None) or [])]
            if len(values) != self._settings.embedding_dimensionality:
                raise ProviderError(
                    "embedding dimensionality does not match EMBEDDING_DIMENSIONALITY"
                )
            vectors.append(values)
        if len(vectors) != len(texts):
            raise ProviderError("embedding count mismatch")
        return vectors

    def _require_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError("Install the gcp extra to call Vertex AI embeddings") from exc
        self._client = genai.Client(
            vertexai=True,
            project=self._settings.gcp_project,
            location=self._settings.gcp_location,
        )
        return self._client

    def _log_embed_usage(self, response: Any, inputs: int, task_type: str) -> None:
        metadata = getattr(response, "metadata", None)
        billable = getattr(metadata, "billable_character_count", None)
        if billable is None:
            return
        logger.info(
            "embedding_usage",
            billable_characters=billable,
            inputs=inputs,
            task=task_type,
            model=self._settings.embedding_model,
        )


class VertexGenerationProvider:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client

    async def generate(self, prompt: str, *, system: str) -> str:
        async def attempt() -> str:
            return await asyncio.to_thread(self._generate_sync, prompt, system)

        text = await call_with_retry(
            attempt,
            attempts=self._settings.provider_attempts,
            timeout_s=self._settings.provider_timeout_s,
            backoff_s=self._settings.provider_backoff_s,
        )
        if not isinstance(text, str):
            raise ProviderError("generation provider returned an unexpected payload")
        return text

    def _generate_sync(self, prompt: str, system: str) -> str:
        client = self._require_client()
        try:
            response = client.models.generate_content(
                model=self._settings.generation_model,
                contents=prompt,
                config={
                    "max_output_tokens": self._settings.generation_max_output_tokens,
                    "system_instruction": system,
                    "temperature": self._settings.generation_temperature,
                },
            )
        except Exception as exc:
            if is_transient(exc):
                raise TransientError(str(exc)) from exc
            raise
        self._log_generation_usage(response)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("model returned an empty answer")
        return text.strip()

    def _require_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError("Install the gcp extra to call Vertex AI Gemini") from exc
        self._client = genai.Client(
            vertexai=True,
            project=self._settings.gcp_project,
            location=self._settings.gcp_location,
        )
        return self._client

    def _log_generation_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        logger.info(
            "generation_usage",
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            model=self._settings.generation_model,
        )


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in text.replace("?", ".").replace("!", ".").split(".")]
    return [part for part in parts if part]


def build_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "fake":
        return FakeEmbeddingProvider(settings.embedding_dimensionality)
    return VertexEmbeddingProvider(settings)


def build_generator(settings: Settings) -> GenerationProvider:
    if settings.generation_provider == "fake":
        return FakeGenerationProvider()
    return VertexGenerationProvider(settings)
