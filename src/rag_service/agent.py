"""Bounded retrieve-and-answer loop.

Step one searches the corpus. If the top score is below the rewrite threshold
and a second step is allowed, the generator rewrites the query once and the
corpus is searched again. The loop then answers from the passages or abstains.
There is no open-ended tool router.
"""

from __future__ import annotations

import asyncio

import structlog
from opentelemetry import trace

from rag_service.cache import ResponseCache
from rag_service.config import Settings
from rag_service.models import Answer, Citation, ScoredChunk
from rag_service.prompts import (
    ABSTAIN_TEXT,
    ANSWER_SYSTEM,
    REWRITE_SYSTEM,
    build_answer_prompt,
    looks_like_abstain,
)
from rag_service.providers import EmbeddingProvider, GenerationProvider
from rag_service.resilience import ProviderError
from rag_service.vectorstore import VectorStore

logger = structlog.get_logger(__name__)
_EXCERPT_CHARS = 500


class RagAgent:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider,
        generator: GenerationProvider,
        store: VectorStore,
        cache: ResponseCache,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._generator = generator
        self._store = store
        self._cache = cache
        self._tracer = tracer or trace.get_tracer("rag_service")

    async def answer(self, query: str, top_k: int | None = None) -> Answer:
        limit = top_k if top_k is not None else self._settings.top_k_default
        with self._tracer.start_as_current_span("rag.answer") as span:
            span.set_attribute("rag.query_length", len(query))
            span.set_attribute("rag.top_k", limit)
            try:
                result = await asyncio.wait_for(
                    self._answer(query, limit, span),
                    self._settings.request_timeout_s,
                )
            except TimeoutError as exc:
                raise ProviderError("request timed out") from exc
            span.set_attribute("rag.abstained", result.abstained)
            span.set_attribute("rag.cached", result.cached)
            span.set_attribute("rag.rewritten", result.rewritten)
            return result

    async def _answer(self, query: str, top_k: int, span: trace.Span) -> Answer:
        generation = self._store.generation()
        variant = f"{self._settings.embedding_model}|{self._settings.generation_model}"
        cached = self._cache.get_exact(query, top_k, generation, variant)
        if cached is not None:
            logger.info("query_answered", cached=True, abstained=cached.abstained, rewritten=False)
            return cached

        embed_calls = 0
        generation_calls = 0
        with self._tracer.start_as_current_span("rag.retrieve"):
            hits, embedding, embed_calls, generation_calls, rewritten = await self._retrieve(
                query, top_k
            )

        semantic = self._cache.get_semantic(embedding, generation)
        if semantic is not None:
            logger.info(
                "query_answered", cached=True, abstained=semantic.abstained, rewritten=False
            )
            return semantic

        top_score = hits[0].score if hits else None
        span.set_attribute("rag.top_score", top_score if top_score is not None else -1.0)
        if not hits or hits[0].score < self._settings.abstain_threshold:
            answer = Answer(answer=ABSTAIN_TEXT, citations=[], abstained=True, rewritten=rewritten)
        else:
            with self._tracer.start_as_current_span("rag.generate"):
                text = await self._generator.generate(
                    build_answer_prompt(query, hits),
                    system=ANSWER_SYSTEM,
                )
            generation_calls += 1
            abstained = looks_like_abstain(text)
            answer = Answer(
                answer=text,
                citations=[] if abstained else [_citation(hit) for hit in hits],
                abstained=abstained,
                rewritten=rewritten,
            )
        self._cache.put(query, top_k, generation, variant, answer, embedding)
        logger.info(
            "query_answered",
            cached=False,
            abstained=answer.abstained,
            rewritten=rewritten,
            citations=len(answer.citations),
            top_score=None if top_score is None else round(top_score, 4),
            embed_calls=embed_calls,
            generation_calls=generation_calls,
        )
        return answer

    async def _retrieve(
        self, query: str, top_k: int
    ) -> tuple[list[ScoredChunk], list[float], int, int, bool]:
        embedding = await self._embed(query)
        hits = await self._store.query(embedding, top_k)
        embed_calls = 1
        generation_calls = 0
        rewritten = False
        weak = not hits or hits[0].score < self._settings.rewrite_threshold
        if weak and self._settings.rewrite_threshold > 0 and self._settings.agent_max_steps > 1:
            rewritten_query = (await self._generator.generate(query, system=REWRITE_SYSTEM)).strip()
            generation_calls += 1
            if rewritten_query and rewritten_query.casefold() != query.casefold():
                rewritten = True
                second_embedding = await self._embed(rewritten_query)
                embed_calls += 1
                second = await self._store.query(second_embedding, top_k)
                hits = _merge(hits, second, top_k)
        return hits, embedding, embed_calls, generation_calls, rewritten

    async def _embed(self, query: str) -> list[float]:
        vectors = await self._embedder.embed([query], task="query")
        if len(vectors) != 1:
            raise ProviderError("embedding count mismatch")
        return vectors[0]


def _merge(first: list[ScoredChunk], second: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    best: dict[str, ScoredChunk] = {}
    for hit in first + second:
        current = best.get(hit.chunk.chunk_id)
        if current is None or hit.score > current.score:
            best[hit.chunk.chunk_id] = hit
    ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def _citation(hit: ScoredChunk) -> Citation:
    text = hit.chunk.text
    excerpt = text if len(text) <= _EXCERPT_CHARS else text[:_EXCERPT_CHARS].rstrip() + "..."
    return Citation(
        chunk_id=hit.chunk.chunk_id,
        document_id=hit.chunk.document_id,
        title=hit.chunk.title,
        score=hit.score,
        excerpt=excerpt,
    )
