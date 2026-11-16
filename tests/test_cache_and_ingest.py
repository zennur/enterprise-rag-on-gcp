import pytest

from rag_service.cache import ResponseCache
from rag_service.config import Settings
from rag_service.ingest import IngestService
from rag_service.models import Answer, Document
from rag_service.prompts import ABSTAIN_TEXT
from rag_service.textutil import hashing_embed
from rag_service.vectorstore import MemoryVectorStore


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str], *, task: str) -> list[list[float]]:
        del task
        self.calls += 1
        return [hashing_embed(text, 32) for text in texts]


def test_exact_cache_is_generation_and_query_sensitive() -> None:
    now = {"value": 0.0}
    settings = Settings(
        cache_ttl_s=10,
        cache_max_entries=2,
        semantic_cache_enabled=False,
        log_level="WARNING",
    )
    cache = ResponseCache(settings, clock=lambda: now["value"])
    answer = Answer(answer="kept", citations=[], abstained=False)
    cache.put("Hello   there", 4, 1, "model", answer)
    hit = cache.get_exact("hello there", 4, 1, "model")
    assert hit is not None
    assert hit.cached is True
    assert cache.get_exact("hello there", 4, 2, "model") is None
    now["value"] = 11
    assert cache.get_exact("hello there", 4, 1, "model") is None


def test_semantic_cache_requires_a_high_score() -> None:
    settings = Settings(
        semantic_cache_enabled=True, semantic_cache_threshold=0.92, log_level="WARNING"
    )
    cache = ResponseCache(settings)
    vector = hashing_embed("sev1 incidentcommander pager", 64)
    answer = Answer(answer="page the incidentcommander", citations=[], abstained=False)
    cache.put("sev1 pager", 4, 1, "model", answer, vector)
    assert cache.get_semantic(vector, 1) is not None
    assert cache.get_semantic(hashing_embed("mealcap financeportal", 64), 1) is None


@pytest.mark.asyncio
async def test_reingest_of_the_same_document_does_not_reembed() -> None:
    settings = Settings(
        embedding_dimensionality=32,
        embed_batch_size=8,
        chunk_size=200,
        chunk_overlap=20,
        log_level="WARNING",
    )
    embedder = CountingEmbedder()
    store = MemoryVectorStore()
    service = IngestService(settings, embedder, store)
    document = Document(
        document_id="policy",
        title="Expenses",
        text="The mealcap is seventyfive dollars. Rideshare receipts go to the financeportal.",
    )
    first = await service.handle_batch([document, document])
    assert first.embedded_documents == 1
    assert embedder.calls == 1
    second = await service.handle_batch([document])
    assert second.embedded_documents == 0
    assert embedder.calls == 1

    updated = document.model_copy(update={"text": "The mealcap is forty dollars."})
    third = await service.handle_batch([updated])
    assert third.embedded_documents == 1
    assert embedder.calls == 2
    hits = await store.query(hashing_embed("mealcap forty", 32), 2)
    assert hits
    assert "forty" in hits[0].chunk.text
    assert all("seventyfive" not in hit.chunk.text for hit in hits)


@pytest.mark.asyncio
async def test_empty_store_abstains_without_generation(settings: Settings) -> None:
    from rag_service.agent import RagAgent
    from rag_service.providers import FakeEmbeddingProvider, FakeGenerationProvider

    agent = RagAgent(
        settings,
        FakeEmbeddingProvider(settings.embedding_dimensionality),
        FakeGenerationProvider(),
        MemoryVectorStore(),
        ResponseCache(settings),
    )
    answer = await agent.answer("What is the mealcap?")
    assert answer.abstained is True
    assert answer.answer == ABSTAIN_TEXT
    assert answer.citations == []
