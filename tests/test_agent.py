import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from tests.conftest import sample_chunk

from rag_service.agent import RagAgent
from rag_service.cache import ResponseCache
from rag_service.models import ScoredChunk
from rag_service.prompts import (
    ANSWER_SYSTEM,
    REWRITE_SYSTEM,
    build_answer_prompt,
    parse_answer_prompt,
)
from rag_service.providers import FakeEmbeddingProvider, FakeGenerationProvider
from rag_service.vectorstore import MemoryVectorStore


class ScriptedStore:
    def __init__(self, responses: list[list[ScoredChunk]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generation(self) -> int:
        return 3

    async def query(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        del vector, top_k
        self.calls += 1
        return self._responses.pop(0)


class ScriptedGenerator:
    def __init__(self) -> None:
        self.systems: list[str] = []

    async def generate(self, prompt: str, *, system: str) -> str:
        del prompt
        self.systems.append(system)
        if system == REWRITE_SYSTEM:
            return "mealcap seventyfive dollars"
        return "The mealcap is seventyfive dollars. [expenses:0:abc]"


class FixedEmbedder:
    async def embed(self, texts: list[str], *, task: str) -> list[list[float]]:
        del task
        return [[0.2, 0.8] for _ in texts]


def test_prompt_round_trip() -> None:
    hit = ScoredChunk(chunk=sample_chunk("expenses"), score=0.8)
    question, chunks = parse_answer_prompt(build_answer_prompt("What is the mealcap?", [hit]))
    assert question == "What is the mealcap?"
    assert chunks == [("expenses:0:abc", hit.chunk.text)]


@pytest.mark.asyncio
async def test_one_rewrite_then_answer(settings) -> None:  # type: ignore[no-untyped-def]
    settings = settings.model_copy(
        update={"rewrite_threshold": 0.4, "abstain_threshold": 0.3, "agent_max_steps": 2}
    )
    weak = ScoredChunk(chunk=sample_chunk("other", "unrelated note"), score=0.1)
    strong = ScoredChunk(chunk=sample_chunk("expenses"), score=0.9)
    store = ScriptedStore([[weak], [strong]])
    generator = ScriptedGenerator()
    agent = RagAgent(settings, FixedEmbedder(), generator, store, ResponseCache(settings))
    answer = await agent.answer("mealcap?")
    assert store.calls == 2
    assert generator.systems == [REWRITE_SYSTEM, ANSWER_SYSTEM]
    assert answer.rewritten is True
    assert answer.abstained is False
    assert answer.citations[0].document_id == "expenses"


@pytest.mark.asyncio
async def test_rewrite_stays_off_when_threshold_is_zero(settings) -> None:  # type: ignore[no-untyped-def]
    settings = settings.model_copy(update={"rewrite_threshold": 0, "abstain_threshold": 0.5})
    store = ScriptedStore([[ScoredChunk(chunk=sample_chunk(), score=0.1)]])
    generator = ScriptedGenerator()
    agent = RagAgent(settings, FixedEmbedder(), generator, store, ResponseCache(settings))
    answer = await agent.answer("mealcap?")
    assert store.calls == 1
    assert generator.systems == []
    assert answer.abstained is True


@pytest.mark.asyncio
async def test_answer_span_is_recorded(settings) -> None:  # type: ignore[no-untyped-def]
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    agent = RagAgent(
        settings,
        FakeEmbeddingProvider(settings.embedding_dimensionality),
        FakeGenerationProvider(),
        MemoryVectorStore(),
        ResponseCache(settings),
        tracer=provider.get_tracer("test"),
    )
    await agent.answer("anything")
    assert "rag.answer" in {span.name for span in exporter.get_finished_spans()}
