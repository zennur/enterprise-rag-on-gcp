import pytest

from rag_service.ingest import IngestService
from rag_service.messaging import PulledMessage, consume_batch
from rag_service.models import Document
from rag_service.providers import VertexEmbeddingProvider
from rag_service.resilience import TransientError
from rag_service.textutil import hashing_embed
from rag_service.vectorstore import MemoryVectorStore


class FakePull:
    def __init__(self, messages: list[PulledMessage]) -> None:
        self._messages = messages
        self.acked: list[str] = []
        self.nacked: list[str] = []

    async def pull(self, max_messages: int) -> list[PulledMessage]:
        del max_messages
        batch = self._messages
        self._messages = []
        return batch

    async def ack(self, ack_ids: list[str]) -> None:
        self.acked.extend(ack_ids)

    async def nack(self, ack_ids: list[str]) -> None:
        self.nacked.extend(ack_ids)


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str], *, task: str) -> list[list[float]]:
        del task
        self.calls += 1
        return [hashing_embed(text, 16) for text in texts]


@pytest.mark.asyncio
async def test_pull_batch_acks_after_a_successful_ingest(settings) -> None:  # type: ignore[no-untyped-def]
    embedder = CountingEmbedder()
    store = MemoryVectorStore()
    ingest = IngestService(settings, embedder, store)
    document = Document(
        document_id="expenses", title="Expenses", text="The mealcap is seventyfive."
    )
    source = FakePull(
        [
            PulledMessage("a1", document),
            PulledMessage("a2", document),
        ]
    )
    count = await consume_batch(source, ingest, max_messages=8)
    assert count == 2
    assert embedder.calls == 1
    assert source.acked == ["a1", "a2"]
    assert source.nacked == []


class FlakyModels:
    def __init__(self) -> None:
        self.calls = 0

    def embed_content(self, **kwargs: object) -> object:
        self.calls += 1
        self.kwargs = kwargs
        if self.calls == 1:
            raise type("ResourceExhausted", (Exception,), {})("quota")

        class Embedding:
            values = [0.1, 0.2]

        class Response:
            embeddings = [Embedding(), Embedding()]
            metadata = None

        return Response()


@pytest.mark.asyncio
async def test_vertex_embedder_retries_and_sends_task_type(settings) -> None:  # type: ignore[no-untyped-def]
    settings = settings.model_copy(
        update={
            "embedding_provider": "vertex",
            "embedding_dimensionality": 2,
            "gcp_project": "demo",
            "provider_attempts": 2,
            "provider_backoff_s": 0,
        }
    )
    client = type("Client", (), {})()
    client.models = FlakyModels()
    provider = VertexEmbeddingProvider(settings, client=client)
    vectors = await provider.embed(["alpha", "beta"], task="query")
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert client.models.calls == 2
    assert client.models.kwargs["config"]["task_type"] == "RETRIEVAL_QUERY"
    assert client.models.kwargs["model"] == settings.embedding_model


@pytest.mark.asyncio
async def test_vertex_embedder_does_not_retry_a_permanent_error(settings) -> None:  # type: ignore[no-untyped-def]
    settings = settings.model_copy(update={"embedding_dimensionality": 2, "provider_attempts": 3})

    class Models:
        calls = 0

        def embed_content(self, **kwargs: object) -> object:
            del kwargs
            self.calls += 1
            raise ValueError("bad request")

    client = type("Client", (), {"models": Models()})()
    provider = VertexEmbeddingProvider(settings, client=client)
    with pytest.raises(ValueError):
        await provider.embed(["alpha"], task="document")
    assert client.models.calls == 1


def test_transient_error_is_distinct_from_value_error() -> None:
    assert not issubclass(ValueError, TransientError)
