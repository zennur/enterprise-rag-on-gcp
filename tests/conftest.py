import pytest

from rag_service.config import Settings
from rag_service.models import Chunk


@pytest.fixture
def settings() -> Settings:
    return Settings(
        role="all",
        log_level="WARNING",
        embedding_provider="fake",
        generation_provider="fake",
        vector_store="memory",
        ingest_transport="inline",
        embedding_dimensionality=128,
        rate_limit_rps=1000,
        rate_limit_burst=1000,
        max_inflight=50,
        provider_backoff_s=0,
        provider_attempts=2,
        provider_timeout_s=2,
        request_timeout_s=5,
        rewrite_threshold=0,
        abstain_threshold=0.05,
        cache_enabled=True,
        semantic_cache_enabled=False,
        api_keys="",
        chunk_size=400,
        chunk_overlap=40,
    )


def sample_chunk(document_id: str = "doc", text: str = "The mealcap is seventyfive.") -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:0:abc",
        document_id=document_id,
        document_hash="hash",
        title="Policy",
        text=text,
        index=0,
    )
