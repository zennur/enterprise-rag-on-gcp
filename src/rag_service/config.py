"""Environment-backed settings. Field names map to uppercase env vars."""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["fake", "vertex"]
VectorStoreName = Literal["memory", "pgvector", "vertex"]
IngestTransport = Literal["inline", "pubsub"]
RoleName = Literal["api", "worker", "all"]
LogFormat = Literal["json", "console"]
DistanceMode = Literal["cosine", "dot"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    role: RoleName = "all"
    log_format: LogFormat = "console"
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8080

    embedding_provider: ProviderName = "fake"
    generation_provider: ProviderName = "fake"
    vector_store: VectorStoreName = "memory"
    ingest_transport: IngestTransport = "inline"
    allow_fake_provider: bool = False
    allow_memory_store: bool = False

    gcp_project: str = ""
    gcp_location: str = "us-central1"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensionality: int = Field(default=768, ge=8, le=3072)
    generation_model: str = "gemini-2.5-flash"
    generation_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    generation_max_output_tokens: int = Field(default=512, ge=32, le=8192)

    database_url: str = ""
    vector_search_index: str = ""
    vector_search_endpoint: str = ""
    vector_search_deployed_index_id: str = ""
    vector_search_distance: DistanceMode = "cosine"

    pubsub_topic: str = ""
    pubsub_subscription: str = ""
    pubsub_push_audience: str = ""
    pubsub_push_service_account: str = ""

    chunk_size: int = Field(default=800, ge=200, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    embed_batch_size: int = Field(default=16, ge=1, le=250)
    pull_batch_size: int = Field(default=8, ge=1, le=100)
    top_k_default: int = Field(default=4, ge=1, le=20)

    # Rewrite only when the first pass would otherwise abstain. Raising
    # rewrite_threshold spends a generation call on more queries.
    abstain_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    rewrite_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    agent_max_steps: int = Field(default=2, ge=1, le=2)

    request_timeout_s: float = Field(default=20.0, gt=0)
    provider_timeout_s: float = Field(default=5.0, gt=0)
    provider_attempts: int = Field(default=2, ge=1, le=5)
    provider_backoff_s: float = Field(default=0.2, ge=0)

    rate_limit_rps: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=10, ge=1)
    max_inflight: int = Field(default=24, ge=1)

    cache_enabled: bool = True
    cache_ttl_s: float = Field(default=300.0, ge=0)
    cache_max_entries: int = Field(default=512, ge=1)
    semantic_cache_enabled: bool = False
    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)

    api_keys: str = ""

    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "rag-service"

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError("unknown log level")
        return normalized

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_fits(cls, value: int, info: ValidationInfo) -> int:
        size = info.data.get("chunk_size")
        if isinstance(size, int) and value >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


def configuration_error(settings: Settings) -> str | None:
    """Return a ready-check reason when this process should not serve traffic."""
    on_cloud_run = bool(os.environ.get("K_SERVICE"))
    fake_selected = settings.embedding_provider == "fake" or settings.generation_provider == "fake"
    if on_cloud_run and fake_selected and not settings.allow_fake_provider:
        return (
            "The fake provider is selected on Cloud Run. Set the Vertex providers, "
            "or set ALLOW_FAKE_PROVIDER=true for a deliberate platform drill."
        )
    if on_cloud_run and settings.role in {"worker", "all"} and not settings.pubsub_push_audience:
        return "PUBSUB_PUSH_AUDIENCE is required on Cloud Run so push requests are authenticated"
    if on_cloud_run and settings.vector_store == "memory" and not settings.allow_memory_store:
        return (
            "VECTOR_STORE=memory is not shared across Cloud Run instances. "
            "Set pgvector or vertex, or set ALLOW_MEMORY_STORE=true for a smoke test."
        )
    vertex_selected = (
        settings.embedding_provider == "vertex" or settings.generation_provider == "vertex"
    )
    if vertex_selected and not settings.gcp_project:
        return "GCP_PROJECT is required when a Vertex provider is selected"
    if settings.vector_store == "pgvector" and not settings.database_url:
        return "DATABASE_URL is required when VECTOR_STORE=pgvector"
    if settings.vector_store == "vertex":
        missing = [
            name
            for name, value in (
                ("VECTOR_SEARCH_INDEX", settings.vector_search_index),
                ("VECTOR_SEARCH_ENDPOINT", settings.vector_search_endpoint),
                ("VECTOR_SEARCH_DEPLOYED_INDEX_ID", settings.vector_search_deployed_index_id),
            )
            if not value
        ]
        if missing:
            return "Missing " + ", ".join(missing)
    if settings.ingest_transport == "pubsub" and not settings.pubsub_topic:
        return "PUBSUB_TOPIC is required when INGEST_TRANSPORT=pubsub"
    return None
