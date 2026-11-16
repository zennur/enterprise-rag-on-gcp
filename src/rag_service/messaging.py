"""In-process and Pub/Sub transports for document ingestion.

Push is the serverless deployment path: one document per HTTP request, with
chunk-level embed batches inside that document. The pull helper accumulates
several documents so embedding calls span a batch. Pull wants a process that
stays up (CPU always allocated, min instances 1). The Terraform stack uses push
so the worker can scale to zero.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import structlog
from pydantic import ValidationError

from rag_service.ingest import IngestService
from rag_service.models import Document, IngestStats
from rag_service.resilience import ProviderError

logger = structlog.get_logger(__name__)


class DocumentPublisher(Protocol):
    async def publish(self, documents: list[Document]) -> None: ...


class InlinePublisher:
    def __init__(self, handle: Callable[[list[Document]], Awaitable[IngestStats]]) -> None:
        self._handle = handle

    async def publish(self, documents: list[Document]) -> None:
        await self._handle(documents)


class PubSubPublisher:
    def __init__(self, topic: str) -> None:
        self._topic = topic
        self._client: Any = None

    async def publish(self, documents: list[Document]) -> None:
        try:
            await asyncio.to_thread(self._publish_sync, documents)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("failed to publish ingest messages") from exc

    def _publish_sync(self, documents: list[Document]) -> None:
        client = self._publisher()
        futures = [
            client.publish(
                self._topic,
                document.model_dump_json().encode(),
                ordering_key=document.document_id,
            )
            for document in documents
        ]
        for future in futures:
            future.result(timeout=10)

    def _publisher(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google.cloud import pubsub_v1
            from google.cloud.pubsub_v1.types import PublisherOptions
        except ImportError as exc:
            raise ProviderError("Install the gcp extra to publish to Pub/Sub") from exc
        self._client = pubsub_v1.PublisherClient(
            publisher_options=PublisherOptions(enable_message_ordering=True)
        )
        return self._client


@dataclass(frozen=True)
class PulledMessage:
    ack_id: str
    document: Document


class PullSource(Protocol):
    async def pull(self, max_messages: int) -> list[PulledMessage]: ...

    async def ack(self, ack_ids: list[str]) -> None: ...

    async def nack(self, ack_ids: list[str]) -> None: ...


class PubSubPull:
    def __init__(self, subscription: str) -> None:
        self._subscription = subscription
        self._client: Any = None

    async def pull(self, max_messages: int) -> list[PulledMessage]:
        raw = await asyncio.to_thread(self._pull_sync, max_messages)
        valid: list[PulledMessage] = []
        invalid: list[str] = []
        for ack_id, data in raw:
            try:
                document = Document.model_validate_json(data)
            except ValidationError:
                invalid.append(ack_id)
                continue
            valid.append(PulledMessage(ack_id=ack_id, document=document))
        if invalid:
            logger.warning("poison_pubsub_messages", count=len(invalid))
            await self.nack(invalid)
        return valid

    async def ack(self, ack_ids: list[str]) -> None:
        if ack_ids:
            await asyncio.to_thread(self._ack_sync, ack_ids)

    async def nack(self, ack_ids: list[str]) -> None:
        if ack_ids:
            await asyncio.to_thread(self._nack_sync, ack_ids)

    def _pull_sync(self, max_messages: int) -> list[tuple[str, bytes]]:
        client = self._subscriber()
        response = client.pull(
            request={"subscription": self._subscription, "max_messages": max_messages},
            timeout=10,
        )
        return [
            (received.ack_id, bytes(received.message.data))
            for received in response.received_messages
        ]

    def _ack_sync(self, ack_ids: list[str]) -> None:
        self._subscriber().acknowledge(
            request={"subscription": self._subscription, "ack_ids": ack_ids}
        )

    def _nack_sync(self, ack_ids: list[str]) -> None:
        self._subscriber().modify_ack_deadline(
            request={
                "subscription": self._subscription,
                "ack_ids": ack_ids,
                "ack_deadline_seconds": 0,
            }
        )

    def _subscriber(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google.cloud import pubsub_v1
        except ImportError as exc:
            raise ProviderError("Install the gcp extra to pull from Pub/Sub") from exc
        self._client = pubsub_v1.SubscriberClient()
        return self._client


async def consume_batch(source: PullSource, ingest: IngestService, max_messages: int) -> int:
    messages = await source.pull(max_messages)
    if not messages:
        return 0
    try:
        await ingest.handle_batch([message.document for message in messages])
    except Exception:
        await source.nack([message.ack_id for message in messages])
        raise
    await source.ack([message.ack_id for message in messages])
    return len(messages)
