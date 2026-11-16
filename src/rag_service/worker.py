"""One-shot file ingest, or a pull loop for larger embedding batches.

File ingest is the Cloud Run job shape: the task sees a local JSONL and exits.
A memory vector store is emptied when the process exits; use pgvector or
Vertex AI Vector Search for anything that must outlive the task.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import structlog

from rag_service.config import Settings
from rag_service.container import build_container
from rag_service.messaging import PubSubPull, consume_batch
from rag_service.models import load_jsonl_documents
from rag_service.observability import configure_logging, configure_tracing

logger = structlog.get_logger(__name__)


async def ingest_file(path: Path, settings: Settings | None = None) -> None:
    selected = settings or Settings()
    configure_logging(selected)
    configure_tracing(selected)
    container = build_container(selected)
    await container.startup()
    if container.ready_reason != "ok":
        raise RuntimeError(container.ready_reason)
    if selected.vector_store == "memory":
        logger.warning("memory_store_discarded_on_exit")
    documents = load_jsonl_documents(path)
    stats = await container.ingest.handle_batch(documents)
    logger.info(
        "file_ingest_complete",
        documents=stats.received,
        embedded_documents=stats.embedded_documents,
        chunks=stats.chunks,
    )
    await container.shutdown()


async def pull_until_stopped(settings: Settings | None = None) -> None:
    selected = settings or Settings()
    if not selected.pubsub_subscription:
        raise RuntimeError("PUBSUB_SUBSCRIPTION is required for pull mode")
    configure_logging(selected)
    configure_tracing(selected)
    container = build_container(selected)
    await container.startup()
    if container.ready_reason != "ok":
        raise RuntimeError(container.ready_reason)
    source = PubSubPull(selected.pubsub_subscription)
    logger.info("pull_worker_started", subscription=selected.pubsub_subscription)
    try:
        while True:
            try:
                count = await consume_batch(source, container.ingest, selected.pull_batch_size)
            except Exception:
                logger.exception("pull_batch_failed")
                await asyncio.sleep(1)
                continue
            if count == 0:
                await asyncio.sleep(0.5)
    finally:
        await container.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="JSONL file of documents to ingest once")
    group.add_argument("--pull", action="store_true", help="Pull Pub/Sub until interrupted")
    args = parser.parse_args(argv)
    if args.pull:
        asyncio.run(pull_until_stopped())
        return 0
    asyncio.run(ingest_file(args.input))
    return 0


if __name__ == "__main__":
    sys.exit(main())
