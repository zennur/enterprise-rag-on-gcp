from pathlib import Path

import pytest

from rag_service.config import Settings
from rag_service.container import build_container
from rag_service.evalrunner import evaluate, load_questions
from rag_service.models import load_jsonl_documents
from rag_service.textutil import cosine, hashing_embed

ROOT = Path(__file__).resolve().parents[1]


def test_related_local_embeddings_outrank_unrelated_text() -> None:
    query = hashing_embed("sev1 incidentcommander pager", 128)
    oncall = hashing_embed(
        "A sev1 incident pages the incidentcommander as soon as the pager is acknowledged.",
        128,
    )
    expenses = hashing_embed("The mealcap is seventyfive dollars per person.", 128)
    assert cosine(query, oncall) > cosine(query, expenses)


@pytest.mark.asyncio
async def test_sample_eval_clears_the_fake_provider() -> None:
    settings = Settings(
        embedding_provider="fake",
        generation_provider="fake",
        vector_store="memory",
        ingest_transport="inline",
        log_level="WARNING",
    )
    container = build_container(settings)
    report = await evaluate(
        container,
        load_jsonl_documents(ROOT / "eval" / "corpus.jsonl"),
        load_questions(ROOT / "eval" / "questions.jsonl"),
        top_k=4,
    )
    misses = [item.question for item in report.results if item.hit is False]
    wrong_decisions = [
        (item.question, item.abstained, item.top_score)
        for item in report.results
        if item.abstained != item.should_abstain
    ]
    assert report.retrieval_hit_rate == 1.0, misses
    assert report.decision_accuracy == 1.0, wrong_decisions
    assert report.mean_groundedness is not None
    assert report.mean_groundedness >= 0.8
