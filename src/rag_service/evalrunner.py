"""Retrieval hit rate and lexical groundedness on a local sample corpus.

The groundedness score is the share of content words in the answer that also
appear in the retrieved passages, after citation brackets are removed. It
checks that the answer stays on the supplied text. It does not judge whether
the passages were the right ones, and it is not an LLM judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag_service.config import Settings
from rag_service.container import Container, build_container
from rag_service.models import Document, load_jsonl_documents
from rag_service.observability import configure_logging
from rag_service.prompts import strip_citations
from rag_service.textutil import content_tokens


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    expected_document_ids: list[str] = Field(default_factory=list)
    should_abstain: bool = False


@dataclass
class QuestionResult:
    question: str
    hit: bool | None
    abstained: bool
    should_abstain: bool
    groundedness: float | None
    top_score: float | None


@dataclass
class EvalReport:
    results: list[QuestionResult]

    @property
    def retrieval_hit_rate(self) -> float:
        return _rate([item.hit for item in self.results if item.hit is not None])

    @property
    def decision_accuracy(self) -> float:
        if not self.results:
            return 0.0
        matches = [item.abstained == item.should_abstain for item in self.results]
        return sum(1 for item in matches if item) / len(matches)

    @property
    def mean_groundedness(self) -> float | None:
        scores = [item.groundedness for item in self.results if item.groundedness is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)


def lexical_groundedness(answer: str, contexts: list[str]) -> float:
    words = content_tokens(strip_citations(answer))
    if not words:
        return 1.0
    context_words = set(content_tokens(" ".join(contexts)))
    return sum(1 for word in words if word in context_words) / len(words)


def load_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                questions.append(EvalQuestion.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid question on line {line_number}") from exc
    return questions


async def evaluate(
    container: Container,
    documents: list[Document],
    questions: list[EvalQuestion],
    top_k: int,
) -> EvalReport:
    await container.ingest.handle_batch(documents)
    results: list[QuestionResult] = []
    for item in questions:
        vectors = await container.embedder.embed([item.question], task="query")
        hits = await container.store.query(vectors[0], top_k)
        top_score = hits[0].score if hits else None
        hit: bool | None = None
        if item.expected_document_ids:
            found = {scored.chunk.document_id for scored in hits}
            hit = any(document_id in found for document_id in item.expected_document_ids)
        answer = await container.agent.answer(item.question, top_k)
        grounded: float | None = None
        if not item.should_abstain and not answer.abstained:
            grounded = lexical_groundedness(
                answer.answer,
                [citation.excerpt for citation in answer.citations],
            )
        results.append(
            QuestionResult(
                question=item.question,
                hit=hit,
                abstained=answer.abstained,
                should_abstain=item.should_abstain,
                groundedness=grounded,
                top_score=top_score,
            )
        )
    return EvalReport(results)


def format_report(report: EvalReport) -> str:
    grounded = "n/a" if report.mean_groundedness is None else f"{report.mean_groundedness:.3f}"
    lines = [
        f"retrieval_hit_rate {report.retrieval_hit_rate:.3f}",
        f"decision_accuracy {report.decision_accuracy:.3f}",
        f"mean_groundedness {grounded}",
    ]
    for item in report.results:
        lines.append(
            json.dumps(
                {
                    "question": item.question,
                    "hit": item.hit,
                    "abstained": item.abstained,
                    "should_abstain": item.should_abstain,
                    "top_score": None if item.top_score is None else round(item.top_score, 3),
                    "groundedness": item.groundedness,
                }
            )
        )
    return "\n".join(lines)


async def run(
    corpus: Path,
    questions: Path,
    *,
    top_k: int = 4,
    strict: bool = False,
) -> int:
    settings = Settings(
        embedding_provider="fake",
        generation_provider="fake",
        vector_store="memory",
        ingest_transport="inline",
        log_level="WARNING",
    )
    configure_logging(settings)
    container = build_container(settings)
    await container.startup()
    report = await evaluate(
        container,
        load_jsonl_documents(corpus),
        load_questions(questions),
        top_k,
    )
    print(format_report(report))
    await container.shutdown()
    if not strict:
        return 0
    grounded = report.mean_groundedness if report.mean_groundedness is not None else 0.0
    if report.retrieval_hit_rate < 1.0 or report.decision_accuracy < 1.0 or grounded < 0.8:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local RAG eval sample")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 unless hit rate and decision accuracy are 1 and groundedness is at least 0.8",
    )
    args = parser.parse_args(argv)
    return asyncio.run(run(args.corpus, args.questions, top_k=args.top_k, strict=args.strict))


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)
