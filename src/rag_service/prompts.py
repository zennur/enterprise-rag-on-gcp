"""Prompt layout shared by Vertex Gemini and the extractive local generator."""

from __future__ import annotations

import re

from rag_service.models import ScoredChunk

ABSTAIN_TEXT = "I don't know based on the available documents."
ANSWER_SYSTEM = (
    "You answer using only the context blocks. If the context does not contain "
    f"the answer, reply exactly: {ABSTAIN_TEXT} "
    "Cite supporting chunk ids in brackets. Do not use prior knowledge."
)
REWRITE_SYSTEM = (
    "Rewrite this search query so it is more specific. Return only the rewritten query."
)

_CHUNK_START = "<<<CHUNK "
_CHUNK_END = "<<<END>>>"
_CITATION = re.compile(r"\[[A-Za-z0-9_.:-]+\]")


def build_answer_prompt(query: str, hits: list[ScoredChunk]) -> str:
    blocks = [
        f"{_CHUNK_START}{hit.chunk.chunk_id}>>>\n{hit.chunk.text}\n{_CHUNK_END}" for hit in hits
    ]
    return f"Question:\n{query}\n\nContext:\n" + "\n".join(blocks)


def parse_answer_prompt(prompt: str) -> tuple[str, list[tuple[str, str]]]:
    question = ""
    body = prompt
    marker = "\n\nContext:\n"
    if prompt.startswith("Question:\n") and marker in prompt:
        question, body = prompt[len("Question:\n") :].split(marker, maxsplit=1)
    chunks: list[tuple[str, str]] = []
    for part in body.split(_CHUNK_START)[1:]:
        header, _, rest = part.partition(">>>\n")
        text, _, _ = rest.partition(f"\n{_CHUNK_END}")
        chunks.append((header.strip(), text.strip()))
    return question.strip(), chunks


def strip_citations(answer: str) -> str:
    return _CITATION.sub(" ", answer)


def looks_like_abstain(answer: str) -> bool:
    return answer.strip().lower().startswith(ABSTAIN_TEXT.lower())
