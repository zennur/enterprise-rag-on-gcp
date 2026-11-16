"""Process-local exact cache, plus an optional linear semantic cache.

Entries die with the instance and are keyed by the local index generation.
Other instances can serve a stale answer until ``cache_ttl_s``. A shared cache
(Memorystore) is the horizontal step; it is intentionally not in this v1.
The semantic cache scans a capped list, which is only reasonable for a small
hot set, and it is off by default because a wrong hit is worse than a miss.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from rag_service.config import Settings
from rag_service.models import Answer
from rag_service.textutil import cosine


@dataclass
class _Entry:
    answer: Answer
    expires_at: float
    generation: int
    embedding: list[float] | None


class ResponseCache:
    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic) -> None:
        self._settings = settings
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def get_exact(self, query: str, top_k: int, generation: int, variant: str) -> Answer | None:
        if not self._settings.cache_enabled:
            return None
        entry = self._entries.get(self._key(query, top_k, generation, variant))
        if entry is None or not self._live(entry, generation):
            return None
        return entry.answer.model_copy(update={"cached": True})

    def get_semantic(self, embedding: list[float], generation: int) -> Answer | None:
        if not self._settings.cache_enabled or not self._settings.semantic_cache_enabled:
            return None
        best: _Entry | None = None
        best_score = self._settings.semantic_cache_threshold
        stale: list[str] = []
        for key, entry in self._entries.items():
            if not self._live(entry, generation) or entry.embedding is None:
                if not self._live(entry, generation):
                    stale.append(key)
                continue
            score = cosine(embedding, entry.embedding)
            if score >= best_score:
                best = entry
                best_score = score
        for key in stale:
            self._entries.pop(key, None)
        if best is None:
            return None
        return best.answer.model_copy(update={"cached": True})

    def put(
        self,
        query: str,
        top_k: int,
        generation: int,
        variant: str,
        answer: Answer,
        embedding: list[float] | None = None,
    ) -> None:
        if not self._settings.cache_enabled or self._settings.cache_ttl_s <= 0:
            return
        stored = answer.model_copy(update={"cached": False})
        vector = list(embedding) if self._settings.semantic_cache_enabled and embedding else None
        self._entries[self._key(query, top_k, generation, variant)] = _Entry(
            answer=stored,
            expires_at=self._clock() + self._settings.cache_ttl_s,
            generation=generation,
            embedding=vector,
        )
        self._evict()

    def _live(self, entry: _Entry, generation: int) -> bool:
        return entry.generation == generation and entry.expires_at > self._clock()

    def _key(self, query: str, top_k: int, generation: int, variant: str) -> str:
        normalized = " ".join(query.lower().split())
        raw = f"{generation}|{variant}|{top_k}|{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _evict(self) -> None:
        while len(self._entries) > self._settings.cache_max_entries:
            self._entries.popitem(last=False)
