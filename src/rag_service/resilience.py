"""Timeouts, jittered retries, a per-instance token bucket, and an inflight cap.

The limiters are process-local. On Cloud Run they shed load on one instance.
A global quota belongs at the edge (Cloud Armor or an API gateway) and in the
platform max-instance setting.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class TransientError(Exception):
    """A provider call failed in a way that a retry might clear."""


class ProviderError(Exception):
    """The model or messaging dependency cannot serve this request."""


_TRANSIENT_NAMES = {
    "Aborted",
    "DeadlineExceeded",
    "InternalServerError",
    "ResourceExhausted",
    "ServiceUnavailable",
    "TooManyRequests",
}
_TRANSIENT_CODES = {"4", "8", "10", "13", "14"}


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TransientError):
        return True
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    code = getattr(exc, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            code = None
    text = str(code) if code is not None else ""
    if text in _TRANSIENT_CODES:
        return True
    upper = text.upper()
    return "UNAVAILABLE" in upper or "RESOURCE_EXHAUSTED" in upper or "DEADLINE" in upper


async def call_with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    timeout_s: float,
    backoff_s: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(func(), timeout_s)
        except TimeoutError as exc:
            last = exc
        except Exception as exc:
            if not is_transient(exc):
                raise
            last = exc
        if attempt + 1 < attempts and backoff_s > 0:
            delay = backoff_s * (2**attempt) + random.uniform(0, backoff_s)
            await sleep(delay)
    assert last is not None
    raise last


class TokenBucketLimiter:
    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("rate and burst must be positive")
        self._rate = rate
        self._burst = float(burst)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        tokens, updated = self._buckets.get(key, (self._burst, now))
        tokens = min(self._burst, tokens + (now - updated) * self._rate)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False
        self._buckets[key] = (tokens - 1.0, now)
        return True


class InflightGate:
    """Reject new work once this process is at its model-call budget."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self._limit = limit
        self._current = 0

    def try_enter(self) -> bool:
        if self._current >= self._limit:
            return False
        self._current += 1
        return True

    def exit(self) -> None:
        if self._current > 0:
            self._current -= 1
