import pytest

from rag_service.resilience import (
    InflightGate,
    TokenBucketLimiter,
    TransientError,
    call_with_retry,
    is_transient,
)
from rag_service.textutil import distance_to_score


@pytest.mark.asyncio
async def test_retries_transient_errors_then_returns() -> None:
    calls = {"count": 0}

    async def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise TransientError("busy")
        return "ok"

    result = await call_with_retry(flaky, attempts=3, timeout_s=1, backoff_s=0)
    assert result == "ok"
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_does_not_retry_permanent_errors() -> None:
    calls = {"count": 0}

    async def broken() -> str:
        calls["count"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await call_with_retry(broken, attempts=3, timeout_s=1, backoff_s=0)
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_timeout_is_retried_then_raised() -> None:
    async def slow() -> str:
        import asyncio

        await asyncio.sleep(0.05)
        return "late"

    with pytest.raises(TimeoutError):
        await call_with_retry(slow, attempts=2, timeout_s=0.01, backoff_s=0)


def test_token_bucket_refills_on_the_injected_clock() -> None:
    now = {"value": 0.0}
    limiter = TokenBucketLimiter(rate=1, burst=1, clock=lambda: now["value"])
    assert limiter.allow("client")
    assert not limiter.allow("client")
    now["value"] = 1.0
    assert limiter.allow("client")


def test_inflight_gate_rejects_at_capacity() -> None:
    gate = InflightGate(1)
    assert gate.try_enter()
    assert not gate.try_enter()
    gate.exit()
    assert gate.try_enter()


def test_distance_conversion() -> None:
    assert distance_to_score(0.25, "cosine") == pytest.approx(0.75)
    assert distance_to_score(0.25, "dot") == pytest.approx(0.25)


def test_transient_classifier_matches_provider_names() -> None:
    assert is_transient(TransientError("x"))
    assert is_transient(type("ResourceExhausted", (Exception,), {})())
    assert not is_transient(ValueError("bad argument"))
