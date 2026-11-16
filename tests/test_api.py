import base64

import pytest
from fastapi.testclient import TestClient

from rag_service.app import create_app
from rag_service.container import build_container
from rag_service.vectorstore import MemoryVectorStore


def test_health_and_query_round_trip(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 200
        ingested = client.post(
            "/v1/documents",
            json={
                "documents": [
                    {
                        "document_id": "expenses",
                        "title": "Expense policy",
                        "text": "The mealcap is seventyfive dollars per person.",
                    }
                ]
            },
        )
        assert ingested.status_code == 202
        assert ingested.json()["processed_inline"] is True
        first = client.post("/v1/query", json={"query": "What is the mealcap?"})
        assert first.status_code == 200
        body = first.json()
        assert body["abstained"] is False
        assert body["citations"][0]["document_id"] == "expenses"
        assert "seventyfive" in body["answer"]
        second = client.post("/v1/query", json={"query": "What is the mealcap?"})
        assert second.json()["cached"] is True
        empty = client.post("/v1/query", json={"query": "Who won the table tennis tournament?"})
        assert empty.status_code == 200
        assert empty.json()["abstained"] is True


def test_api_key_rate_limit_and_backpressure(settings) -> None:  # type: ignore[no-untyped-def]
    secured = settings.model_copy(update={"api_keys": "topsecret"})
    with TestClient(create_app(secured)) as client:
        rejected = client.post("/v1/query", json={"query": "hello"})
        assert rejected.status_code == 401
        allowed = client.post(
            "/v1/query",
            json={"query": "hello"},
            headers={"x-api-key": "topsecret"},
        )
        assert allowed.status_code == 200
        assert client.get("/healthz").status_code == 200

    limited = settings.model_copy(update={"rate_limit_rps": 0.0001, "rate_limit_burst": 1})
    with TestClient(create_app(limited)) as client:
        assert client.post("/v1/query", json={"query": "hello"}).status_code == 200
        blocked = client.post("/v1/query", json={"query": "hello"})
        assert blocked.status_code == 429
        assert blocked.headers["retry-after"] == "1"
        assert blocked.json()["error"]["code"] == "rate_limited"

    container = build_container(settings.model_copy(update={"max_inflight": 1}))
    assert container.gate.try_enter()
    with TestClient(create_app(settings, container)) as client:
        busy = client.post("/v1/query", json={"query": "hello"})
        assert busy.status_code == 429
        assert "in-flight" in busy.json()["error"]["message"]


def test_pubsub_push_is_idempotent_and_authenticated(settings) -> None:  # type: ignore[no-untyped-def]
    document = {
        "document_id": "oncall",
        "title": "On-call",
        "text": "A sev1 incident pages the incidentcommander.",
    }
    encoded = base64.b64encode(json_bytes(document)).decode()
    with TestClient(create_app(settings)) as client:
        pushed = client.post(
            "/internal/pubsub", json={"message": {"data": encoded, "messageId": "m1"}}
        )
        assert pushed.status_code == 204
        again = client.post(
            "/internal/pubsub", json={"message": {"data": encoded, "messageId": "m2"}}
        )
        assert again.status_code == 204
        found = client.post("/v1/query", json={"query": "Who is the incidentcommander for sev1?"})
        assert found.json()["abstained"] is False
        assert found.json()["citations"][0]["document_id"] == "oncall"
        poison = client.post(
            "/internal/pubsub", json={"message": {"data": base64.b64encode(b"{}").decode()}}
        )
        assert poison.status_code == 503

    guarded = settings.model_copy(update={"pubsub_push_audience": "rag-worker"})
    with TestClient(create_app(guarded)) as client:
        missing = client.post("/internal/pubsub", json={"message": {"data": encoded}})
        assert missing.status_code == 401


def test_worker_role_hides_the_query_route(settings) -> None:  # type: ignore[no-untyped-def]
    with TestClient(create_app(settings.model_copy(update={"role": "worker"}))) as client:
        assert client.post("/v1/query", json={"query": "hello"}).status_code == 404
        assert client.get("/healthz").status_code == 200


def test_ready_fails_when_the_store_ping_fails(settings) -> None:  # type: ignore[no-untyped-def]
    class DownStore(MemoryVectorStore):
        async def ping(self) -> bool:
            return False

    container = build_container(settings, store=DownStore())
    with TestClient(create_app(settings, container)) as client:
        assert client.get("/healthz").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["status"] == "not_ready"


def json_bytes(document: dict[str, str]) -> bytes:
    import json

    return json.dumps(document).encode()


@pytest.mark.asyncio
async def test_configuration_refuses_fake_provider_on_cloud_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_service.config import Settings, configuration_error

    monkeypatch.setenv("K_SERVICE", "rag-api")
    reason = configuration_error(Settings(log_level="WARNING"))
    assert reason is not None
    assert "fake provider" in reason
    allowed = configuration_error(
        Settings(log_level="WARNING", allow_fake_provider=True, allow_memory_store=True, role="api")
    )
    assert allowed is None
