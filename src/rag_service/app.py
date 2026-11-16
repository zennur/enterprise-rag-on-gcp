"""FastAPI application: query, enqueue, Pub/Sub push, and health."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.base import RequestResponseEndpoint

from rag_service import __version__
from rag_service.config import Settings
from rag_service.container import Container, build_container
from rag_service.models import Answer, Citation, Document
from rag_service.observability import configure_logging, configure_tracing
from rag_service.resilience import ProviderError, TransientError

logger = structlog.get_logger(__name__)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    cached: bool
    rewritten: bool


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[Document] = Field(min_length=1, max_length=20)


class IngestResponse(BaseModel):
    accepted: int
    document_ids: list[str]
    processed_inline: bool


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    reason: str


def create_app(
    settings: Settings | None = None,
    container: Container | None = None,
) -> FastAPI:
    selected = settings if settings is not None else Settings()
    if container is None:
        container = build_container(selected)
    configure_logging(container.settings)
    configure_tracing(container.settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await app.state.container.startup()
        yield
        await app.state.container.shutdown()

    app = FastAPI(
        title="Enterprise RAG",
        version=__version__,
        summary="Grounded question answering over an internal corpus, shaped for Cloud Run.",
        lifespan=lifespan,
    )
    app.state.container = container

    @app.middleware("http")
    async def access_log(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = (
            request.headers.get("x-request-id")
            or hashlib.sha256(f"{time.time_ns()}:{id(request)}".encode()).hexdigest()[:16]
        )
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        tracer = trace.get_tracer("rag_service")
        started = time.perf_counter()
        status = 500
        try:
            with tracer.start_as_current_span("http.server") as span:
                span.set_attribute("http.request.method", request.method)
                span.set_attribute("url.path", request.url.path)
                response = await call_next(request)
                status = response.status_code
                span.set_attribute("http.response.status_code", status)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        codes = {
            401: "unauthorized",
            404: "not_found",
            429: "rate_limited",
            503: "unavailable",
        }
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": codes.get(exc.status_code, "error"), "message": message}},
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(ProviderError)
    async def provider_error(_request: Request, exc: ProviderError) -> JSONResponse:
        logger.warning("provider_error", error=str(exc))
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "unavailable", "message": str(exc)}},
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(TransientError)
    async def transient_error(_request: Request, _exc: TransientError) -> JSONResponse:
        logger.warning("provider_transient_exhausted")
        return JSONResponse(
            status_code=503,
            content={
                "error": {"code": "unavailable", "message": "Upstream temporarily unavailable"}
            },
            headers={"Retry-After": "1"},
        )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz(active: Container = Depends(get_container)) -> JSONResponse:
        ok, reason = await active.ready()
        body = ReadyResponse(status="ok" if ok else "not_ready", reason=reason)
        return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())

    @app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(limit_request)])
    async def query(
        payload: QueryRequest,
        active: Container = Depends(get_container),
    ) -> QueryResponse:
        _require_role(active.settings.role, "api")
        answer = await active.agent.answer(payload.query, payload.top_k)
        return _query_response(answer)

    @app.post(
        "/v1/documents",
        response_model=IngestResponse,
        status_code=202,
        dependencies=[Depends(limit_request)],
    )
    async def documents(
        payload: IngestRequest,
        active: Container = Depends(get_container),
    ) -> IngestResponse:
        _require_role(active.settings.role, "api")
        await active.publisher.publish(payload.documents)
        return IngestResponse(
            accepted=len(payload.documents),
            document_ids=[document.document_id for document in payload.documents],
            processed_inline=active.settings.ingest_transport == "inline",
        )

    @app.post("/internal/pubsub", status_code=204)
    async def pubsub_push(
        request: Request,
        active: Container = Depends(get_container),
    ) -> Response:
        _require_role(active.settings.role, "worker")
        await _verify_push_token(request, active.settings)
        message_id = ""
        try:
            envelope = await request.json()
            message = envelope.get("message", {}) if isinstance(envelope, dict) else {}
            message_id = str(message.get("messageId", ""))
            data = message.get("data")
            if not isinstance(data, str):
                raise ValueError("missing data")
            document = Document.model_validate_json(base64.b64decode(data))
        except (ValidationError, ValueError) as exc:
            logger.warning(
                "poison_pubsub_message", message_id=message_id, reason=type(exc).__name__
            )
            raise HTTPException(status_code=503, detail="Invalid ingest message") from exc
        try:
            await active.ingest.handle_batch([document])
        except (ProviderError, TransientError):
            raise
        except Exception as exc:
            logger.exception("ingest_failed", document_id=document.document_id)
            raise HTTPException(status_code=503, detail="Ingest failed") from exc
        return Response(status_code=204)

    return app


def get_container(request: Request) -> Container:
    container = request.app.state.container
    if not isinstance(container, Container):
        raise RuntimeError("application container is not configured")
    return container


async def limit_request(
    request: Request, active: Container = Depends(get_container)
) -> AsyncIterator[None]:
    _require_api_key(request, active.settings)
    if not active.limiter.allow(_client_key(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
            headers={"Retry-After": "1"},
        )
    if not active.gate.try_enter():
        raise HTTPException(
            status_code=429,
            detail="Too many in-flight requests",
            headers={"Retry-After": "1"},
        )
    try:
        yield
    finally:
        active.gate.exit()


def _require_role(role: str, accepted: str) -> None:
    if role != "all" and role != accepted:
        raise HTTPException(status_code=404, detail="Not found")


def _require_api_key(request: Request, settings: Settings) -> None:
    configured = [item.strip() for item in settings.api_keys.split(",") if item.strip()]
    if not configured:
        return
    presented = request.headers.get("x-api-key", "")
    if any(secrets.compare_digest(presented, key) for key in configured):
        return
    raise HTTPException(status_code=401, detail="Missing or invalid API key")


def _client_key(request: Request) -> str:
    api_key = request.headers.get("x-api-key", "")
    if api_key:
        digest = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        return f"key:{digest}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


async def _verify_push_token(request: Request, settings: Settings) -> None:
    if not settings.pubsub_push_audience:
        return
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing push token")
    token = header[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing push token")
    try:
        email = await asyncio.to_thread(
            _google_email,
            token,
            settings.pubsub_push_audience,
        )
    except Exception as exc:
        logger.warning("push_token_rejected", error=type(exc).__name__)
        raise HTTPException(status_code=401, detail="Invalid push token") from exc
    expected = settings.pubsub_push_service_account
    if expected and email != expected:
        raise HTTPException(status_code=401, detail="Unexpected push identity")


def _google_email(token: str, audience: str) -> str:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    info = id_token.verify_oauth2_token(token, google_requests.Request(), audience=audience)
    email = info.get("email", "")
    if not isinstance(email, str):
        return ""
    return email


def _query_response(answer: Answer) -> QueryResponse:
    return QueryResponse(
        answer=answer.answer,
        citations=answer.citations,
        abstained=answer.abstained,
        cached=answer.cached,
        rewritten=answer.rewritten,
    )
