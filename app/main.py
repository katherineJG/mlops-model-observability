"""FastAPI application: lifespan model load, routes, request context middleware, error mapping."""

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, load_settings
from app.inference import predict as run_inference
from app.instrumentation import Metrics, record_observations
from app.logging_config import LOGGER_NAME, configure_logging
from app.model_loader import LoadedModel, ModelLoadError, load_model
from app.observation_sink import JsonlSink, ObservationSink
from app.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)

MAX_BODY_BYTES = 256 * 1024
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
KNOWN_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

log = logging.getLogger(LOGGER_NAME)


class ModelNotReady(Exception):
    """Raised by routes that need the model while none is loaded."""


class BatchTooLarge(Exception):
    """Batch exceeds the configured MAX_BATCH_SIZE (the schema enforces the hard ceiling of 100)."""


def app_version() -> str:
    try:
        return version("mlops-model-observability")
    except PackageNotFoundError:
        return "unknown"


def error_response(
    status: int,
    code: str,
    message: str,
    request: Request | Scope,
    details: list[ErrorDetail] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    scope = request.scope if isinstance(request, Request) else request
    body = ErrorResponse.model_validate(
        {
            "error": {
                "code": code,
                "message": message,
                "request_id": scope["state"]["request_id"],
                "details": details or [],
            }
        }
    )
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


class RequestContextMiddleware:
    """Pure ASGI middleware: request ID, body size cap, access log, metrics, last-resort 500."""

    def __init__(
        self,
        app: Callable[[Scope, Receive, Send], Awaitable[None]],
        metrics: Metrics,
        model_version: Callable[[], str | None],
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.model_version = model_version

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        supplied = Headers(scope=scope).get("x-request-id", "")
        request_id = supplied if REQUEST_ID_RE.match(supplied) else str(uuid.uuid4())
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id

        status = 500
        started = False
        exceeded = False
        crashed = False
        seen = 0

        async def emit(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        async def guarded_send(message: Message) -> None:
            if not exceeded:  # once the cap is hit we replace whatever the app produces with a 413
                await emit(message)

        async def counting_receive() -> Message:
            nonlocal seen, exceeded
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > MAX_BODY_BYTES:
                    exceeded = True
                    raise RuntimeError("request body too large")
            return message

        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            exceeded = True
        else:
            try:
                await self.app(scope, counting_receive, guarded_send)
            except Exception:
                if not exceeded:
                    crashed = True
                    log.exception("unhandled exception", extra={"request_id": request_id})

        if not started:
            if exceeded:
                response = error_response(
                    413, "payload_too_large", "Request body is too large.", scope
                )
            else:
                crashed = True
                response = error_response(500, "internal_error", "Internal server error.", scope)
            await response(scope, receive, emit)

        self._record(scope, state, request_id, status, time.perf_counter() - start, crashed)

    def _record(
        self,
        scope: Scope,
        state: dict[str, Any],
        request_id: str,
        status: int,
        elapsed: float,
        crashed: bool,
    ) -> None:
        method = scope["method"] if scope["method"] in KNOWN_METHODS else "OTHER"
        route = getattr(scope.get("route"), "path", "unmatched")
        self.metrics.requests.labels(method, route, f"{status // 100}xx").inc()
        self.metrics.latency.labels(method, route).observe(elapsed)
        batch = state.get("batch_size")
        if batch and status == 200:
            self.metrics.instances.inc(batch)
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": method,
                "route": route,
                "status": status,
                "latency_ms": round(elapsed * 1000, 2),
                "model_version": self.model_version(),
                "batch_size": batch,
                "outcome": "error"
                if crashed or status >= 500
                else "ok"
                if status < 400
                else "rejected",
            },
        )


def create_app(settings: Settings | None = None, sink: ObservationSink | None = None) -> FastAPI:
    settings = settings or load_settings()
    sink = sink or JsonlSink(settings.observation_dir)
    configure_logging(settings.log_level)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = None
        try:
            app.state.model = load_model(settings.model_dir)
            log.info(
                "model loaded", extra={"model_version": app.state.model.manifest.model_version}
            )
        except ModelLoadError as exc:
            # Stay up but not ready: liveness stays green, readiness and /predict return 503.
            log.error("model load failed", extra={"reason": exc.reason})
        metrics.model_loaded.set(1 if app.state.model else 0)
        yield

    app = FastAPI(title="mlops-model-observability", version=app_version(), lifespan=lifespan)

    def current_model() -> LoadedModel | None:
        model: LoadedModel | None = app.state.model
        return model

    app.add_middleware(
        RequestContextMiddleware,
        metrics=metrics,
        model_version=lambda: m.manifest.model_version if (m := current_model()) else None,
    )

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Only location and message: the offending input value is deliberately not echoed.
        details = [
            ErrorDetail(field=".".join(str(p) for p in e["loc"][1:]) or "body", issue=e["msg"])
            for e in exc.errors()
        ]
        return error_response(
            422, "validation_error", "Request validation failed.", request, details
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code in (404, 405):
            headers = dict(exc.headers or {})
            return error_response(
                exc.status_code, "not_found", "Resource not found.", request, headers=headers
            )
        return error_response(500, "internal_error", "Internal server error.", request)

    @app.exception_handler(ModelNotReady)
    async def on_not_ready(request: Request, exc: ModelNotReady) -> JSONResponse:
        return error_response(
            503, "model_not_ready", "Model is not ready.", request, headers={"Retry-After": "5"}
        )

    @app.exception_handler(BatchTooLarge)
    async def on_batch(request: Request, exc: BatchTooLarge) -> JSONResponse:
        issue = f"At most {settings.max_batch_size} instances are allowed"
        return error_response(
            422,
            "validation_error",
            "Request validation failed.",
            request,
            [ErrorDetail(field="instances", issue=issue)],
        )

    @app.get("/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready", response_model=HealthResponse, responses={503: {"model": HealthResponse}}
    )
    def ready() -> Response:
        if current_model() is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.get("/version", response_model=VersionResponse)
    def get_version() -> VersionResponse:
        model = current_model()
        return VersionResponse(
            app_version=app_version(),
            model_id=model.manifest.model_id if model else None,
            model_version=model.manifest.model_version if model else None,
            source_revision=settings.source_revision,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def predict(body: PredictRequest, request: Request, response: Response) -> PredictResponse:
        model = current_model()
        if model is None:
            raise ModelNotReady
        if len(body.instances) > settings.max_batch_size:
            raise BatchTooLarge
        request.state.batch_size = len(body.instances)
        started = time.perf_counter()
        predictions = run_inference(model, body.instances)
        latency_ms = (time.perf_counter() - started) * 1000
        for prediction in predictions:
            metrics.prediction_score.observe(prediction.churn_probability)
        event_id = uuid.uuid4().hex
        record_observations(sink, metrics, model, event_id, body.instances, predictions, latency_ms)
        response.headers["X-Event-ID"] = event_id
        return PredictResponse(
            model_id=model.manifest.model_id,
            model_version=model.manifest.model_version,
            predictions=predictions,
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
