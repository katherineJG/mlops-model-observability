"""Prometheus metrics and the failure-isolated observation hook."""

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from app.logging_config import LOGGER_NAME
from app.model_loader import LoadedModel
from app.observation_sink import Observation, ObservationSink
from app.schemas import Customer, Prediction

log = logging.getLogger(LOGGER_NAME)


class Metrics:
    """Prometheus metrics on a private registry; labels: route template, method, status class."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests",
            ["method", "route", "status_class"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "route"],
            registry=self.registry,
        )
        self.instances = Counter(
            "predict_instances_total", "Customers scored", registry=self.registry
        )
        self.model_loaded = Gauge(
            "model_loaded", "1 if a verified model is loaded", registry=self.registry
        )
        self.prediction_score = Histogram(
            "prediction_score",
            "Predicted churn probability per scored row",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )
        self.sink_failures = Counter(
            "observation_sink_failures_total",
            "Observation batches that could not be written",
            registry=self.registry,
        )


def record_observations(
    sink: ObservationSink,
    metrics: Metrics,
    model: LoadedModel,
    event_id: str,
    customers: Sequence[Customer],
    predictions: Sequence[Prediction],
    latency_ms: float,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    """Write one observation per scored row. Never raises: monitoring must not break inference.

    Failures are counted and logged by exception type only, since messages can carry payload.
    """
    try:
        timestamp = clock().isoformat(timespec="milliseconds")
        manifest = model.manifest
        sink.write(
            [
                Observation(
                    event_id=f"{event_id}.{i}",
                    timestamp=timestamp,
                    model_id=manifest.model_id,
                    model_version=manifest.model_version,
                    churn_probability=p.churn_probability,
                    churn=p.churn,
                    latency_ms=round(latency_ms, 2),
                    features={k: v for k, v in c.model_dump().items() if k in manifest.features},
                )
                for i, (c, p) in enumerate(zip(customers, predictions, strict=True))
            ]
        )
    except Exception as exc:
        metrics.sink_failures.inc()
        log.error("observation write failed", extra={"reason": type(exc).__name__})
