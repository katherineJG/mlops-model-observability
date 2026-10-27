"""Window results as Prometheus metrics, pushed to a Pushgateway.

The drift job is a batch process Prometheus cannot scrape, so it pushes. Label values are limited
to status names, manifest feature names and issue codes: bounded, and never data values.
"""

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from monitoring.delayed_evaluation import EvaluationResult
from monitoring.drift_report import DriftResult

WINDOW_STATUSES = ("no_drift", "drift", "insufficient_data", "invalid_input")
QUALITY_STATUSES = ("ok", "warning", "hard_failure", "insufficient_data")
JOB = "drift_report"


def build_registry(result: DriftResult, now: float) -> CollectorRegistry:
    registry = CollectorRegistry()
    status = Gauge(
        "model_window_status", "1 for the window's drift status", ["status"], registry=registry
    )
    quality = Gauge(
        "model_window_quality_status",
        "1 for the window's data-quality status",
        ["status"],
        registry=registry,
    )
    for s in WINDOW_STATUSES:
        status.labels(s).set(float(result.status == s))
    for s in QUALITY_STATUSES:
        quality.labels(s).set(float(result.quality.status == s))
    Gauge("model_window_rows", "Rows in the analysed window", registry=registry).set(
        result.current_rows
    )
    Gauge(
        "model_monitoring_last_run_timestamp_seconds",
        "Unix time of the last run",
        registry=registry,
    ).set(now)
    issues = Gauge(
        "model_quality_issue_fraction",
        "Fraction of rows (or count for missing_column) hit by a data-quality issue",
        ["code", "column"],
        registry=registry,
    )
    for issue in result.quality.issues:
        issues.labels(issue.code, issue.column or "").set(issue.value)
    if result.drifted_share is not None:
        Gauge(
            "model_drifted_feature_share",
            "Share of model inputs flagged as drifted",
            registry=registry,
        ).set(result.drifted_share)
        effect = Gauge(
            "model_feature_drift_effect_size",
            "KS statistic or Cramer's V per input",
            ["feature"],
            registry=registry,
        )
        flagged = Gauge(
            "model_feature_drifted", "1 if the input is flagged", ["feature"], registry=registry
        )
        for c in result.columns:
            effect.labels(c.column).set(c.effect_size)
            flagged.labels(c.column).set(float(c.drifted))
    if result.score is not None:
        Gauge(
            "model_score_drifted", "1 if the score distribution is flagged", registry=registry
        ).set(float(result.score.drifted))
    return registry


def push(result: DriftResult, url: str, now: float) -> None:
    """Replace this job's metrics on the gateway. Raises on network failure."""
    push_to_gateway(url, job=JOB, registry=build_registry(result, now), timeout=5)


PERFORMANCE_STATUSES = ("ok", "degraded", "insufficient_labels", "invalid_input", "no_predictions")
PERFORMANCE_JOB = "delayed_evaluation"


def build_performance_registry(result: EvaluationResult, now: float) -> CollectorRegistry:
    registry = CollectorRegistry()
    status = Gauge(
        "model_performance_status",
        "1 for the window's performance status",
        ["status"],
        registry=registry,
    )
    for s in PERFORMANCE_STATUSES:
        status.labels(s).set(float(result.status == s))
    plain = {
        "model_performance_last_run_timestamp_seconds": now,
        "model_predictions_in_window": result.predictions,
        "model_labelled_rows": result.labelled,
        "model_pending_labels": result.pending,
        "model_baseline_roc_auc": result.baseline.roc_auc,
        "model_label_coverage": result.coverage,
        "model_roc_auc_drop": result.auc_drop,
        "model_roc_auc": result.metrics.roc_auc if result.metrics else None,
        "model_precision": result.metrics.precision if result.metrics else None,
        "model_recall": result.metrics.recall if result.metrics else None,
    }
    for name, value in plain.items():
        if value is not None:  # an undefined metric is absent, never reported as 0
            Gauge(name, name.removeprefix("model_").replace("_", " "), registry=registry).set(value)
    return registry


def push_performance(result: EvaluationResult, url: str, now: float) -> None:
    """Replace this job's metrics on the gateway. Raises on network failure."""
    push_to_gateway(
        url, job=PERFORMANCE_JOB, registry=build_performance_registry(result, now), timeout=5
    )
