"""Join delayed ground-truth labels to earlier predictions and report model performance.

Only labelled rows count. Unlabelled predictions are never treated as correct or incorrect, and
every metric is reported next to the label coverage it was computed on. Labels that have not
arrived by `as_of` are pending, not wrong. A window with a hard data-quality failure is invalid.
"""

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from monitoring.config import MonitoringConfig
from monitoring.data_quality import QualityResult, check_quality
from monitoring.reference import load_reference
from monitoring.window import load_window

Status = Literal["ok", "degraded", "insufficient_labels", "invalid_input", "no_predictions"]
REQUIRED = {"event_id", "timestamp", "churn_probability", "churn"}
NOTE = (
    "Metrics use labelled predictions only, at the coverage shown. Unlabelled predictions are "
    "pending, not correct. A fall in ROC AUC against the reference baseline is the evidence of "
    "degradation; input drift alone is not."
)


@dataclass(frozen=True)
class PerformanceMetrics:
    roc_auc: float | None  # undefined when the labelled rows contain one class only
    precision: float | None  # undefined when nothing was predicted positive
    recall: float | None  # undefined when no labelled row is positive
    f1: float | None
    positive_rate: float  # share of labelled rows that churned


@dataclass(frozen=True)
class LabelSet:
    frame: pd.DataFrame  # event_id, label, labelled_at (NaT when the file gives no arrival time)
    rejected: int  # event IDs dropped: unparseable label, or conflicting duplicates


@dataclass(frozen=True)
class EvaluationResult:
    status: Status
    as_of: str
    predictions: int
    labelled: int
    pending: int  # predictions with no usable label as of `as_of`
    late_labels: int  # of the pending ones, labels that exist but arrive after `as_of`
    coverage: float | None  # labelled / predictions
    labels_outside_window: int  # labels that arrived by `as_of` for events not in this window
    rejected_labels: int
    labelled_before_prediction: int  # excluded: a label cannot predate its prediction
    median_label_delay_hours: float | None
    metrics: PerformanceMetrics | None
    baseline: PerformanceMetrics
    auc_drop: float | None
    quality: QualityResult
    note: str = NOTE


def compute_metrics(
    y_true: np.ndarray, prob: np.ndarray, predicted: np.ndarray
) -> PerformanceMetrics:
    both_classes = len(np.unique(y_true)) == 2
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PerformanceMetrics(
        roc_auc=float(roc_auc_score(y_true, prob)) if both_classes else None,
        precision=precision,
        recall=recall,
        f1=f1,
        positive_rate=float(y_true.mean()),
    )


def load_labels(path: Path) -> LabelSet:
    """Read a CSV with columns event_id, label (0 or 1) and optionally labelled_at (ISO 8601).

    Rows with an unparseable label are rejected. An event ID given conflicting labels is rejected
    entirely, since there is no way to know which is right; identical repeats collapse to one.
    """
    raw = pd.read_csv(path, dtype=str)
    if not {"event_id", "label"} <= set(raw.columns):
        raise ValueError("labels file needs columns event_id and label")
    label = pd.to_numeric(raw["label"], errors="coerce")
    valid = label.isin([0, 1])
    frame = pd.DataFrame({"event_id": raw["event_id"], "label": label})[valid]
    if "labelled_at" in raw.columns:
        frame["labelled_at"] = pd.to_datetime(raw["labelled_at"], utc=True, format="ISO8601")[valid]
    else:  # no arrival times: every label counts as arrived. Keep the dtype tz-aware.
        frame["labelled_at"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    rejected_ids = set(raw["event_id"][~valid])
    conflicting = frame.groupby("event_id")["label"].nunique()
    conflicting_ids = set(conflicting[conflicting > 1].index)
    rejected_ids |= conflicting_ids
    frame = frame[~frame["event_id"].isin(conflicting_ids)]
    frame = frame.sort_values("labelled_at").drop_duplicates("event_id", keep="first")
    return LabelSet(frame=frame.reset_index(drop=True), rejected=len(rejected_ids))


def evaluate(
    predictions: pd.DataFrame,
    labels: LabelSet,
    reference: pd.DataFrame,
    features: Mapping[str, Mapping[str, Any]],
    cfg: MonitoringConfig,
    as_of: datetime,
    decision_threshold: float,
) -> EvaluationResult:
    baseline = compute_metrics(
        reference["churn"].to_numpy(),
        reference["churn_probability"].to_numpy(),
        (reference["churn_probability"] >= decision_threshold).to_numpy().astype(int),
    )
    quality = check_quality(predictions, features, cfg)
    n = len(predictions)

    def result(status: Status, **values: Any) -> EvaluationResult:
        defaults: dict[str, Any] = {
            "labelled": 0, "pending": n, "late_labels": 0, "coverage": None,
            "labels_outside_window": 0, "labelled_before_prediction": 0,
            "median_label_delay_hours": None, "metrics": None, "auc_drop": None,
        }  # fmt: skip
        return EvaluationResult(
            status=status, as_of=as_of.isoformat(), predictions=n,
            rejected_labels=labels.rejected, baseline=baseline, quality=quality,
            **{**defaults, **values},
        )  # fmt: skip

    if n == 0:
        return result("no_predictions")
    if quality.status == "hard_failure" or not REQUIRED <= set(predictions.columns):
        return result("invalid_input")

    preds = predictions.drop_duplicates("event_id")
    n = len(preds)
    known = labels.frame
    arrived = known[known["labelled_at"].isna() | (known["labelled_at"] <= as_of)]
    in_window = set(preds["event_id"])
    late = known[known["labelled_at"] > as_of]
    late_labels = int(late["event_id"].isin(in_window).sum())
    outside = int((~arrived["event_id"].isin(in_window)).sum())

    joined = preds.merge(arrived, on="event_id", how="left")
    has_label = joined["label"].notna()
    before = has_label & (joined["labelled_at"] < joined["timestamp"])
    used = joined[has_label & ~before]
    delay = (used["labelled_at"] - used["timestamp"]).dropna()
    delay_hours = float(delay.dt.total_seconds().median() / 3600) if len(delay) else None
    common: dict[str, Any] = {
        "labelled": len(used), "pending": n - len(used), "late_labels": late_labels,
        "coverage": len(used) / n, "labels_outside_window": outside,
        "labelled_before_prediction": int(before.sum()), "median_label_delay_hours": delay_hours,
    }  # fmt: skip
    if len(used) < cfg.min_labels:
        return result("insufficient_labels", **common)

    metrics = compute_metrics(
        used["label"].to_numpy().astype(int),
        used["churn_probability"].to_numpy(),
        used["churn"].to_numpy().astype(int),
    )
    if metrics.roc_auc is None or baseline.roc_auc is None:
        return result("insufficient_labels", metrics=metrics, **common)  # one class only
    drop = baseline.roc_auc - metrics.roc_auc
    status: Status = "degraded" if drop > cfg.auc_drop_threshold else "ok"
    return result(status, metrics=metrics, auc_drop=drop, **common)


def format_markdown(result: EvaluationResult, start: datetime, end: datetime) -> str:
    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    m, b = result.metrics, result.baseline
    rows = [
        ("ROC AUC", fmt(m.roc_auc if m else None), fmt(b.roc_auc)),
        ("Precision", fmt(m.precision if m else None), fmt(b.precision)),
        ("Recall", fmt(m.recall if m else None), fmt(b.recall)),
        ("F1", fmt(m.f1 if m else None), fmt(b.f1)),
    ]
    coverage = "n/a" if result.coverage is None else f"{result.coverage:.1%}"
    delay = fmt(result.median_label_delay_hours)
    lines = [
        "# Delayed-label performance report",
        "",
        f"Window {start.isoformat()} to {end.isoformat()}, labels as of {result.as_of}.",
        f"**Status: {result.status}**",
        "",
        f"- Predictions in window: {result.predictions}",
        f"- Labelled: {result.labelled} (coverage {coverage}); pending: {result.pending}, "
        f"of which labels arriving after the cut-off: {result.late_labels}",
        f"- Median label delay (hours): {delay}",
        f"- Labels for events outside the window: {result.labels_outside_window}; "
        f"rejected labels: {result.rejected_labels}; "
        f"labels dated before their prediction (excluded): {result.labelled_before_prediction}",
        "",
        "| Metric | This window (labelled rows only) | Reference baseline |",
        "|---|---|---|",
        *[f"| {name} | {value} | {base} |" for name, value, base in rows],
        "",
        result.note,
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Evaluate a window of predictions against delayed labels."
    )
    p.add_argument("--observations", type=Path, default=Path("observations"))
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path("reference"))
    p.add_argument("--model-dir", type=Path, default=Path("models/churn-v1"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--start", required=True, help="ISO 8601 with offset")
    p.add_argument("--end", required=True, help="ISO 8601 with offset")
    p.add_argument("--as-of", help="labels arriving after this time are pending; default is now")
    p.add_argument("--min-labels", type=int, default=MonitoringConfig().min_labels)
    p.add_argument("--pushgateway", help="URL of a Pushgateway to publish the result to")
    args = p.parse_args(argv)

    cfg = MonitoringConfig(min_labels=args.min_labels)
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    reference, provenance = load_reference(args.reference)
    window = load_window(args.observations, start, end)
    result = evaluate(
        window.frame, load_labels(args.labels), reference, manifest["features"], cfg, as_of,
        manifest["decision_threshold"],
    )  # fmt: skip

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"performance-{start.astimezone(UTC):%Y%m%dT%H%M%SZ}"
    document = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "reference": provenance,
        "malformed_lines": window.malformed_lines,
        "thresholds": asdict(cfg),
        **asdict(result),
    }
    (args.out / f"{stem}.json").write_text(json.dumps(document, indent=2, default=float) + "\n")
    (args.out / f"{stem}.md").write_text(format_markdown(result, start, end), encoding="utf-8")
    print(f"{result.status}: {args.out / stem}.json")
    if args.pushgateway:
        from monitoring.metrics_export import push_performance

        try:
            push_performance(result, args.pushgateway, time.time())
        except Exception as exc:  # the report is already written; only the publish failed
            print(f"pushgateway publish failed: {type(exc).__name__}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
