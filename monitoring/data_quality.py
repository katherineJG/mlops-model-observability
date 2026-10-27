"""Data-quality checks for one window of observations, against the model manifest.

Hard failures mean the window cannot be trusted; warnings are reported and evaluation continues.
A window below the minimum sample size is `insufficient_data`, never `ok`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from monitoring.config import MonitoringConfig

Status = Literal["ok", "warning", "hard_failure", "insufficient_data"]
Severity = Literal["hard", "warning"]


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    column: str | None
    value: float  # a fraction of rows, or a count for missing_column
    detail: str


@dataclass(frozen=True)
class QualityResult:
    status: Status
    rows: int
    issues: tuple[Issue, ...]


def check_quality(
    frame: pd.DataFrame, features: Mapping[str, Mapping[str, Any]], cfg: MonitoringConfig
) -> QualityResult:
    rows = len(frame)
    if rows < cfg.min_samples:
        return QualityResult("insufficient_data", rows, ())
    issues: list[Issue] = []
    for name, spec in features.items():
        if name not in frame.columns:
            issues.append(Issue("missing_column", "hard", name, 1, "column absent from window"))
            continue
        issues.extend(_check_column(name, frame[name], spec, cfg))
    if "event_id" in frame.columns:
        dup = float(frame["event_id"].duplicated().mean())
        if dup > 0:
            issues.append(Issue("duplicate_event_ids", "warning", "event_id", dup, "repeated IDs"))
    if any(i.severity == "hard" for i in issues):
        status: Status = "hard_failure"
    else:
        status = "warning" if issues else "ok"
    return QualityResult(status, rows, tuple(issues))


def _check_column(
    name: str, series: pd.Series, spec: Mapping[str, Any], cfg: MonitoringConfig
) -> list[Issue]:
    issues: list[Issue] = []
    missing = float(series.isna().mean())
    if missing > cfg.missing_warn_fraction:
        issues.append(Issue("missing_values", "warning", name, missing, "null or absent values"))
    present = series.dropna()
    if present.empty:
        return issues
    allowed = spec.get("allowed")
    if allowed is not None:
        unseen = float((~present.isin(allowed)).mean())
        if unseen > cfg.unseen_warn_fraction:
            issues.append(
                Issue("unseen_category", "warning", name, unseen, "value not in manifest")
            )
        return issues
    numeric = pd.to_numeric(present, errors="coerce")
    bad_type = numeric.isna()
    if spec.get("type") == "int":
        bad_type = bad_type | (numeric % 1 != 0)
    if bad_type.any():
        issues.append(
            Issue("invalid_type", "hard", name, float(bad_type.mean()), "not a valid number")
        )
    valid = numeric[~bad_type]
    low, high = spec.get("min"), spec.get("max")
    outside = pd.Series(False, index=valid.index)
    if low is not None:
        outside |= valid < low
    if high is not None:
        outside |= valid > high
    out_fraction = float(outside.sum() / len(present))
    if out_fraction > cfg.range_warn_fraction:
        issues.append(
            Issue("out_of_range", "warning", name, out_fraction, "outside manifest range")
        )
    return issues
