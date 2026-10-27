"""Window analysis: data quality, then drift against the reference.

The machine-readable result comes from the tests below, so thresholds are explicit and testable.
The HTML report is rendered by Evidently for people; its own drift flags use Evidently's default
methods and may disagree with ours. The JSON result is authoritative.

Drift is not evidence that the model got worse: only labelled predictions show that.
"""

import argparse
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from scipy import stats

from monitoring.config import MonitoringConfig
from monitoring.data_quality import QualityResult, check_quality
from monitoring.reference import load_reference
from monitoring.window import load_window, prune

NOTE = (
    "Drift means the inputs differ from the reference. It does not show that the model is less "
    "accurate; that needs labelled predictions (see the delayed-label evaluation)."
)
SCORE = "churn_probability"
Status = Literal["no_drift", "drift", "insufficient_data", "invalid_input"]


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    test: str
    statistic: float
    effect_size: float  # KS statistic for numeric columns, Cramer's V for categorical ones
    p_value: float
    p_adjusted: float
    drifted: bool


@dataclass(frozen=True)
class DriftResult:
    status: Status
    reference_rows: int
    current_rows: int
    columns: tuple[ColumnDrift, ...]
    drifted_share: float | None
    score: ColumnDrift | None
    quality: QualityResult
    note: str = NOTE


def _categorical(ref: pd.Series, cur: pd.Series) -> tuple[str, float, float, float]:
    counts = pd.DataFrame(
        {"ref": ref.astype(str).value_counts(), "cur": cur.astype(str).value_counts()}
    ).fillna(0)
    if len(counts) < 2:
        return "chi_square", 0.0, 1.0, 0.0
    chi2, p, _, _ = stats.chi2_contingency(counts.to_numpy().T, correction=False)
    return "chi_square", float(chi2), float(p), math.sqrt(chi2 / float(counts.to_numpy().sum()))


def _numeric(ref: pd.Series, cur: pd.Series) -> tuple[str, float, float, float]:
    result = stats.ks_2samp(
        pd.to_numeric(ref, errors="coerce").dropna(), pd.to_numeric(cur, errors="coerce").dropna()
    )
    return "ks_2samp", float(result.statistic), float(result.pvalue), float(result.statistic)


def _column(
    name: str, ref: pd.Series, cur: pd.Series, spec: Mapping[str, Any], m: int, alpha: float
) -> ColumnDrift:
    test = _categorical if spec.get("allowed") is not None else _numeric
    label, statistic, p, effect = test(ref, cur)
    adjusted = min(1.0, p * m)
    return ColumnDrift(name, label, statistic, effect, p, adjusted, adjusted < alpha)


def analyze_window(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Mapping[str, Mapping[str, Any]],
    cfg: MonitoringConfig,
) -> DriftResult:
    quality = check_quality(current, features, cfg)

    def result(
        status: Status,
        columns: tuple[ColumnDrift, ...] = (),
        share: float | None = None,
        score: ColumnDrift | None = None,
    ) -> DriftResult:
        return DriftResult(status, len(reference), len(current), columns, share, score, quality)

    if quality.status == "insufficient_data":
        return result("insufficient_data")
    if quality.status == "hard_failure":
        return result("invalid_input")
    m = len(features)  # Bonferroni over the model inputs
    columns = tuple(
        _column(n, reference[n], current[n], spec, m, cfg.alpha) for n, spec in features.items()
    )
    share = sum(c.drifted for c in columns) / m
    score = None
    if SCORE in current.columns and SCORE in reference.columns:
        score = _column(SCORE, reference[SCORE], current[SCORE], {}, 1, cfg.alpha)
    status: Status = "drift" if share >= cfg.drift_share_threshold else "no_drift"
    return result(status, columns, share, score)


def write_html(
    reference: pd.DataFrame, current: pd.DataFrame, features: Mapping[str, Any], path: Path
) -> None:
    """Evidently data-drift report for human readers."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    categorical = [n for n, s in features.items() if s.get("allowed") is not None]
    numerical = [n for n in features if n not in categorical] + [SCORE]
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    columns = [*features, SCORE]
    ref = Dataset.from_pandas(reference[columns], data_definition=definition)
    cur = Dataset.from_pandas(current[columns], data_definition=definition)
    Report([DataDriftPreset()]).run(cur, ref).save_html(str(path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyse one window of observations for drift.")
    p.add_argument("--observations", type=Path, default=Path("observations"))
    p.add_argument("--reference", type=Path, default=Path("reference"))
    p.add_argument("--model-dir", type=Path, default=Path("models/churn-v1"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--start", help="ISO 8601 with offset; default is one window before --end")
    p.add_argument("--end", help="ISO 8601 with offset; default is now")
    p.add_argument("--window-hours", type=float, default=1.0)
    p.add_argument("--min-samples", type=int, default=200)
    p.add_argument("--prune", action="store_true", help="apply the retention policy first")
    p.add_argument("--pushgateway", help="URL of a Pushgateway to publish the result to")
    args = p.parse_args(argv)

    cfg = MonitoringConfig(window_hours=args.window_hours, min_samples=args.min_samples)
    end = datetime.fromisoformat(args.end) if args.end else datetime.now(UTC)
    start = (
        datetime.fromisoformat(args.start)
        if args.start
        else end - timedelta(hours=cfg.window_hours)
    )
    features = json.loads((args.model_dir / "manifest.json").read_text())["features"]
    if args.prune:
        for gone in prune(args.observations, cfg.retention_days, datetime.now(UTC).date()):
            print(f"pruned {gone.name}")

    reference, provenance = load_reference(args.reference)
    window = load_window(args.observations, start, end)
    result = analyze_window(reference, window.frame, features, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"drift-{start.astimezone(UTC):%Y%m%dT%H%M%SZ}"
    document = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "reference": provenance,
        "malformed_lines": window.malformed_lines,
        "thresholds": asdict(cfg),
        **asdict(result),
    }
    (args.out / f"{stem}.json").write_text(
        json.dumps(document, indent=2, default=float) + "\n", encoding="utf-8", newline="\n"
    )
    if result.status in ("drift", "no_drift"):
        write_html(reference, window.frame, features, args.out / f"{stem}.html")
    print(f"{result.status}: {args.out / stem}.json")
    if args.pushgateway:
        from monitoring.metrics_export import push

        try:
            push(result, args.pushgateway, time.time())
        except Exception as exc:  # the report is already written; only the publish failed
            print(f"pushgateway publish failed: {type(exc).__name__}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
