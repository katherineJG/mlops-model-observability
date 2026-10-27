from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from app.model_loader import LoadedModel, load_model
from monitoring.config import MonitoringConfig
from monitoring.delayed_evaluation import EvaluationResult, evaluate, load_labels
from monitoring.drift_report import DriftResult, analyze_window, main
from monitoring.metrics_export import build_performance_registry, build_registry
from monitoring.reference import load_reference
from monitoring.simulate_incident import simulate
from monitoring.synthetic import generate, observations_from_frame, score
from monitoring.window import load_window
from tests.conftest import FEATURES, REAL_MODEL_DIR, REFERENCE_DIR

CFG = MonitoringConfig()


@pytest.fixture(scope="module")
def model() -> LoadedModel:
    return load_model(REAL_MODEL_DIR)


def result_for(
    model: LoadedModel, rows: int, shifted: bool, *, bad_category: bool = False
) -> DriftResult:
    reference, _ = load_reference(REFERENCE_DIR)
    frame = generate(rows, seed=7, shifted=shifted)
    frame["churn_probability"] = score(model, frame)
    if bad_category:
        frame.loc[:20, "contract"] = "five-year"
    return analyze_window(reference, frame, FEATURES, CFG)


def samples(registry: CollectorRegistry) -> dict[str, dict[tuple[tuple[str, str], ...], float]]:
    out: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    for family in registry.collect():
        for s in family.samples:
            out.setdefault(s.name, {})[tuple(sorted(s.labels.items()))] = s.value
    return out


def test_shifted_window_is_exported_with_one_hot_status(model: LoadedModel) -> None:
    got = samples(build_registry(result_for(model, 500, True), now=1234.0))
    assert got["model_window_status"][(("status", "drift"),)] == 1.0
    assert sum(got["model_window_status"].values()) == 1.0  # exactly one state is set
    assert got["model_window_quality_status"][(("status", "ok"),)] == 1.0
    assert got["model_window_rows"][()] == 500
    assert got["model_monitoring_last_run_timestamp_seconds"][()] == 1234.0
    assert got["model_drifted_feature_share"][()] >= 0.3
    assert got["model_feature_drifted"][(("feature", "tenure_months"),)] == 1.0
    assert got["model_feature_drifted"][(("feature", "gender"),)] == 0.0
    assert got["model_score_drifted"][()] == 1.0
    assert len(got["model_feature_drift_effect_size"]) == len(FEATURES)


def test_insufficient_window_exports_no_drift_metrics(model: LoadedModel) -> None:
    got = samples(build_registry(result_for(model, 50, True), now=1.0))
    assert got["model_window_status"][(("status", "insufficient_data"),)] == 1.0
    assert got["model_window_rows"][()] == 50
    for absent in ("model_drifted_feature_share", "model_feature_drifted", "model_score_drifted"):
        assert absent not in got


def test_quality_issues_are_exported_by_code_and_column(model: LoadedModel) -> None:
    got = samples(build_registry(result_for(model, 500, False, bad_category=True), now=1.0))
    assert got["model_window_quality_status"][(("status", "warning"),)] == 1.0
    assert (("code", "unseen_category"), ("column", "contract")) in got[
        "model_quality_issue_fraction"
    ]


def test_labels_are_bounded_and_carry_no_data_values(model: LoadedModel) -> None:
    registry = build_registry(result_for(model, 500, False, bad_category=True), now=1.0)
    allowed = {"status", "feature", "code", "column"}
    for family in registry.collect():
        for s in family.samples:
            assert set(s.labels) <= allowed
            if "feature" in s.labels:
                assert s.labels["feature"] in FEATURES
    text = generate_latest(registry).decode()
    assert "five-year" not in text and "electronic-check" not in text


def test_failed_publish_still_writes_the_report_and_returns_nonzero(
    tmp_path: Path, model: LoadedModel, capsys: pytest.CaptureFixture[str]
) -> None:
    start = datetime(2026, 10, 27, 10, tzinfo=UTC)
    from app.observation_sink import JsonlSink

    frame: pd.DataFrame = generate(300, seed=7, shifted=False)
    JsonlSink(tmp_path / "obs").write(observations_from_frame(frame, model, start, 1.0, "s"))
    code = main(
        [
            "--observations", str(tmp_path / "obs"),
            "--reference", str(REFERENCE_DIR),
            "--model-dir", str(REAL_MODEL_DIR),
            "--out", str(tmp_path / "reports"),
            "--start", start.isoformat(),
            "--end", "2026-10-27T11:00:00+00:00",
            "--pushgateway", "http://127.0.0.1:9",  # nothing listens on the discard port
        ]
    )  # fmt: skip
    assert code == 1
    assert "pushgateway publish failed" in capsys.readouterr().out
    assert list((tmp_path / "reports").glob("*.json"))


def performance_for(tmp_path: Path, scenario: str, early: bool) -> EvaluationResult:
    sim = simulate(scenario, tmp_path, model_dir=REAL_MODEL_DIR)
    reference, _ = load_reference(REFERENCE_DIR)
    window = load_window(sim.observations_dir, sim.window_start, sim.window_end)
    as_of = sim.window_end + timedelta(hours=5) if early else sim.last_label_at
    return evaluate(
        window.frame, load_labels(sim.labels_path), reference, FEATURES, CFG, as_of, 0.5
    )


def test_degraded_performance_is_exported(tmp_path: Path) -> None:
    got = samples(
        build_performance_registry(performance_for(tmp_path, "concept_shift", False), 9.0)
    )
    assert got["model_performance_status"][(("status", "degraded"),)] == 1.0
    assert sum(got["model_performance_status"].values()) == 1.0
    assert got["model_roc_auc_drop"][()] > 0.08
    assert got["model_roc_auc"][()] < got["model_baseline_roc_auc"][()]
    assert 0 < got["model_label_coverage"][()] <= 1
    assert got["model_labelled_rows"][()] + got["model_pending_labels"][()] == 500
    assert got["model_performance_last_run_timestamp_seconds"][()] == 9.0


def test_undefined_performance_metrics_are_absent_not_zero(tmp_path: Path) -> None:
    got = samples(build_performance_registry(performance_for(tmp_path, "healthy", True), 1.0))
    assert got["model_performance_status"][(("status", "insufficient_labels"),)] == 1.0
    for absent in ("model_roc_auc", "model_roc_auc_drop", "model_precision", "model_recall"):
        assert absent not in got
    assert got["model_baseline_roc_auc"][()] > 0.5  # the reference baseline is always known


def test_performance_labels_are_bounded(tmp_path: Path) -> None:
    registry = build_performance_registry(performance_for(tmp_path, "schema_break", False), 1.0)
    for family in registry.collect():
        for s in family.samples:
            assert set(s.labels) <= {"status"}
