import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.model_loader import LoadedModel, load_model
from app.observation_sink import JsonlSink
from monitoring.config import MonitoringConfig
from monitoring.drift_report import NOTE, analyze_window, main
from monitoring.reference import build_reference, load_reference
from monitoring.synthetic import generate, observations_from_frame, score
from tests.conftest import FEATURES, REAL_MODEL_DIR, REFERENCE_DIR

CFG = MonitoringConfig()
START = datetime(2026, 10, 27, 10, tzinfo=UTC)


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return load_reference(REFERENCE_DIR)[0]


@pytest.fixture(scope="module")
def model() -> LoadedModel:
    return load_model(REAL_MODEL_DIR)


def window(model: LoadedModel, seed: int, shifted: bool, n: int = 500) -> pd.DataFrame:
    frame = generate(n, seed, shifted)
    frame["churn_probability"] = score(model, frame)
    return frame


def test_stable_window_is_not_drift(reference: pd.DataFrame, model: LoadedModel) -> None:
    result = analyze_window(reference, window(model, seed=1001, shifted=False), FEATURES, CFG)
    assert result.status == "no_drift" and result.drifted_share is not None
    assert result.drifted_share < CFG.drift_share_threshold
    assert result.quality.status == "ok" and result.note == NOTE


def test_shifted_window_is_drift_and_names_the_shifted_inputs(
    reference: pd.DataFrame,
    model: LoadedModel,
) -> None:
    result = analyze_window(reference, window(model, seed=7, shifted=True), FEATURES, CFG)
    flagged = {c.column for c in result.columns if c.drifted}
    assert result.status == "drift"
    assert {"tenure_months", "monthly_charges"} <= flagged
    assert "gender" not in flagged  # not shifted
    assert result.score is not None and result.score.drifted
    assert "accurate" in result.note  # drift is never presented as an accuracy statement


def test_p_values_are_bonferroni_adjusted_over_the_inputs(
    reference: pd.DataFrame, model: LoadedModel
) -> None:
    result = analyze_window(reference, window(model, seed=1001, shifted=False), FEATURES, CFG)
    for c in result.columns:
        assert c.p_adjusted == min(1.0, c.p_value * len(FEATURES))
    assert result.score is not None and result.score.p_adjusted == result.score.p_value


def test_effect_sizes_are_reported_and_ordered(reference: pd.DataFrame, model: LoadedModel) -> None:
    result = analyze_window(reference, window(model, seed=7, shifted=True), FEATURES, CFG)
    by_name = {c.column: c for c in result.columns}
    assert by_name["tenure_months"].test == "ks_2samp" and by_name["contract"].test == "chi_square"
    assert by_name["tenure_months"].effect_size > by_name["gender"].effect_size


def test_undersized_window_reports_insufficient_data_and_no_drift_signal(
    reference: pd.DataFrame,
    model: LoadedModel,
) -> None:
    result = analyze_window(reference, window(model, seed=7, shifted=True, n=199), FEATURES, CFG)
    assert result.status == "insufficient_data"
    assert result.columns == () and result.drifted_share is None and result.score is None


def test_hard_quality_failure_blocks_drift_evaluation(
    reference: pd.DataFrame, model: LoadedModel
) -> None:
    current = window(model, seed=7, shifted=True).drop(columns=["contract"])
    result = analyze_window(reference, current, FEATURES, CFG)
    assert result.status == "invalid_input" and result.columns == ()
    assert result.quality.status == "hard_failure"


def test_unseen_category_is_counted_as_drift_in_that_column(
    reference: pd.DataFrame, model: LoadedModel
) -> None:
    current = window(model, seed=1001, shifted=False)
    current.loc[:99, "payment_method"] = "crypto"  # 20 %: warning, not hard, so drift runs
    result = analyze_window(reference, current, FEATURES, CFG)
    column = next(c for c in result.columns if c.column == "payment_method")
    assert result.quality.status == "warning" and column.drifted


def test_reference_matches_a_fresh_deterministic_build(tmp_path: Path) -> None:
    build_reference(tmp_path, REAL_MODEL_DIR)
    for suffix in ("csv", "json"):
        name = f"reference-v1.{suffix}"
        assert (tmp_path / name).read_bytes() == (REFERENCE_DIR / name).read_bytes()


def test_reference_with_wrong_checksum_is_rejected(tmp_path: Path) -> None:
    shutil.copytree(REFERENCE_DIR, tmp_path / "ref")
    with (tmp_path / "ref" / "reference-v1.csv").open("a") as f:
        f.write("tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        load_reference(tmp_path / "ref")


def run_cli(
    tmp_path: Path, model: LoadedModel, shifted: bool, rows: int
) -> tuple[dict[str, Any], Path]:
    obs_dir, out = tmp_path / "obs", tmp_path / "reports"
    frame = generate(rows, seed=7, shifted=shifted)
    JsonlSink(obs_dir).write(observations_from_frame(frame, model, START, 1.0, "sim"))
    code = main(
        [
            "--observations", str(obs_dir),
            "--reference", str(REFERENCE_DIR),
            "--model-dir", str(REAL_MODEL_DIR),
            "--out", str(out),
            "--start", START.isoformat(),
            "--end", "2026-10-27T11:00:00+00:00",
        ]
    )  # fmt: skip
    assert code == 0
    report = next(out.glob("*.json"))
    return json.loads(report.read_text()), out


def test_cli_writes_json_and_html_for_a_shifted_window(tmp_path: Path, model: LoadedModel) -> None:
    document, out = run_cli(tmp_path, model, shifted=True, rows=500)
    assert document["status"] == "drift" and document["note"] == NOTE
    assert document["reference"]["version"] == "reference-v1"
    assert document["window"] == {"start": START.isoformat(), "end": "2026-10-27T11:00:00+00:00"}
    html = next(out.glob("*.html")).read_text(encoding="utf-8")
    assert "tenure_months" in html


def test_cli_for_an_undersized_window_writes_json_only(tmp_path: Path, model: LoadedModel) -> None:
    document, out = run_cli(tmp_path, model, shifted=True, rows=50)
    assert document["status"] == "insufficient_data" and document["current_rows"] == 50
    assert not list(out.glob("*.html"))


def test_healthy_baseline_false_alarm_rate(reference: pd.DataFrame, model: LoadedModel) -> None:
    """100 seeded healthy windows of 500 rows: the documented baseline in docs/ARCHITECTURE.md."""
    results = [
        analyze_window(reference, window(model, seed, shifted=False), FEATURES, CFG)
        for seed in range(1000, 1100)
    ]
    window_drift = sum(r.status == "drift" for r in results)
    any_feature = sum(any(c.drifted for c in r.columns) for r in results)
    score_flagged = sum(bool(r.score and r.score.drifted) for r in results)
    print(f"BASELINE window_drift={window_drift} any_feature={any_feature} score={score_flagged}")
    assert window_drift == 0
    assert any_feature <= 5 and score_flagged <= 5
