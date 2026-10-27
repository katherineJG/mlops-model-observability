from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from app.model_loader import load_model
from monitoring.config import MonitoringConfig
from monitoring.delayed_evaluation import (
    EvaluationResult,
    LabelSet,
    compute_metrics,
    evaluate,
    load_labels,
    main,
)
from monitoring.reference import load_reference
from monitoring.simulate_incident import simulate
from monitoring.synthetic import generate, score
from tests.conftest import FEATURES, REAL_MODEL_DIR, REFERENCE_DIR

T0 = pd.Timestamp("2026-10-27T10:00:00Z")
AS_OF = datetime(2026, 10, 28, tzinfo=UTC)
CFG = MonitoringConfig(min_samples=10, min_labels=10)
INPUTS = list(FEATURES)


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return load_reference(REFERENCE_DIR)[0]


def make_preds(n: int = 40, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """n predictions with valid features, plus true labels that loosely follow the scores."""
    rng = np.random.default_rng(seed)
    prob = rng.random(n)
    preds = generate(n, seed)[INPUTS].copy()
    preds["event_id"] = [f"e{i}" for i in range(n)]
    preds["timestamp"] = T0 + pd.to_timedelta(np.arange(n), unit="min")
    preds["churn_probability"] = prob
    preds["churn"] = prob >= 0.5
    return preds, (rng.random(n) < prob).astype(int)


def labelset(
    ids: list[str], ys: list[int] | np.ndarray, at: str | None = None, rejected: int = 0
) -> LabelSet:
    stamp = pd.Timestamp(at) if at is not None else pd.NaT
    frame = pd.DataFrame(
        {
            "event_id": ids,
            "label": np.asarray(ys, dtype=float),
            "labelled_at": pd.Series([stamp] * len(ids), dtype="datetime64[ns, UTC]"),
        }
    )
    return LabelSet(frame=frame, rejected=rejected)


def run(
    preds: pd.DataFrame,
    labels: LabelSet,
    reference: pd.DataFrame,
    cfg: MonitoringConfig = CFG,
    as_of: datetime = AS_OF,
) -> EvaluationResult:
    return evaluate(preds, labels, reference, FEATURES, cfg, as_of, decision_threshold=0.5)


def test_metrics_use_labelled_rows_only_and_coverage_is_reported(reference: pd.DataFrame) -> None:
    preds, y = make_preds(40)
    labelled = list(range(0, 40, 2))  # half the predictions have a label
    result = run(
        preds, labelset([f"e{i}" for i in labelled], y[labelled], "2026-10-27T12:00:00Z"), reference
    )
    assert (result.labelled, result.pending, result.coverage) == (20, 20, 0.5)
    assert result.metrics is not None
    assert result.metrics.roc_auc == pytest.approx(
        roc_auc_score(y[labelled], preds["churn_probability"].to_numpy()[labelled])
    )


def test_unlabelled_predictions_cannot_change_the_metrics(reference: pd.DataFrame) -> None:
    preds, y = make_preds(40)
    labelled = list(range(0, 40, 2))
    labels = labelset([f"e{i}" for i in labelled], y[labelled], "2026-10-27T12:00:00Z")
    before = run(preds, labels, reference).metrics
    scrambled = preds.copy()
    odd = scrambled.index[1::2]
    scrambled.loc[odd, "churn_probability"] = 1.0 - scrambled.loc[odd, "churn_probability"]
    scrambled.loc[odd, "churn"] = scrambled.loc[odd, "churn_probability"] >= 0.5
    assert run(scrambled, labels, reference).metrics == before


def test_labels_arriving_after_the_cutoff_are_pending_not_wrong(reference: pd.DataFrame) -> None:
    preds, y = make_preds(40)
    ids = [f"e{i}" for i in range(40)]
    arrival = pd.Series(
        T0 + pd.to_timedelta(np.arange(40) + 60, unit="min")
    )  # 1 h after each prediction
    labels = LabelSet(
        pd.DataFrame({"event_id": ids, "label": y.astype(float), "labelled_at": arrival}),
        rejected=0,
    )
    cutoff = T0.to_pydatetime() + timedelta(minutes=60 + 19)  # the first 20 have arrived
    result = run(preds, labels, reference, as_of=cutoff)
    assert (result.labelled, result.pending, result.late_labels) == (20, 20, 20)
    assert result.median_label_delay_hours == pytest.approx(1.0)
    later = run(preds, labels, reference, as_of=AS_OF)
    assert (later.labelled, later.pending, later.late_labels) == (40, 0, 0)


def test_a_label_cannot_predate_its_prediction(reference: pd.DataFrame) -> None:
    preds, y = make_preds(40)
    ids = [f"e{i}" for i in range(40)]
    start = T0.to_pydatetime()
    arrival = [
        start - timedelta(hours=1) if i < 5 else start + timedelta(hours=3) for i in range(40)
    ]
    labels = LabelSet(
        pd.DataFrame(
            {"event_id": ids, "label": y.astype(float), "labelled_at": pd.Series(arrival)}
        ),
        rejected=0,
    )
    result = run(preds, labels, reference)
    assert result.labelled_before_prediction == 5 and result.labelled == 35


def test_labels_for_other_windows_are_counted_not_joined(reference: pd.DataFrame) -> None:
    preds, y = make_preds(40)
    ids = [f"e{i}" for i in range(40)] + ["elsewhere.1", "elsewhere.2"]
    labels = labelset(ids, [*y, 1, 0], "2026-10-27T12:00:00Z")
    result = run(preds, labels, reference)
    assert result.labels_outside_window == 2 and result.labelled == 40


def test_too_few_labels_gives_no_verdict(reference: pd.DataFrame) -> None:
    preds, y = make_preds(40)
    result = run(preds, labelset([f"e{i}" for i in range(5)], y[:5]), reference)
    assert result.status == "insufficient_labels" and result.metrics is None
    assert result.labelled == 5 and result.coverage == 5 / 40


def test_labels_with_one_class_give_no_auc_and_no_verdict(reference: pd.DataFrame) -> None:
    preds, _ = make_preds(40)
    result = run(preds, labelset([f"e{i}" for i in range(40)], [1] * 40), reference)
    assert result.status == "insufficient_labels"
    assert result.metrics is not None and result.metrics.roc_auc is None


def test_no_predictions_is_reported_as_such(reference: pd.DataFrame) -> None:
    assert run(pd.DataFrame(), labelset([], []), reference).status == "no_predictions"


def test_hard_quality_failure_invalidates_the_evaluation(reference: pd.DataFrame) -> None:
    preds, y = make_preds(40)
    result = run(
        preds.drop(columns=["contract"]), labelset([f"e{i}" for i in range(40)], y), reference
    )
    assert result.status == "invalid_input" and result.metrics is None
    assert result.quality.status == "hard_failure"


def test_duplicate_prediction_rows_are_not_double_counted(reference: pd.DataFrame) -> None:
    preds, y = make_preds(40)
    doubled = pd.concat([preds, preds.head(10)], ignore_index=True)
    result = run(
        doubled, labelset([f"e{i}" for i in range(40)], y, "2026-10-27T12:00:00Z"), reference
    )
    assert result.predictions == 40 and result.labelled == 40


def test_undefined_metrics_are_none_not_zero() -> None:
    y, prob = np.array([1, 0, 1, 0]), np.array([0.1, 0.2, 0.3, 0.4])
    never_positive = compute_metrics(y, prob, np.zeros(4, dtype=int))
    assert never_positive.precision is None and never_positive.f1 is None
    assert never_positive.recall == 0.0 and never_positive.roc_auc is not None
    all_wrong = compute_metrics(y, prob, np.array([0, 1, 0, 1]))
    assert (all_wrong.precision, all_wrong.recall, all_wrong.f1) == (0.0, 0.0, 0.0)
    no_positives = compute_metrics(np.zeros(4, dtype=int), prob, np.ones(4, dtype=int))
    assert no_positives.recall is None and no_positives.roc_auc is None


def test_load_labels_rejects_bad_rows_and_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "event_id,label,labelled_at\n"
        "a,1,2026-10-27T12:00:00+00:00\n"
        "a,1,2026-10-27T13:00:00+00:00\n"  # identical repeat: collapses to the earliest
        "b,1,2026-10-27T12:00:00+00:00\n"
        "b,0,2026-10-27T12:30:00+00:00\n"  # conflicting: both rejected
        "c,maybe,2026-10-27T12:00:00+00:00\n"  # unparseable
        "d,2,2026-10-27T12:00:00+00:00\n"  # not 0 or 1
        "e,0,2026-10-27T12:00:00+00:00\n"
    )
    labels = load_labels(path)
    assert sorted(labels.frame["event_id"]) == ["a", "e"]
    assert labels.rejected == 3  # b, c, d
    a = labels.frame[labels.frame["event_id"] == "a"].iloc[0]
    assert a["labelled_at"] == pd.Timestamp("2026-10-27T12:00:00Z")


def test_labels_without_arrival_times_count_as_arrived(
    tmp_path: Path, reference: pd.DataFrame
) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("event_id,label\n" + "".join(f"e{i},{i % 2}\n" for i in range(40)))
    preds, _ = make_preds(40)
    result = run(preds, load_labels(path), reference)
    assert result.labelled == 40 and result.median_label_delay_hours is None


def test_labels_file_needs_the_documented_columns(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("id,churned\n1,0\n")
    with pytest.raises(ValueError, match="event_id and label"):
        load_labels(path)


def test_healthy_windows_are_rarely_flagged_as_degraded(reference: pd.DataFrame) -> None:
    """The calibration behind min_labels=300 and a 0.08 AUC drop (docs/ARCHITECTURE.md 3.7)."""
    model = load_model(REAL_MODEL_DIR)
    cfg = MonitoringConfig()
    flagged, drops = 0, []
    for seed in range(5000, 5100):
        frame = generate(300, seed)
        preds = frame[INPUTS].copy()
        preds["event_id"] = [f"w{seed}.{i}" for i in range(300)]
        preds["timestamp"] = T0
        preds["churn_probability"] = score(model, frame)
        preds["churn"] = preds["churn_probability"] >= 0.5
        labels = labelset(
            list(preds["event_id"]), frame["churn"].to_numpy(), "2026-10-27T20:00:00Z"
        )
        result = run(preds, labels, reference, cfg=cfg)
        assert result.auc_drop is not None
        drops.append(result.auc_drop)
        flagged += result.status == "degraded"
    print(
        f"BASELINE degraded={flagged}/100 mean_drop={np.mean(drops):.3f} max_drop={max(drops):.3f}"
    )
    assert flagged <= 3


def test_cli_writes_json_and_markdown_and_reports_a_failed_push(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sim = simulate("concept_shift", tmp_path / "sim", model_dir=REAL_MODEL_DIR)
    code = main(
        [
            "--observations", str(sim.observations_dir),
            "--labels", str(sim.labels_path),
            "--reference", str(REFERENCE_DIR),
            "--model-dir", str(REAL_MODEL_DIR),
            "--out", str(tmp_path / "reports"),
            "--start", sim.window_start.isoformat(),
            "--end", sim.window_end.isoformat(),
            "--as-of", (sim.last_label_at + timedelta(minutes=1)).isoformat(),
            "--pushgateway", "http://127.0.0.1:9",  # nothing listens on the discard port
        ]
    )  # fmt: skip
    assert code == 1 and "pushgateway publish failed" in capsys.readouterr().out
    report = next((tmp_path / "reports").glob("*.json")).read_text()
    assert '"status": "degraded"' in report
    markdown = next((tmp_path / "reports").glob("*.md")).read_text()
    assert "Status: degraded" in markdown and "labelled rows only" in markdown
