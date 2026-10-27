"""The simulations must be deterministic and must produce the signals the runbook promises."""

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from monitoring.config import MonitoringConfig
from monitoring.delayed_evaluation import evaluate, load_labels
from monitoring.drift_report import analyze_window
from monitoring.reference import load_reference
from monitoring.simulate_incident import (
    DROPPED_FIELD,
    SCENARIOS,
    Simulation,
    error_burst,
    main,
    simulate,
)
from monitoring.window import load_window
from tests.conftest import FEATURES, REAL_MODEL_DIR, REFERENCE_DIR
from tests.contract.test_endpoints import CUSTOMER

CFG = MonitoringConfig()
THRESHOLD = 0.5

# scenario -> (drift status, performance status once the labels have arrived)
EXPECTED = {
    "healthy": ("no_drift", "ok"),
    "input_shift": ("drift", "degraded"),
    "benign_shift": ("drift", "ok"),
    "concept_shift": ("no_drift", "degraded"),
    "schema_break": ("invalid_input", "invalid_input"),
}


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return load_reference(REFERENCE_DIR)[0]


def analyse(
    sim: Simulation, reference: pd.DataFrame, as_of_offset_hours: float | None
) -> tuple[str, str, int]:
    window = load_window(sim.observations_dir, sim.window_start, sim.window_end)
    drift = analyze_window(reference, window.frame, FEATURES, CFG)
    as_of = (
        sim.last_label_at
        if as_of_offset_hours is None
        else sim.window_end + timedelta(hours=as_of_offset_hours)
    )
    performance = evaluate(
        window.frame, load_labels(sim.labels_path), reference, FEATURES, CFG, as_of, THRESHOLD
    )
    return drift.status, performance.status, performance.labelled


def test_every_scenario_has_an_expected_outcome() -> None:
    assert set(EXPECTED) == set(SCENARIOS)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_produces_the_documented_signals(
    scenario: str, tmp_path: Path, reference: pd.DataFrame
) -> None:
    sim = simulate(scenario, tmp_path, model_dir=REAL_MODEL_DIR)
    drift, performance, _ = analyse(sim, reference, None)
    assert (drift, performance) == EXPECTED[scenario]


def test_drift_and_accuracy_loss_are_independent(tmp_path: Path, reference: pd.DataFrame) -> None:
    """The point of the four scenarios: neither signal implies the other."""
    outcomes = {
        s: analyse(simulate(s, tmp_path / s, model_dir=REAL_MODEL_DIR), reference, None)[:2]
        for s in ("input_shift", "benign_shift", "concept_shift")
    }
    assert outcomes["benign_shift"] == ("drift", "ok")  # drift without accuracy loss
    assert outcomes["concept_shift"] == ("no_drift", "degraded")  # accuracy loss without drift
    assert outcomes["input_shift"] == ("drift", "degraded")  # and sometimes both


def test_early_cutoff_leaves_most_labels_pending_and_gives_no_verdict(
    tmp_path: Path, reference: pd.DataFrame
) -> None:
    sim = simulate("concept_shift", tmp_path, model_dir=REAL_MODEL_DIR)
    _, performance, labelled = analyse(sim, reference, 5)
    assert performance == "insufficient_labels"
    assert 0 < labelled < sim.labelled  # some labels have arrived, not all of them


def test_same_seed_gives_identical_files_and_a_different_seed_does_not(tmp_path: Path) -> None:
    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
        }

    simulate("input_shift", tmp_path / "a", model_dir=REAL_MODEL_DIR, seed=7)
    simulate("input_shift", tmp_path / "b", model_dir=REAL_MODEL_DIR, seed=7)
    simulate("input_shift", tmp_path / "c", model_dir=REAL_MODEL_DIR, seed=8)
    assert snapshot(tmp_path / "a") == snapshot(tmp_path / "b")
    assert snapshot(tmp_path / "a") != snapshot(tmp_path / "c")


def test_label_settings_never_change_the_observations(tmp_path: Path) -> None:
    simulate("healthy", tmp_path / "a", model_dir=REAL_MODEL_DIR, label_coverage=0.8)
    simulate("healthy", tmp_path / "b", model_dir=REAL_MODEL_DIR, label_coverage=0.3)
    day = "2026-10-27.jsonl"
    assert (tmp_path / "a/observations" / day).read_bytes() == (
        tmp_path / "b/observations" / day
    ).read_bytes()
    assert (tmp_path / "a/labels.csv").read_bytes() != (tmp_path / "b/labels.csv").read_bytes()


def test_schema_break_removes_exactly_one_field_from_every_observation(tmp_path: Path) -> None:
    sim = simulate("schema_break", tmp_path, model_dir=REAL_MODEL_DIR)
    lines = [
        json.loads(x) for x in (sim.observations_dir / "2026-10-27.jsonl").read_text().splitlines()
    ]
    assert len(lines) == sim.rows
    assert all(set(FEATURES) - set(line["features"]) == {DROPPED_FIELD} for line in lines)


def test_simulated_observations_carry_no_identifiers_beyond_the_event_id(tmp_path: Path) -> None:
    sim = simulate("healthy", tmp_path, model_dir=REAL_MODEL_DIR)
    first = json.loads((sim.observations_dir / "2026-10-27.jsonl").read_text().splitlines()[0])
    assert set(first) == {
        "event_id", "timestamp", "model_id", "model_version",
        "churn_probability", "churn", "latency_ms", "features",
    }  # fmt: skip
    assert first["event_id"].startswith("sim-healthy-7.")


def test_unknown_scenario_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        simulate("meltdown", tmp_path, model_dir=REAL_MODEL_DIR)


def broken_and_healthy_clients(tmp_path: Path) -> tuple[TestClient, TestClient]:
    broken = create_app(load_settings({"MODEL_DIR": str(tmp_path / "missing")}))
    healthy = create_app(load_settings({"MODEL_DIR": str(REAL_MODEL_DIR)}))
    return TestClient(broken), TestClient(healthy)


def test_error_burst_against_a_service_with_no_model_is_all_5xx(tmp_path: Path) -> None:
    broken, healthy = broken_and_healthy_clients(tmp_path)
    with broken, healthy:

        def send_to(client: TestClient) -> Callable[[], int]:
            return lambda: client.post("/predict", json={"instances": [CUSTOMER]}).status_code

        assert dict(error_burst(send_to(broken), 20)) == {"5xx": 20}
        assert dict(error_burst(send_to(healthy), 20)) == {"2xx": 20}
        text = broken.get("/metrics").text
        assert 'http_requests_total{method="POST",route="/predict",status_class="5xx"} 20.0' in text
        assert "model_loaded 0.0" in text  # the signals the 5xx and ModelNotLoaded alerts watch


def test_cli_prints_the_follow_up_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["benign_shift", "--out", str(tmp_path / "sim"), "--model-dir", str(REAL_MODEL_DIR)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "monitoring.drift_report" in out and "monitoring.delayed_evaluation" in out
    assert out.count("--as-of") == 2


def test_cli_needs_an_output_directory_for_offline_scenarios() -> None:
    with pytest.raises(SystemExit):
        main(["healthy"])
