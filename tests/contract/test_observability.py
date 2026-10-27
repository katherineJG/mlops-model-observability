"""Observations, event IDs, failure isolation, label cardinality and data minimisation."""

import json
import logging
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigError, load_settings
from app.logging_config import JsonFormatter
from app.main import create_app
from app.observation_sink import JsonlSink, Observation
from tests.contract.test_endpoints import CUSTOMER

LABELS_RE = re.compile(r"\{([^}]*)\}")


@pytest.fixture
def obs_dir(tmp_path: Path) -> Path:
    return tmp_path / "observations"


@pytest.fixture
def client(model_dir: Path, obs_dir: Path) -> Iterator[TestClient]:
    settings = load_settings({"MODEL_DIR": str(model_dir), "OBSERVATION_DIR": str(obs_dir)})
    with TestClient(create_app(settings)) as c:
        yield c


def read_all(obs_dir: Path) -> list[dict[str, object]]:
    files = sorted(obs_dir.glob("*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines()]


def test_predict_writes_one_observation_per_row_and_returns_event_id(
    client: TestClient, obs_dir: Path
) -> None:
    r = client.post("/predict", json={"instances": [CUSTOMER, {**CUSTOMER, "tenure_months": 40}]})
    assert r.status_code == 200
    batch = r.headers["X-Event-ID"]
    rows = read_all(obs_dir)
    assert [o["event_id"] for o in rows] == [f"{batch}.0", f"{batch}.1"]
    assert [o["churn_probability"] for o in rows] == [
        p["churn_probability"] for p in r.json()["predictions"]
    ]
    assert rows[1]["features"]["tenure_months"] == 40  # type: ignore[index]
    assert rows[0]["model_id"] == "churn-v1" and rows[0]["model_version"] == "1"
    assert float(rows[0]["latency_ms"]) >= 0  # type: ignore[arg-type]


def test_rejected_and_unready_requests_write_nothing(
    client: TestClient, obs_dir: Path, tmp_path: Path
) -> None:
    bad = client.post("/predict", json={"instances": [{**CUSTOMER, "tenure_months": -1}]})
    assert bad.status_code == 422 and "X-Event-ID" not in bad.headers
    degraded = create_app(
        load_settings({"MODEL_DIR": str(tmp_path / "none"), "OBSERVATION_DIR": str(obs_dir)})
    )
    with TestClient(degraded) as c:
        assert c.post("/predict", json={"instances": [CUSTOMER]}).status_code == 503
    assert not obs_dir.exists()


def test_sink_failure_does_not_break_inference(model_dir: Path) -> None:
    class Broken:
        def write(self, observations: Sequence[Observation]) -> None:
            raise RuntimeError("boom")

    settings = load_settings({"MODEL_DIR": str(model_dir)})
    with TestClient(create_app(settings, sink=Broken())) as c:
        r = c.post("/predict", json={"instances": [CUSTOMER]})
        assert r.status_code == 200 and len(r.json()["predictions"]) == 1
        assert "observation_sink_failures_total 1.0" in c.get("/metrics").text


def test_unwritable_directory_does_not_break_inference(model_dir: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a file where the sink needs a directory
    settings = load_settings({"MODEL_DIR": str(model_dir)})
    with TestClient(create_app(settings, sink=JsonlSink(blocker / "obs"))) as c:
        assert c.post("/predict", json={"instances": [CUSTOMER]}).status_code == 200
        assert "observation_sink_failures_total 1.0" in c.get("/metrics").text


def test_metric_labels_are_bounded_and_carry_no_request_data(client: TestClient) -> None:
    event_ids = []
    for i in range(30):
        r = client.post(
            "/predict",
            json={"instances": [{**CUSTOMER, "tenure_months": i, "monthly_charges": 50.0 + i}]},
            headers={"X-Request-ID": f"req-{i}"},
        )
        event_ids.append(r.headers["X-Event-ID"])
    text = client.get("/metrics").text
    label_names = {
        pair.split("=")[0]
        for line in text.splitlines()
        if not line.startswith("#")
        for group in LABELS_RE.findall(line)
        for pair in group.split(",")
        if "=" in pair
    }
    assert label_names <= {"method", "route", "status_class", "le"}
    series = [line for line in text.splitlines() if line.startswith("http_requests_total{")]
    assert len(series) == 1  # 30 distinct requests collapse into one series
    for leaked in [*event_ids, "req-3", "electronic-check", "fiber"]:
        assert leaked not in text


def test_logs_exclude_event_ids_and_feature_values(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger("app")
    logger.addHandler(caplog.handler)  # app logger does not propagate
    try:
        with caplog.at_level(logging.DEBUG, logger="app"):
            r = client.post("/predict", json={"instances": [CUSTOMER]})
    finally:
        logger.removeHandler(caplog.handler)
    emitted = "".join(JsonFormatter().format(rec) for rec in caplog.records)
    assert caplog.records  # the request was logged, so the checks below mean something
    for leaked in (r.headers["X-Event-ID"], "electronic-check", "month-to-month", "fiber"):
        assert leaked not in emitted


def test_observation_dir_must_be_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "f"
    f.write_text("x")
    with pytest.raises(ConfigError, match="OBSERVATION_DIR"):
        load_settings({"OBSERVATION_DIR": str(f)})
