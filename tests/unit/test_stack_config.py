"""Static checks on the Prometheus rules, Grafana dashboard and runbook. No Docker needed."""

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.instrumentation import Metrics
from app.model_loader import load_model
from monitoring.config import MonitoringConfig
from monitoring.delayed_evaluation import evaluate, load_labels
from monitoring.drift_report import analyze_window
from monitoring.metrics_export import build_performance_registry, build_registry
from monitoring.reference import load_reference
from monitoring.simulate_incident import simulate
from monitoring.synthetic import generate, score
from monitoring.window import load_window
from tests.conftest import FEATURES, REAL_MODEL_DIR, REFERENCE_DIR

ROOT = Path(__file__).resolve().parent.parent.parent
RULES = [
    r
    for g in yaml.safe_load((ROOT / "prometheus" / "alerts.yml").read_text())["groups"]
    for r in g["rules"]
]
DASHBOARD: dict[str, Any] = json.loads(
    (ROOT / "dashboards" / "model-service-overview.json").read_text()
)
METRIC_NAME = re.compile(
    r"\b(?:http_|predict_|model_|prediction_|observation_)[a-z_]+\b|\bup\b|\bALERTS\b"
)


def slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")


def emitted_metric_names() -> set[str]:
    reference, _ = load_reference(REFERENCE_DIR)
    model = load_model(REAL_MODEL_DIR)
    frame = generate(500, seed=7, shifted=True)
    frame["churn_probability"] = score(model, frame)
    frame.loc[:20, "contract"] = "five-year"  # so the issue gauge is emitted too
    result = analyze_window(reference, frame, FEATURES, MonitoringConfig())
    with tempfile.TemporaryDirectory() as tmp:  # a fully labelled window, so every gauge is set
        sim = simulate("concept_shift", Path(tmp), model_dir=REAL_MODEL_DIR)
        window = load_window(sim.observations_dir, sim.window_start, sim.window_end)
        performance = evaluate(
            window.frame,
            load_labels(sim.labels_path),
            reference,
            FEATURES,
            MonitoringConfig(),
            sim.last_label_at,
            0.5,
        )
    names = {"up", "ALERTS"}
    suffixes = {"counter": ["_total"], "histogram": ["_bucket", "_sum", "_count"]}
    for registry in (
        Metrics().registry,
        build_registry(result, now=1.0),
        build_performance_registry(performance, now=1.0),
    ):
        for family in registry.collect():
            # a labelled metric with no children yet emits no samples, so derive its names
            names.add(family.name)
            names.update(family.name + s for s in suffixes.get(family.type, []))
            names.update(s.name for s in family.samples)
    return names


def test_every_alert_has_severity_description_and_a_runbook_section() -> None:
    headings = {
        slug(line.removeprefix("## "))
        for line in (ROOT / "docs" / "RUNBOOK.md").read_text().splitlines()
        if line.startswith("## ")
    }
    assert len(RULES) == 9
    for rule in RULES:
        assert rule["labels"]["severity"] in {"page", "warn"}, rule["alert"]
        notes = rule["annotations"]
        assert {"summary", "description", "runbook_url"} <= set(notes), rule["alert"]
        assert len(notes["description"]) > 40, rule["alert"]
        path, _, anchor = notes["runbook_url"].partition("#")
        assert path == "docs/RUNBOOK.md" and anchor in headings, rule["alert"]


def test_drift_alert_never_claims_accuracy_loss() -> None:
    drift = next(r for r in RULES if r["alert"] == "DataDriftDetected")
    assert "does not show that the model is less accurate" in drift["annotations"]["description"]


@pytest.mark.parametrize("source", ["alerts", "dashboard"])
def test_every_metric_used_is_one_that_something_emits(source: str) -> None:
    if source == "alerts":
        exprs = [r["expr"] for r in RULES]
    else:
        exprs = [t["expr"] for p in DASHBOARD["panels"] for t in p.get("targets", [])]
    emitted = emitted_metric_names()
    used = {m for e in exprs for m in METRIC_NAME.findall(e)}
    assert used, "no metrics found; the extraction pattern is broken"
    assert used <= emitted, f"unknown metrics: {sorted(used - emitted)}"


def test_dashboard_is_provisionable() -> None:
    assert DASHBOARD["uid"] == "model-service-overview"
    panels = [p for p in DASHBOARD["panels"] if p["type"] != "row"]
    assert len(panels) == 21  # plus 4 row headers
    assert len({p["id"] for p in DASHBOARD["panels"]}) == len(DASHBOARD["panels"])
    for p in panels:
        assert p["datasource"] == {"type": "prometheus", "uid": "prometheus"}, p["title"]
        assert p["targets"] and all(t["expr"] for t in p["targets"]), p["title"]
    datasource = yaml.safe_load(
        (ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text()
    )["datasources"][0]
    assert datasource["uid"] == "prometheus"


def test_compose_pins_every_third_party_image_by_digest() -> None:
    services = yaml.safe_load((ROOT / "compose.yaml").read_text())["services"]
    for name, spec in services.items():
        if "build" not in spec:
            assert re.search(r"@sha256:[0-9a-f]{64}$", spec["image"]), name
    for name, spec in services.items():  # nothing is exposed beyond loopback
        for port in spec.get("ports", []):
            assert port.startswith("127.0.0.1:"), (name, port)
