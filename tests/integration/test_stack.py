"""Brings up the real Docker Compose stack and checks what an operator would rely on.

Skipped unless RUN_STACK_TESTS=1 (needs Docker, network for the first build, and free loopback
ports 8000, 9090, 9091 and 3000). In PowerShell:
    $env:RUN_STACK_TESTS = "1"; uv run python -m pytest tests/integration -v
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from monitoring.simulate_incident import simulate
from monitoring.synthetic import generate

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_STACK_TESTS") != "1", reason="set RUN_STACK_TESTS=1"),
]

ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT = "mlobs-test"
SERVICE, PROM, PUSH, GRAFANA = (f"http://127.0.0.1:{p}" for p in (8000, 9090, 9091, 3000))
GRAFANA_AUTH = "Basic " + base64.b64encode(b"admin:local-demo-only").decode()
PROM_DIGEST = "sha256:efd719c99d83b060d9daefdcf00360461adf279f45ef5391f8d111892118753e"
PROM_IMAGE = f"prom/prometheus:v3.15.0@{PROM_DIGEST}"
SHIFTED_ROWS = 300


def http(url: str, body: Any = None, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def wait_for(check: Callable[[], Any], what: str, timeout: float = 90, interval: float = 3) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            last = check()
            if last:
                return last
        except (urllib.error.URLError, ConnectionError, OSError, KeyError, IndexError):
            pass
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}; last value: {last!r}")


def prom(query: str) -> list[dict[str, Any]]:
    _, body = http(f"{PROM}/api/v1/query?query={urllib.parse.quote(query)}")
    result: list[dict[str, Any]] = body["data"]["result"]
    return result


def scalar(query: str) -> float | None:
    result = prom(query)
    return float(result[0]["value"][1]) if result else None


def compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-p", PROJECT, *args],
        cwd=ROOT, capture_output=True, text=True, timeout=600, check=False,
    )  # fmt: skip


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Start the stack, send shifted traffic, run the drift job. Yields the copied observations."""
    work = tmp_path_factory.mktemp("stack")
    try:
        up = compose("up", "-d", "--build", "--wait", "--wait-timeout", "240")
        assert up.returncode == 0, up.stderr[-2000:]
        wait_for(lambda: http(f"{SERVICE}/health/ready")[0] == 200, "service ready")

        rows = (
            generate(SHIFTED_ROWS, seed=7, shifted=True).drop(columns=["churn"]).to_dict("records")
        )
        for i in range(0, SHIFTED_ROWS, 100):
            status, _ = http(f"{SERVICE}/predict", {"instances": rows[i : i + 100]})
            assert status == 200
        for _ in range(3):  # rejected requests, so the 4xx series exists
            assert http(f"{SERVICE}/predict", {"instances": [{"tenure_months": -1}]})[0] == 422

        copied = compose("cp", "service:/data/observations", str(work / "observations"))
        assert copied.returncode == 0, copied.stderr
        now = datetime.now(UTC)
        job = subprocess.run(
            [
                sys.executable, "-m", "monitoring.drift_report",
                "--observations", str(work / "observations"),
                "--out", str(work / "reports"),
                "--start", (now - timedelta(hours=1)).isoformat(),
                "--end", (now + timedelta(minutes=5)).isoformat(),
                "--pushgateway", PUSH,
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=300, check=False,
        )  # fmt: skip
        assert job.returncode == 0, job.stdout + job.stderr[-2000:]
        assert job.stdout.startswith("drift:"), job.stdout

        # Performance needs labels, which the live service has none of: use the seeded
        # concept-shift simulation (accuracy falls, inputs do not drift) and push its evaluation.
        sim = simulate("concept_shift", work / "sim", model_dir=ROOT / "models" / "churn-v1")
        evaluation = subprocess.run(
            [
                sys.executable, "-m", "monitoring.delayed_evaluation",
                "--observations", str(sim.observations_dir),
                "--labels", str(sim.labels_path),
                "--out", str(work / "performance"),
                "--start", sim.window_start.isoformat(),
                "--end", sim.window_end.isoformat(),
                "--as-of", sim.last_label_at.isoformat(),
                "--pushgateway", PUSH,
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=300, check=False,
        )  # fmt: skip
        assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr[-2000:]
        assert evaluation.stdout.startswith("degraded:"), evaluation.stdout
        yield work / "observations"
    finally:
        compose("logs", "--tail", "40", "service")
        compose("down", "-v", "--remove-orphans")


def test_promtool_accepts_the_config_and_all_rule_tests_pass() -> None:
    def promtool(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", "run", "--rm", "-v", f"{ROOT / 'prometheus'}:/etc/prometheus:ro",
             "-w", "/etc/prometheus", "--entrypoint", "promtool", PROM_IMAGE, *args],
            capture_output=True, text=True, timeout=120, check=False,
        )  # fmt: skip

    config = promtool("check", "config", "/etc/prometheus/prometheus.yml")
    assert config.returncode == 0 and "9 rules found" in config.stdout, (
        config.stdout + config.stderr
    )
    tests = promtool("test", "rules", "alerts_test.yml")
    assert tests.returncode == 0 and "SUCCESS" in tests.stdout, tests.stdout + tests.stderr


def test_observations_were_written_by_the_non_root_container(stack: Path) -> None:
    files = list(stack.glob("*.jsonl"))
    assert files
    lines = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    assert len(lines) == SHIFTED_ROWS
    assert all("customer_id" not in line["features"] for line in lines)


def test_prometheus_scrapes_both_targets(stack: Path) -> None:
    def targets_up() -> bool:
        _, body = http(f"{PROM}/api/v1/targets")
        active = body["data"]["activeTargets"]
        return len(active) == 2 and all(t["health"] == "up" for t in active)

    wait_for(targets_up, "both scrape targets up")


def test_service_metrics_reach_prometheus(stack: Path) -> None:
    ok = wait_for(
        lambda: scalar(
            'sum(http_requests_total{job="model-service",route="/predict",status_class="2xx"})'
        ),
        "2xx request counter",
    )
    assert ok == SHIFTED_ROWS / 100
    wait_for(
        lambda: (
            scalar(
                'sum(http_requests_total{job="model-service",route="/predict",status_class="4xx"})'
            )
            == 3
        ),
        "4xx request counter",
    )
    assert scalar('model_loaded{job="model-service"}') == 1
    assert scalar('sum(prediction_score_count{job="model-service"})') == SHIFTED_ROWS
    assert scalar('sum(predict_instances_total{job="model-service"})') == SHIFTED_ROWS
    buckets = 'sum by (le) (http_request_duration_seconds_bucket{route="/predict"})'
    assert prom(f"histogram_quantile(0.95, {buckets})")
    labels = {k for series in prom("http_requests_total") for k in series["metric"]}
    assert labels <= {"__name__", "job", "instance", "method", "route", "status_class"}


def test_alert_rules_are_loaded_and_healthy(stack: Path) -> None:
    # A rule's health is "unknown" until its group has been evaluated once (every 15 s), which a
    # slow runner may not have reached yet. Wait for that, and name the rules if it never happens.
    deadline = time.monotonic() + 60
    while True:
        _, body = http(f"{PROM}/api/v1/rules")
        health = {r["name"]: r["health"] for g in body["data"]["groups"] for r in g["rules"]}
        if len(health) == 9 and set(health.values()) == {"ok"}:
            break
        assert time.monotonic() < deadline, f"alert rules not healthy after 60 s: {health}"
        time.sleep(3)


def test_drift_result_is_pushed_and_the_alert_fires(stack: Path) -> None:
    wait_for(lambda: scalar('model_window_status{status="drift"}') == 1, "drift status scraped")
    assert scalar('model_window_rows{job="drift_report"}') == SHIFTED_ROWS
    assert scalar('model_feature_drifted{feature="tenure_months"}') == 1
    assert scalar('model_feature_drifted{feature="gender"}') == 0

    def drift_alert_firing() -> bool:
        _, body = http(f"{PROM}/api/v1/alerts")
        return any(
            a["labels"]["alertname"] == "DataDriftDetected" and a["state"] == "firing"
            for a in body["data"]["alerts"]
        )

    wait_for(drift_alert_firing, "DataDriftDetected firing", timeout=90)
    _, body = http(f"{PROM}/api/v1/alerts")
    alert = next(
        a for a in body["data"]["alerts"] if a["labels"]["alertname"] == "DataDriftDetected"
    )
    assert alert["annotations"]["runbook_url"] == "docs/RUNBOOK.md#data-drift-detected"


def test_performance_result_is_pushed_and_the_alert_fires(stack: Path) -> None:
    wait_for(
        lambda: scalar('model_performance_status{status="degraded"}') == 1, "performance scraped"
    )
    assert (scalar('model_roc_auc_drop{job="delayed_evaluation"}') or 0) > 0.08
    assert scalar('model_predictions_in_window{job="delayed_evaluation"}') == 500

    def firing() -> bool:
        _, body = http(f"{PROM}/api/v1/alerts")
        return any(
            a["labels"]["alertname"] == "ModelPerformanceDegraded" and a["state"] == "firing"
            for a in body["data"]["alerts"]
        )

    wait_for(firing, "ModelPerformanceDegraded firing", timeout=90)
    _, body = http(f"{PROM}/api/v1/alerts")
    alert = next(
        a for a in body["data"]["alerts"] if a["labels"]["alertname"] == "ModelPerformanceDegraded"
    )
    assert alert["annotations"]["runbook_url"] == "docs/RUNBOOK.md#model-performance-degraded"


def grafana_query(query: dict[str, Any], auth: dict[str, str]) -> tuple[int, Any]:
    """Run one query through Grafana. Retries for 30 s, because Grafana can answer with an error
    body for a moment after start-up; the last status and body are returned either way."""
    deadline = time.monotonic() + 30
    while True:
        status, result = http(
            f"{GRAFANA}/api/ds/query",
            {"queries": [query], "from": "now-30m", "to": "now"},
            headers=auth,
        )
        if (status == 200 and "results" in (result or {})) or time.monotonic() > deadline:
            return status, result
        time.sleep(2)


def test_grafana_dashboard_is_provisioned_and_every_panel_query_runs(stack: Path) -> None:
    auth = {"Authorization": GRAFANA_AUTH}
    wait_for(lambda: http(f"{GRAFANA}/api/health")[0] == 200, "grafana healthy")
    status, ds = http(f"{GRAFANA}/api/datasources/uid/prometheus", headers=auth)
    assert status == 200 and ds["type"] == "prometheus"
    status, loaded = http(f"{GRAFANA}/api/dashboards/uid/model-service-overview", headers=auth)
    assert status == 200
    panels = [p for p in loaded["dashboard"]["panels"] if p["type"] != "row"]
    assert len(panels) == 21

    # rate() needs two scrapes inside its window, and the fixture sent one burst. Keep a trickle
    # going across a few 15 s scrape intervals so the rate panels have something to show.
    row = generate(1, seed=3).drop(columns=["churn"]).to_dict("records")
    for _ in range(9):
        assert http(f"{SERVICE}/predict", {"instances": row})[0] == 200
        time.sleep(5)

    with_data: dict[str, bool] = {}
    for panel in panels:
        target = panel["targets"][0]
        instant = bool(target.get("instant"))
        query = {
            "refId": "A", "datasource": {"type": "prometheus", "uid": "prometheus"},
            "expr": target["expr"], "instant": instant, "range": not instant,
            "intervalMs": 15000, "maxDataPoints": 200,
        }  # fmt: skip
        status, result = grafana_query(query, auth)
        assert status == 200 and "results" in (result or {}), (panel["title"], status, result)
        outcome = result["results"]["A"]
        assert "error" not in outcome, (panel["title"], outcome.get("error"))
        frames = outcome.get("frames", [])
        with_data[panel["title"]] = any(
            f["data"]["values"] and f["data"]["values"][0] for f in frames
        )

    for title in (
        "Model loaded", "Service up", "Request rate by route", "Rows scored per minute",
        "Window drift status", "Drifted input share (flag at 30%)", "Rows in last window",
        "Drift effect size by input (KS or Cramer's V)",
        "Performance status", "Label coverage", "Labelled rows (verdict needs 300)",
        "ROC AUC drop vs baseline (alert above 0.08)",
    ):  # fmt: skip
        assert with_data[title], f"no data in panel {title!r}"


def test_error_burst_and_recovery_follow_the_runbook(stack: Path) -> None:
    """Breaks the service on purpose (no model), checks the signals, then recovers it.

    Runs last. The alerts themselves need 2 to 5 minutes of sustained failure, so this checks
    the metrics they are built on; the alert timing is covered by the promtool rule tests.
    """
    broken = compose(
        "-f", "compose.yaml", "-f", "compose.incident.yaml", "up", "-d", "--no-deps", "service"
    )
    assert broken.returncode == 0, broken.stderr[-2000:]
    wait_for(lambda: http(f"{SERVICE}/health/live")[0] == 200, "broken service answering")
    assert http(f"{SERVICE}/health/ready")[0] == 503

    burst = subprocess.run(
        [sys.executable, "-m", "monitoring.simulate_incident", "error_burst",
         "--requests", "45", "--spacing", "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    assert burst.returncode == 0 and "{'5xx': 45}" in burst.stdout, burst.stdout + burst.stderr

    wait_for(lambda: scalar('model_loaded{job="model-service"}') == 0, "model_loaded drops to 0")
    ratio = (
        'sum(rate(http_requests_total{job="model-service",status_class="5xx"}[1m]))'
        ' / sum(rate(http_requests_total{job="model-service"}[1m]))'
    )
    assert wait_for(lambda: (scalar(ratio) or 0) > 0.5, "5xx ratio far above the 2% threshold")

    fixed = compose("up", "-d", "--no-deps", "--wait", "service")  # compose.yaml alone
    assert fixed.returncode == 0, fixed.stderr[-2000:]
    wait_for(lambda: http(f"{SERVICE}/health/ready")[0] == 200, "service ready again")
    wait_for(lambda: scalar('model_loaded{job="model-service"}') == 1, "model_loaded back to 1")
    assert (
        http(
            f"{SERVICE}/predict",
            {"instances": generate(1, seed=3).drop(columns=["churn"]).to_dict("records")},
        )[0]
        == 200
    )
