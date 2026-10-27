"""Contract tests: every endpoint's status code and body shape, success and failure."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

import app.main as main
from app.config import load_settings
from app.main import MAX_BODY_BYTES, create_app

CUSTOMER = {
    "senior_citizen": 0,
    "tenure_months": 3,
    "monthly_charges": 80.0,
    "total_charges": 240.0,
    "gender": "female",
    "partner": "no",
    "contract": "month-to-month",
    "internet_service": "fiber",
    "payment_method": "electronic-check",
    "paperless_billing": "yes",
}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@pytest.fixture
def client(model_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(load_settings({"MODEL_DIR": str(model_dir)}))) as c:
        yield c


@pytest.fixture
def degraded(tmp_path: Path) -> Iterator[TestClient]:
    """App whose model directory does not exist: alive but never ready."""
    with TestClient(create_app(load_settings({"MODEL_DIR": str(tmp_path / "missing")}))) as c:
        yield c


def assert_error(response: Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]
    error: dict[str, Any] = body["error"]
    return error


# ---- health and version ----


def test_live(client: TestClient) -> None:
    r = client.get("/health/live")
    assert (r.status_code, r.json()) == (200, {"status": "ok"})


def test_ready(client: TestClient) -> None:
    r = client.get("/health/ready")
    assert (r.status_code, r.json()) == (200, {"status": "ready"})


def test_version(client: TestClient) -> None:
    r = client.get("/version")
    body = r.json()
    assert r.status_code == 200
    assert set(body) == {"app_version", "model_id", "model_version", "source_revision"}
    assert (body["model_id"], body["model_version"]) == ("churn-v1", "1")
    assert body["source_revision"] is None


def test_degraded_service_is_alive_but_not_ready(degraded: TestClient) -> None:
    assert degraded.get("/health/live").status_code == 200
    r = degraded.get("/health/ready")
    assert (r.status_code, r.json()) == (503, {"status": "not_ready"})
    v = degraded.get("/version").json()
    assert (v["model_id"], v["model_version"]) == (None, None)


def test_predict_when_not_ready_is_503(degraded: TestClient) -> None:
    r = degraded.post("/predict", json={"instances": [CUSTOMER]})
    assert_error(r, 503, "model_not_ready")
    assert r.headers["Retry-After"] == "5"


# ---- predict success ----


def test_predict_single_and_batch(client: TestClient) -> None:
    r = client.post("/predict", json={"instances": [CUSTOMER, {**CUSTOMER, "tenure_months": 70}]})
    body = r.json()
    assert r.status_code == 200
    assert set(body) == {"model_id", "model_version", "predictions"}
    assert len(body["predictions"]) == 2
    first, second = body["predictions"]
    assert set(first) == {"churn_probability", "churn"}
    assert first["churn"] is True
    assert first["churn_probability"] > second["churn_probability"]


# ---- predict failures ----


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"instances": [{**CUSTOMER, "tenure_months": 999}]}, "instances.0.tenure_months"),
        ({"instances": [{**CUSTOMER, "contract": "SECRETVALUE"}]}, "instances.0.contract"),
        (
            {"instances": [{k: v for k, v in CUSTOMER.items() if k != "gender"}]},
            "instances.0.gender",
        ),
        ({"instances": [{**CUSTOMER, "surprise": 1}]}, "instances.0.surprise"),
        ({"instances": []}, "instances"),
        ({"instances": [CUSTOMER] * 101}, "instances"),
        ({}, "instances"),
    ],
)
def test_validation_errors_are_structured(
    client: TestClient, payload: dict[str, Any], field: str
) -> None:
    r = client.post("/predict", json=payload)
    err = assert_error(r, 422, "validation_error")
    assert field in [d["field"] for d in err["details"]]
    # input values are never echoed; the random request_id is excluded because it can contain "999"
    echoed = err["message"] + str(err["details"])
    assert "SECRETVALUE" not in echoed and "999" not in echoed


def test_malformed_json_is_422(client: TestClient) -> None:
    r = client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    assert_error(r, 422, "validation_error")


def test_configured_batch_limit_is_enforced(model_dir: Path) -> None:
    settings = load_settings({"MODEL_DIR": str(model_dir), "MAX_BATCH_SIZE": "2"})
    with TestClient(create_app(settings)) as c:
        assert c.post("/predict", json={"instances": [CUSTOMER] * 2}).status_code == 200
        err = assert_error(
            c.post("/predict", json={"instances": [CUSTOMER] * 3}), 422, "validation_error"
        )
    assert err["details"][0]["field"] == "instances"


def test_unknown_path_and_wrong_method(client: TestClient) -> None:
    assert_error(client.get("/nope"), 404, "not_found")
    assert_error(client.get("/predict"), 405, "not_found")


def test_body_over_limit_declared_length(client: TestClient) -> None:
    r = client.post("/predict", content=b"x" * (MAX_BODY_BYTES + 1))
    assert_error(r, 413, "payload_too_large")


def test_body_over_limit_chunked(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(MAX_BODY_BYTES // 1024 + 2):
            yield b"x" * 1024

    r = client.post("/predict", content=chunks())
    assert_error(r, 413, "payload_too_large")


def test_unhandled_exception_does_not_leak(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*_: object) -> None:
        raise RuntimeError("secret /home/user/key")

    monkeypatch.setattr(main, "run_inference", boom)
    r = client.post("/predict", json={"instances": [CUSTOMER]})
    err = assert_error(r, 500, "internal_error")
    assert err["details"] == [] and err["message"] == "Internal server error."
    assert "secret" not in r.text and "/home" not in r.text and "Traceback" not in r.text


# ---- request id, logs, metrics ----


def test_request_id_is_reused_or_generated(client: TestClient) -> None:
    assert (
        client.get("/health/live", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"]
        == "abc-123"
    )
    generated = client.get("/health/live").headers["X-Request-ID"]
    assert UUID_RE.match(generated)
    replaced = client.get("/health/live", headers={"X-Request-ID": "bad id!\t"}).headers[
        "X-Request-ID"
    ]
    assert UUID_RE.match(replaced)


def test_logs_have_request_fields_and_no_payload(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger("app")
    logger.addHandler(
        caplog.handler
    )  # app logger does not propagate, so attach the capture handler
    try:
        with caplog.at_level(logging.INFO, logger="app"):
            client.post(
                "/predict",
                json={"instances": [{**CUSTOMER, "partner": "yes"}]},
                headers={"X-Request-ID": "log-test-1"},
            )
    finally:
        logger.removeHandler(caplog.handler)
    record = next(r for r in caplog.records if getattr(r, "request_id", "") == "log-test-1")
    assert (record.route, record.status, record.model_version, record.batch_size) == (  # type: ignore[attr-defined]
        "/predict",
        200,
        "1",
        1,
    )
    assert "electronic-check" not in "".join(r.getMessage() for r in caplog.records)


def test_metrics_use_route_templates_and_no_request_ids(client: TestClient) -> None:
    client.get("/health/live", headers={"X-Request-ID": "metric-leak-check"})
    client.get("/some/random/path/12345")
    client.post("/predict", json={"instances": [CUSTOMER]})
    text = client.get("/metrics").text
    assert 'route="/predict"' in text and 'route="/health/live"' in text
    assert 'route="unmatched"' in text
    # raw paths and request ids must never become labels (checked by name, not by digits that
    # latency values could contain by chance)
    assert "metric-leak-check" not in text and "some/random" not in text
    assert "predict_instances_total 1.0" in text
    assert "model_loaded 1.0" in text
