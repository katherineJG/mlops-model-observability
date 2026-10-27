import logging
from collections.abc import Sequence

import pytest

from app.instrumentation import Metrics, record_observations
from app.logging_config import JsonFormatter
from app.model_loader import load_model
from app.observation_sink import Observation
from app.schemas import Customer, Prediction
from tests.conftest import REAL_MODEL_DIR

CUSTOMER = Customer(
    senior_citizen=0,
    tenure_months=3,
    monthly_charges=80.0,
    total_charges=240.0,
    gender="female",
    partner="no",
    contract="month-to-month",
    internet_service="fiber",
    payment_method="electronic-check",
    paperless_billing="yes",
)
PREDICTION = Prediction(churn_probability=0.7, churn=True)


class Capture:
    def __init__(self) -> None:
        self.written: list[Observation] = []

    def write(self, observations: Sequence[Observation]) -> None:
        self.written.extend(observations)


class Failing:
    def write(self, observations: Sequence[Observation]) -> None:
        raise OSError(f"disk full while writing {observations[0].features['contract']}")


def test_records_manifest_features_only_with_row_ids() -> None:
    model, sink = load_model(REAL_MODEL_DIR), Capture()
    record_observations(
        sink, Metrics(), model, "abc", [CUSTOMER, CUSTOMER], [PREDICTION, PREDICTION], 3.456
    )
    assert [o.event_id for o in sink.written] == ["abc.0", "abc.1"]
    assert set(sink.written[0].features) == set(model.manifest.features)
    assert sink.written[0].latency_ms == 3.46


def test_sink_failure_is_swallowed_counted_and_logged_without_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model, metrics = load_model(REAL_MODEL_DIR), Metrics()
    logger = logging.getLogger("app")
    logger.addHandler(caplog.handler)  # app logger does not propagate
    try:
        record_observations(Failing(), metrics, model, "abc", [CUSTOMER], [PREDICTION], 1.0)
    finally:
        logger.removeHandler(caplog.handler)
    assert metrics.sink_failures._value.get() == 1.0
    emitted = "".join(JsonFormatter().format(r) for r in caplog.records)
    assert "observation write failed" in emitted and "OSError" in emitted
    assert "month-to-month" not in emitted and "abc" not in emitted
