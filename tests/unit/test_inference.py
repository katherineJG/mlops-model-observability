from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.inference import predict
from app.model_loader import LoadedModel, load_model
from app.schemas import Customer

HIGH_RISK = {
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
LOW_RISK = {
    **HIGH_RISK,
    "tenure_months": 70,
    "total_charges": 1400.0,
    "monthly_charges": 20.0,
    "contract": "two-year",
    "internet_service": "none",
    "payment_method": "card",
    "paperless_billing": "no",
}


@pytest.fixture
def model(model_dir: Path) -> LoadedModel:
    return load_model(model_dir)


def test_probability_matches_known_value(model: LoadedModel) -> None:
    # 0.943 was observed by running the pipeline directly on this row during scaffolding
    (result,) = predict(model, [Customer.model_validate(HIGH_RISK)])
    assert result.churn_probability == pytest.approx(0.943, abs=0.001)
    assert result.churn is True


def test_order_is_preserved_and_threshold_applied(model: LoadedModel) -> None:
    customers = [Customer.model_validate(r) for r in (LOW_RISK, HIGH_RISK, LOW_RISK)]
    results = predict(model, customers)
    assert [r.churn for r in results] == [False, True, False]
    assert results[0].churn_probability == results[2].churn_probability


def test_empty_input_rejected(model: LoadedModel) -> None:
    with pytest.raises(ValueError):
        predict(model, [])


def test_concurrent_calls_give_identical_results(model: LoadedModel) -> None:
    customers = [Customer.model_validate(r) for r in (HIGH_RISK, LOW_RISK)]
    expected = predict(model, customers)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: predict(model, customers), range(200)))
    assert all(r == expected for r in results)
