import pytest
from pydantic import ValidationError

from app.schemas import MAX_BATCH_SIZE, Customer, PredictRequest

VALID: dict[str, object] = {
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


def test_valid_customer_parses() -> None:
    assert Customer.model_validate(VALID).tenure_months == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenure_months", -1),
        ("tenure_months", 73),
        ("monthly_charges", -0.01),
        ("contract", "weekly"),
        ("senior_citizen", 2),
        ("tenure_months", "abc"),
    ],
)
def test_invalid_value_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Customer.model_validate({**VALID, field: value})


def test_missing_and_unknown_fields_rejected() -> None:
    missing = {k: v for k, v in VALID.items() if k != "gender"}
    with pytest.raises(ValidationError):
        Customer.model_validate(missing)
    with pytest.raises(ValidationError):
        Customer.model_validate({**VALID, "extra": 1})


def test_batch_size_limits() -> None:
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"instances": []})
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"instances": [VALID] * (MAX_BATCH_SIZE + 1)})
    assert (
        len(PredictRequest.model_validate({"instances": [VALID] * MAX_BATCH_SIZE}).instances)
        == MAX_BATCH_SIZE
    )
