"""Request and response schemas; mirrors docs/API_CONTRACT.md and the model manifest."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_SIZE = 100  # hard ceiling; MAX_BATCH_SIZE in config can only lower it


class Customer(BaseModel):
    """One customer record. Unknown fields are rejected so typos fail loudly."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    senior_citizen: Literal[0, 1]
    tenure_months: int = Field(ge=0, le=72)
    monthly_charges: float = Field(ge=0, le=500)
    total_charges: float = Field(ge=0, le=50_000)
    gender: Literal["female", "male"]
    partner: Literal["yes", "no"]
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal["card", "bank-transfer", "electronic-check", "mailed-check"]
    paperless_billing: Literal["yes", "no"]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[Customer] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn: bool  # churn_probability >= the model's decision threshold


class PredictResponse(BaseModel):
    model_id: str
    model_version: str
    predictions: list[Prediction]  # same order and length as the request's instances


class HealthResponse(BaseModel):
    status: Literal["ok", "ready", "not_ready"]


class VersionResponse(BaseModel):
    app_version: str
    model_id: str | None  # null when no model is loaded
    model_version: str | None
    source_revision: str | None = None


class ErrorDetail(BaseModel):
    field: str  # dotted location in the request body, e.g. "instances.0.tenure_months"
    issue: str  # validation message; the offending input value is never echoed


class ErrorBody(BaseModel):
    code: Literal[
        "validation_error", "payload_too_large", "not_found", "model_not_ready", "internal_error"
    ]
    message: str
    request_id: str
    details: list[ErrorDetail] = []


class ErrorResponse(BaseModel):
    error: ErrorBody
