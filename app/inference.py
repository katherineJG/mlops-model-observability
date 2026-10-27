"""Inference service independent of HTTP routing."""

from collections.abc import Sequence

import pandas as pd

from app.model_loader import LoadedModel
from app.schemas import Customer, Prediction


def predict(model: LoadedModel, customers: Sequence[Customer]) -> list[Prediction]:
    """Score already-validated customers. Output order and length match the input."""
    if not customers:
        raise ValueError("at least one customer is required")
    columns = list(model.manifest.features)
    frame = pd.DataFrame([c.model_dump() for c in customers])[columns]
    probabilities = model.pipeline.predict_proba(frame)[:, 1]
    threshold = model.manifest.decision_threshold
    return [
        Prediction(churn_probability=float(p), churn=bool(p >= threshold)) for p in probabilities
    ]
