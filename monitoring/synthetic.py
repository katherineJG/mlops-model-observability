"""Deterministic healthy and shifted synthetic data.

The healthy generator is a copy of Project 01's `generate_synthetic` (this repo is meant to run on
its own), extended with an optional input shift. The same (n_rows, seed, shifted) always gives the
same frame.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.model_loader import LoadedModel
from app.observation_sink import Observation

CONTRACTS = ["month-to-month", "one-year", "two-year"]
INTERNET = ["dsl", "fiber", "none"]
PAYMENT = ["card", "bank-transfer", "electronic-check", "mailed-check"]


def generate(
    n_rows: int,
    seed: int,
    shifted: bool = False,
    concept_shift: bool = False,
    benign_shift: bool = False,
) -> pd.DataFrame:
    """Telco-like churn data with a `churn` label.

    `shifted` moves the input distribution: newer customers, more month-to-month contracts and
    higher monthly charges. How churn depends on the inputs is unchanged, but the population is
    narrower, so the model ranks it less well: drift and a fall in accuracy happen together here.

    `benign_shift` moves only inputs that do not drive the label (gender, senior_citizen, partner).
    Drift is detected and accuracy is unaffected.

    `concept_shift` keeps the inputs exactly as they would be without it but reverses how tenure,
    charges and contract drive churn. The model, trained on the old relationship, then ranks
    customers the wrong way round: accuracy falls while the inputs show no drift. It is an
    extreme case chosen to make the effect unmistakable, not a realistic one.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 25 if shifted else 73, n_rows)
    contract = rng.choice(
        CONTRACTS, n_rows, p=[0.85, 0.10, 0.05] if shifted else [0.55, 0.25, 0.20]
    )
    internet = rng.choice(INTERNET, n_rows, p=[0.35, 0.45, 0.20])
    base = np.select([internet == "dsl", internet == "fiber"], [45.0, 80.0], default=20.0)
    base = base + (15.0 if shifted else 0.0)
    monthly = np.round(base + rng.normal(0, 5, n_rows).clip(-10, 10), 2)
    total = np.round(monthly * tenure + rng.normal(0, 3, n_rows).clip(0, None) * (tenure > 0), 2)
    paperless = rng.choice(["yes", "no"], n_rows, p=[0.6, 0.4])
    payment = rng.choice(PAYMENT, n_rows)
    logit = (
        -0.5
        - 0.045 * tenure
        + 0.02 * (monthly - 60)
        + 1.2 * (contract == "month-to-month")
        - 0.8 * (contract == "two-year")
        + 0.5 * (payment == "electronic-check")
        + 0.3 * (paperless == "yes")
    )
    if concept_shift:  # same terms with the signs of tenure, charges and contract reversed
        logit = (
            -0.5
            + 0.045 * tenure
            - 0.02 * (monthly - 60)
            - 1.2 * (contract == "month-to-month")
            + 0.8 * (contract == "two-year")
            + 0.5 * (payment == "electronic-check")
            + 0.3 * (paperless == "yes")
        )
    churn = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "gender": rng.choice(
                ["female", "male"], n_rows, p=[0.9, 0.1] if benign_shift else None
            ),
            "senior_citizen": rng.choice(
                [0, 1], n_rows, p=[0.5, 0.5] if benign_shift else [0.84, 0.16]
            ),
            "partner": rng.choice(["yes", "no"], n_rows, p=[0.9, 0.1] if benign_shift else None),
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "monthly_charges": monthly,
            "total_charges": total,
            "churn": churn,
        }
    )


def score(model: LoadedModel, frame: pd.DataFrame) -> np.ndarray:
    """Churn probabilities from the real model, in the frame's row order."""
    columns = list(model.manifest.features)
    return np.asarray(model.pipeline.predict_proba(frame[columns])[:, 1], dtype=float)


def observations_from_frame(
    frame: pd.DataFrame,
    model: LoadedModel,
    start: datetime,
    seconds_between: float,
    batch_prefix: str,
) -> list[Observation]:
    """Turn generated rows into observations as the service would have written them.

    Row i gets timestamp `start + i * seconds_between` and event_id `<batch_prefix>.<i>`.
    """
    manifest = model.manifest
    columns = list(manifest.features)
    probabilities = score(model, frame)
    return [
        Observation(
            event_id=f"{batch_prefix}.{i}",
            timestamp=(start + timedelta(seconds=i * seconds_between)).isoformat(
                timespec="milliseconds"
            ),
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            churn_probability=float(p),
            churn=bool(p >= manifest.decision_threshold),
            latency_ms=1.0,
            features={c: row[c] for c in columns},
        )
        for i, (row, p) in enumerate(
            zip(frame[columns].to_dict("records"), probabilities, strict=True)
        )
    ]
