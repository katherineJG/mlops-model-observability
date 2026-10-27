"""Model artifact loading and validation. No HTTP or FastAPI imports."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, ValidationError
from sklearn.pipeline import Pipeline

MANIFEST_NAME = "manifest.json"

# One fixed row, valid under models/<id>/manifest.json, used to prove the loaded model predicts.
SMOKE_ROW: dict[str, Any] = {
    "senior_citizen": 0,
    "tenure_months": 12,
    "monthly_charges": 50.0,
    "total_charges": 600.0,
    "gender": "female",
    "partner": "no",
    "contract": "one-year",
    "internet_service": "dsl",
    "payment_method": "card",
    "paperless_billing": "no",
}


class ModelLoadError(Exception):
    """The model cannot be served. `reason` is a stable code for logs, not for callers."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class _TrainedWith(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scikit_learn: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_id: str
    model_version: str
    artifact: str
    sha256: str
    trained_with: _TrainedWith
    decision_threshold: float
    features: dict[str, dict[str, Any]]  # insertion order = model input column order


@dataclass(frozen=True)
class LoadedModel:
    """A verified model. Treated as immutable after load; see docs/ARCHITECTURE.md."""

    pipeline: Pipeline
    manifest: Manifest


def read_manifest(model_dir: Path) -> Manifest:
    path = model_dir / MANIFEST_NAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        raise ModelLoadError("manifest_missing") from None
    try:
        return Manifest.model_validate(json.loads(raw))
    except (ValueError, ValidationError):
        raise ModelLoadError("manifest_invalid") from None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(model_dir: Path) -> LoadedModel:
    """Verify and load the model in `model_dir`. Raises ModelLoadError, never a partial model.

    joblib is pickle: the checksum is verified BEFORE the file is deserialised, and `model_dir`
    must still be treated as trusted input (see docs/ARCHITECTURE.md).
    """
    manifest = read_manifest(model_dir)

    if Path(manifest.artifact).name != manifest.artifact:  # no path separators or traversal
        raise ModelLoadError("manifest_invalid", "artifact must be a plain file name")
    artifact = model_dir / manifest.artifact
    if not artifact.is_file():
        raise ModelLoadError("artifact_missing")

    if _sha256(artifact) != manifest.sha256.lower():
        raise ModelLoadError("checksum_mismatch")

    if sklearn.__version__ != manifest.trained_with.scikit_learn:
        raise ModelLoadError(
            "version_mismatch",
            f"artifact trained with scikit-learn {manifest.trained_with.scikit_learn}, "
            f"runtime has {sklearn.__version__}",
        )

    try:
        pipeline = joblib.load(artifact)
    except Exception as exc:  # unpickling can raise almost anything
        raise ModelLoadError("artifact_unreadable", type(exc).__name__) from None
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
        raise ModelLoadError("artifact_incompatible", "not a fitted sklearn Pipeline")

    expected = set(getattr(pipeline, "feature_names_in_", []))
    if expected != set(manifest.features):
        raise ModelLoadError("schema_mismatch", "manifest features differ from model inputs")

    model = LoadedModel(pipeline=pipeline, manifest=manifest)
    try:
        _smoke_test(model)
    except Exception as exc:
        raise ModelLoadError("smoke_test_failed", type(exc).__name__) from None
    return model


def _smoke_test(model: LoadedModel) -> None:
    frame = pd.DataFrame([SMOKE_ROW])[list(model.manifest.features)]
    proba = model.pipeline.predict_proba(frame)
    if proba.shape != (1, 2) or not 0.0 <= float(proba[0, 1]) <= 1.0:
        raise ValueError("unexpected predict_proba output")
