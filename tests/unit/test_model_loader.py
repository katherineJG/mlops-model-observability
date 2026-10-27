import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pytest

from app.model_loader import MANIFEST_NAME, ModelLoadError, load_model


def _edit_manifest(model_dir: Path, **changes: Any) -> None:
    path = model_dir / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _replace_artifact(model_dir: Path, obj: object) -> None:
    """Swap in a different artifact with a *correct* checksum, so only later checks can catch it."""
    artifact = model_dir / "model.joblib"
    joblib.dump(obj, artifact)
    _edit_manifest(model_dir, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())


def test_real_artifact_loads(model_dir: Path) -> None:
    model = load_model(model_dir)
    assert model.manifest.model_id == "churn-v1"
    assert model.manifest.model_version == "1"


def test_missing_manifest(model_dir: Path) -> None:
    (model_dir / MANIFEST_NAME).unlink()
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "manifest_missing"


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError) as exc:
        load_model(tmp_path / "does-not-exist")
    assert exc.value.reason == "manifest_missing"


@pytest.mark.parametrize("content", ["not json", "[]", '{"model_id": "x"}'])
def test_invalid_manifest(model_dir: Path, content: str) -> None:
    (model_dir / MANIFEST_NAME).write_text(content, encoding="utf-8")
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "manifest_invalid"


def test_missing_artifact(model_dir: Path) -> None:
    (model_dir / "model.joblib").unlink()
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "artifact_missing"


def test_artifact_name_cannot_escape_directory(model_dir: Path) -> None:
    _edit_manifest(model_dir, artifact="../churn-v1/model.joblib")
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "manifest_invalid"


def test_corrupted_artifact_fails_checksum(model_dir: Path) -> None:
    with (model_dir / "model.joblib").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "checksum_mismatch"


def test_garbage_with_matching_checksum_is_unreadable(model_dir: Path) -> None:
    artifact = model_dir / "model.joblib"
    artifact.write_bytes(b"this is not a pickle")
    _edit_manifest(model_dir, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "artifact_unreadable"


def test_wrong_object_type_is_incompatible(model_dir: Path) -> None:
    _replace_artifact(model_dir, {"not": "a model"})
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "artifact_incompatible"


def test_sklearn_version_mismatch(model_dir: Path) -> None:
    _edit_manifest(model_dir, trained_with={"scikit_learn": "0.0.1"})
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "version_mismatch"


def test_manifest_schema_disagrees_with_model(model_dir: Path) -> None:
    manifest = json.loads((model_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    features = dict(manifest["features"])
    del features["gender"]
    _edit_manifest(model_dir, features=features)
    with pytest.raises(ModelLoadError) as exc:
        load_model(model_dir)
    assert exc.value.reason == "schema_mismatch"
