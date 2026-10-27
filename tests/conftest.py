import json
import shutil
from pathlib import Path
from typing import Any

import pytest

REAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "churn-v1"


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A writable copy of the real artifact so tests can corrupt it freely."""
    target = tmp_path / "churn-v1"
    shutil.copytree(REAL_MODEL_DIR, target)
    return target


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative defaults such as observations/ must never land in the repo during tests."""
    monkeypatch.chdir(tmp_path)


REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
FEATURES: dict[str, dict[str, Any]] = json.loads((REAL_MODEL_DIR / "manifest.json").read_text())[
    "features"
]
