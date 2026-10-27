"""The committed example reports must be what the documented commands produce today.

JSON is compared with a float tolerance (CI runs on Linux, development on Windows); Markdown
exactly. The Evidently HTML is not compared: it embeds random identifiers.
"""

from pathlib import Path
from typing import Any

import pytest

from monitoring import delayed_evaluation, drift_report
from monitoring.simulate_incident import simulate
from tests.conftest import REAL_MODEL_DIR, REFERENCE_DIR

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "reports" / "examples"
START, END = "2026-10-27T10:00:00+00:00", "2026-10-27T11:00:00+00:00"
AS_OF = "2026-10-28T00:00:00+00:00"
STEM = "20261027T100000Z"


def assert_same(actual: Any, expected: Any, where: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), where
        for key in expected:
            assert_same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_same(a, e, f"{where}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), where
    else:
        assert actual == expected, where


@pytest.mark.parametrize("scenario", ["healthy", "input_shift", "concept_shift"])
def test_example_reports_match_the_documented_commands(scenario: str, tmp_path: Path) -> None:
    import json

    sim = simulate(scenario, tmp_path / "sim", model_dir=REAL_MODEL_DIR)
    out = tmp_path / "out"
    common = ["--reference", str(REFERENCE_DIR), "--model-dir", str(REAL_MODEL_DIR)]
    window = ["--start", START, "--end", END, "--out", str(out)]
    obs = ["--observations", str(sim.observations_dir)]
    assert drift_report.main([*obs, *common, *window]) == 0
    labels = ["--labels", str(sim.labels_path), "--as-of", AS_OF]
    assert delayed_evaluation.main([*obs, *labels, *common, *window]) == 0

    for name in (f"drift-{STEM}.json", f"performance-{STEM}.json"):
        assert_same(
            json.loads((out / name).read_text()),
            json.loads((EXAMPLES / scenario / name).read_text()),
            name,
        )
    md = f"performance-{STEM}.md"
    assert (out / md).read_text() == (EXAMPLES / scenario / md).read_text()


def test_the_html_pair_exists_for_a_healthy_and_a_drifted_window() -> None:
    for scenario in ("healthy", "input_shift"):
        html = (EXAMPLES / scenario / f"drift-{STEM}.html").read_text(encoding="utf-8")
        assert "tenure_months" in html and "<html" in html.lower()
