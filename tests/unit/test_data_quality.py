import pandas as pd
import pytest

from monitoring.config import MonitoringConfig
from monitoring.data_quality import QualityResult, check_quality
from monitoring.synthetic import generate
from tests.conftest import FEATURES

CFG = MonitoringConfig()  # min_samples 200
INPUTS = list(FEATURES)


def healthy(n: int = 400) -> pd.DataFrame:
    frame = generate(n, seed=1)[INPUTS]
    frame.insert(0, "event_id", [f"e.{i}" for i in range(n)])
    return frame


def codes(result: QualityResult) -> dict[str, str]:
    return {f"{i.code}:{i.column}": i.severity for i in result.issues}


def test_healthy_window_is_ok() -> None:
    result = check_quality(healthy(), FEATURES, CFG)
    assert (result.status, result.rows, result.issues) == ("ok", 400, ())


@pytest.mark.parametrize("rows", [0, 199])
def test_undersized_window_is_not_evaluated(rows: int) -> None:
    frame = healthy(400).head(rows).drop(columns=["contract"])  # would be a hard failure
    result = check_quality(frame, FEATURES, CFG)
    assert result.status == "insufficient_data" and result.issues == ()


def test_exactly_minimum_size_is_evaluated() -> None:
    assert check_quality(healthy(200), FEATURES, CFG).status == "ok"


def test_missing_column_is_a_hard_failure() -> None:
    result = check_quality(healthy().drop(columns=["contract"]), FEATURES, CFG)
    assert result.status == "hard_failure"
    assert codes(result) == {"missing_column:contract": "hard"}


def test_unparseable_and_non_integer_values_are_hard_failures() -> None:
    frame = healthy().astype({"tenure_months": object, "monthly_charges": object})
    frame.loc[0, "monthly_charges"] = "abc"
    frame.loc[1, "tenure_months"] = 3.5  # int feature
    result = check_quality(frame, FEATURES, CFG)
    assert result.status == "hard_failure"
    assert codes(result) == {
        "invalid_type:monthly_charges": "hard",
        "invalid_type:tenure_months": "hard",
    }


def test_unseen_category_warns_only_above_threshold() -> None:
    frame = healthy()
    frame.loc[:3, "contract"] = "five-year"  # 4 of 400 = 1.0 %, not above the 1 % threshold
    assert check_quality(frame, FEATURES, CFG).status == "ok"
    frame.loc[:4, "contract"] = "five-year"  # 5 of 400 = 1.25 %
    result = check_quality(frame, FEATURES, CFG)
    assert result.status == "warning" and codes(result) == {"unseen_category:contract": "warning"}


def test_out_of_range_warns() -> None:
    frame = healthy()
    frame.loc[:9, "tenure_months"] = 500  # manifest max is 72
    result = check_quality(frame, FEATURES, CFG)
    assert result.status == "warning" and codes(result) == {"out_of_range:tenure_months": "warning"}


def test_missing_values_warn_above_five_percent() -> None:
    frame = healthy()
    frame.loc[:19, "gender"] = None  # exactly 5 %: not above
    assert check_quality(frame, FEATURES, CFG).status == "ok"
    frame.loc[:20, "gender"] = None
    assert codes(check_quality(frame, FEATURES, CFG)) == {"missing_values:gender": "warning"}


def test_duplicate_event_ids_warn() -> None:
    frame = healthy()
    frame.loc[1, "event_id"] = frame.loc[0, "event_id"]
    result = check_quality(frame, FEATURES, CFG)
    assert result.status == "warning" and codes(result) == {
        "duplicate_event_ids:event_id": "warning"
    }


def test_hard_failure_outranks_warning() -> None:
    frame = healthy().drop(columns=["contract"])
    frame.loc[:20, "gender"] = None
    assert check_quality(frame, FEATURES, CFG).status == "hard_failure"
