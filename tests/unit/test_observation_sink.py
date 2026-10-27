import json
from pathlib import Path

from app.observation_sink import JsonlSink, Observation

T1 = "2026-10-27T10:00:00.000+00:00"


def obs(event_id: str, timestamp: str) -> Observation:
    return Observation(
        event_id=event_id,
        timestamp=timestamp,
        model_id="churn-v1",
        model_version="1",
        churn_probability=0.25,
        churn=False,
        latency_ms=1.5,
        features={"tenure_months": 3, "contract": "one-year"},
    )


def test_writes_one_json_line_per_observation(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "obs")
    sink.write([obs("a.0", T1), obs("a.1", T1)])
    sink.write([obs("b.0", "2026-10-27T11:00:00.000+00:00")])
    lines = (tmp_path / "obs" / "2026-10-27.jsonl").read_text().splitlines()
    assert [json.loads(line)["event_id"] for line in lines] == ["a.0", "a.1", "b.0"]
    assert json.loads(lines[0]) == {
        "event_id": "a.0",
        "timestamp": T1,
        "model_id": "churn-v1",
        "model_version": "1",
        "churn_probability": 0.25,
        "churn": False,
        "latency_ms": 1.5,
        "features": {"tenure_months": 3, "contract": "one-year"},
    }


def test_file_is_chosen_by_observation_date(tmp_path: Path) -> None:
    JsonlSink(tmp_path).write(
        [obs("a.0", "2026-10-27T23:59:59.999+00:00"), obs("b.0", "2026-10-28T00:00:00.000+00:00")]
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-10-27.jsonl", "2026-10-28.jsonl"]


def test_empty_write_creates_nothing(tmp_path: Path) -> None:
    JsonlSink(tmp_path / "obs").write([])
    assert not (tmp_path / "obs").exists()
