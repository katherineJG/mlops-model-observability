import json
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.observation_sink import JsonlSink, Observation
from monitoring.window import load_window, prune

START = datetime(2026, 10, 27, 10, tzinfo=UTC)
END = START + timedelta(hours=1)


def obs(event_id: str, when: datetime) -> Observation:
    return Observation(
        event_id=event_id,
        timestamp=when.isoformat(timespec="milliseconds"),
        model_id="churn-v1",
        model_version="1",
        churn_probability=0.5,
        churn=True,
        latency_ms=1.0,
        features={"tenure_months": 3, "contract": "one-year"},
    )


def test_window_is_half_open_and_flattens_features(tmp_path: Path) -> None:
    JsonlSink(tmp_path).write(
        [
            obs("before", START - timedelta(milliseconds=1)),
            obs("at-start", START),
            obs("inside", START + timedelta(minutes=30)),
            obs("at-end", END),
        ]
    )
    window = load_window(tmp_path, START, END)
    assert list(window.frame["event_id"]) == ["at-start", "inside"]
    assert (
        window.frame.loc[0, "tenure_months"] == 3 and window.frame.loc[0, "contract"] == "one-year"
    )
    assert "features" not in window.frame.columns
    assert window.malformed_lines == 0


def test_adjacent_windows_do_not_overlap_or_drop_rows(tmp_path: Path) -> None:
    JsonlSink(tmp_path).write([obs(f"e{i}", START + timedelta(minutes=10 * i)) for i in range(12)])
    first = load_window(tmp_path, START, END).frame
    second = load_window(tmp_path, END, END + timedelta(hours=1)).frame
    assert len(first) == 6 and len(second) == 6
    assert set(first["event_id"]).isdisjoint(second["event_id"])


def test_window_spanning_midnight_reads_both_files(tmp_path: Path) -> None:
    start = datetime(2026, 10, 27, 23, 30, tzinfo=UTC)
    JsonlSink(tmp_path).write(
        [obs("late", start + timedelta(minutes=10)), obs("early", start + timedelta(minutes=40))]
    )
    frame = load_window(tmp_path, start, start + timedelta(hours=1)).frame
    assert list(frame["event_id"]) == ["late", "early"]


def test_non_utc_bounds_are_converted(tmp_path: Path) -> None:
    JsonlSink(tmp_path).write([obs("a", START + timedelta(minutes=5))])
    plus2 = timezone(timedelta(hours=2))
    frame = load_window(tmp_path, START.astimezone(plus2), END.astimezone(plus2)).frame
    assert list(frame["event_id"]) == ["a"]


def test_malformed_lines_are_counted_not_fatal(tmp_path: Path) -> None:
    JsonlSink(tmp_path).write([obs("ok", START)])
    path = tmp_path / "2026-10-27.jsonl"
    good = path.read_text()
    bad = [
        "not json",
        "[1, 2]",
        json.dumps({"event_id": "no-timestamp"}),
        json.dumps({"timestamp": "x"}),
    ]
    path.write_text("\n".join(bad) + "\n" + good)
    window = load_window(tmp_path, START, END)
    assert list(window.frame["event_id"]) == ["ok"] and window.malformed_lines == 4


def test_empty_directory_gives_empty_frame(tmp_path: Path) -> None:
    window = load_window(tmp_path, START, END)
    assert window.frame.empty and window.malformed_lines == 0


def test_naive_or_reversed_bounds_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone"):
        load_window(tmp_path, START.replace(tzinfo=None), END)
    with pytest.raises(ValueError, match="after"):
        load_window(tmp_path, END, START)


def test_prune_deletes_only_expired_observation_files(tmp_path: Path) -> None:
    for name in (
        "2026-10-19.jsonl",
        "2026-10-20.jsonl",
        "2026-10-27.jsonl",
        "notes.jsonl",
        "x.txt",
    ):
        (tmp_path / name).write_text("{}\n")
    deleted = prune(tmp_path, retention_days=7, today=date(2026, 10, 27))
    assert [p.name for p in deleted] == ["2026-10-19.jsonl"]  # the 20th is exactly 7 days old
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "2026-10-20.jsonl",
        "2026-10-27.jsonl",
        "notes.jsonl",
        "x.txt",
    ]
