"""Read a time window of observations from the JSONL files and apply retention."""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_COLUMNS = ["event_id", "timestamp", "model_id", "model_version", "churn_probability", "churn"]


@dataclass(frozen=True)
class Window:
    frame: pd.DataFrame  # one row per observation, features flattened into columns
    malformed_lines: int  # lines skipped because they were not a JSON object


def load_window(directory: Path, start: datetime, end: datetime) -> Window:
    """Observations with `start <= timestamp < end` (half-open, so adjacent windows never overlap).

    Both bounds must be timezone-aware. Timestamps that cannot be parsed are counted as malformed.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if end <= start:
        raise ValueError("end must be after start")
    records: list[dict[str, object]] = []
    malformed = 0
    day = start.astimezone(UTC).date()
    last = (end.astimezone(UTC) - timedelta(microseconds=1)).date()
    while day <= last:
        path = directory / f"{day.isoformat()}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    ts = datetime.fromisoformat(row["timestamp"])
                    if not isinstance(row, dict) or ts.tzinfo is None:
                        raise ValueError
                except (ValueError, KeyError, TypeError):
                    malformed += 1
                    continue
                if start <= ts < end:
                    features = row.pop("features", None)
                    records.append({**row, **(features if isinstance(features, dict) else {})})
        day += timedelta(days=1)
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return Window(frame=frame, malformed_lines=malformed)


def prune(directory: Path, retention_days: int, today: date) -> list[Path]:
    """Delete observation files dated before `today - retention_days`. Returns what was deleted."""
    cutoff = today - timedelta(days=retention_days)
    deleted: list[Path] = []
    for path in sorted(directory.glob("*.jsonl")):
        try:
            file_day = date.fromisoformat(path.stem)
        except ValueError:
            continue  # not one of ours; never delete files we did not write
        if file_day < cutoff:
            path.unlink()
            deleted.append(path)
    return deleted
