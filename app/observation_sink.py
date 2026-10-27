"""Safe prediction observations and a replaceable sink.

An Observation holds only what docs/ARCHITECTURE.md section 4 allows: model inputs, the
prediction, and operational context. Nothing outside the model's feature set is accepted.
"""

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Observation:
    event_id: str  # "<batch id>.<row index>"; the join key for delayed labels
    timestamp: str  # UTC ISO 8601
    model_id: str
    model_version: str
    churn_probability: float
    churn: bool
    latency_ms: float  # inference time of the whole request, repeated on each row of a batch
    features: Mapping[str, object]  # manifest features only


class ObservationSink(Protocol):
    def write(self, observations: Sequence[Observation]) -> None: ...


class JsonlSink:
    """Appends one JSON line per observation to `<directory>/YYYY-MM-DD.jsonl` (UTC date).

    shortcut: a lock serialises writers inside one process only; run one service process per
    directory, or swap in a store that handles concurrent writers.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()

    def write(self, observations: Sequence[Observation]) -> None:
        if not observations:
            return
        by_day: dict[str, list[str]] = {}
        for o in observations:
            by_day.setdefault(o.timestamp[:10], []).append(
                json.dumps(asdict(o), separators=(",", ":")) + "\n"
            )
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            for day, lines in by_day.items():
                with (self.directory / f"{day}.jsonl").open(
                    "a", encoding="utf-8", newline="\n"
                ) as f:
                    f.writelines(lines)
