"""Deterministic incident simulations.

Offline scenarios write one hour of observations plus a delayed-label file, exactly as the
service and a labelling process would have, so the analysis jobs can be run on them:

  healthy        nothing wrong: the baseline
  input_shift    newer, month-to-month customers: drift, and accuracy falls with it
  benign_shift   only inputs the label ignores move: drift, accuracy holds
  concept_shift  inputs look normal, the label relationship reverses: accuracy falls, no drift
  schema_break   a producer on another version omits the `contract` field: hard quality failure

Together these show why drift alone is never taken as proof of degradation: it can come with a
fall in accuracy (input_shift), without one (benign_shift), and accuracy can fall with no drift
(concept_shift). Only labelled predictions tell them apart.

`error_burst` is live: it sends requests to a running service. Break the service first with
`compose.incident.yaml` (see docs/RUNBOOK.md); this script never injects faults itself.

  uv run python -m monitoring.simulate_incident input_shift --out sim/input_shift
"""

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.model_loader import load_model
from app.observation_sink import JsonlSink
from monitoring.synthetic import generate, observations_from_frame

SCENARIOS = ("healthy", "input_shift", "benign_shift", "concept_shift", "schema_break")
DEFAULT_START = datetime(2026, 10, 27, 10, tzinfo=UTC)
DROPPED_FIELD = "contract"


@dataclass(frozen=True)
class Simulation:
    scenario: str
    observations_dir: Path
    labels_path: Path
    window_start: datetime
    window_end: datetime
    rows: int
    labelled: int
    last_label_at: datetime


def simulate(
    scenario: str,
    out_dir: Path,
    model_dir: Path = Path("models/churn-v1"),
    seed: int = 7,
    rows: int = 500,
    start: datetime = DEFAULT_START,
    label_coverage: float = 0.8,
    label_delay_hours: float = 6.0,
) -> Simulation:
    """Write `<out_dir>/observations/*.jsonl` and `<out_dir>/labels.csv`.

    The same arguments always produce the same bytes.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")
    frame = generate(
        rows,
        seed,
        shifted=scenario == "input_shift",
        benign_shift=scenario == "benign_shift",
        concept_shift=scenario == "concept_shift",
    )
    model = load_model(model_dir)
    observations = observations_from_frame(
        frame, model, start, 3600 / rows, f"sim-{scenario}-{seed}"
    )
    if scenario == "schema_break":
        observations = [
            replace(o, features={k: v for k, v in o.features.items() if k != DROPPED_FIELD})
            for o in observations
        ]
    JsonlSink(out_dir / "observations").write(observations)

    rng = np.random.default_rng(
        seed + 1
    )  # labels use their own stream, so they never move the data
    labelled = rng.random(rows) < label_coverage
    delay = rng.uniform(0.5 * label_delay_hours, 1.5 * label_delay_hours, rows)
    stamps = [datetime.fromisoformat(o.timestamp) for o in observations]
    labels = pd.DataFrame(
        {
            "event_id": [o.event_id for o in observations],
            "label": frame["churn"].to_numpy(),
            "labelled_at": [
                (t + timedelta(hours=h)).isoformat() for t, h in zip(stamps, delay, strict=True)
            ],
        }
    )[labelled]
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"
    labels.to_csv(labels_path, index=False, lineterminator="\n")
    last = max(datetime.fromisoformat(s) for s in labels["labelled_at"]) if len(labels) else start
    return Simulation(
        scenario=scenario,
        observations_dir=out_dir / "observations",
        labels_path=labels_path,
        window_start=start,
        window_end=start + timedelta(hours=1),
        rows=rows,
        labelled=len(labels),
        last_label_at=last,
    )


def error_burst(
    send: Callable[[], int], requests: int, spacing_seconds: float = 0.0
) -> Counter[str]:
    """Call `send` (which returns an HTTP status) `requests` times; count by status class."""
    import time

    counts: Counter[str] = Counter()
    for _ in range(requests):
        counts[f"{send() // 100}xx"] += 1
        if spacing_seconds:
            time.sleep(spacing_seconds)
    return counts


def http_sender(url: str) -> Callable[[], int]:
    row = generate(1, seed=3).drop(columns=["churn"]).to_dict("records")
    body = json.dumps({"instances": row}).encode()

    def send() -> int:
        request = urllib.request.Request(
            f"{url}/predict", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as r:
                return int(r.status)
        except urllib.error.HTTPError as e:
            return e.code

    return send


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a deterministic incident simulation.")
    p.add_argument("scenario", choices=[*SCENARIOS, "error_burst"])
    p.add_argument("--out", type=Path, help="output directory (offline scenarios)")
    p.add_argument("--model-dir", type=Path, default=Path("models/churn-v1"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rows", type=int, default=500)
    p.add_argument("--service-url", default="http://127.0.0.1:8000", help="error_burst only")
    p.add_argument("--requests", type=int, default=60, help="error_burst only")
    p.add_argument("--spacing", type=float, default=1.0, help="seconds between requests")
    args = p.parse_args(argv)

    if args.scenario == "error_burst":
        counts = error_burst(http_sender(args.service_url), args.requests, args.spacing)
        print("responses by class:", dict(sorted(counts.items())))
        return 0
    if args.out is None:
        p.error("--out is required for offline scenarios")
    sim = simulate(args.scenario, args.out, args.model_dir, seed=args.seed, rows=args.rows)
    start, end = sim.window_start.isoformat(), sim.window_end.isoformat()
    print(
        f"{sim.scenario}: {sim.rows} observations, {sim.labelled} labels, window {start} to {end}"
    )
    obs, labels = sim.observations_dir, sim.labels_path
    print("analyse the inputs:")
    print(
        f"  uv run python -m monitoring.drift_report --observations {obs} "
        f"--start {start} --end {end}"
    )
    early = (sim.window_end + timedelta(hours=5)).isoformat()
    for tag, when in (("early", early), ("complete", sim.last_label_at.isoformat())):
        print(f"evaluate against labels that have arrived ({tag}):")
        print(
            f"  uv run python -m monitoring.delayed_evaluation --observations {obs} "
            f"--labels {labels} --start {start} --end {end} --as-of {when}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
