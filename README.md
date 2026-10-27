# Model observability and drift detection

A monitoring layer for an ML inference service: a churn model served over HTTP, with Prometheus, Grafana and a set of offline jobs that watch the data going in and, later, the outcomes coming back. It runs locally with free tools, and everything is seeded, so a failure can be replayed and produces the same numbers each time.

The idea it is built around: **drift is not proof that a model got worse.** Inputs can move and accuracy hold, accuracy can fall while the inputs look normal, and sometimes both happen together. Only predictions that have a ground-truth label can tell these apart, and labels arrive late. So the project keeps four kinds of signal apart and never reports one as another.

This is project 3 of a series. Project 1 trains the model, project 2 serves it. `app/` here is a copy of project 2's service plus the instrumentation this project adds, so the repo runs on its own.

## The four layers

| Layer | Question it answers | Needs labels | Where you see it |
|---|---|---|---|
| Service reliability | Is it up, fast and error-free? | No | Prometheus alerts, dashboard top row |
| Data quality | Is the input valid and complete? | No | JSON report, dashboard, alert on hard failure |
| Data drift | Have the inputs moved away from the reference? | No | JSON and HTML report, dashboard, alert |
| Model performance | Are the predictions right? | **Yes, and they arrive late** | Markdown and JSON report, dashboard, alert |

```text
 caller --HTTP--> inference service (app/)
                    |   /metrics (request, error, latency, score)     one JSONL line per scored row
                    v                                                  v
               Prometheus <--scrape-- Pushgateway <--push--  monitoring jobs (offline, run on a window)
                    |                                          data_quality.py  ->  hard failure / warning
                    v                                          drift_report.py  ->  JSON + HTML report
                 Grafana + alert rules                         delayed_evaluation.py  <-- labels, later
```

Monitoring never sits in the request path: writing an observation happens after the prediction is computed, and a failure there is counted and logged but never reaches the caller.

## Quick start

You need Docker with Compose v2 and [uv](https://docs.astral.sh/uv/) (0.13 or newer; it installs Python 3.12 for you). The first build downloads images and Python packages, so it needs internet and takes a few minutes. Ports 8000, 9090, 9091 and 3000 must be free.

**Resources, measured on Windows with Docker Desktop:** the idle stack uses about 730 MiB of RAM (Grafana about 530, the service about 150, Prometheus about 40, Pushgateway about 10) and under 1% CPU. The images take about 3.5 GB of disk. Nothing needs an account or a paid service.

```bash
# 1. Install and check. (make is not required; these are the Makefile's lines.)
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy app monitoring tests
uv run python -m pytest -q

# 2. Start the stack: service :8000, Prometheus :9090, Pushgateway :9091, Grafana :3000
docker compose up --build -d

# 3. Score one customer. (In PowerShell use curl.exe.) The reply has an X-Event-ID header.
curl -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @docs/example_request.json

# 4. Replay a seeded incident, analyse it, and put the results on the dashboard
uv run python -m monitoring.simulate_incident input_shift --out sim/input_shift
uv run python -m monitoring.drift_report --observations sim/input_shift/observations --start 2026-10-27T10:00:00+00:00 --end 2026-10-27T11:00:00+00:00 --out sim/input_shift/reports --pushgateway http://127.0.0.1:9091
uv run python -m monitoring.delayed_evaluation --observations sim/input_shift/observations --labels sim/input_shift/labels.csv --start 2026-10-27T10:00:00+00:00 --end 2026-10-27T11:00:00+00:00 --as-of 2026-10-28T00:00:00+00:00 --out sim/input_shift/reports --pushgateway http://127.0.0.1:9091

# 5. Look: Grafana http://127.0.0.1:3000 (dashboard "Model service overview"), alerts at http://127.0.0.1:9090/alerts

# 6. Stop and delete the stack's data
docker compose down -v
```

Reports land in `sim/input_shift/reports/`: `drift-*.json` and `.html`, `performance-*.json` and `.md`. Committed examples are in [`reports/examples/`](reports/examples/).

To analyse the live service instead of a simulation, copy its observations out and run the same jobs on a window that has at least 200 rows:

```bash
docker compose cp service:/data/observations ./observations
uv run python -m monitoring.drift_report --pushgateway http://127.0.0.1:9091   # the last hour
```

The Grafana admin password in `compose.yaml` is a fixed local value and anonymous viewing is on. Both are acceptable only because every port is bound to 127.0.0.1. Do not expose this stack.

## Incident simulations

Five seeded scenarios write one hour of observations (500 rows) and a labels file in which 80% of rows get a label 3 to 9 hours later. The same seed gives byte-identical files. With the defaults (seed 7):

| Scenario | What changes | Drift | Quality | Performance once labels arrive (AUC; baseline 0.809) |
|---|---|---|---|---|
| `healthy` | nothing | none | ok | ok (0.779) |
| `input_shift` | newer, month-to-month customers | drift, 4 of 10 inputs | ok | **degraded** (0.689) |
| `benign_shift` | only inputs the label ignores (gender, senior, partner) | drift, 3 of 10 inputs | ok | ok (0.784) |
| `concept_shift` | how tenure, charges and contract drive churn reverses | none | ok | **degraded** (0.196) |
| `schema_break` | a field is missing from every observation | not computed | hard failure | not computed |

The point of the set is that the second and third columns disagree: drift came with a fall in accuracy once, without one once, and accuracy fell with no drift once. `error_burst` is the sixth, live one: it starts the service without a model and sends requests at it. Each scenario has diagnosis, mitigation and recovery steps in [docs/RUNBOOK.md](docs/RUNBOOK.md#incident-walkthroughs).

If you evaluate the same window before the labels have arrived (`--as-of 2026-10-27T16:00:00+00:00`), only 182 of 500 predictions have a label, below the 300 needed, so the status is `insufficient_labels`. That means "no verdict yet", not "fine".

## Thresholds

All are initial values that I then measured on healthy synthetic windows; the measurements and their costs are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#3-signal-catalogue) sections 3.5 and 3.7.

| Signal | Rule | Value |
|---|---|---|
| Window size | rows needed to evaluate anything | 200 (below: `insufficient_data`) |
| Drifted input | Bonferroni-adjusted p across the 10 inputs | below 0.01 (KS test for numeric, chi-square for categorical) |
| Window drift | share of inputs flagged | 30% or more |
| Performance verdict | labelled rows needed | 300 (below: `insufficient_labels`) |
| Performance degraded | ROC AUC below the reference baseline by | more than 0.08 |
| Data quality hard failure | missing column, unparseable value | any |
| Data quality warning | unseen category, out of range / missing values / duplicate event IDs | over 1% / over 5% / any |
| Alerts | 5xx ratio over 2% for 5 min (page); `/predict` p95 over 0.5 s for 10 min (warn); no model for 2 min (page); service down for 2 min (page); sink failing (warn) | see `prometheus/alerts.yml` |

What the calibration showed: with a 0.05 AUC drop and 200 labels, 13% of healthy windows would have been flagged, which is too noisy. At 0.08 and 300 labels it is about 1%. The price is that a real degradation smaller than about 0.08 AUC goes unflagged.

## Design choices, and what I decided against

- **Drift tests written out, Evidently for the report.** The JSON result comes from explicit KS and chi-square tests so the thresholds are visible and testable; Evidently draws the HTML for people. Using Evidently's own flags for everything was the alternative. It would have been less code, but I could not state or test the rule being applied. The two can disagree, and the JSON is the one to trust.
- **Pushgateway for the offline jobs.** Prometheus cannot scrape a batch job. Alternatives were the node-exporter textfile collector (another service, files shared across containers) or having Grafana read the JSON reports (loses alerts). The cost of Pushgateway is that results stay until the next run, so a stopped job leaves a stale alert; there is no staleness alert because nothing schedules the jobs here.
- **JSONL files behind a small `ObservationSink` interface.** A database would handle concurrent writers and queries, but would be a service to run for a demo. The sink is replaceable; its limit (one writer process per directory) is stated in the code.
- **A copy of project 2's service, not an import.** Each repo stays runnable on its own. The cost is that fixes to the serving code have to be made twice.
- **Labels join on a service-generated event ID, not a customer ID.** `X-Event-ID` is returned per request and rows are `<id>.<row>`. Joining on a business key would have meant storing an identifier.
- **A hard data-quality failure stops the analysis** (`invalid_input`) instead of warning and carrying on, because drift and performance numbers from a window with a missing column would look valid and not be.
- **No Alertmanager.** It is optional in the brief and would route alerts nowhere here. Alerts are visible in Prometheus and on the dashboard.

## Limitations

- All data is synthetic, including the labels. Nothing has been tested on real traffic or a real labelling process.
- The performance baseline is the model's training-set AUC, so it is slightly optimistic: healthy windows average about 0.01 below it.
- `concept_shift` reverses the relationship, which is extreme. It shows the mechanism, not a realistic size of drop.
- The `benign_shift` window sits exactly on the 30% drift threshold, so its `drift` result belongs to this seed.
- Inputs are not independent (`total_charges` follows tenure and charges), so one real shift can flag several inputs and inflate the share.
- Bonferroni is conservative, so a small real shift in one column can be missed at 500 rows; with very large windows the opposite happens and tiny shifts become significant. Effect sizes are reported next to the p-values for that reason.
- The service validates requests, so a schema break can reach the observation files only from a different producer. The simulation writes the files directly.
- Docker Hub tag dates are rebuild times, so "pinned to a release that is a few weeks old" was a judgement.
- Pandas is pinned below 3 because Evidently 0.7.23 does not import under pandas 3.1.0.

## Privacy and data minimization

Observations hold the ten model inputs, the prediction, the model version, the latency and an event ID, and nothing else. `customer_id` is never stored. Raw request bodies and headers are never logged. Metric labels are limited to method, route template, status class, status names, manifest feature names and issue codes; no request ID, event ID or feature value is ever a label. Tests assert all of this against `/metrics`, the formatted logs and the pushed metrics. Observation files older than 7 days can be removed with `drift_report --prune` (not scheduled). The local files are not encrypted, which is fine for synthetic data and not a production design.

## Tests and CI

```bash
uv run python -m pytest -q                       # 150 tests, no Docker needed
$env:RUN_STACK_TESTS = "1"                        # bash: export RUN_STACK_TESTS=1
uv run python -m pytest tests/integration -v     # 9 tests: builds and starts the real stack, about 3 minutes
```

The stack tests run `promtool` on the config and on 13 alert scenarios (including ones that must stay silent), send traffic, run both jobs against the live stack, check that the drift and performance alerts fire, run every dashboard query through Grafana, and then break and restore the service.

`.github/workflows/ci.yml` runs the checks and then the stack tests on every pull request. It has been linted with `actionlint` but not yet run on GitHub, because this repo has no remote.

A manual checklist for the parts a test cannot see (the rendered dashboard) is in [docs/MANUAL_CHECKLIST.md](docs/MANUAL_CHECKLIST.md). The owner has viewed the dashboard and approved how it looks; the checklist itself has not been worked through, and there are no dashboard screenshots yet.

## Layout

```text
app/                 the inference service (copy of project 2) + instrumentation.py, observation_sink.py
monitoring/          window.py, data_quality.py, drift_report.py, delayed_evaluation.py,
                     simulate_incident.py, metrics_export.py, reference.py, synthetic.py
reference/           versioned reference dataset with a checksum and provenance
models/churn-v1/     the model artifact, verified by SHA-256 before it is loaded
prometheus/          config, alert rules, and promtool tests for the rules
grafana/, dashboards/  provisioned datasource and the "Model service overview" dashboard
reports/examples/    healthy, drifted and degraded example reports
docs/                ARCHITECTURE.md, RUNBOOK.md, ACCEPTANCE.md, MANUAL_CHECKLIST.md
compose.yaml         the local stack;  compose.incident.yaml  starts the service with no model
```

Acceptance status, with the evidence for each criterion and what is still unverified, is in [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md).

## Licence

[MIT](LICENSE), the same as the other projects in the series. The third-party images and packages this stack runs (Prometheus, Grafana, Pushgateway, Evidently and the Python libraries) keep their own licences. Grafana is AGPL-3.0; this repo only configures it and does not redistribute it.
