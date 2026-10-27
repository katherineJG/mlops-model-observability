# Architecture and signals

Written as the initial design and updated as the system was built. Sections 1 to 3.4 and 4 to 5 are the original design; sections 3.5 to 3.7 and 6 record what was built and measured. Where a measurement changed a threshold, the later section says so: the performance thresholds in 3.4 are superseded by 3.7. The Pushgateway between the offline jobs and Prometheus was added after the first design and is not in the diagram below; see 3.6. What passed and what is still unverified is in `ACCEPTANCE.md`.

## 1. Monitoring layers

Four layers, kept separate because each answers a different question and has a different failure meaning.

```text
                    ┌──────────────────────────────────────────────┐
 caller ──HTTP───►  │ inference service (copy of Project 02 app/)  │
                    │   instrumentation.py   observation_sink.py   │
                    └───────┬──────────────────────────┬───────────┘
                            │ /metrics                 │ one JSONL record per scored row
                            ▼                          ▼
                       Prometheus                observations/YYYY-MM-DD.jsonl
                            │                          │
                            ▼                          ▼  (windowing job)
                         Grafana ◄── alerts     data_quality.py ─► hard failure / warning
                                                drift_report.py ─► drift JSON + HTML
 labels (later) ───────────────────────────────► delayed_evaluation.py ─► performance report
```

| Layer | Question | Needs labels? | Failure means |
|---|---|---|---|
| Service reliability | Is the service up, fast and error-free? | No | Callers are being hurt now |
| Data quality | Is the input valid and complete? | No | Inputs are broken; predictions untrustworthy |
| Data drift | Has the input distribution moved from the reference? | No | Inputs changed; **performance unknown** |
| Model performance | Are predictions right? | **Yes, delayed** | The model is measurably worse |

Principle carried through every output: drift is a prompt to investigate, never evidence of degraded accuracy. Reports and alerts for drift must say so, and only the performance layer may report accuracy-type metrics, and only over labelled rows.

## 2. Inputs from Project 02

Project 02 already exposes `/metrics` on a private registry: `http_requests_total{method,route,status_class}`, `http_request_duration_seconds{method,route}`, `predict_instances_total`, `model_loaded`. This project reuses those and adds a monitoring-failure counter plus observation output. The model has 10 input features declared in `models/churn-v1/manifest.json` (4 numeric: `senior_citizen`, `tenure_months`, `monthly_charges`, `total_charges`; 6 categorical: `gender`, `partner`, `contract`, `internet_service`, `payment_method`, `paperless_billing`). That manifest is the schema for data-quality checks.

## 3. Signal catalogue

Initial thresholds only. Operator actions refer to sections of `docs/RUNBOOK.md`.

### 3.1 Service reliability (Prometheus, no labels needed)

| Signal | Source | Unit | Labels | Window | Threshold (initial) | Severity | Operator action |
|---|---|---|---|---|---|---|---|
| Request rate | `rate(http_requests_total)` | req/s | route, status_class | 5 m | none, dashboard only | n/a | n/a |
| 5xx ratio | `http_requests_total{status_class="5xx"}` / total | ratio | route | 5 m | > 2 % for 5 m | page | RUNBOOK: elevated errors |
| p95 latency | `histogram_quantile(0.95, http_request_duration_seconds)` | s | route | 5 m | > 0.5 s for 10 m | warn | RUNBOOK: latency |
| Model not loaded | `model_loaded` | 0/1 | none | instant | == 0 for 2 m | page | RUNBOOK: model not ready |
| Scrape target down | `up` | 0/1 | job | instant | == 0 for 2 m | page | RUNBOOK: service down |
| Monitoring sink failures | `observation_sink_failures_total` (added with the service instrumentation) | count | none | 15 m | > 0 | warn | RUNBOOK: sink failing |

### 3.2 Data quality (windowed job over observations)

| Signal | Class | Unit | Window | Threshold (initial) | Operator action |
|---|---|---|---|---|---|
| Missing required column | hard failure | count of columns | window | any | RUNBOOK: schema change |
| Wrong type / unparseable value | hard failure | fraction of rows | window | > 0 | RUNBOOK: schema change |
| Out-of-range numeric (vs manifest min/max) | warning | fraction of rows | window | > 1 % | RUNBOOK: upstream data fault |
| Unseen category (vs manifest `allowed`) | warning | fraction of rows | window | > 1 % | RUNBOOK: new category |
| Missing values | warning | fraction per column | window | > 5 % | RUNBOOK: upstream data fault |
| Duplicate event IDs | warning | fraction of rows | window | > 0 | RUNBOOK: duplicate delivery |
| Empty or undersized window | not evaluated | row count | window | < minimum sample size | none: report `insufficient_data`, never a pass |

Hard failures mean the row or window cannot be trusted and downstream drift and performance results for it are marked invalid. Warnings are reported but evaluation continues.

### 3.3 Data drift (windowed job, reference vs current)

| Signal | Source | Unit | Window | Threshold (initial) | Severity | Operator action |
|---|---|---|---|---|---|---|
| Per-feature drift flag | numeric: two-sample KS; categorical: chi-square on category counts (computed with scipy; Evidently only renders the HTML report) | p-value and effect size per feature | 1 h, min 200 rows | p < 0.01 after Bonferroni over the 10 features | warn | RUNBOOK: drift investigation |
| Share of drifted features | count flagged / features tested | ratio | same | >= 30 % | warn | RUNBOOK: drift investigation |
| Prediction score distribution shift | KS on predicted probability | statistic | same | p < 0.01 | warn | RUNBOOK: drift investigation |

Known limitations, to be repeated in the README: with large windows trivial shifts become statistically significant, so each result carries an effect size alongside the p-value; testing 10 features inflates false positives, hence the correction; windows below the minimum sample size return `insufficient_data` and raise no drift signal.

### 3.4 Model performance (delayed labels)

| Signal | Source | Unit | Window | Threshold (initial) | Operator action |
|---|---|---|---|---|---|
| Label coverage | labelled predictions / predictions in window | ratio | prediction-time window | report only; metrics suppressed below 50 labels | RUNBOOK: label delay |
| ROC AUC, precision, recall, F1 | labelled rows only | score | prediction-time window | superseded: see 3.7 (a verdict needs 300 labels and an AUC drop of more than 0.08 against the reference baseline) | RUNBOOK: performance drop |
| Unlabelled count | predictions with no label yet | count | window | report only | n/a |

Windows are by prediction time, not label arrival time. Metrics are computed only on rows that have a label and always shown with the coverage they were computed on. Unlabelled predictions are never counted as correct or incorrect.

### 3.5 Data quality and drift: implementation and measured baseline

`monitoring/window.py` (half-open `[start, end)` windows, malformed-line count, retention pruning), `data_quality.py`, `drift_report.py` (tests, Bonferroni, JSON result, Evidently HTML, CLI), `reference.py` (versioned reference with SHA-256 provenance), `synthetic.py` (seeded healthy and shifted data). Run `uv run python -m monitoring.drift_report --start <ISO> --end <ISO>`.

Window statuses: `insufficient_data` (below 200 rows, nothing else evaluated), `invalid_input` (a hard data-quality failure; drift not computed), `no_drift`, `drift` (share of drifted inputs >= 0.3). A flagged column needs a Bonferroni-adjusted p-value below 0.01; effect size is reported next to it but does not gate the flag. The prediction-score test is a single unadjusted test and does not count toward the share.

Measured baseline (`tests/unit/test_drift_report.py::test_healthy_baseline_false_alarm_rate`): 100 seeded healthy windows of 500 rows against `reference-v1` gave 0 window-level drift results, 3 windows with at least one input flagged, and 1 window with the score flagged. A shifted window (seed 7, 500 rows) is `drift` with 4 of 10 inputs flagged (`tenure_months`, `monthly_charges`, `total_charges`, `contract`). Thresholds were left at their initial values; this is evidence they are workable on synthetic data, not tuning for real traffic.

Limitations observed:
- Inputs are not independent: `total_charges` is derived from tenure and charges, so one real shift can flag several columns and inflate the share. The share is a heuristic.
- Bonferroni is conservative, so small real shifts in one column can be missed at 500 rows; with very large windows the opposite happens and tiny shifts become significant, which is why effect sizes are reported.
- Evidently renders the HTML with its own default drift methods; its flags may differ from the JSON, which is authoritative.
- Categorical values are compared as strings, so `1` and `1.0` would count as different categories.
- Evidently 0.7.23 does not import under pandas 3.1.0, so pandas is pinned below 3.

### 3.6 Local stack

`docker compose up --build -d` starts the service, Prometheus (scrapes `/metrics` every 15 s and loads `prometheus/alerts.yml`), a Pushgateway, and Grafana with the datasource and `dashboards/model-service-overview.json` provisioned from files. Every port binds to 127.0.0.1. There is no Alertmanager (optional in the spec), so alerts are visible in Prometheus and on the dashboard only; nothing is routed anywhere.

Pushgateway is an addition to the diagram in section 1. The drift job is a batch process Prometheus cannot scrape, and a Pushgateway is the standard way to get a batch job's result into Grafana and into alert rules. It is optional: without `--pushgateway` the job still writes its JSON and HTML. The service also gained a label-free `prediction_score` histogram so the dashboard has a live model signal that needs no labels.

Eight alert rules, each with a severity (`page` or `warn`), a static description that names the first operator step, and a `runbook_url` pointing at a heading in `docs/RUNBOOK.md`: ServiceDown, ModelNotLoaded, ServiceErrorRatioHigh, ServiceLatencyHigh, ObservationSinkFailing, DataDriftDetected, DataQualityHardFailure, DataQualityWarning. The drift alert's text says it does not show lower accuracy. Behaviour is covered by `prometheus/alerts_test.yml` (promtool, including cases that must stay silent). The runbook URLs are repo-relative; they become links once the repo is hosted.

Differences from the signal tables above: the 5xx ratio is aggregated across routes (not per route), and the latency alert watches `/predict` only, because health checks would dilute it. The data-quality rows in 3.2 are exported as the window's quality status and as per-issue gauges rather than separate alerts per check; only hard failure and warnings alert.

Image pins (tag and digest): Prometheus v3.15.0, Pushgateway v1.11.4, Grafana 13.2.4, the same Python base image and uv as Project 02. Docker Hub tag dates are rebuild times, not release dates, so "a few weeks old" was a judgement and not a measurement. Bumping an image means re-resolving the digest and re-running the stack tests.

Limitations:
- The drift metrics on the dashboard are the last pushed run. They stay until the next run replaces them, so a stopped job leaves a stale drift alert and a stale "last run" timestamp. There is no alert on a stale job, because in this demo nothing schedules it.
- Grafana runs with an anonymous viewer and a fixed local admin password. That is acceptable only because it is loopback-only; do not expose it.
- Prometheus keeps 2 days of data and Grafana keeps no state, so the stack is for a demo, not for retaining history.
- Dashboard screenshots have not been taken (see `docs/ACCEPTANCE.md`).

### 3.7 Delayed labels, performance and incident simulation

Implemented in `monitoring/delayed_evaluation.py` (join, metrics, JSON and Markdown report, optional Pushgateway export), `monitoring/simulate_incident.py` (five seeded scenarios plus a live `error_burst`), and `compose.incident.yaml`.

Supersedes the 3.4 table: the thresholds there were placeholders. Calibrated values are a minimum of **300 labelled rows** and a verdict of `degraded` when ROC AUC is more than **0.08** below the reference baseline.

How labels are handled:
- Labels are `(event_id, label[, labelled_at])`. The join key is the `event_id` the service produced (`<batch id>.<row index>`).
- A label that arrives after the `--as-of` time is pending, not wrong, and is ignored until you evaluate again with a later time. A label dated before its own prediction is excluded and counted. A label for an event outside the window is counted, not joined. An event given conflicting labels is rejected entirely.
- Metrics use labelled rows only and are always reported with the coverage behind them. With fewer than 300 labels, or labels of one class only, the status is `insufficient_labels` and no verdict is given. Precision, recall and F1 are `None` when undefined, never 0.
- A window with a hard data-quality failure is `invalid_input`: neither drift nor performance is computed from it.

Calibration (measured, 300 draws per size, healthy synthetic windows against the baseline AUC of 0.809): the mean drop is +0.010 (the baseline is the model's training-set performance, so it is slightly optimistic) and the spread shrinks with labels, from a standard deviation of 0.035 at 200 labels to 0.016 at 1000. A 0.05 drop would flag 13% of healthy windows at 200 labels, which was too noisy. At 0.08 the false-alarm rate is about 1% at 300 labels, 0.3% at 500 and 0% at 1000. The committed check (`test_healthy_windows_are_rarely_flagged_as_degraded`, 100 seeded windows of 300) flagged 1 of 100. Cost of this choice: a real degradation smaller than about 0.08 AUC is not flagged, and a small window needs 300 labels before it can be judged at all.

What the simulations establish (seed 7, 500 observations, 412 labelled):

| Scenario | Drift | Performance (AUC vs 0.809) |
|---|---|---|
| healthy | none | ok (0.779) |
| input_shift | drift, 4 of 10 inputs | degraded (0.689) |
| benign_shift | drift, 3 of 10 inputs | ok (0.784) |
| concept_shift | none | degraded (0.196) |
| schema_break | not computed | not computed |

So drift can come with a fall in accuracy, without one, and accuracy can fall with no drift. The first run of `input_shift` contradicted my own expectation that it would show no loss: the shifted population is narrower, so ranking quality drops even though how churn depends on the inputs is unchanged. `benign_shift`, which moves only inputs the label ignores, was added to show drift without loss.

Limits:
- `concept_shift` is extreme (the relationship is reversed). It shows the mechanism, not a realistic magnitude.
- The `benign_shift` window sits exactly on the 30% drift threshold (3 of 10 inputs), so its `drift` status is a property of this seed; a different seed could land at 2 of 10.
- The service validates requests, so a schema break cannot reach observation files through it. `schema_break` models a different producer writing the files.
- The baseline is computed on training data, not on held-out data.

## 4. Observation record and privacy boundary

One JSON line per scored row, written by `observation_sink.py` behind a small interface so the file store can be replaced.

Stored: `event_id`, `timestamp`, `model_id`, `model_version`, predicted probability, predicted label, `latency_ms`, and the 10 manifest features.

Never stored: raw request bodies, headers, IP addresses, any field outside the manifest, and `customer_id` (present in the training data, not a model input). Stored feature values are the model inputs only, because drift detection needs them.

Rules, each covered by a test:
- No feature value, `event_id` or request ID is ever a metric label. Metric labels are limited to method, route template, status class.
- No observation content appears in application logs.
- Retention: observation files older than a configurable number of days (default 7) are deleted by the windowing job. Local files are not encrypted; they hold synthetic data only. This is a stated limitation, not a production design.

## 5. Failure isolation

Writing an observation happens after the prediction is computed and is wrapped so any exception is caught, logged without payload content, counted in `observation_sink_failures_total`, and never reaches the caller. No monitoring failure is classified as safety-critical in this project, so inference always continues.

The windowing job is a separate process and cannot affect the service.

## 6. Decisions and open questions

Decisions already agreed: copy Project 02's `app/` rather than depend on it; JSONL files behind a replaceable `ObservationSink`; Evidently for reports.

Settled before the instrumentation was built:
1. **Label join key.** The service generates a batch ID per successful `/predict` call and returns it in the `X-Event-ID` response header. Each row's `event_id` is `<batch id>.<row index>`, with the index matching the order of `instances` in the request. Labels arrive as `(event_id, label)`. A business key such as `customer_id` was rejected on privacy grounds. Failed (4xx/5xx) requests get no event ID and write nothing.
2. **Stored numerics.** The model's exact values are kept, since inputs are synthetic and drift tests need them. Revisit if real data is ever used.
3. **Window default.** 1 hour with a 200-row minimum, both configurable; the incident simulations generate enough rows per window to cross it.

Implemented: `app/observation_sink.py` (`Observation`, `ObservationSink` protocol, `JsonlSink`), `app/instrumentation.py` (`Metrics`, `record_observations`), `OBSERVATION_DIR` setting (default `observations/`). `latency_ms` on a row is the inference time of its whole request, so rows of one batch share the value. Retention deletion is `monitoring.window.prune`, run by `drift_report --prune`; it is not scheduled automatically.
