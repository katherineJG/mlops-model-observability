# Runbook

Operator steps for each alert in `prometheus/alerts.yml`. Every alert's `runbook_url` points at a heading here. The [incident walkthroughs](#incident-walkthroughs) at the end replay each failure with seeded data and show the diagnosis, mitigation and recovery.

Commands assume the local stack from `compose.yaml`, run from the repo root. Alerts are visible at http://127.0.0.1:9090/alerts and as the "Alerts firing" tile on the Grafana dashboard. There is no Alertmanager, so nothing is sent anywhere.

## Service down

Alert `ServiceDown` (page): Prometheus cannot scrape `service:8000`.

1. `docker compose ps service`. If it is not running or is restarting, read why: `docker compose logs --tail 100 service`.
2. If it is running, check it answers: `curl http://127.0.0.1:8000/health/live`. A response means the problem is the network between Prometheus and the service, not the service.
3. Restart with `docker compose up -d service`. If it exits again at startup, go to [Model not loaded](#model-not-loaded).

## Model not loaded

Alert `ModelNotLoaded` (page): the service is alive but `/health/ready` returns 503 and `/predict` is refused.

1. Find the cause in the log: `docker compose logs service | findstr "model load failed"`. The `reason` field is one of:
   - `manifest_missing`, `manifest_invalid`, `artifact_missing`: the `models/churn-v1` directory in the image is incomplete. Restore it from version control and rebuild.
   - `checksum_mismatch`: `model.joblib` does not match the SHA-256 in `manifest.json`. Do not edit the manifest to make it pass. Restore the artifact; the file was changed or corrupted.
   - `version_mismatch`: the scikit-learn in the image differs from the one that wrote the artifact. Restore the pinned versions in `pyproject.toml` or re-export the model.
   - `artifact_unreadable`, `artifact_incompatible`, `schema_mismatch`, `smoke_test_failed`: the artifact and the service disagree. Treat it as a bad release.
2. Rebuild and restart: `docker compose up --build -d service`.
3. Confirm: `curl http://127.0.0.1:8000/health/ready` returns `{"status":"ready"}`.

## Elevated errors

Alert `ServiceErrorRatioHigh` (page): more than 2% of requests returned 5xx for 5 minutes.

1. Find the route: in Prometheus run `sum by (route) (rate(http_requests_total{status_class="5xx"}[5m]))`.
2. Read the errors: `docker compose logs --tail 200 service`. Entries with `"outcome": "error"` carry the request ID, which is also in the `X-Request-ID` response header. Callers never see stack traces; the log does.
3. A 503 on `/predict` is a model problem: [Model not loaded](#model-not-loaded). A 500 is a bug: capture the log entry and the request ID, then roll back to the previous image.
4. 4xx responses are not counted here. A rise in 422s means callers are sending invalid data, which the data-quality check sees as well.

## High latency

Alert `ServiceLatencyHigh` (warn): p95 of `/predict` above 0.5 s for 10 minutes.

1. Check batch size: callers sending batches near the limit of 100 rows are slower. Compare `rate(predict_instances_total[5m])` with the request rate.
2. Check the host: `docker stats --no-stream`. The service is a single process with scikit-learn running on CPU.
3. If the load is legitimate, run more replicas behind a load balancer; that is outside this local stack.

## Observation sink failing

Alert `ObservationSinkFailing` (warn): the service could not write observations. Inference still works.

1. The log has `observation write failed` with an exception type (for example `PermissionError` or `OSError`).
2. Check space: `docker compose exec service df -h /data/observations`.
3. Check the volume exists and is writable by user 10001; recreating it (`docker compose down` then `up`, without `-v` unless you accept losing observations) restores the image's ownership.
4. Until fixed, drift windows will have gaps and the delayed-label join will miss events. Windows that fall below 200 rows report `insufficient_data` instead of a false "no drift".

## Data drift detected

Alert `DataDriftDetected` (warn): in the latest window at least 30% of the model inputs differ from the reference at a Bonferroni-adjusted p < 0.01.

Drift is not proof the model has got worse. It means the inputs changed.

1. Open the JSON report from `reports/` (printed by `monitoring.drift_report`) and read `columns`: which inputs are flagged and how large the effect size is. A flagged input with a tiny effect size on a large window is likely noise from sample size.
2. Remember related inputs flag together: `total_charges` follows tenure and charges.
3. Decide whether the change is expected (a new customer segment, a campaign) or a fault (an upstream change in how a field is filled).
4. Check whether the model is actually worse by comparing against labelled outcomes: run `monitoring.delayed_evaluation` on the same window (see [Model performance degraded](#model-performance-degraded)). The simulations show drift with a fall in accuracy, drift with none, and a fall with no drift at all, so the drift alert cannot answer this question.
5. If few labels have arrived yet, the evaluation says `insufficient_labels`. That is not good news; it means wait, or check smaller signals such as the score distribution.
6. Do not retrain or swap the model on this alert alone.

The alert reflects the last pushed run and clears on the next run that is not `drift`.

## Model performance degraded

Alert `ModelPerformanceDegraded` (warn): on labelled predictions, ROC AUC is more than 0.08 below the reference baseline, with at least 300 labels. This is measured against outcomes, so unlike drift it is direct evidence the model is worse.

1. Open the `performance-*.json` report (and the `.md` beside it). Read `labelled`, `coverage` and `median_label_delay_hours` first: the verdict is only as good as the labels behind it.
2. Compare the drift report for the same window. Inputs that moved (`drift`) point to a population change; inputs that did not move point to a change in how the outcome relates to them (concept change), or to a fault in the labels.
3. Rule out the labels: check `rejected_labels`, `labelled_before_prediction` and `labels_outside_window` in the report, and whether labels come from a new process.
4. The baseline is the model's performance on its own training data, so it is a little optimistic: healthy windows average about 0.01 below it, and about 1 in 100 windows of 300 labels crosses the 0.08 line by chance. A single borderline window is not an incident; two in a row is.
5. Mitigation is a release decision, not an automatic one: roll back to the previous model version, or retrain on recent data and release it through the normal process. Nothing in this stack does either by itself.
6. The alert reflects the last pushed evaluation and clears on the next one that is not `degraded`.

## Labels are late or missing

No alert: this is a property of the data, shown by the "Label coverage" and "Labelled rows" tiles.

- `insufficient_labels` means fewer than 300 predictions in the window have a label yet. No metric is reported as a verdict; pending predictions are not counted as correct.
- Labels arriving after the `--as-of` time are listed as `late_labels` and are ignored until you evaluate again with a later time. Re-run when more have arrived.
- If coverage stays low, find out whether the labelling process is stuck: `median_label_delay_hours` shows the usual lag. Labels for events outside the window (`labels_outside_window`) are normal when labels for several windows share one file.
- Labels with conflicting values for one event are rejected, not guessed at.

## Data quality hard failure

Alert `DataQualityHardFailure` (page): the latest window had a missing column or unparseable values. Drift was not computed.

1. In the JSON report, `quality.issues` lists each failed check with its column. `missing_column` and `invalid_type` are the hard ones.
2. A missing column or wrong type nearly always means a caller or an upstream system changed its payload. The service validates requests, so also check for a service version that does not match the manifest.
3. Fix the source. Re-run the window analysis once clean data arrives; the alert clears on the next run.

## Data quality warning

Alert `DataQualityWarning` (warn): a window crossed a warning threshold. Drift was still computed.

1. In the JSON report, `quality.issues` shows the check: `unseen_category` or `out_of_range` above 1% of rows, `missing_values` above 5% of a column, or `duplicate_event_ids` above 0.
2. Duplicate event IDs mean the same observation was delivered twice; delayed labels would join to both.
3. Unseen categories and out-of-range values are rejected by the service's own validation, so seeing them here suggests the observation files were not written by this service version.

## Incident walkthroughs

Each scenario is seeded, so the numbers below are what you will see with the defaults (seed 7, 500 observations in one hour, 80% of them eventually labelled, labels arriving 3 to 9 hours later). Run from the repo root; replace `NAME` with the scenario:

```bash
uv run python -m monitoring.simulate_incident NAME --out sim/NAME
```

It writes `sim/NAME/observations/` and `sim/NAME/labels.csv` and prints the two follow-up commands: `monitoring.drift_report` for the inputs and `monitoring.delayed_evaluation` for performance, the latter twice, once with an early `--as-of` (labels still arriving) and once with a complete one. Add `--pushgateway http://127.0.0.1:9091` to either to put the result on the dashboard.

Reading the results, the point of the set is that drift and accuracy are independent:

| Scenario | Drift | Quality | Performance once labels arrive |
|---|---|---|---|
| `healthy` | none | ok | ok: AUC 0.779 vs baseline 0.809 (412 of 500 labelled) |
| `input_shift` | drift: 4 of 10 inputs | ok | degraded: AUC 0.689 (drop 0.119) |
| `benign_shift` | drift: 3 of 10 inputs | ok | ok: AUC 0.784 (drop 0.024) |
| `concept_shift` | none | ok | degraded: AUC 0.196 (drop 0.612) |
| `schema_break` | not computed | hard failure | not computed |

With the early `--as-of` (5 hours after the window ends) only 182 of 500 predictions have a label, below the 300 needed, so every scenario reports `insufficient_labels`. That is the correct answer for that moment, not a pass.

### Input shift

Newer customers on month-to-month contracts with higher charges. Alert: `DataDriftDetected`, then `ModelPerformanceDegraded` once labels arrive.

- Diagnose: the drift report flags `tenure_months`, `monthly_charges`, `total_charges` and `contract`; `total_charges` follows the other two. The score distribution shifts as well. Once 300 or more labels are in, AUC falls to 0.689. This population is also narrower, so the model ranks it less well even though how churn depends on the inputs has not changed.
- Mitigate: confirm the new segment is real and expected. If it is, the model needs data from that segment; a roll back will not help, because the earlier model was trained on the older population too.
- Recover: retrain on data that includes the new segment, release it, and re-run both jobs on a fresh window. Both alerts clear on the next run that is not `drift` / `degraded`.

### Benign shift

Only `gender`, `senior_citizen` and `partner` move, none of which drive churn here. Alert: `DataDriftDetected` only.

- Diagnose: the drift report flags exactly those three inputs (share 0.3, the threshold); the score distribution does not move. With labels, AUC is 0.784, within noise of the baseline, and no performance alert fires.
- Mitigate: nothing is wrong with the model. Record why the inputs moved and, if it recurs, consider whether drift on those inputs should count toward the window flag.
- Recover: the alert clears on the next window that is not `drift`. This is the scenario that justifies not retraining on a drift alert alone.

### Concept shift

The inputs look as before, but how they relate to churn has reversed. Alert: `ModelPerformanceDegraded` only; drift is silent.

- Diagnose: no input is flagged and quality is ok, so the inputs are not the explanation. AUC is 0.196, worse than a coin flip. The causes to check are a change in the business (a new retention programme), a change in how churn is defined or labelled, or broken labels (see step 3 of [Model performance degraded](#model-performance-degraded)). The scenario is deliberately extreme.
- Mitigate: roll back to the previous model if it was last known good, or fall back to a rule while the cause is found. Retraining on the new relationship only helps once the cause is understood.
- Recover: a new model released through the normal process, then a fresh evaluation with enough labels.

### Schema break

A producer on another version stops sending `contract`. The service itself rejects such requests with a 422, so this reaches the observation files only when something else writes them; the simulator writes them directly. Alert: `DataQualityHardFailure`.

- Diagnose: the drift report is `invalid_input` and `quality.issues` shows `missing_column` for `contract`. The performance report is `invalid_input` too: neither is computed from a window that cannot be trusted.
- Mitigate: find the producer that changed and roll it back or fix the field name. Meanwhile do not read anything into drift or performance for that window.
- Recover: re-run both jobs on a window written after the fix. Rows from the broken period stay in their files and are excluded by analysing later windows.

### Error burst

The service is alive but has no model, so every `/predict` returns 503. Alerts: `ModelNotLoaded` after 2 minutes, `ServiceErrorRatioHigh` after 5.

```bash
docker compose -f compose.yaml -f compose.incident.yaml up -d service   # starts the service with no model
uv run python -m monitoring.simulate_incident error_burst --requests 60 --spacing 1
```

- Diagnose: the burst prints `responses by class: {'5xx': 60}`. In Prometheus `model_loaded` is 0 and the 5xx ratio is 100%; the dashboard tiles "Model loaded" and "5xx ratio" show it. The service log says `model load failed` with a reason. Follow [Model not loaded](#model-not-loaded).
- Mitigate: nothing fixes the service from the outside; restore the model artifact or roll back the image.
- Recover: `docker compose up -d service` recreates the service from `compose.yaml` alone. Confirm `curl http://127.0.0.1:8000/health/ready` and that `model_loaded` returns to 1. No observations were written during the outage because rejected requests are never scored, so the drift jobs see a gap, not bad data.
