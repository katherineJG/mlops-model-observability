# Manual verification checklist

What the automated tests cannot see: how the dashboard looks, and whether the signals make sense to a person reading them. The tests check that every dashboard query runs and returns data (`tests/integration/test_stack.py`); nothing checks the rendered page. Work through this once after any change to `dashboards/` or `prometheus/`, tick what you saw, and note the date. The look of the dashboard has been approved by the owner (see the record at the bottom); the step-by-step checks below have not been worked through yet.

Time needed: about 10 minutes plus the stack's start-up.

## Set-up

```bash
docker compose up --build -d
uv run python -m monitoring.simulate_incident concept_shift --out sim/concept_shift
uv run python -m monitoring.simulate_incident input_shift --out sim/input_shift
```

Open http://127.0.0.1:3000 and the "Model service overview" dashboard. The default view is the last 30 minutes, refreshing every 10 seconds.

## 1. Healthy but quiet

- [ ] "Model loaded" is green and says Loaded; "Service up" is green and says Up.
- [ ] "Alerts firing" shows 0 and is not red.
- [ ] The drift, data-quality and performance rows show "No data" (nothing has been pushed yet). They are not showing zeros that look like good news.
- [ ] Each row has a title that says where its data comes from (scrape of `/metrics`, or pushed by a named job).

## 2. Traffic

Send some requests (a few every few seconds for a minute):

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @docs/example_request.json
```

- [ ] "Request rate by route" shows a line for `/predict` and, from Prometheus scraping, `/metrics`.
- [ ] "Rows scored per minute" and "Mean predicted churn probability" move.
- [ ] "p95 latency" is a number in seconds, well under the 0.5 s alert level.
- [ ] Hovering the "Mean predicted churn probability" title shows a description saying that a score is a prediction and not an outcome.

## 3. Drift and performance, as separate things

Push the `input_shift` result (drift and a fall in accuracy), then `concept_shift` (accuracy falls, no drift). Use the commands each simulation prints, adding `--pushgateway http://127.0.0.1:9091` to both, or run:

```bash
uv run python -m monitoring.drift_report --observations sim/input_shift/observations --start 2026-10-27T10:00:00+00:00 --end 2026-10-27T11:00:00+00:00 --pushgateway http://127.0.0.1:9091
uv run python -m monitoring.delayed_evaluation --observations sim/input_shift/observations --labels sim/input_shift/labels.csv --start 2026-10-27T10:00:00+00:00 --end 2026-10-27T11:00:00+00:00 --as-of 2026-10-28T00:00:00+00:00 --pushgateway http://127.0.0.1:9091
```

- [ ] "Window drift status" turns orange and says `drift`; "Drifted input share" reads 40%.
- [ ] "Drift effect size by input" lists the inputs and the bars for `tenure_months` and `monthly_charges` are visibly longer than for `gender`.
- [ ] "Performance status" is red and says `degraded`; "ROC AUC drop" is red (above 0.08).
- [ ] "Label coverage" and "Labelled rows" are shown next to the performance numbers, so the verdict is not read without the coverage behind it.
- [ ] Re-run the performance command with `--as-of 2026-10-27T16:00:00+00:00`: "Performance status" says `insufficient_labels` in blue, not green.
- [ ] Repeat both commands with `sim/concept_shift` in place of `sim/input_shift`: drift status goes back to `no_drift` (green) while performance stays `degraded`, and "ROC AUC drop" grows from about 0.12 to about 0.61.

## 4. Alerts

- [ ] http://127.0.0.1:9090/alerts lists nine rules. After step 3, `DataDriftDetected` and `ModelPerformanceDegraded` are firing.
- [ ] Each firing alert shows its description and a `runbook_url` that matches a heading in `docs/RUNBOOK.md`.
- [ ] The "Alerts firing" tile on the dashboard counts them.
- [ ] The description of `DataDriftDetected` says it does not show lower accuracy.

## 5. Failure and recovery

```bash
docker compose -f compose.yaml -f compose.incident.yaml up -d service
uv run python -m monitoring.simulate_incident error_burst --requests 60 --spacing 1
```

- [ ] "Model loaded" turns red and says NOT LOADED; the 5xx ratio panel climbs towards 100%.
- [ ] After about 2 minutes `ModelNotLoaded` fires, and after about 5 `ServiceErrorRatioHigh`.
- [ ] `docker compose up -d service` brings it back; "Model loaded" returns to green and the alerts clear.

## Screenshots

The README deliverables list dashboard screenshots. Capture, at 1200 px wide to match the design system's container: the healthy view (step 2), the drift and performance rows after step 3, and the failure view (step 5). Save them under `docs/screenshots/` and link them from the README. None exist yet.

## Record

| Date | Who | Steps ticked | Notes |
|---|---|---|---|
| 2026-10-27 | Project owner | None individually | Viewed the running dashboard with the `input_shift` incident loaded and live traffic, and approved how it looks ("simple yet functional"). This was a look at one state. The healthy and failure states (steps 1 and 5), the alert text on the page (step 4) and the screenshots were not covered. |
