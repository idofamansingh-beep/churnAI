# Roadmap

## Prototype complete

- Streamlit dashboard entry point at `app/dashboard/app.py`
- CSV upload and sample CSV download
- Data readiness checks and documented schema
- Heuristic churn risk scoring
- Executive summary and markdown report export
- Retention priorities and Customer 360 drill-down
- Segment insights and score backtesting
- Retention offer what-if simulator
- Budget planner and campaign plan exports
- Dataset-aware Ask ChurnAI assistant
- Theme configuration, helper scripts, deployment docs, CI, and tests

## Recommended production next steps

- Replace the heuristic score with a trained model using historical customer data.
- Add model training, evaluation, and versioning workflows under `models/` and `reports/`.
- Store uploaded data in a managed database or warehouse when multi-user persistence is needed.
- Add authentication before exposing customer data outside local or private environments.
- Add privacy review, retention policy, and PII handling rules for real customer datasets.
- Add richer monitoring for model drift, false positives, false negatives, and campaign outcomes.
- Add browser-based visual regression checks for the Streamlit UI before public releases.

## Release checklist

Before sharing a build, run:

```powershell
.\scripts\preflight.cmd
```

Expected result:

```text
No broken requirements found.
Ran 8 tests ... OK
```
