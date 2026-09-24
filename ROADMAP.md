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
- Real IBM Telco Customer Churn dataset (7,043 customers) loaded into SQLite with analytical SQL queries (`src/churn_db.py`)
- Trained scikit-learn model (Logistic Regression vs. Random Forest, best model selected by ROC-AUC) with saved metrics and confusion matrix (`src/churn_model.py`)
- Experiment tracking log (`reports/experiment_log.csv`) appending accuracy/precision/recall/F1/ROC-AUC per candidate model on every training run

## Recommended production next steps

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
Ran 14 tests ... OK
```
