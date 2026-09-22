# ChurnAI

ChurnAI is a Streamlit dashboard for customer churn analysis, retention prioritization, campaign planning, and heuristic score backtesting.

## Run locally

```powershell
.\scripts\run_dashboard.cmd
```

This runs `.\venv\Scripts\streamlit.exe run .\app\dashboard\app.py` from the repository root.

Or create a fresh environment:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\streamlit.exe run .\app\dashboard\app.py
```

The app opens at <http://localhost:8501> by default.

The canonical Streamlit entry point is `app/dashboard/app.py`. The older `app/aap.py` file is only a compatibility pointer.

## Test

```powershell
.\scripts\test.cmd
```

GitHub Actions runs dependency checks, Python compilation, and the unittest suite with `.github/workflows/tests.yml`.

For a fuller local readiness check before sharing, run:

```powershell
.\scripts\preflight.cmd
```

## Deploy

See [DEPLOYMENT.md](DEPLOYMENT.md) for Streamlit Community Cloud settings and pre-share validation steps.

See [ROADMAP.md](ROADMAP.md) for completed prototype scope and recommended production next steps.

## CSV columns

For the full retention workflow, upload a CSV with these columns:

- `customer_id`
- `tenure_months`
- `monthly_charges`
- `contract`
- `churn` optional, for backtesting

The app also accepts close variants such as `Customer ID`, `customer`, `tenure`, `charges`, `monthly-charge`, `monthly_charge`, `Contract Type`, and `contract_type`.

You can try the full workflow with `data/sample_customers.csv`.

See [DATA_SCHEMA.md](DATA_SCHEMA.md) for accepted header variants, data types, and churn-label handling.

## Current features

- Sample CSV download
- Upload readiness checks
- Executive summary
- Churn risk estimator
- Model assumptions
- Retention priorities and Customer 360
- Segment insights
- Score backtesting with threshold comparison
- Retention campaign plan export
- Dataset-aware Ask ChurnAI assistant
