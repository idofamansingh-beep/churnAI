# ChurnAI

ChurnAI is a Streamlit dashboard for customer churn analysis, retention prioritization, campaign planning, and heuristic score backtesting. It also includes a real-data pipeline: SQLite + SQL analytics and a trained scikit-learn model on the IBM Telco Customer Churn dataset (7,043 real customers).

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

## Real data, SQL, and trained model

The dashboard's built-in heuristic works on any small CSV, but the project also ships a real-data pipeline built on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns):

```powershell
.\scripts\build_database.cmd   # loads data/telco_customer_churn_raw.csv into SQLite at data/churnai.db
.\scripts\train_model.cmd      # trains Logistic Regression and Random Forest, saves the best model to models/
```

After running both, the dashboard's "Real data & trained ML model" section shows:

- model accuracy, precision, recall, and ROC-AUC from `reports/model_metrics.json`
- top churn drivers (feature importance) from `reports/feature_importance.csv`
- SQL-derived churn rate by contract type and internet service (`src/churn_db.py`)
- SQL-derived revenue at risk by payment method for churned customers
- the top 10 highest-risk real customers scored by the trained model

Model training and evaluation logic lives in `src/churn_model.py`; SQL queries live in `src/churn_db.py`.

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
