# Deployment

## Streamlit Community Cloud

Use these settings when deploying from a Git repository:

- Main file path: `app/dashboard/app.py`
- Python version: 3.11 or newer
- Dependencies file: `requirements.txt`
- Secrets: optional `app_password` (see below); the app runs without any secrets configured

After deployment, open the app and upload `data/sample_customers.csv` to verify the full workflow.

## Optional authentication

To require a password before the dashboard is visible, add this to `.streamlit/secrets.toml` (or the platform's secrets manager):

```toml
app_password = "your-chosen-password"
```

If `app_password` is not set, the dashboard skips the login gate entirely, which keeps local development frictionless.

## Local production-style run

```powershell
.\scripts\run_dashboard.cmd
```

The app runs on Streamlit's default port, usually <http://localhost:8501>.

## Validation before sharing

```powershell
.\scripts\preflight.cmd
```

Expected result:

```text
No broken requirements found.
Ran 19 tests ... OK
```

## Real data, SQL, and trained model (optional)

For the "Real data & trained ML model" section to appear, run once after cloning:

```powershell
.\scripts\build_database.cmd
.\scripts\train_model.cmd
```

This generates `data/churnai.db` and `models/churn_model.joblib`, both already committed in this repo so the section works out of the box. Re-run these scripts only if `data/telco_customer_churn_raw.csv` changes. Each training run appends a row per candidate model to `reports/experiment_log.csv` for comparison over time.

## Supported input schema

The full workflow needs these fields:

- `customer_id`
- `tenure_months`
- `monthly_charges`
- `contract`
- `churn` optional, for backtesting

The app also accepts common header variants like `Customer ID`, `Monthly-Charges`, and `Contract Type`.
