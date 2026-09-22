# Deployment

## Streamlit Community Cloud

Use these settings when deploying from a Git repository:

- Main file path: `app/dashboard/app.py`
- Python version: 3.11 or newer
- Dependencies file: `requirements.txt`
- Secrets: none required for the current prototype

After deployment, open the app and upload `data/sample_customers.csv` to verify the full workflow.

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
Ran 8 tests ... OK
```

## Supported input schema

The full workflow needs these fields:

- `customer_id`
- `tenure_months`
- `monthly_charges`
- `contract`
- `churn` optional, for backtesting

The app also accepts common header variants like `Customer ID`, `Monthly-Charges`, and `Contract Type`.
