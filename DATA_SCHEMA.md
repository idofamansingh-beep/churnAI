# Data schema

ChurnAI accepts customer-level CSV files. The dashboard normalizes column headers by lowercasing them and treating punctuation or spaces as underscores, so `Customer ID`, `customer-id`, and `customer_id` are equivalent.

## Required fields

| Field | Accepted examples | Type | Notes |
| --- | --- | --- | --- |
| `customer_id` | `Customer ID`, `customer`, `customer-id` | Text | Unique customer identifier shown in priority tables and Customer 360. |
| `tenure_months` | `Tenure Months`, `tenure` | Number | Customer age in months. Nonnumeric rows are excluded from retention scoring. |
| `monthly_charges` | `Monthly Charges`, `monthly-charge`, `charges` | Number | Monthly customer revenue or subscription charge. Nonnumeric rows are excluded from retention scoring. |
| `contract` | `Contract`, `Contract Type`, `contract_type` | Text | Values containing `month`, `annual`, `year`, `two`, or `2` affect heuristic risk scoring. |

## Optional fields

| Field | Accepted examples | Type | Notes |
| --- | --- | --- | --- |
| `churn` | `Churn`, `Has Churned?`, `churned` | Boolean, number, or text | Enables churn-rate display and score backtesting. Numeric values greater than `0` count as churned. Text values `1`, `true`, `yes`, `y`, `churn`, and `churned` count as churned. |

## Example

```csv
customer_id,tenure_months,monthly_charges,contract,churn
C-1001,18,72.50,Annual,0
C-1002,4,91.20,Monthly,1
```

Use `data/sample_customers.csv` for a fuller sample dataset.
