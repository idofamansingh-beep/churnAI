"""SQLite persistence and SQL analytics for the real Telco customer churn dataset."""

import sqlite3
from pathlib import Path

import pandas as pd

TABLE_NAME = "customers"


def clean_raw_dataframe(data: pd.DataFrame) -> pd.DataFrame:
	"""Apply the standard cleaning steps for the IBM Telco Customer Churn dataset."""
	cleaned = data.copy()
	cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
	# New customers (tenure == 0) have blank TotalCharges in the source data; they have not been billed yet.
	cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0.0)
	cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)
	return cleaned


def build_database(csv_path: Path, db_path: Path) -> int:
	"""Load the raw CSV into a SQLite database table and return the row count written."""
	data = clean_raw_dataframe(pd.read_csv(csv_path))
	db_path.parent.mkdir(parents=True, exist_ok=True)
	connection = sqlite3.connect(db_path)
	try:
		data.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)
		connection.commit()
	finally:
		connection.close()
	return len(data)


def run_query(db_path: Path, sql: str) -> pd.DataFrame:
	"""Run a read-only SQL query against the customers database and return a DataFrame."""
	connection = sqlite3.connect(db_path)
	try:
		return pd.read_sql_query(sql, connection)
	finally:
		connection.close()


def churn_rate_by_contract(db_path: Path) -> pd.DataFrame:
	sql = f"""
		SELECT
			Contract AS contract,
			COUNT(*) AS customers,
			ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate,
			ROUND(SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_at_risk
		FROM {TABLE_NAME}
		GROUP BY Contract
		ORDER BY churn_rate DESC
	"""
	return run_query(db_path, sql)


def churn_rate_by_internet_service(db_path: Path) -> pd.DataFrame:
	sql = f"""
		SELECT
			InternetService AS internet_service,
			COUNT(*) AS customers,
			ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 4) AS churn_rate
		FROM {TABLE_NAME}
		GROUP BY InternetService
		ORDER BY churn_rate DESC
	"""
	return run_query(db_path, sql)


def tenure_and_charges_by_churn(db_path: Path) -> pd.DataFrame:
	sql = f"""
		SELECT
			Churn AS churn,
			COUNT(*) AS customers,
			ROUND(AVG(tenure), 1) AS avg_tenure_months,
			ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
		FROM {TABLE_NAME}
		GROUP BY Churn
	"""
	return run_query(db_path, sql)


def revenue_at_risk_by_payment_method(db_path: Path) -> pd.DataFrame:
	sql = f"""
		SELECT
			PaymentMethod AS payment_method,
			COUNT(*) AS churned_customers,
			ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_at_risk
		FROM {TABLE_NAME}
		WHERE Churn = 'Yes'
		GROUP BY PaymentMethod
		ORDER BY monthly_revenue_at_risk DESC
	"""
	return run_query(db_path, sql)
