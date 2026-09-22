import pandas as pd


def normalize_column_name(column: object) -> str:
	characters = [character.lower() if character.isalnum() else "_" for character in str(column).strip()]
	return "_".join(part for part in "".join(characters).split("_") if part)


def find_churn_column(data: pd.DataFrame) -> str | None:
	for column in data.columns:
		if normalize_column_name(column) == "churn":
			return column
	for column in data.columns:
		if "churn" in normalize_column_name(column):
			return column
	return None


def churn_rate(data: pd.DataFrame, churn_column: str | None) -> float | None:
	if churn_column is None:
		return None

	values = data[churn_column]
	if pd.api.types.is_numeric_dtype(values):
		return float(values.mean())

	normalized = values.astype(str).str.lower().str.strip()
	churned = normalized.isin({"1", "true", "yes", "y", "churn", "churned"})
	return float(churned.mean())


def normalized_churn_labels(data: pd.DataFrame, churn_column: str | None) -> pd.Series | None:
	if churn_column is None:
		return None

	values = data[churn_column]
	if pd.api.types.is_numeric_dtype(values):
		return pd.to_numeric(values, errors="coerce").fillna(0).gt(0)

	normalized = values.astype(str).str.lower().str.strip()
	return normalized.isin({"1", "true", "yes", "y", "churn", "churned"})


def estimate_churn_probability(tenure_months: int, monthly_charges: float, contract: str) -> float:
	probability = 0.18

	if tenure_months < 6:
		probability += 0.28
	elif tenure_months < 12:
		probability += 0.18
	elif tenure_months < 24:
		probability += 0.08
	else:
		probability -= 0.05

	if monthly_charges >= 100:
		probability += 0.18
	elif monthly_charges >= 75:
		probability += 0.10
	elif monthly_charges < 50:
		probability -= 0.04

	contract_name = contract.lower()
	if "month" in contract_name:
		probability += 0.25
	elif "annual" in contract_name or "year" in contract_name:
		probability -= 0.08
	elif "two" in contract_name or "2" in contract_name:
		probability -= 0.16

	return min(max(probability, 0.03), 0.95)


def risk_level(probability: float) -> str:
	if probability >= 0.70:
		return "Critical"
	if probability >= 0.50:
		return "High"
	if probability >= 0.30:
		return "Medium"
	return "Low"


def retention_recommendation(probability: float) -> str:
	if probability >= 0.70:
		return "Immediate retention intervention recommended."
	if probability >= 0.50:
		return "Targeted retention campaign recommended."
	if probability >= 0.30:
		return "Increase customer engagement."
	return "Continue normal customer engagement."


def find_column(data: pd.DataFrame, candidates: set[str]) -> str | None:
	for column in data.columns:
		normalized = normalize_column_name(column)
		if normalized in candidates:
			return column
	return None


def retention_priorities(data: pd.DataFrame) -> pd.DataFrame | None:
	customer_column = find_column(data, {"customer_id", "customer"})
	tenure_column = find_column(data, {"tenure_months", "tenure"})
	charges_column = find_column(data, {"monthly_charges", "monthly_charge", "charges"})
	contract_column = find_column(data, {"contract", "contract_type"})

	if not all([customer_column, tenure_column, charges_column, contract_column]):
		return None

	priorities = data[[customer_column, tenure_column, charges_column, contract_column]].copy()
	priorities = priorities.rename(
		columns={
			customer_column: "customer_id",
			tenure_column: "tenure_months",
			charges_column: "monthly_charges",
			contract_column: "contract",
		}
	)
	priorities["tenure_months"] = pd.to_numeric(priorities["tenure_months"], errors="coerce")
	priorities["monthly_charges"] = pd.to_numeric(priorities["monthly_charges"], errors="coerce")
	priorities = priorities.dropna(subset=["tenure_months", "monthly_charges"])
	if priorities.empty:
		return None

	priorities["estimated_churn_risk"] = priorities.apply(
		lambda row: estimate_churn_probability(
			int(row["tenure_months"]),
			float(row["monthly_charges"]),
			str(row["contract"]),
		),
		axis=1,
	)
	priorities["risk_level"] = priorities["estimated_churn_risk"].apply(risk_level)
	priorities["revenue_at_risk"] = priorities["monthly_charges"] * priorities["estimated_churn_risk"]
	return priorities.sort_values("estimated_churn_risk", ascending=False)
