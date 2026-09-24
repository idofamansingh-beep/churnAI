from io import BytesIO
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.churn_db import (
	churn_rate_by_contract,
	churn_rate_by_internet_service,
	revenue_at_risk_by_payment_method,
	tenure_and_charges_by_churn,
)
from src.churn_model import load_model, predict_churn_probability
from src.churnai_core import (
	churn_rate,
	estimate_churn_probability,
	find_churn_column,
	find_column,
	normalized_churn_labels,
	retention_priorities,
	retention_recommendation,
	risk_level,
)


st.set_page_config(page_title="ChurnAI", page_icon="🤖", layout="wide")

REAL_DATA_CSV = ROOT_DIR / "data" / "telco_customer_churn_raw.csv"
REAL_DATABASE = ROOT_DIR / "data" / "churnai.db"
MODEL_PATH = ROOT_DIR / "models" / "churn_model.joblib"
MODEL_METRICS_PATH = ROOT_DIR / "reports" / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "reports" / "feature_importance.csv"


@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
	return pd.read_csv(BytesIO(file_bytes))


def sample_data() -> pd.DataFrame:
	sample_path = Path(__file__).resolve().parents[2] / "data" / "sample_customers.csv"
	if sample_path.exists():
		return pd.read_csv(sample_path)

	return pd.DataFrame(
		{
			"customer_id": ["C-1001", "C-1002", "C-1003", "C-1004", "C-1005"],
			"tenure_months": [18, 4, 32, 9, 25],
			"monthly_charges": [72.5, 91.2, 58.4, 104.9, 66.0],
			"contract": ["Annual", "Monthly", "Annual", "Monthly", "Two year"],
			"churn": [0, 1, 0, 1, 0],
		}
	)


def retention_offer_scenarios(tenure_months: int, monthly_charges: float, contract: str, discount_percent: int) -> pd.DataFrame:
	baseline_risk = estimate_churn_probability(tenure_months, monthly_charges, contract)
	discounted_charges = monthly_charges * (1 - discount_percent / 100)
	scenarios = [
		("Current plan", monthly_charges, contract),
		(f"{discount_percent}% discount", discounted_charges, contract),
		("Annual-plan offer", discounted_charges, "Annual"),
	]

	rows = []
	for scenario, scenario_charges, scenario_contract in scenarios:
		scenario_risk = estimate_churn_probability(tenure_months, scenario_charges, scenario_contract)
		rows.append(
			{
				"scenario": scenario,
				"contract": scenario_contract,
				"monthly_charges": scenario_charges,
				"estimated_churn_risk": scenario_risk,
				"risk_reduction": max(baseline_risk - scenario_risk, 0),
				"recommendation": retention_recommendation(scenario_risk),
			}
		)

	return pd.DataFrame(rows)


def budget_plan(priorities: pd.DataFrame, discount_percent: int, monthly_budget: float) -> pd.DataFrame:
	plan = priorities.copy()
	plan["offer_cost"] = plan["monthly_charges"] * (discount_percent / 100)
	plan["post_offer_risk"] = plan.apply(
		lambda row: estimate_churn_probability(
			int(row["tenure_months"]),
			float(row["monthly_charges"] - row["offer_cost"]),
			"Annual" if "month" in str(row["contract"]).lower() else str(row["contract"]),
		),
		axis=1,
	)
	plan["risk_reduction"] = (plan["estimated_churn_risk"] - plan["post_offer_risk"]).clip(lower=0)
	plan["risk_reduction_per_dollar"] = plan["risk_reduction"] / plan["offer_cost"].replace(0, pd.NA)
	plan = plan.sort_values(["risk_reduction_per_dollar", "estimated_churn_risk"], ascending=False)
	plan["cumulative_offer_cost"] = plan["offer_cost"].cumsum()
	return plan[plan["cumulative_offer_cost"] <= monthly_budget]


def budget_summary(plan: pd.DataFrame, monthly_budget: float) -> dict[str, str]:
	spent = plan["offer_cost"].sum() if not plan.empty else 0.0
	return {
		"targeted_customers": f"{len(plan):,}",
		"budget_used": f"${spent:,.2f}",
		"budget_remaining": f"${max(monthly_budget - spent, 0):,.2f}",
		"average_risk_reduction": f"{plan['risk_reduction'].mean():.0%}" if not plan.empty else "0%",
	}


def retention_action(probability: float, contract: str) -> str:
	contract_name = contract.lower()
	if probability >= 0.70:
		return "Schedule same-day save call"
	if probability >= 0.50 and "month" in contract_name:
		return "Offer annual-plan incentive"
	if probability >= 0.50:
		return "Assign retention specialist"
	if probability >= 0.30:
		return "Send engagement check-in"
	return "Monitor in regular cadence"


def campaign_priority(probability: float) -> str:
	if probability >= 0.70:
		return "Urgent"
	if probability >= 0.50:
		return "High"
	if probability >= 0.30:
		return "Standard"
	return "Monitor"


def readiness_report(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
	requirements = [
		("Customer ID", find_column(data, {"customer_id", "customer"}), False),
		("Tenure months", find_column(data, {"tenure_months", "tenure"}), True),
		("Monthly charges", find_column(data, {"monthly_charges", "monthly_charge", "charges"}), True),
		("Contract", find_column(data, {"contract", "contract_type"}), False),
	]

	rows = []
	invalid_numeric_rows = 0
	for label, column, needs_numeric in requirements:
		status = "Ready" if column is not None else "Missing"
		invalid_values = 0
		if column is not None and needs_numeric:
			invalid_values = int(pd.to_numeric(data[column], errors="coerce").isna().sum())
			invalid_numeric_rows = max(invalid_numeric_rows, invalid_values)
			if invalid_values:
				status = "Needs cleanup"

		rows.append(
			{
				"field": label,
				"matched_column": column or "-",
				"status": status,
				"invalid_values": invalid_values,
			}
		)

	return pd.DataFrame(rows), invalid_numeric_rows


def segment_summary(priorities: pd.DataFrame) -> pd.DataFrame:
	return (
		priorities.groupby("contract", as_index=False)
		.agg(
			customers=("customer_id", "count"),
			average_risk=("estimated_churn_risk", "mean"),
			revenue_at_risk=("revenue_at_risk", "sum"),
		)
		.sort_values("revenue_at_risk", ascending=False)
	)


def campaign_plan(priorities: pd.DataFrame) -> pd.DataFrame:
	plan = priorities.copy()
	plan["campaign_priority"] = plan["estimated_churn_risk"].apply(campaign_priority)
	plan["recommended_action"] = plan.apply(
		lambda row: retention_action(float(row["estimated_churn_risk"]), str(row["contract"])),
		axis=1,
	)
	plan["recommended_timing"] = plan["estimated_churn_risk"].apply(
		lambda probability: "Within 24 hours" if probability >= 0.70 else "This week" if probability >= 0.50 else "This month"
	)
	return plan[
		[
			"customer_id",
			"risk_level",
			"estimated_churn_risk",
			"revenue_at_risk",
			"campaign_priority",
			"recommended_action",
			"recommended_timing",
		]
	]


def executive_summary(priorities: pd.DataFrame | None) -> dict[str, str] | None:
	if priorities is None or priorities.empty:
		return None

	top_customer = priorities.iloc[0]
	segments = segment_summary(priorities)
	top_segment = segments.iloc[0]
	immediate_actions = int((priorities["estimated_churn_risk"] >= 0.70).sum())
	revenue_at_risk = priorities["revenue_at_risk"].sum()
	return {
		"top_customer": f"{top_customer['customer_id']} ({top_customer['estimated_churn_risk']:.0%})",
		"focus_segment": f"{top_segment['contract']} (${top_segment['revenue_at_risk']:,.2f})",
		"immediate_actions": f"{immediate_actions:,}",
		"revenue_at_risk": f"${revenue_at_risk:,.2f}",
		"recommendation": retention_recommendation(float(top_customer["estimated_churn_risk"])),
	}


def executive_report(priorities: pd.DataFrame, summary: dict[str, str], rate: float | None) -> str:
	churn_text = "Not detected" if rate is None else f"{rate:.1%}"
	top_customers = priorities.head(5)
	segments = segment_summary(priorities).head(3)
	lines = [
		"# ChurnAI executive report",
		"",
		f"- Detected churn rate: {churn_text}",
		f"- Top risk customer: {summary['top_customer']}",
		f"- Focus segment: {summary['focus_segment']}",
		f"- Urgent actions: {summary['immediate_actions']}",
		f"- Revenue at risk: {summary['revenue_at_risk']}",
		f"- Recommendation: {summary['recommendation']}",
		"",
		"## Top retention priorities",
	]

	for _, customer in top_customers.iterrows():
		lines.append(
			f"- {customer['customer_id']}: {customer['estimated_churn_risk']:.0%} risk, "
			f"{customer['risk_level']} level, ${customer['revenue_at_risk']:.2f} revenue at risk"
		)

	lines.extend(["", "## Segment focus"])
	for _, segment in segments.iterrows():
		lines.append(
			f"- {segment['contract']}: {int(segment['customers']):,} customers, "
			f"{segment['average_risk']:.0%} average risk, ${segment['revenue_at_risk']:.2f} revenue at risk"
		)

	return "\n".join(lines)


def score_backtest(data: pd.DataFrame, priorities: pd.DataFrame | None, churn_column: str | None, threshold: float) -> dict[str, object] | None:
	labels = normalized_churn_labels(data, churn_column)
	if priorities is None or labels is None:
		return None

	evaluation = priorities.join(labels.rename("actual_churn"), how="inner")
	if evaluation.empty:
		return None

	evaluation["predicted_churn"] = evaluation["estimated_churn_risk"] >= threshold
	actual = evaluation["actual_churn"]
	predicted = evaluation["predicted_churn"]
	true_positive = int((actual & predicted).sum())
	false_positive = int((~actual & predicted).sum())
	false_negative = int((actual & ~predicted).sum())
	true_negative = int((~actual & ~predicted).sum())
	accuracy = float((actual == predicted).mean())
	precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
	recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
	by_actual = (
		evaluation.groupby("actual_churn", as_index=False)
		.agg(customers=("customer_id", "count"), average_estimated_risk=("estimated_churn_risk", "mean"))
		.replace({"actual_churn": {False: "Retained", True: "Churned"}})
	)
	confusion = pd.DataFrame(
		[
			{"actual": "Churned", "predicted": "High risk", "customers": true_positive},
			{"actual": "Churned", "predicted": "Lower risk", "customers": false_negative},
			{"actual": "Retained", "predicted": "High risk", "customers": false_positive},
			{"actual": "Retained", "predicted": "Lower risk", "customers": true_negative},
		]
	)
	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"threshold": threshold,
		"by_actual": by_actual,
		"confusion": confusion,
	}


def threshold_sweep(data: pd.DataFrame, priorities: pd.DataFrame | None, churn_column: str | None) -> pd.DataFrame | None:
	rows = []
	for step in range(1, 10):
		threshold = step / 10
		result = score_backtest(data, priorities, churn_column, threshold)
		if result is None:
			return None

		rows.append(
			{
				"threshold": threshold,
				"accuracy": result["accuracy"],
				"precision": result["precision"],
				"recall": result["recall"],
			}
		)

	return pd.DataFrame(rows)


def answer_churn_question(question: str, priorities: pd.DataFrame | None, rate: float | None) -> str:
	if priorities is None:
		if rate is None:
			return "I can summarize uploaded customer data, but I need churn or retention-priority columns first."
		return f"The detected churn rate is {rate:.1%}. Upload customer_id, tenure_months, monthly_charges, and contract columns to unlock customer-level priorities."

	question_lower = question.lower()
	high_risk = priorities[priorities["estimated_churn_risk"] >= 0.50]
	total_revenue_at_risk = priorities["revenue_at_risk"].sum()

	if "highest" in question_lower or "top" in question_lower or "risk" in question_lower:
		top_customers = priorities.head(3)
		lines = ["Top retention priorities:"]
		for _, customer in top_customers.iterrows():
			lines.append(
				f"- {customer['customer_id']}: {customer['estimated_churn_risk']:.0%} risk, {customer['risk_level']} level, ${customer['revenue_at_risk']:.2f} revenue at risk"
			)
		return "\n".join(lines)

	if "revenue" in question_lower or "money" in question_lower:
		return f"Estimated monthly revenue at risk is ${total_revenue_at_risk:,.2f} across {len(priorities):,} customers."

	if "how many" in question_lower or "count" in question_lower or "customers" in question_lower:
		return f"There are {len(priorities):,} customers with usable priority data, including {len(high_risk):,} at high or critical risk."

	if "churn rate" in question_lower or "churn" in question_lower:
		if rate is None:
			return "I did not detect a churn column, but I can estimate customer-level risk from tenure, charges, and contract."
		return f"The detected churn rate is {rate:.1%}, and {len(high_risk):,} customers are currently estimated at high or critical risk."

	return "Ask me about top risk customers, revenue at risk, customer counts, or churn rate."


@st.cache_resource
def load_real_model():
	return load_model(MODEL_PATH.parent)


@st.cache_data
def load_model_metrics() -> dict | None:
	if not MODEL_METRICS_PATH.exists():
		return None
	return json.loads(MODEL_METRICS_PATH.read_text())


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
	if not FEATURE_IMPORTANCE_PATH.exists():
		return None
	return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data
def load_real_customer_data() -> pd.DataFrame:
	return pd.read_csv(REAL_DATA_CSV)


@st.cache_data
def top_real_risk_customers(_pipeline, real_data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
	from src.churn_db import clean_raw_dataframe

	cleaned = clean_raw_dataframe(real_data)
	cleaned = cleaned.assign(predicted_churn_risk=predict_churn_probability(_pipeline, cleaned))
	columns = ["customerID", "tenure", "MonthlyCharges", "Contract", "predicted_churn_risk"]
	return cleaned[columns].sort_values("predicted_churn_risk", ascending=False).head(top_n)


def configured_app_password() -> str | None:
	# Auth is opt-in: no app_password secret means local/dev runs stay frictionless.
	try:
		return st.secrets.get("app_password")
	except Exception:
		return None


def require_login() -> None:
	expected_password = configured_app_password()
	if not expected_password or st.session_state.get("authenticated", False):
		return

	st.title("🔒 ChurnAI")
	st.write("Enter the access password to continue.")
	password = st.text_input("Password", type="password")
	if st.button("Log in"):
		if password == expected_password:
			st.session_state["authenticated"] = True
			st.rerun()
		else:
			st.error("Incorrect password.")
	st.stop()


require_login()

st.title("🤖 ChurnAI")
st.caption("Customer churn dashboard")

with st.sidebar:
	st.header("Data")
	uploaded_file = st.file_uploader("Upload customer CSV", type="csv")
	st.download_button(
		"Download sample CSV",
		sample_data().to_csv(index=False),
		file_name="churnai_sample_customers.csv",
		mime="text/csv",
	)

if uploaded_file is None:
	data = sample_data()
	st.info("Upload a customer CSV to replace the sample dashboard data.")
else:
	data = load_csv(uploaded_file.getvalue())

churn_column = find_churn_column(data)
rate = churn_rate(data, churn_column)
numeric_columns = data.select_dtypes(include="number").columns.tolist()
readiness, invalid_numeric_rows = readiness_report(data)
priorities = retention_priorities(data)

filtered_priorities = priorities
if priorities is not None:
	with st.sidebar:
		st.header("Retention filters")
		risk_levels = ["Critical", "High", "Medium", "Low"]
		selected_risk_levels = st.multiselect("Risk levels", risk_levels, default=["Critical", "High", "Medium"])
		minimum_risk = st.slider("Minimum estimated risk", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.0f%%")

	filtered_priorities = priorities[
		priorities["risk_level"].isin(selected_risk_levels)
		& (priorities["estimated_churn_risk"] >= minimum_risk)
	]

with st.container(horizontal=True):
	st.metric("Customers", f"{len(data):,}", border=True)
	st.metric("Columns", f"{len(data.columns):,}", border=True)
	if rate is None:
		st.metric("Churn rate", "Not found", border=True)
	else:
		st.metric("Churn rate", f"{rate:.1%}", border=True)

with st.container(border=True):
	st.subheader("Executive summary")
	summary = executive_summary(filtered_priorities)
	if summary is None:
		st.write("Upload customer priority fields to generate an executive summary.")
	else:
		with st.container(horizontal=True):
			st.metric("Top risk customer", summary["top_customer"], border=True)
			st.metric("Focus segment", summary["focus_segment"], border=True)
			st.metric("Urgent actions", summary["immediate_actions"], border=True)
			st.metric("Revenue at risk", summary["revenue_at_risk"], border=True)
		st.write(summary["recommendation"])
		st.download_button(
			"Download executive report",
			executive_report(filtered_priorities, summary, rate),
			file_name="churnai_executive_report.md",
			mime="text/markdown",
		)

with st.container(border=True):
	st.subheader("Churn risk estimator")
	estimator_left, estimator_middle, estimator_right = st.columns(3)
	with estimator_left:
		tenure_months = st.number_input("Tenure in months", min_value=0, max_value=120, value=6)
	with estimator_middle:
		monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=89.0, step=5.0)
	with estimator_right:
		contract = st.selectbox("Contract", ["Monthly", "Annual", "Two year"])

	estimated_probability = estimate_churn_probability(int(tenure_months), float(monthly_charges), contract)
	estimated_level = risk_level(estimated_probability)
	st.metric("Estimated churn risk", f"{estimated_probability:.0%}", estimated_level, border=True)
	st.write(retention_recommendation(estimated_probability))

	discount_percent = st.slider("Retention offer discount", min_value=0, max_value=40, value=15, step=5, format="%d%%")
	offer_scenarios = retention_offer_scenarios(int(tenure_months), float(monthly_charges), contract, discount_percent)
	st.dataframe(
		offer_scenarios,
		hide_index=True,
		column_config={
			"monthly_charges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
			"estimated_churn_risk": st.column_config.ProgressColumn(
				"Estimated churn risk",
				format="%.0f%%",
				min_value=0,
				max_value=1,
			),
			"risk_reduction": st.column_config.ProgressColumn(
				"Risk reduction",
				format="%.0f%%",
				min_value=0,
				max_value=1,
			),
		},
	)

with st.container(border=True):
	st.subheader("Model assumptions")
	st.dataframe(
		pd.DataFrame(
			[
				{"input": "Short tenure", "effect": "Raises churn risk most strongly below 6 months."},
				{"input": "Monthly charges", "effect": "Raises risk at higher spend levels and lowers it slightly below $50."},
				{"input": "Contract", "effect": "Monthly contracts raise risk; annual and two-year contracts lower it."},
				{"input": "Backtest", "effect": "Compares heuristic risk against actual churn labels when a churn column exists."},
			]
		),
		hide_index=True,
	)
	st.info("Use the backtest section to calibrate the threshold before making campaign decisions on real data.")

with st.container(border=True):
	st.subheader("Data readiness")
	ready_fields = int((readiness["status"] == "Ready").sum())
	needs_cleanup = int((readiness["status"] == "Needs cleanup").sum())
	missing_fields = int((readiness["status"] == "Missing").sum())
	with st.container(horizontal=True):
		st.metric("Ready fields", f"{ready_fields}/4", border=True)
		st.metric("Missing fields", missing_fields, border=True)
		st.metric("Fields needing cleanup", needs_cleanup, border=True)
	st.dataframe(readiness, hide_index=True)
	if missing_fields:
		st.warning("Add the missing fields to generate customer-level retention priorities.")
	elif invalid_numeric_rows:
		st.warning("Clean nonnumeric tenure or monthly charge values before relying on priority scores.")
	else:
		st.success("This dataset is ready for retention scoring.")

left, right = st.columns(2)

with left:
	with st.container(border=True):
		st.subheader("Churn breakdown")
		if churn_column is None:
			st.write("No churn column detected yet.")
		else:
			churn_counts = data[churn_column].value_counts().rename_axis("status").reset_index(name="customers")
			st.bar_chart(churn_counts, x="status", y="customers")

with right:
	with st.container(border=True):
		st.subheader("Numeric overview")
		if not numeric_columns:
			st.write("No numeric columns detected yet.")
		else:
			selected_metric = st.selectbox("Column", numeric_columns)
			st.bar_chart(data[selected_metric].head(50))

with st.container(border=True):
	st.subheader("Segment insights")
	if priorities is None:
		st.write("Retention segment insights will appear after customer-level priority fields are available.")
	elif filtered_priorities.empty:
		st.write("No segment insights match the current retention filters.")
	else:
		segments = segment_summary(filtered_priorities)
		segment_left, segment_right = st.columns(2)
		with segment_left:
			st.bar_chart(segments, x="contract", y="revenue_at_risk", y_label="Revenue at risk")
		with segment_right:
			risk_distribution = (
				filtered_priorities["risk_level"]
				.value_counts()
				.rename_axis("risk_level")
				.reset_index(name="customers")
			)
			st.bar_chart(risk_distribution, x="risk_level", y="customers")

		st.dataframe(
			segments,
			hide_index=True,
			column_config={
				"average_risk": st.column_config.ProgressColumn(
					"Average risk",
					format="%.0f%%",
					min_value=0,
					max_value=1,
				),
				"revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.2f"),
			},
		)

with st.container(border=True):
	st.subheader("Score backtest")
	backtest_threshold = st.slider("High-risk decision threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05, format="%.0f%%")
	backtest = score_backtest(data, priorities, churn_column, backtest_threshold)
	if backtest is None:
		st.write("Add a churn column plus retention-priority fields to compare estimated risk against actual churn labels.")
	else:
		with st.container(horizontal=True):
			st.metric("Threshold", f"{backtest['threshold']:.0%}", border=True)
			st.metric("Accuracy", f"{backtest['accuracy']:.0%}", border=True)
			st.metric("Precision", f"{backtest['precision']:.0%}", border=True)
			st.metric("Recall", f"{backtest['recall']:.0%}", border=True)
		backtest_left, backtest_right = st.columns(2)
		with backtest_left:
			st.bar_chart(backtest["by_actual"], x="actual_churn", y="average_estimated_risk", y_label="Average estimated risk")
		with backtest_right:
			st.dataframe(backtest["confusion"], hide_index=True)

		sweep = threshold_sweep(data, priorities, churn_column)
		if sweep is not None:
			st.dataframe(
				sweep,
				hide_index=True,
				column_config={
					"threshold": st.column_config.ProgressColumn(
						"Threshold",
						format="%.0f%%",
						min_value=0,
						max_value=1,
					),
					"accuracy": st.column_config.ProgressColumn(
						"Accuracy",
						format="%.0f%%",
						min_value=0,
						max_value=1,
					),
					"precision": st.column_config.ProgressColumn(
						"Precision",
						format="%.0f%%",
						min_value=0,
						max_value=1,
					),
					"recall": st.column_config.ProgressColumn(
						"Recall",
						format="%.0f%%",
						min_value=0,
						max_value=1,
					),
				},
			)

with st.container(border=True):
	st.subheader("Retention priorities")
	if priorities is None:
		st.write("Upload data with customer_id, tenure_months, monthly_charges, and contract columns to generate retention priorities.")
	elif filtered_priorities.empty:
		st.write("No customers match the current retention filters.")
	else:
		with st.container(horizontal=True):
			st.metric("Matching customers", f"{len(filtered_priorities):,}", border=True)
			st.metric("Revenue at risk", f"${filtered_priorities['revenue_at_risk'].sum():,.2f}", border=True)
			st.metric("Average risk", f"{filtered_priorities['estimated_churn_risk'].mean():.0%}", border=True)

		st.dataframe(
			filtered_priorities.head(10),
			hide_index=True,
			column_config={
				"monthly_charges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
				"estimated_churn_risk": st.column_config.ProgressColumn(
					"Estimated churn risk",
					format="%.0f%%",
					min_value=0,
					max_value=1,
				),
				"revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.2f"),
			},
		)

		st.download_button(
			"Download retention priorities",
			filtered_priorities.to_csv(index=False),
			file_name="retention_priorities.csv",
			mime="text/csv",
		)

		selected_customer = st.selectbox("Customer 360", filtered_priorities["customer_id"].astype(str))
		customer = filtered_priorities[filtered_priorities["customer_id"].astype(str) == selected_customer].iloc[0]
		profile_left, profile_middle, profile_right = st.columns(3)
		with profile_left:
			st.metric("Risk level", customer["risk_level"], border=True)
		with profile_middle:
			st.metric("Estimated churn risk", f"{customer['estimated_churn_risk']:.0%}", border=True)
		with profile_right:
			st.metric("Revenue at risk", f"${customer['revenue_at_risk']:.2f}", border=True)

		st.write(retention_recommendation(float(customer["estimated_churn_risk"])))

with st.container(border=True):
	st.subheader("Retention budget planner")
	if priorities is None:
		st.write("Budget planning will appear after retention priorities are available.")
	elif filtered_priorities.empty:
		st.write("No customers match the current retention filters.")
	else:
		budget_left, budget_right = st.columns(2)
		with budget_left:
			planner_discount = st.slider("Planner discount", min_value=5, max_value=40, value=15, step=5, format="%d%%")
		with budget_right:
			monthly_budget = st.number_input("Monthly retention budget", min_value=0.0, value=250.0, step=50.0)

		budget_targets = budget_plan(filtered_priorities, int(planner_discount), float(monthly_budget))
		summary = budget_summary(budget_targets, float(monthly_budget))
		with st.container(horizontal=True):
			st.metric("Targeted customers", summary["targeted_customers"], border=True)
			st.metric("Budget used", summary["budget_used"], border=True)
			st.metric("Budget remaining", summary["budget_remaining"], border=True)
			st.metric("Average risk reduction", summary["average_risk_reduction"], border=True)

		if budget_targets.empty:
			st.write("Increase the monthly budget or lower the discount to target customers.")
		else:
			st.dataframe(
				budget_targets[
					[
						"customer_id",
						"estimated_churn_risk",
						"post_offer_risk",
						"risk_reduction",
						"offer_cost",
						"cumulative_offer_cost",
					]
				],
				hide_index=True,
				column_config={
					"estimated_churn_risk": st.column_config.ProgressColumn("Current risk", format="%.0f%%", min_value=0, max_value=1),
					"post_offer_risk": st.column_config.ProgressColumn("Post-offer risk", format="%.0f%%", min_value=0, max_value=1),
					"risk_reduction": st.column_config.ProgressColumn("Risk reduction", format="%.0f%%", min_value=0, max_value=1),
					"offer_cost": st.column_config.NumberColumn("Offer cost", format="$%.2f"),
					"cumulative_offer_cost": st.column_config.NumberColumn("Cumulative cost", format="$%.2f"),
				},
			)

with st.container(border=True):
	st.subheader("Retention campaign plan")
	if priorities is None:
		st.write("Campaign planning will appear after retention priorities are available.")
	elif filtered_priorities.empty:
		st.write("No campaign actions match the current retention filters.")
	else:
		campaigns = campaign_plan(filtered_priorities)
		st.dataframe(
			campaigns.head(15),
			hide_index=True,
			column_config={
				"estimated_churn_risk": st.column_config.ProgressColumn(
					"Estimated churn risk",
					format="%.0f%%",
					min_value=0,
					max_value=1,
				),
				"revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.2f"),
			},
		)
		st.download_button(
			"Download campaign plan",
			campaigns.to_csv(index=False),
			file_name="retention_campaign_plan.csv",
			mime="text/csv",
		)

with st.container(border=True):
	st.subheader("Ask ChurnAI")
	if "churnai_messages" not in st.session_state:
		st.session_state.churnai_messages = [
			{
				"role": "assistant",
				"content": "Ask me about top risk customers, revenue at risk, customer counts, or churn rate.",
			}
		]

	for message in st.session_state.churnai_messages:
		with st.chat_message(message["role"]):
			st.write(message["content"])

	if prompt := st.chat_input("Ask ChurnAI about this dataset"):
		st.session_state.churnai_messages.append({"role": "user", "content": prompt})
		response = answer_churn_question(prompt, filtered_priorities, rate)
		st.session_state.churnai_messages.append({"role": "assistant", "content": response})
		st.rerun()

with st.container(border=True):
	st.subheader("Real data & trained ML model")
	st.caption("Trained on the IBM Telco Customer Churn dataset (7,043 real customers) instead of the heuristic used above.")

	if not MODEL_PATH.exists() or not REAL_DATABASE.exists():
		st.info(
			"Model or database not found. Run `.\\scripts\\build_database.cmd` then `.\\scripts\\train_model.cmd` "
			"to generate `data/churnai.db` and `models/churn_model.joblib`."
		)
	else:
		metrics = load_model_metrics()
		if metrics is not None:
			best_model_metrics = metrics["models"][metrics["best_model"]]
			st.write(f"Best model: **{metrics['best_model'].replace('_', ' ').title()}**")
			with st.container(horizontal=True):
				st.metric("Accuracy", f"{best_model_metrics['accuracy']:.0%}", border=True)
				st.metric("Precision", f"{best_model_metrics['precision']:.0%}", border=True)
				st.metric("Recall", f"{best_model_metrics['recall']:.0%}", border=True)
				st.metric("ROC-AUC", f"{best_model_metrics['roc_auc']:.0%}", border=True)

		feature_importance = load_feature_importance()
		if feature_importance is not None:
			st.write("**Top churn drivers (feature importance)**")
			st.bar_chart(
				feature_importance.head(10).set_index("feature"),
				horizontal=True,
				y_label="Relative importance",
			)

		real_data = load_real_customer_data()
		pipeline = load_real_model()
		sql_left, sql_right = st.columns(2)
		with sql_left:
			st.write("**Churn rate by contract (SQL)**")
			st.dataframe(
				churn_rate_by_contract(REAL_DATABASE),
				hide_index=True,
				column_config={
					"churn_rate": st.column_config.ProgressColumn("Churn rate", format="%.0f%%", min_value=0, max_value=1),
					"monthly_revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.2f"),
				},
			)
		with sql_right:
			st.write("**Churn rate by internet service (SQL)**")
			st.dataframe(
				churn_rate_by_internet_service(REAL_DATABASE),
				hide_index=True,
				column_config={
					"churn_rate": st.column_config.ProgressColumn("Churn rate", format="%.0f%%", min_value=0, max_value=1),
				},
			)

		st.write("**Top 10 highest-risk real customers (model predictions)**")
		st.dataframe(
			top_real_risk_customers(pipeline, real_data),
			hide_index=True,
			column_config={
				"MonthlyCharges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
				"predicted_churn_risk": st.column_config.ProgressColumn(
					"Predicted churn risk", format="%.0f%%", min_value=0, max_value=1
				),
			},
		)

		sql_left_2, sql_right_2 = st.columns(2)
		with sql_left_2:
			st.write("**Tenure and charges by churn status (SQL)**")
			st.dataframe(
				tenure_and_charges_by_churn(REAL_DATABASE),
				hide_index=True,
				column_config={"avg_monthly_charges": st.column_config.NumberColumn("Avg monthly charges", format="$%.2f")},
			)
		with sql_right_2:
			st.write("**Revenue at risk by payment method, churned customers only (SQL)**")
			st.dataframe(
				revenue_at_risk_by_payment_method(REAL_DATABASE),
				hide_index=True,
				column_config={
					"monthly_revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.2f"),
				},
			)

with st.container(border=True):
	st.subheader("Customer data")
	st.dataframe(data, hide_index=True)
