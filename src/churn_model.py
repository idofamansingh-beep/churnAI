"""Training, evaluation, and inference for the real Telco churn prediction model."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn_db import clean_raw_dataframe

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
	"gender",
	"Partner",
	"Dependents",
	"PhoneService",
	"MultipleLines",
	"InternetService",
	"OnlineSecurity",
	"OnlineBackup",
	"DeviceProtection",
	"TechSupport",
	"StreamingTV",
	"StreamingMovies",
	"Contract",
	"PaperlessBilling",
	"PaymentMethod",
]


def load_features_and_target(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
	data = clean_raw_dataframe(pd.read_csv(csv_path))
	features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
	target = (data[TARGET_COLUMN] == "Yes").astype(int)
	return features, target


def build_pipeline(classifier) -> Pipeline:
	preprocessor = ColumnTransformer(
		transformers=[
			("numeric", StandardScaler(), NUMERIC_FEATURES),
			("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
		]
	)
	return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate(pipeline: Pipeline, features_test: pd.DataFrame, target_test: pd.Series) -> dict:
	predicted = pipeline.predict(features_test)
	probability = pipeline.predict_proba(features_test)[:, 1]
	return {
		"accuracy": float(accuracy_score(target_test, predicted)),
		"precision": float(precision_score(target_test, predicted)),
		"recall": float(recall_score(target_test, predicted)),
		"f1_score": float(f1_score(target_test, predicted)),
		"roc_auc": float(roc_auc_score(target_test, probability)),
	}


def train_and_save(csv_path: Path, models_dir: Path, reports_dir: Path, random_state: int = 42) -> dict:
	"""Train candidate models, keep the best by ROC-AUC, and persist the model plus reports."""
	features, target = load_features_and_target(csv_path)
	features_train, features_test, target_train, target_test = train_test_split(
		features, target, test_size=0.2, random_state=random_state, stratify=target
	)

	candidates = {
		"logistic_regression": build_pipeline(LogisticRegression(max_iter=1000, random_state=random_state)),
		"random_forest": build_pipeline(RandomForestClassifier(n_estimators=300, random_state=random_state)),
	}

	results = {}
	for name, pipeline in candidates.items():
		pipeline.fit(features_train, target_train)
		results[name] = evaluate(pipeline, features_test, target_test)

	best_name = max(results, key=lambda name: results[name]["roc_auc"])
	best_pipeline = candidates[best_name]

	models_dir.mkdir(parents=True, exist_ok=True)
	reports_dir.mkdir(parents=True, exist_ok=True)

	joblib.dump(best_pipeline, models_dir / "churn_model.joblib")

	predicted_best = best_pipeline.predict(features_test)
	matrix = confusion_matrix(target_test, predicted_best)
	pd.DataFrame(
		matrix,
		index=["actual_retained", "actual_churned"],
		columns=["predicted_retained", "predicted_churned"],
	).to_csv(reports_dir / "confusion_matrix.csv")

	metrics_report = {
		"best_model": best_name,
		"models": results,
		"training_rows": len(features_train),
		"test_rows": len(features_test),
	}
	(reports_dir / "model_metrics.json").write_text(json.dumps(metrics_report, indent=2))

	return metrics_report


def load_model(models_dir: Path) -> Pipeline:
	return joblib.load(models_dir / "churn_model.joblib")


def predict_churn_probability(pipeline: Pipeline, data: pd.DataFrame) -> pd.Series:
	"""Predict churn probability for rows already containing the required feature columns."""
	features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
	return pd.Series(pipeline.predict_proba(features)[:, 1], index=data.index)
