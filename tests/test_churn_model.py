import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.churn_model import (
	extract_feature_importance,
	load_features_and_target,
	passes_quality_gate,
	predict_churn_probability,
	train_and_save,
)

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
YES_NO = ["Yes", "No"]


def _make_synthetic_telco_csv(path: Path, rows: int = 60, seed: int = 7) -> None:
	rng = random.Random(seed)
	records = []
	for index in range(rows):
		tenure = rng.randint(0, 72)
		monthly_charges = round(rng.uniform(20, 120), 2)
		contract = rng.choice(CONTRACTS)
		# Bias churn so month-to-month, short tenure, high charges churn more often.
		churn_score = (0.5 if contract == "Month-to-month" else 0.1) + (0.3 if tenure < 6 else 0.0) + (monthly_charges / 400)
		churn = "Yes" if rng.random() < churn_score else "No"
		records.append(
			{
				"customerID": f"{1000 + index}-TEST",
				"gender": rng.choice(["Male", "Female"]),
				"SeniorCitizen": rng.choice([0, 1]),
				"Partner": rng.choice(YES_NO),
				"Dependents": rng.choice(YES_NO),
				"tenure": tenure,
				"PhoneService": rng.choice(YES_NO),
				"MultipleLines": rng.choice(YES_NO),
				"InternetService": rng.choice(INTERNET_SERVICES),
				"OnlineSecurity": rng.choice(YES_NO),
				"OnlineBackup": rng.choice(YES_NO),
				"DeviceProtection": rng.choice(YES_NO),
				"TechSupport": rng.choice(YES_NO),
				"StreamingTV": rng.choice(YES_NO),
				"StreamingMovies": rng.choice(YES_NO),
				"Contract": contract,
				"PaperlessBilling": rng.choice(YES_NO),
				"PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"]),
				"MonthlyCharges": monthly_charges,
				"TotalCharges": "" if tenure == 0 else str(round(monthly_charges * tenure, 2)),
				"Churn": churn,
			}
		)
	pd.DataFrame(records).to_csv(path, index=False)


class ChurnModelTest(unittest.TestCase):
	def test_load_features_and_target_shapes(self) -> None:
		with TemporaryDirectory() as tmp_dir:
			csv_path = Path(tmp_dir) / "raw.csv"
			_make_synthetic_telco_csv(csv_path)

			features, target = load_features_and_target(csv_path)

			self.assertEqual(len(features), len(target))
			self.assertEqual({0, 1}, set(target.unique()))
			self.assertNotIn("customerID", features.columns)
			self.assertNotIn("Churn", features.columns)

	def test_train_and_save_produces_model_and_reports(self) -> None:
		with TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			csv_path = tmp_path / "raw.csv"
			models_dir = tmp_path / "models"
			reports_dir = tmp_path / "reports"
			_make_synthetic_telco_csv(csv_path, rows=120)

			report = train_and_save(csv_path, models_dir, reports_dir)

			self.assertIn(report["best_model"], {"logistic_regression", "random_forest"})
			self.assertTrue((models_dir / "churn_model.joblib").exists())
			self.assertTrue((reports_dir / "model_metrics.json").exists())
			self.assertTrue((reports_dir / "confusion_matrix.csv").exists())
			self.assertTrue((reports_dir / "experiment_log.csv").exists())
			self.assertTrue((reports_dir / "feature_importance.csv").exists())
			for metrics in report["models"].values():
				self.assertGreaterEqual(metrics["roc_auc"], 0.0)
				self.assertLessEqual(metrics["roc_auc"], 1.0)

	def test_extract_feature_importance_ranks_all_features(self) -> None:
		with TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			csv_path = tmp_path / "raw.csv"
			models_dir = tmp_path / "models"
			reports_dir = tmp_path / "reports"
			_make_synthetic_telco_csv(csv_path, rows=120)
			train_and_save(csv_path, models_dir, reports_dir)

			from src.churn_model import load_model

			pipeline = load_model(models_dir)
			importance = extract_feature_importance(pipeline)

			self.assertIn("tenure", importance["feature"].values)
			self.assertTrue((importance["importance"] >= 0).all())
			self.assertTrue(importance["importance"].is_monotonic_decreasing)

	def test_passes_quality_gate_accepts_metrics_above_threshold(self) -> None:
		metrics_report = {"best_model": "logistic_regression", "models": {"logistic_regression": {"roc_auc": 0.84}}}

		passed, message = passes_quality_gate(metrics_report, min_roc_auc=0.75)

		self.assertTrue(passed)
		self.assertIn("0.8400", message)

	def test_passes_quality_gate_rejects_metrics_below_threshold(self) -> None:
		metrics_report = {"best_model": "logistic_regression", "models": {"logistic_regression": {"roc_auc": 0.60}}}

		passed, message = passes_quality_gate(metrics_report, min_roc_auc=0.75)

		self.assertFalse(passed)
		self.assertIn("below", message)

	def test_repeated_training_appends_to_experiment_log(self) -> None:
		with TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			csv_path = tmp_path / "raw.csv"
			models_dir = tmp_path / "models"
			reports_dir = tmp_path / "reports"
			_make_synthetic_telco_csv(csv_path, rows=120)

			train_and_save(csv_path, models_dir, reports_dir)
			train_and_save(csv_path, models_dir, reports_dir)

			log = pd.read_csv(reports_dir / "experiment_log.csv")
			self.assertEqual(4, len(log))
			self.assertEqual({"logistic_regression", "random_forest"}, set(log["model"].unique()))

	def test_predict_churn_probability_returns_values_in_range(self) -> None:
		with TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			csv_path = tmp_path / "raw.csv"
			models_dir = tmp_path / "models"
			reports_dir = tmp_path / "reports"
			_make_synthetic_telco_csv(csv_path, rows=120)
			train_and_save(csv_path, models_dir, reports_dir)

			from src.churn_db import clean_raw_dataframe
			from src.churn_model import load_model

			pipeline = load_model(models_dir)
			data = clean_raw_dataframe(pd.read_csv(csv_path))
			probabilities = predict_churn_probability(pipeline, data)

			self.assertEqual(len(data), len(probabilities))
			self.assertTrue((probabilities >= 0).all())
			self.assertTrue((probabilities <= 1).all())


if __name__ == "__main__":
	unittest.main()
