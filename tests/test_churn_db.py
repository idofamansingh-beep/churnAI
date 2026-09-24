import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.churn_db import build_database, churn_rate_by_contract, clean_raw_dataframe, tenure_and_charges_by_churn

SAMPLE_ROWS = [
	{
		"customerID": "1000-AAAAA",
		"gender": "Female",
		"SeniorCitizen": 0,
		"Partner": "Yes",
		"Dependents": "No",
		"tenure": 1,
		"PhoneService": "Yes",
		"MultipleLines": "No",
		"InternetService": "DSL",
		"OnlineSecurity": "No",
		"OnlineBackup": "Yes",
		"DeviceProtection": "No",
		"TechSupport": "No",
		"StreamingTV": "No",
		"StreamingMovies": "No",
		"Contract": "Month-to-month",
		"PaperlessBilling": "Yes",
		"PaymentMethod": "Electronic check",
		"MonthlyCharges": 70.0,
		"TotalCharges": "",
		"Churn": "Yes",
	},
	{
		"customerID": "1000-BBBBB",
		"gender": "Male",
		"SeniorCitizen": 0,
		"Partner": "No",
		"Dependents": "No",
		"tenure": 40,
		"PhoneService": "Yes",
		"MultipleLines": "Yes",
		"InternetService": "Fiber optic",
		"OnlineSecurity": "Yes",
		"OnlineBackup": "Yes",
		"DeviceProtection": "Yes",
		"TechSupport": "Yes",
		"StreamingTV": "Yes",
		"StreamingMovies": "Yes",
		"Contract": "Two year",
		"PaperlessBilling": "No",
		"PaymentMethod": "Mailed check",
		"MonthlyCharges": 90.0,
		"TotalCharges": "3600.0",
		"Churn": "No",
	},
]


class ChurnDbTest(unittest.TestCase):
	def test_clean_raw_dataframe_fills_blank_total_charges(self) -> None:
		data = pd.DataFrame(SAMPLE_ROWS)
		cleaned = clean_raw_dataframe(data)

		self.assertEqual(0.0, cleaned.loc[0, "TotalCharges"])
		self.assertEqual(3600.0, cleaned.loc[1, "TotalCharges"])
		self.assertTrue(pd.api.types.is_integer_dtype(cleaned["SeniorCitizen"]))

	def test_build_database_and_run_queries(self) -> None:
		with TemporaryDirectory() as tmp_dir:
			tmp_path = Path(tmp_dir)
			csv_path = tmp_path / "raw.csv"
			db_path = tmp_path / "churnai.db"
			pd.DataFrame(SAMPLE_ROWS).to_csv(csv_path, index=False)

			rows_written = build_database(csv_path, db_path)
			self.assertEqual(2, rows_written)

			contract_summary = churn_rate_by_contract(db_path)
			self.assertEqual(2, len(contract_summary))
			self.assertIn("churn_rate", contract_summary.columns)

			tenure_summary = tenure_and_charges_by_churn(db_path)
			self.assertEqual({"No", "Yes"}, set(tenure_summary["churn"]))
			self.assertIn("avg_tenure_months", tenure_summary.columns)


if __name__ == "__main__":
	unittest.main()
