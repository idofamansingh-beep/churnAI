from pathlib import Path
import logging
import subprocess
import sys
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from src import churnai_core


ROOT = Path(__file__).resolve().parents[1]


class DashboardSmokeTest(unittest.TestCase):
	def test_dashboard_runs_without_exceptions(self) -> None:
		logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
		app = AppTest.from_file(str(ROOT / "app" / "dashboard" / "app.py"))
		app.run(timeout=10)
		self.assertEqual([], [exception.value for exception in app.exception])

	def test_dashboard_imports_core_when_run_from_app_directory(self) -> None:
		dashboard_dir = ROOT / "app" / "dashboard"
		result = subprocess.run(
			[sys.executable, "app.py"],
			cwd=dashboard_dir,
			capture_output=True,
			text=True,
			timeout=15,
			check=False,
		)

		self.assertEqual(0, result.returncode, result.stderr)
		self.assertNotIn("ModuleNotFoundError", result.stderr)

	def test_sample_csv_has_required_columns(self) -> None:
		data = pd.read_csv(ROOT / "data" / "sample_customers.csv")
		required_columns = {"customer_id", "tenure_months", "monthly_charges", "contract", "churn"}
		self.assertTrue(required_columns.issubset(data.columns))
		self.assertEqual(10, len(data))
		self.assertGreaterEqual(data["contract"].nunique(), 3)
		self.assertEqual({0, 1}, set(data["churn"].unique()))

		priorities = churnai_core.retention_priorities(data)
		self.assertIsNotNone(priorities)
		self.assertEqual(len(data), len(priorities))

	def test_messy_uploaded_headers_are_matched(self) -> None:
		data = pd.DataFrame(
			{
				"Customer ID": ["C-2001", "C-2002"],
				"Tenure Months": [3, 30],
				"Monthly-Charges": [110.0, 62.0],
				"Contract Type": ["Monthly", "Two year"],
				"Has Churned?": ["yes", "no"],
			}
		)

		self.assertEqual("Has Churned?", churnai_core.find_churn_column(data))
		priorities = churnai_core.retention_priorities(data)
		self.assertIsNotNone(priorities)
		self.assertEqual(["customer_id", "tenure_months", "monthly_charges", "contract"], list(priorities.columns[:4]))


if __name__ == "__main__":
	unittest.main()
