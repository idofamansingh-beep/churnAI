import unittest

import pandas as pd

from src import churnai_core


class ChurnAICoreTest(unittest.TestCase):
	def test_normalize_column_name_handles_spacing_and_punctuation(self) -> None:
		self.assertEqual("monthly_charges", churnai_core.normalize_column_name(" Monthly-Charges "))
		self.assertEqual("contract_type", churnai_core.normalize_column_name("Contract Type"))
		self.assertEqual("has_churned", churnai_core.normalize_column_name("Has Churned?"))

	def test_risk_score_reflects_contract_and_customer_profile(self) -> None:
		monthly_risk = churnai_core.estimate_churn_probability(3, 110.0, "Monthly")
		annual_risk = churnai_core.estimate_churn_probability(30, 60.0, "Annual")

		self.assertGreater(monthly_risk, annual_risk)
		self.assertEqual("Critical", churnai_core.risk_level(monthly_risk))
		self.assertEqual("Low", churnai_core.risk_level(annual_risk))

	def test_churn_labels_accept_text_and_numeric_values(self) -> None:
		data = pd.DataFrame({"churn": [1, 0, None], "churn_text": ["yes", "no", "churned"]})

		numeric_labels = churnai_core.normalized_churn_labels(data, "churn")
		text_labels = churnai_core.normalized_churn_labels(data, "churn_text")

		self.assertEqual([True, False, False], numeric_labels.tolist())
		self.assertEqual([True, False, True], text_labels.tolist())

	def test_retention_priorities_drop_invalid_numeric_rows(self) -> None:
		data = pd.DataFrame(
			{
				"Customer ID": ["C-1", "C-2"],
				"Tenure Months": ["bad", "8"],
				"Monthly Charges": ["90.00", "75.00"],
				"Contract Type": ["Monthly", "Annual"],
			}
		)

		priorities = churnai_core.retention_priorities(data)

		self.assertIsNotNone(priorities)
		self.assertEqual(["C-2"], priorities["customer_id"].tolist())
		self.assertIn("estimated_churn_risk", priorities.columns)


if __name__ == "__main__":
	unittest.main()
