"""CLI: train the churn prediction model on the real Telco dataset."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.churn_model import train_and_save


def main() -> None:
	report = train_and_save(
		ROOT_DIR / "data" / "telco_customer_churn_raw.csv",
		ROOT_DIR / "models",
		ROOT_DIR / "reports",
	)
	print(f"Best model: {report['best_model']}")
	print(report["models"][report["best_model"]])


if __name__ == "__main__":
	main()
