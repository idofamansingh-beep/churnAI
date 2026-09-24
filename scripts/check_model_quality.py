"""CLI: fail the build if the trained model's ROC-AUC drops below the quality threshold."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.churn_model import passes_quality_gate


def main() -> None:
	metrics_path = ROOT_DIR / "reports" / "model_metrics.json"
	metrics_report = json.loads(metrics_path.read_text())

	passed, message = passes_quality_gate(metrics_report)
	print(message)
	if not passed:
		sys.exit(1)


if __name__ == "__main__":
	main()
