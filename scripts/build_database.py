"""CLI: build the SQLite customers database from the raw Telco CSV."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.churn_db import build_database


def main() -> None:
	rows = build_database(ROOT_DIR / "data" / "telco_customer_churn_raw.csv", ROOT_DIR / "data" / "churnai.db")
	print(f"Loaded {rows} rows into data/churnai.db")


if __name__ == "__main__":
	main()
