from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.data_pipeline import run_cleaning_pipeline


def main() -> None:
    result = run_cleaning_pipeline()
    print("Data cleaning completed successfully.")
    print(f"Rows: {result['n_rows']}")
    print(f"Coverage: {result['start_date']:%Y-%m-%d} to {result['end_date']:%Y-%m-%d}")
    print(f"Duplicate date rows: {result['duplicate_date_rows']}")
    print(f"Total missing values: {result['total_missing_values']}")


if __name__ == "__main__":
    main()

