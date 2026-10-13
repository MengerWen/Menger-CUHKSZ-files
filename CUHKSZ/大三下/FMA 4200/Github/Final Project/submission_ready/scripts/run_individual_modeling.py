from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.univariate_modeling import run_individual_modeling_pipeline


def main() -> None:
    result = run_individual_modeling_pipeline()
    print("Individual portfolio modeling completed successfully.")
    print(f"Portfolios modeled: {result['n_portfolios']}")
    print(f"Test summary: {result['test_summary_path']}")
    print(f"Model summary: {result['model_summary_path']}")
    print(f"GARCH summary: {result['garch_summary_path']}")


if __name__ == "__main__":
    main()
