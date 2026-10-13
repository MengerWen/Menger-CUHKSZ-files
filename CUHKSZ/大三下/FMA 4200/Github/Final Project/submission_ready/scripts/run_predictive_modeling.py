from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.predictive_modeling import run_predictive_modeling_pipeline


def main() -> None:
    result = run_predictive_modeling_pipeline()
    print("Predictive modeling completed successfully.")
    print(f"Portfolios modeled: {result['n_portfolios']}")
    print(f"Predictor source: {result['predictor_source']}")
    print(f"Predictor dataset: {result['predictor_dataset_path']}")
    print(f"Predictive summary: {result['predictive_summary_path']}")
    print(f"Predictive metrics: {result['predictive_metrics_path']}")
    print(f"Predictive forecasts: {result['predictive_forecasts_path']}")


if __name__ == "__main__":
    main()
