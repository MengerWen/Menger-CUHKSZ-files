from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.data_pipeline import run_cleaning_pipeline
from fma4200_project.environment import run_environment_check
from fma4200_project.final_report_builder import build_final_report
from fma4200_project.predictive_modeling import run_predictive_modeling_pipeline
from fma4200_project.trading_strategies import run_trading_strategies_pipeline
from fma4200_project.univariate_modeling import run_individual_modeling_pipeline


def main() -> None:
    package_versions = run_environment_check()
    result = run_cleaning_pipeline()
    modeling_result = run_individual_modeling_pipeline()
    predictive_result = run_predictive_modeling_pipeline()
    trading_result = run_trading_strategies_pipeline()
    report_result = build_final_report()

    print("Full pipeline completed successfully.")
    print("Verified imports:")
    for name, version in package_versions:
        print(f"- {name}=={version}")
    print(f"Rows: {result['n_rows']}")
    print(f"Coverage: {result['start_date']:%Y-%m-%d} to {result['end_date']:%Y-%m-%d}")
    print(f"Duplicate date rows: {result['duplicate_date_rows']}")
    print(f"Total missing values: {result['total_missing_values']}")
    print(f"Portfolios modeled: {modeling_result['n_portfolios']}")
    print(f"Predictor source: {predictive_result['predictor_source']}")
    print(f"Trading Section 4 report: {trading_result['section_04_path']}")
    print(f"Final report source: {report_result['final_report_md']}")


if __name__ == "__main__":
    main()
