from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fma4200_project.trading_strategies import run_trading_strategies_pipeline


def main() -> None:
    result = run_trading_strategies_pipeline()
    print("Trading strategies pipeline completed successfully.")
    print(f"Selected VAR lag: {result['selected_var_lag']}")
    print(f"Selected cointegration rank: {result['selected_cointegration_rank']}")
    print(f"Strategy metrics: {result['strategy_metrics_path']}")
    print(f"Strategy returns: {result['strategy_returns_path']}")
    print(f"Section 4 report: {result['section_04_path']}")


if __name__ == "__main__":
    main()
