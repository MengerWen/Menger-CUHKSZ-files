"""Run the Stage 7 OOS portfolio backtest workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Stage 3 and Stage 5 OOS model signals as monthly portfolios.")
    parser.add_argument(
        "--config",
        default="configs/portfolio_features500.yaml",
        help="Project-relative Stage 7 portfolio config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_backtest(config_path=config_path, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
