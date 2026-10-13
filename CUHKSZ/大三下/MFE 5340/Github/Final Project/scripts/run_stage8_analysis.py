"""Run Stage 8 interpretability, robustness, and report-ready analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stage8_analysis import run_stage8_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 8 interpretability and robustness analysis.")
    parser.add_argument(
        "--config",
        default="configs/stage8_features500.yaml",
        help="Project-relative Stage 8 config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_stage8_analysis(config_path=config_path, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
