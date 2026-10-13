"""Run the Stage 6 unified model comparison workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.model_comparison import run_stage6_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare Stage 3 and Stage 5 model outputs.")
    parser.add_argument(
        "--config",
        default="configs/evaluation_features500.yaml",
        help="Project-relative Stage 6 evaluation config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_stage6_evaluation(config_path=config_path, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
