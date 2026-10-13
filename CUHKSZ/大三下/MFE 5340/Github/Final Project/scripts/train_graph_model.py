"""Run Stage 5 graph-enhanced conditional latent factor model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.graph_model_pipeline import run_stage5_graph_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Stage 5 graph conditional latent factor model.")
    parser.add_argument(
        "--config",
        default="configs/graph_model_features500.yaml",
        help="Project-relative Stage 5 graph model config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_stage5_graph_model(config_path=config_path, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
