"""Build Stage 4 monthly stock graphs from available project data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graphs.monthly_graphs import run_stage4_graph_construction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly similarity graphs for Stage 4.")
    parser.add_argument(
        "--config",
        default="configs/graphs_features500.yaml",
        help="Project-relative graph config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_stage4_graph_construction(config_path=config_path, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
