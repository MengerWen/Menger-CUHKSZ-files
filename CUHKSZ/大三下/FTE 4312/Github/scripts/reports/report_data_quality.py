"""Compatibility wrapper for the unified report-data-quality CLI command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from funding_arb.cli import run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory data analysis and data-quality reports.")
    parser.add_argument("--config", default="configs/reports/data_quality.yaml", help="Path to a YAML config file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_command("report-data-quality", args.config, args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())