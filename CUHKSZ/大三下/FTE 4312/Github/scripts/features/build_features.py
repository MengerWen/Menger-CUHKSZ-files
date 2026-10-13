"""Compatibility wrapper for the unified build-features CLI command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from funding_arb.cli import run_command



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features for the arbitrage pipeline.")
    parser.add_argument("--config", default="configs/features/default.yaml", help="Path to a YAML config file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    return run_command("build-features", args.config, args.log_level)



if __name__ == "__main__":
    raise SystemExit(main())