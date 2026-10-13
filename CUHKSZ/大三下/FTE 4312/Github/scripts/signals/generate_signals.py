"""Compatibility wrapper for the unified generate-signals CLI command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from funding_arb.cli import run_command



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate normalized trading signals from model outputs.")
    parser.add_argument("--config", default="configs/signals/default.yaml", help="Path to a YAML config file.")
    parser.add_argument("--source", default=None, help="Optional source override, e.g. baseline or dl.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    return run_command("generate-signals", args.config, args.log_level, source_override=args.source)



if __name__ == "__main__":
    raise SystemExit(main())
