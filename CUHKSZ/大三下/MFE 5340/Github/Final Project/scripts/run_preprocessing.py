"""Preprocessing configuration validation entry point.

This validates the intended configuration but does not build a panel yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PreprocessConfig, validate_preprocess_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate preprocessing settings.")
    parser.add_argument("--feature-universe", default="features500", choices=["features500", "features"])
    parser.add_argument("--target-horizon-months", default=1, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessConfig(
        feature_universe=args.feature_universe,
        target_horizon_months=args.target_horizon_months,
    )
    validate_preprocess_config(config)
    print(f"Preprocessing config is valid: {config}")
    print("Use scripts/build_panel.py to build the Stage 2 cleaned panel artifact.")


if __name__ == "__main__":
    main()


