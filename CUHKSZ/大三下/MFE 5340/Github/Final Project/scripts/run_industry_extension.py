"""Run the static industry-classification graph extension."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.industry_extension import run_industry_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/industry_extension_features500.yaml", help="Project-relative extension config path.")
    parser.add_argument("--smoke", action="store_true", help="Run a short smoke test under outputs/industry_extension_smoke.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run_industry_extension(config_path, PROJECT_ROOT, smoke=args.smoke)


if __name__ == "__main__":
    main()
