"""Inspect Stage 2 data schemas and write a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inspection import inspect_data, write_inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect required Stage 2 data files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metadata" / "data_inspection_stage2.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_data(PROJECT_ROOT)
    write_inspection(summary, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote inspection JSON to {args.output}")


if __name__ == "__main__":
    main()
