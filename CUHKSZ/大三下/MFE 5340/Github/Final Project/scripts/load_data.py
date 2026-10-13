"""Pandas-based data tree inspection entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_pickle_table, summarize_data_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect project data locations with pandas.")
    parser.add_argument("--load-pickle", type=Path, help="Optional pickle path to load with pandas.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_pickle is None:
        summary = summarize_data_tree(PROJECT_ROOT)
        print(summary.to_string(index=False))
        return

    obj = load_pickle_table(args.load_pickle)
    print(type(obj).__name__)
    if hasattr(obj, "shape"):
        print(f"shape={obj.shape}")
    if hasattr(obj, "head"):
        print(obj.head().to_string())


if __name__ == "__main__":
    main()
