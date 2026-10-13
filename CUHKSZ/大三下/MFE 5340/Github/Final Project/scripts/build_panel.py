"""Build the cleaned monthly Stage 2 panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import build_monthly_panel, load_cleaning_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaned monthly stock panel.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cleaning_features500.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_cleaning_config(args.config)
    result = build_monthly_panel(config=config, root=PROJECT_ROOT)
    feature_meta = result.metadata["features"]
    print(json.dumps({
        "panel_path": str(result.panel_path),
        "metadata_path": str(result.metadata_path),
        "panel_shape": result.metadata.get("panel_shape"),
        "features": {
            "raw_count": feature_meta["raw_count"],
            "kept_count": feature_meta["kept_count"],
            "dropped_count": feature_meta["dropped_count"],
            "raw_missing_top10": feature_meta["raw_missing_top10"],
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


