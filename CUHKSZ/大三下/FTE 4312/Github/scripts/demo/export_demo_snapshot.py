"""Compatibility wrapper for exporting a future demo snapshot."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from funding_arb.demo.pipeline import export_demo_snapshot
from funding_arb.utils.config import load_yaml_config
from funding_arb.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export demo artifacts for the frontend.")
    parser.add_argument("--config", default="configs/demo/default.yaml", help="Path to a YAML config file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_yaml_config(args.config)
    title = config.get("demo", {}).get("title", "Funding-Rate Arbitrage Prototype Dashboard")
    artifact_dir = config.get("demo", {}).get("artifact_dir", "data/artifacts/demo")
    artifacts = export_demo_snapshot(config)
    LOGGER.info("Command: export-demo-snapshot")
    LOGGER.info("Config path: %s", args.config)
    LOGGER.info("Demo snapshot ready for '%s' with artifact directory '%s'.", title, artifact_dir)
    LOGGER.info("Artifact snapshot: %s", artifacts.artifact_snapshot_path)
    LOGGER.info("Frontend snapshot: %s", artifacts.public_snapshot_path)
    LOGGER.info("Frontend assets: %s", artifacts.public_assets_dir)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
