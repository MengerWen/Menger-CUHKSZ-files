"""Compatibility wrapper for the isolated synthetic demo showcase."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from funding_arb.demo_showcase import (
    DemoShowcaseSettings,
    describe_demo_showcase_job,
    run_demo_showcase,
)
from funding_arb.utils.logging import configure_logging
from funding_arb.utils.config import load_yaml_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fully isolated synthetic demo showcase bundle."
    )
    parser.add_argument(
        "--config",
        default="configs/demo/showcase.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level, e.g. INFO or DEBUG.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = DemoShowcaseSettings.model_validate(load_yaml_config(args.config))
    LOGGER.info("Command: build-demo-showcase")
    LOGGER.info("Config path: %s", args.config)
    LOGGER.info(describe_demo_showcase_job(config))
    artifacts = run_demo_showcase(config)
    LOGGER.info("Data root: %s", artifacts.data_root)
    LOGGER.info("Report root: %s", artifacts.report_root)
    LOGGER.info("Frontend public dir: %s", artifacts.frontend_public_dir)
    LOGGER.info("Snapshot: %s", artifacts.snapshot_path)
    LOGGER.info("Frontend snapshot: %s", artifacts.frontend_snapshot_path)
    if artifacts.final_report_path is not None:
        LOGGER.info("Final report markdown: %s", artifacts.final_report_path)
    if artifacts.final_report_html_path is not None:
        LOGGER.info("Final report HTML: %s", artifacts.final_report_html_path)
    LOGGER.info("Manifest: %s", artifacts.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

