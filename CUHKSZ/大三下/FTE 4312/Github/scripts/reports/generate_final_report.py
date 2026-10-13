"""Compatibility wrapper for generating the project final report."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from funding_arb.config.loader import load_settings
from funding_arb.config.models import FinalReportSettings
from funding_arb.reporting.final_report import describe_final_report_job, run_final_report
from funding_arb.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the project final report.")
    parser.add_argument(
        "--config",
        default="configs/reports/final_report.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_settings(args.config, FinalReportSettings)
    LOGGER.info("Command: generate-final-report")
    LOGGER.info("Config path: %s", args.config)
    LOGGER.info(describe_final_report_job(config))
    artifacts = run_final_report(config)
    LOGGER.info("Artifact output dir: %s", artifacts.artifact_output_dir)
    if artifacts.markdown_report_path is not None:
        LOGGER.info("Markdown report: %s", artifacts.markdown_report_path)
    if artifacts.html_report_path is not None:
        LOGGER.info("HTML report: %s", artifacts.html_report_path)
    if artifacts.summary_json_path is not None:
        LOGGER.info("Summary JSON: %s", artifacts.summary_json_path)
    if artifacts.public_report_dir is not None:
        LOGGER.info("Frontend public report dir: %s", artifacts.public_report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
