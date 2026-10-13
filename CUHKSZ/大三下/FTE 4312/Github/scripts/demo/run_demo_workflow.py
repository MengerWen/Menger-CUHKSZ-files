"""Compatibility wrapper for the end-to-end demo workflow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from funding_arb.config.loader import load_settings
from funding_arb.config.models import DemoWorkflowSettings
from funding_arb.demo.workflow import describe_demo_workflow_job, run_demo_workflow
from funding_arb.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full end-to-end demo workflow."
    )
    parser.add_argument(
        "--config",
        default="configs/demo/workflow.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Optional logging level override, e.g. INFO or DEBUG.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_settings(args.config, DemoWorkflowSettings)
    configure_logging(args.log_level or config.output.log_level)
    LOGGER.info("Command: run-demo")
    LOGGER.info("Config path: %s", args.config)
    LOGGER.info(describe_demo_workflow_job(config))
    artifacts = run_demo_workflow(config)
    LOGGER.info("Completed stages: %s", artifacts.completed_stage_count)
    if artifacts.summary_json_path is not None:
        LOGGER.info("Workflow summary JSON: %s", artifacts.summary_json_path)
    if artifacts.markdown_report_path is not None:
        LOGGER.info("Workflow markdown report: %s", artifacts.markdown_report_path)
    if artifacts.artifact_snapshot_path is not None:
        LOGGER.info("Artifact snapshot: %s", artifacts.artifact_snapshot_path)
    if artifacts.frontend_snapshot_path is not None:
        LOGGER.info("Frontend snapshot: %s", artifacts.frontend_snapshot_path)
    if artifacts.overall_status == "failed":
        LOGGER.error("Demo workflow failed at stage: %s", artifacts.failed_stage)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
