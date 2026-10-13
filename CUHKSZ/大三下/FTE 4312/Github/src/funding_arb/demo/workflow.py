"""Orchestrate a presentation-friendly end-to-end demo workflow."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from funding_arb.config.models import DemoWorkflowSettings
from funding_arb.utils.config import load_config
from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class DemoWorkflowStagePlan:
    """One runnable stage in the end-to-end demo workflow."""

    key: str
    label: str
    optional: bool
    command: list[str]


@dataclass(frozen=True)
class DemoWorkflowStageResult:
    """Execution result for one stage."""

    key: str
    label: str
    optional: bool
    status: str
    return_code: int | None
    duration_seconds: float
    command: list[str]


@dataclass(frozen=True)
class DemoWorkflowArtifacts:
    """Summary paths produced by the demo workflow."""

    overall_status: str
    summary_json_path: str | None
    markdown_report_path: str | None
    frontend_snapshot_path: str | None
    artifact_snapshot_path: str | None
    failed_stage: str | None
    completed_stage_count: int


STAGE_SEQUENCE: tuple[tuple[str, str], ...] = (
    ("fetch_data", "Fetch and normalize market data"),
    ("report_data_quality", "Generate data-quality report"),
    ("build_features", "Build feature table"),
    ("build_labels", "Build supervised dataset"),
    ("build_exploratory_dataset", "Build exploratory DL dataset"),
    ("train_baseline", "Train baseline models"),
    ("train_deep_learning", "Train deep-learning model"),
    ("compare_deep_learning", "Compare deep-learning model zoo"),
    ("compare_exploratory_deep_learning", "Compare exploratory deep-learning showcase bundle"),
    ("compare_exploratory_direction", "Compare exploratory direction-classification bundle"),
    ("generate_baseline_signals", "Generate baseline signals"),
    ("generate_deep_learning_signals", "Generate deep-learning signals"),
    ("generate_exploratory_signals", "Generate exploratory deep-learning signals"),
    ("backtest", "Run backtest"),
    ("backtest_exploratory", "Run exploratory deep-learning backtest"),
    ("sync_vault", "Prepare or submit vault update"),
    ("report_exploratory", "Generate exploratory deep-learning report"),
    ("export_demo_snapshot", "Export frontend demo snapshot"),
)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_path(*path.parts)


def _cli_command(
    command_name: str, config_path: str, log_level: str, source_override: str | None = None
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.main",
        command_name,
        "--config",
        config_path,
        "--log-level",
        log_level,
    ]
    if source_override is not None:
        command.extend(["--source", source_override])
    return command


def _script_command(script_path: str, config_path: str, log_level: str) -> list[str]:
    return [
        sys.executable,
        script_path,
        "--config",
        config_path,
        "--log-level",
        log_level,
    ]


def build_stage_plan(config: DemoWorkflowSettings) -> list[DemoWorkflowStagePlan]:
    """Build the ordered stage plan from config toggles."""
    log_level = config.output.log_level
    command_paths = config.commands
    stage_settings = config.stages

    stage_commands: dict[str, list[str]] = {
        "fetch_data": _cli_command(
            "fetch-data", command_paths.fetch_data_config_path, log_level
        ),
        "report_data_quality": _cli_command(
            "report-data-quality",
            command_paths.report_data_quality_config_path,
            log_level,
        ),
        "build_features": _cli_command(
            "build-features", command_paths.features_config_path, log_level
        ),
        "build_labels": _cli_command(
            "build-labels", command_paths.labels_config_path, log_level
        ),
        "build_exploratory_dataset": _cli_command(
            "build-exploratory-dl-dataset",
            command_paths.exploratory_dataset_config_path,
            log_level,
        ),
        "train_baseline": _cli_command(
            "train-baseline", command_paths.baseline_config_path, log_level
        ),
        "train_deep_learning": _cli_command(
            "train-dl", command_paths.deep_learning_config_path, log_level
        ),
        "compare_deep_learning": _cli_command(
            "compare-dl",
            command_paths.deep_learning_comparison_config_path,
            log_level,
        ),
        "compare_exploratory_deep_learning": _cli_command(
            "compare-dl",
            command_paths.exploratory_deep_learning_comparison_config_path,
            log_level,
        ),
        "compare_exploratory_direction": _cli_command(
            "compare-dl",
            command_paths.exploratory_direction_comparison_config_path,
            log_level,
        ),
        "generate_baseline_signals": _cli_command(
            "generate-signals",
            command_paths.signals_config_path,
            log_level,
            source_override="baseline",
        ),
        "generate_deep_learning_signals": _cli_command(
            "generate-signals",
            command_paths.signals_config_path,
            log_level,
            source_override="dl",
        ),
        "generate_exploratory_signals": _cli_command(
            "generate-exploratory-dl-signals",
            command_paths.exploratory_signals_config_path,
            log_level,
        ),
        "backtest": _cli_command(
            "backtest", command_paths.backtest_config_path, log_level
        ),
        "backtest_exploratory": _cli_command(
            "backtest", command_paths.exploratory_backtest_config_path, log_level
        ),
        "sync_vault": _cli_command(
            "sync-vault", command_paths.integration_config_path, log_level
        ),
        "report_exploratory": _cli_command(
            "generate-exploratory-dl-report",
            command_paths.exploratory_report_config_path,
            log_level,
        ),
        "export_demo_snapshot": _script_command(
            str(repo_path("scripts", "demo", "export_demo_snapshot.py")),
            command_paths.demo_snapshot_config_path,
            log_level,
        ),
    }

    plans: list[DemoWorkflowStagePlan] = []
    for key, label in STAGE_SEQUENCE:
        stage = getattr(stage_settings, key)
        if not stage.enabled:
            continue
        plans.append(
            DemoWorkflowStagePlan(
                key=key,
                label=label,
                optional=stage.optional,
                command=stage_commands[key],
            )
        )
    return plans


def describe_demo_workflow_job(config: DemoWorkflowSettings) -> str:
    """Describe the enabled demo workflow stages."""
    stage_plan = build_stage_plan(config)
    stage_labels = [
        f"{plan.label}{' (optional)' if plan.optional else ''}" for plan in stage_plan
    ]
    return (
        f"Demo workflow '{config.output.run_name}' will run "
        f"{len(stage_plan)} stages: {', '.join(stage_labels)}."
    )


def _stage_status_line(result: DemoWorkflowStageResult) -> str:
    command_text = " ".join(f'"{part}"' if " " in part else part for part in result.command)
    return (
        f"| {result.label} | {result.status} | {'yes' if result.optional else 'no'} | "
        f"{result.return_code if result.return_code is not None else 'n/a'} | "
        f"{result.duration_seconds:.1f}s | `{command_text}` |"
    )


def _artifact_exists(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def _stage_existing_artifact_paths(
    stage_key: str, config: DemoWorkflowSettings
) -> list[Path]:
    if stage_key == "fetch_data":
        feature_config = load_config(_resolve_path(config.commands.features_config_path))
        return [_resolve_path(feature_config["input"]["dataset_path"])]
    if stage_key == "report_data_quality":
        demo_snapshot_config = load_config(
            _resolve_path(config.commands.demo_snapshot_config_path)
        )
        return [
            _resolve_path(demo_snapshot_config["inputs"]["data_quality_summary_path"])
        ]
    if stage_key == "build_features":
        label_config = load_config(_resolve_path(config.commands.labels_config_path))
        return [_resolve_path(label_config["input"]["feature_table_path"])]
    if stage_key == "build_labels":
        baseline_config = load_config(
            _resolve_path(config.commands.baseline_config_path)
        )
        return [_resolve_path(baseline_config["input"]["dataset_path"])]
    if stage_key == "build_exploratory_dataset":
        exploratory_config = load_config(
            _resolve_path(config.commands.exploratory_dataset_config_path)
        )
        return [
            _resolve_path(exploratory_config["output"]["output_dir"])
            / exploratory_config["input"]["provider"]
            / exploratory_config["input"]["symbol"].lower()
            / exploratory_config["input"]["frequency"]
            / exploratory_config["output"]["artifact_name"]
        ]
    if stage_key == "train_baseline":
        signal_config = load_config(_resolve_path(config.commands.signals_config_path))
        return [_resolve_path(signal_config["input"]["baseline_predictions_path"])]
    if stage_key == "train_deep_learning":
        signal_config = load_config(_resolve_path(config.commands.signals_config_path))
        return [_resolve_path(signal_config["input"]["dl_predictions_path"])]
    if stage_key == "compare_deep_learning":
        comparison_config = load_config(
            _resolve_path(config.commands.deep_learning_comparison_config_path)
        )
        first_run_config = load_config(_resolve_path(comparison_config["runs"][0]["config_path"]))
        output_dir = _resolve_path(comparison_config["output"]["output_dir"])
        return [
            output_dir
            / first_run_config["input"]["provider"]
            / first_run_config["input"]["symbol"].lower()
            / first_run_config["input"]["frequency"]
            / comparison_config["output"]["run_name"]
            / "comparison_summary.parquet",
        ]
    if stage_key == "compare_exploratory_deep_learning":
        comparison_config = load_config(
            _resolve_path(config.commands.exploratory_deep_learning_comparison_config_path)
        )
        first_run_config = load_config(_resolve_path(comparison_config["runs"][0]["config_path"]))
        output_dir = _resolve_path(comparison_config["output"]["output_dir"])
        return [
            output_dir
            / first_run_config["input"]["provider"]
            / first_run_config["input"]["symbol"].lower()
            / first_run_config["input"]["frequency"]
            / comparison_config["output"]["run_name"]
            / "comparison_summary.parquet",
        ]
    if stage_key == "compare_exploratory_direction":
        comparison_config = load_config(
            _resolve_path(config.commands.exploratory_direction_comparison_config_path)
        )
        first_run_config = load_config(_resolve_path(comparison_config["runs"][0]["config_path"]))
        output_dir = _resolve_path(comparison_config["output"]["output_dir"])
        return [
            output_dir
            / first_run_config["input"]["provider"]
            / first_run_config["input"]["symbol"].lower()
            / first_run_config["input"]["frequency"]
            / comparison_config["output"]["run_name"]
            / "comparison_summary.parquet",
        ]
    if stage_key == "generate_baseline_signals":
        backtest_config = load_config(_resolve_path(config.commands.backtest_config_path))
        return [_resolve_path(backtest_config["input"]["signal_path"])]
    if stage_key == "generate_deep_learning_signals":
        signal_config = load_config(_resolve_path(config.commands.signals_config_path))
        signal_output = signal_config["output"]
        signal_input = signal_config["input"]
        return [
            _resolve_path(signal_output["output_dir"])
            / signal_input["provider"]
            / signal_input["symbol"].lower()
            / signal_input["frequency"]
            / "dl"
            / signal_output["artifact_name"]
        ]
    if stage_key == "generate_exploratory_signals":
        signal_config = load_config(_resolve_path(config.commands.exploratory_signals_config_path))
        signal_output = signal_config["output"]
        signal_input = signal_config["input"]
        return [
            _resolve_path(signal_output["output_dir"])
            / signal_input["provider"]
            / signal_input["symbol"].lower()
            / signal_input["frequency"]
            / "exploratory"
            / signal_output["artifact_name"]
        ]
    if stage_key == "backtest":
        demo_snapshot_config = load_config(
            _resolve_path(config.commands.demo_snapshot_config_path)
        )
        return [
            _resolve_path(demo_snapshot_config["inputs"]["backtest_manifest_path"]),
            _resolve_path(demo_snapshot_config["inputs"]["backtest_leaderboard_path"]),
        ]
    if stage_key == "backtest_exploratory":
        exploratory_config = load_config(
            _resolve_path(config.commands.exploratory_backtest_config_path)
        )
        output_dir = _resolve_path(exploratory_config["reporting"]["output_dir"])
        return [
            output_dir
            / exploratory_config["input"]["provider"]
            / exploratory_config["input"]["symbol"].lower()
            / exploratory_config["input"]["frequency"]
            / exploratory_config["reporting"]["run_name"]
            / "leaderboard.parquet"
        ]
    if stage_key == "sync_vault":
        demo_snapshot_config = load_config(
            _resolve_path(config.commands.demo_snapshot_config_path)
        )
        return [
            _resolve_path(demo_snapshot_config["inputs"]["integration_plan_path"]),
            _resolve_path(
                demo_snapshot_config["inputs"]["integration_call_summary_path"]
            ),
        ]
    if stage_key == "report_exploratory":
        report_config = load_config(
            _resolve_path(config.commands.exploratory_report_config_path)
        )
        return [
            _resolve_path(report_config["output"]["output_dir"])
            / "binance"
            / "btcusdt"
            / "1h"
            / "exploratory_dl_summary.json"
        ]
    if stage_key == "export_demo_snapshot":
        artifact_snapshot_path, frontend_snapshot_path = _load_snapshot_locations(config)
        paths: list[Path] = []
        if artifact_snapshot_path is not None:
            paths.append(Path(artifact_snapshot_path))
        if frontend_snapshot_path is not None:
            paths.append(Path(frontend_snapshot_path))
        return paths
    return []


def _load_snapshot_locations(config: DemoWorkflowSettings) -> tuple[str | None, str | None]:
    snapshot_config_path = _resolve_path(config.commands.demo_snapshot_config_path)
    if not snapshot_config_path.exists():
        return None, None
    snapshot_config = load_config(snapshot_config_path)
    demo_config = snapshot_config.get("demo", {})
    artifact_dir = demo_config.get("artifact_dir")
    frontend_dir = demo_config.get("frontend_public_dir")
    artifact_snapshot = None
    frontend_snapshot = None
    if artifact_dir:
        artifact_snapshot = str(_resolve_path(artifact_dir) / "demo_snapshot.json")
    if frontend_dir:
        frontend_snapshot = str(_resolve_path(frontend_dir) / "demo_snapshot.json")
    return artifact_snapshot, frontend_snapshot


def _write_summary_files(
    config: DemoWorkflowSettings,
    stage_results: list[DemoWorkflowStageResult],
    overall_status: str,
    failed_stage: str | None,
) -> tuple[str | None, str | None]:
    output_dir = ensure_directory(
        _resolve_path(config.output.output_dir) / config.output.run_name
    )
    artifact_snapshot_path, frontend_snapshot_path = _load_snapshot_locations(config)

    successful_statuses = {"completed", "reused_existing_artifacts"}
    summary = {
        "run_name": config.output.run_name,
        "overall_status": overall_status,
        "failed_stage": failed_stage,
        "generated_at_epoch_seconds": time.time(),
        "completed_stage_count": sum(
            1 for result in stage_results if result.status in successful_statuses
        ),
        "stage_results": [
            {
                "key": result.key,
                "label": result.label,
                "optional": result.optional,
                "status": result.status,
                "return_code": result.return_code,
                "duration_seconds": result.duration_seconds,
                "command": result.command,
            }
            for result in stage_results
        ],
        "artifact_snapshot_path": artifact_snapshot_path,
        "frontend_snapshot_path": frontend_snapshot_path,
        "frontend_ready": bool(
            frontend_snapshot_path and Path(frontend_snapshot_path).exists()
        ),
        "frontend_dir": str(_resolve_path(config.frontend.frontend_dir)),
        "frontend_dashboard_url": config.frontend.dashboard_url,
        "frontend_dev_command": config.frontend.dev_command,
        "notes": config.notes,
    }

    summary_path: str | None = None
    if config.output.write_json:
        summary_file = output_dir / "demo_workflow_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_path = str(summary_file)

    report_path: str | None = None
    if config.output.write_markdown_report:
        report_lines = [
            "# End-to-End Demo Workflow",
            "",
            f"- Run name: `{config.output.run_name}`",
            f"- Status: `{overall_status}`",
            f"- Failed stage: `{failed_stage or 'none'}`",
            "",
            "## Stage Results",
            "",
            "| Stage | Status | Optional | Return Code | Duration | Command |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        report_lines.extend(_stage_status_line(result) for result in stage_results)
        report_lines.extend(
            [
                "",
                "## Demo Artifacts",
                "",
                f"- Artifact snapshot: `{artifact_snapshot_path or 'not configured'}`",
                f"- Frontend snapshot: `{frontend_snapshot_path or 'not configured'}`",
                "",
                "## Next Steps",
                "",
                "1. If the frontend snapshot exists, start the dashboard:",
                f"   `cd {config.frontend.frontend_dir}`",
                f"   `{config.frontend.dev_command}`",
                "2. If you want a live local-chain update instead of dry-run sync, deploy the vault, set `VAULT_ADDRESS` and `PRIVATE_KEY`, enable broadcast in `configs/integration/default.yaml`, then rerun the sync stage.",
            ]
        )
        report_file = output_dir / "demo_workflow_report.md"
        report_file.write_text("\n".join(report_lines), encoding="utf-8")
        report_path = str(report_file)

    return summary_path, report_path


def run_demo_workflow(config: DemoWorkflowSettings) -> DemoWorkflowArtifacts:
    """Run the configured end-to-end demo workflow."""
    stage_plan = build_stage_plan(config)
    stage_results: list[DemoWorkflowStageResult] = []
    failed_stage: str | None = None
    overall_status = "completed"
    had_warnings = False

    for stage in stage_plan:
        started = time.perf_counter()
        completed = subprocess.run(
            stage.command,
            cwd=repo_path(),
            check=False,
        )
        duration = time.perf_counter() - started
        status = "completed" if completed.returncode == 0 else "failed"
        result = DemoWorkflowStageResult(
            key=stage.key,
            label=stage.label,
            optional=stage.optional,
            status=status,
            return_code=completed.returncode,
            duration_seconds=duration,
            command=stage.command,
        )
        stage_results.append(result)
        if completed.returncode == 0:
            continue
        existing_artifacts = _stage_existing_artifact_paths(stage.key, config)
        if _artifact_exists(existing_artifacts):
            # Presentation runs should remain usable when an upstream stage is
            # temporarily unavailable but its last successful artifacts are
            # already present locally.
            stage_results[-1] = DemoWorkflowStageResult(
                key=stage.key,
                label=stage.label,
                optional=stage.optional,
                status="reused_existing_artifacts",
                return_code=completed.returncode,
                duration_seconds=duration,
                command=stage.command,
            )
            had_warnings = True
            continue
        if stage.optional and config.execution.continue_on_optional_failure:
            had_warnings = True
            continue
        failed_stage = stage.key
        overall_status = "failed"
        break

    if overall_status != "failed" and had_warnings:
        overall_status = "completed_with_warnings"

    summary_json_path, markdown_report_path = _write_summary_files(
        config,
        stage_results,
        overall_status,
        failed_stage,
    )
    artifact_snapshot_path, frontend_snapshot_path = _load_snapshot_locations(config)

    return DemoWorkflowArtifacts(
        overall_status=overall_status,
        summary_json_path=summary_json_path,
        markdown_report_path=markdown_report_path,
        frontend_snapshot_path=frontend_snapshot_path,
        artifact_snapshot_path=artifact_snapshot_path,
        failed_stage=failed_stage,
        completed_stage_count=sum(
            1
            for result in stage_results
            if result.status in {"completed", "reused_existing_artifacts"}
        ),
    )
