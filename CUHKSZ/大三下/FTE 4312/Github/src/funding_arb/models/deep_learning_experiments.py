"""Compact experiment orchestration for multi-model deep-learning comparisons."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from funding_arb.config.models import (
    DeepLearningComparisonRunSettings,
    DeepLearningComparisonSettings,
    DeepLearningSettings,
)
from funding_arb.models.deep_learning import run_deep_learning_pipeline
from funding_arb.utils.config import load_config
from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class DeepLearningComparisonArtifacts:
    """Paths produced by a multi-model deep-learning comparison run."""

    output_dir: str
    comparison_summary_path: str
    comparison_summary_csv_path: str | None
    validation_leaderboard_path: str
    validation_leaderboard_csv_path: str | None
    test_leaderboard_path: str
    test_leaderboard_csv_path: str | None
    strategy_leaderboard_path: str
    strategy_leaderboard_csv_path: str | None
    report_path: str | None
    manifest_path: str
    figure_paths: list[str]


@dataclass(frozen=True)
class _ResolvedComparisonRun:
    """One resolved model run used inside the comparison orchestrator."""

    label: str
    config_path: str
    resolved_config_path: str
    settings: DeepLearningSettings
    manifest: dict[str, Any]
    leaderboard: pd.DataFrame
    artifact_reused: bool


def _resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else repo_path(*path.parts)


def _deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _model_output_dir(settings: DeepLearningSettings) -> Path:
    return (
        _resolve_path(settings.output.model_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
        / settings.output.run_name
    )


def _comparison_output_dir(
    settings: DeepLearningComparisonSettings,
    reference: DeepLearningSettings,
) -> Path:
    return ensure_directory(
        _resolve_path(settings.output.output_dir)
        / reference.input.provider
        / reference.input.symbol.lower()
        / reference.input.frequency
        / settings.output.run_name
    )


def _metric_direction(metric_name: str) -> str:
    lowered = metric_name.lower()
    if any(token in lowered for token in ["loss", "rmse", "mae", "error"]):
        return "min"
    return "max"


def _model_group(model_name: str) -> str:
    if model_name in {"lstm", "gru"}:
        return "recurrent"
    if model_name == "tcn":
        return "convolutional"
    if model_name == "transformer_encoder":
        return "attention"
    return "other"


def _run_spec_label(
    run_spec: DeepLearningComparisonRunSettings,
    settings: DeepLearningSettings,
) -> str:
    return run_spec.name or settings.model.name


def _resolve_run_settings(
    run_spec: DeepLearningComparisonRunSettings,
) -> tuple[Path, DeepLearningSettings]:
    resolved_config_path = _resolve_path(run_spec.config_path)
    raw_base = load_config(resolved_config_path)
    merged = _deep_merge_dicts(raw_base, run_spec.overrides)
    settings = DeepLearningSettings.model_validate(merged)
    return resolved_config_path, settings


def _load_existing_run_artifacts(
    settings: DeepLearningSettings,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    manifest_path = _model_output_dir(settings) / "dl_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leaderboard_path = Path(manifest["leaderboard_path"])
    if not leaderboard_path.exists():
        return None
    return manifest, pd.read_parquet(leaderboard_path)


def _ensure_run_result(
    run_spec: DeepLearningComparisonRunSettings,
    comparison_settings: DeepLearningComparisonSettings,
) -> _ResolvedComparisonRun:
    resolved_config_path, run_settings = _resolve_run_settings(run_spec)
    reuse_existing = (
        not comparison_settings.runner.force_retrain_all and not run_spec.force_retrain
    )
    existing = _load_existing_run_artifacts(run_settings) if reuse_existing else None

    if existing is None:
        if not comparison_settings.runner.train_if_missing and reuse_existing:
            raise FileNotFoundError(
                f"Missing deep-learning artifacts for '{run_settings.output.run_name}' "
                "and train_if_missing is disabled."
            )
        artifacts = run_deep_learning_pipeline(run_settings)
        manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))
        leaderboard = pd.read_parquet(artifacts.leaderboard_path)
        artifact_reused = False
    else:
        manifest, leaderboard = existing
        artifact_reused = True

    return _ResolvedComparisonRun(
        label=_run_spec_label(run_spec, run_settings),
        config_path=run_spec.config_path,
        resolved_config_path=str(resolved_config_path),
        settings=run_settings,
        manifest=manifest,
        leaderboard=leaderboard,
        artifact_reused=artifact_reused,
    )


def _validate_run_compatibility(runs: list[_ResolvedComparisonRun]) -> None:
    if not runs:
        raise ValueError("Deep-learning comparison received no resolved runs.")
    reference = runs[0].settings
    for run in runs[1:]:
        current = run.settings
        mismatches: list[str] = []
        if current.input.provider != reference.input.provider:
            mismatches.append("provider")
        if current.input.symbol != reference.input.symbol:
            mismatches.append("symbol")
        if current.input.frequency != reference.input.frequency:
            mismatches.append("frequency")
        if current.target.task != reference.target.task:
            mismatches.append("task")
        if current.target.column != reference.target.column:
            mismatches.append("target.column")
        if mismatches:
            raise ValueError(
                "Deep-learning comparison bundles should compare like-with-like. "
                f"Run '{run.label}' mismatched fields: {', '.join(mismatches)}."
            )


def _leaderboard_rows_by_split(leaderboard: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for split_name in ["validation", "test"]:
        split_frame = leaderboard.loc[leaderboard["split"] == split_name]
        rows[split_name] = {} if split_frame.empty else split_frame.iloc[0].to_dict()
    return rows


def _build_summary_frame(
    comparison_settings: DeepLearningComparisonSettings,
    runs: list[_ResolvedComparisonRun],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        leaderboard_rows = _leaderboard_rows_by_split(run.leaderboard)
        manifest = run.manifest
        selected_hyperparameters = manifest.get("selected_hyperparameters", {})
        row: dict[str, Any] = {
            "experiment_name": comparison_settings.experiment_name,
            "run_label": run.label,
            "model_name": run.settings.model.name,
            "model_group": _model_group(run.settings.model.name),
            "task": run.settings.target.task,
            "target_column": run.settings.target.column,
            "lookback_steps": run.settings.sequence.lookback_steps,
            "best_epoch": manifest.get("best_epoch"),
            "checkpoint_selection_metric": manifest.get("best_checkpoint_metric"),
            "checkpoint_selection_metric_value": manifest.get("best_checkpoint_metric_value"),
            "checkpoint_selection_effective_metric": manifest.get(
                "best_checkpoint_effective_metric"
            ),
            "checkpoint_selection_effective_metric_value": manifest.get(
                "best_checkpoint_effective_metric_value"
            ),
            "checkpoint_selection_fallback_used": manifest.get(
                "checkpoint_metric_fallback_used"
            ),
            "status": manifest.get("status", "ok"),
            "reason": manifest.get("reason"),
            "degenerate_experiment": bool(manifest.get("degenerate_experiment", False)),
            "degenerate_stage": manifest.get("degenerate_stage"),
            "degenerate_reason": manifest.get("degenerate_reason"),
            "fallback_used": bool(manifest.get("fallback_used", False)),
            "fallback_reason": manifest.get("fallback_reason"),
            "selected_threshold": manifest.get("selected_threshold"),
            "selected_threshold_objective": manifest.get("selected_threshold_objective"),
            "selected_threshold_objective_value": manifest.get(
                "selected_threshold_objective_value"
            ),
            "selected_loss": manifest.get("selected_loss"),
            "prediction_mode": manifest.get("prediction_mode"),
            "preprocessing_scaler": run.settings.preprocessing.scaler,
            "winsorize_lower_quantile": run.settings.preprocessing.winsorize_lower_quantile,
            "winsorize_upper_quantile": run.settings.preprocessing.winsorize_upper_quantile,
            "selected_hyperparameters_json": json.dumps(
                selected_hyperparameters, sort_keys=True
            ),
            "hidden_size": run.settings.model.hidden_size,
            "num_layers": run.settings.model.num_layers,
            "dropout": run.settings.model.dropout,
            "bidirectional": run.settings.model.bidirectional,
            "tcn_hidden_channels": run.settings.model.tcn_hidden_channels,
            "tcn_num_blocks": run.settings.model.tcn_num_blocks,
            "tcn_kernel_size": run.settings.model.tcn_kernel_size,
            "transformer_d_model": run.settings.model.transformer_d_model,
            "transformer_nhead": run.settings.model.transformer_nhead,
            "transformer_num_layers": run.settings.model.transformer_num_layers,
            "transformer_dim_feedforward": run.settings.model.transformer_dim_feedforward,
            "training_batch_size": run.settings.training.batch_size,
            "learning_rate": run.settings.training.learning_rate,
            "weight_decay": run.settings.training.weight_decay,
            "config_path": run.config_path,
            "resolved_config_path": run.resolved_config_path,
            "run_name": run.settings.output.run_name,
            "artifact_reused": run.artifact_reused,
            "checkpoint_path": manifest.get("checkpoint_path"),
            "history_path": manifest.get("history_path"),
            "predictions_path": manifest.get("predictions_path"),
            "metrics_path": manifest.get("metrics_path"),
            "leaderboard_path": manifest.get("leaderboard_path"),
            "report_path": manifest.get("report_path"),
            "manifest_path": str(_model_output_dir(run.settings) / "dl_manifest.json"),
        }
        for split_name, metrics_row in leaderboard_rows.items():
            for key, value in metrics_row.items():
                if key in {"model_name", "model_family", "task", "split"}:
                    continue
                row[f"{split_name}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _rank_table(
    frame: pd.DataFrame,
    metric_name: str,
    *,
    metric_prefix: str,
) -> pd.DataFrame:
    metric_column = f"{metric_prefix}_{metric_name}"
    leaderboard = frame.copy()
    if metric_column not in leaderboard.columns:
        leaderboard[metric_column] = pd.NA
    ascending = _metric_direction(metric_name) == "min"
    leaderboard = leaderboard.sort_values(
        by=[metric_column, "run_label"],
        ascending=[ascending, True],
        na_position="last",
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    leaderboard.insert(1, "ranking_metric", metric_name)
    leaderboard.insert(2, "ranking_metric_value", leaderboard[metric_column])
    return leaderboard


def _narrow_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].copy()


def _build_leaderboard_tables(
    summary_frame: pd.DataFrame,
    comparison_settings: DeepLearningComparisonSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation = _rank_table(
        summary_frame,
        comparison_settings.ranking.validation_metric,
        metric_prefix="validation",
    )
    test = _rank_table(
        summary_frame,
        comparison_settings.ranking.test_metric,
        metric_prefix="test",
    )
    strategy_prefix = comparison_settings.ranking.strategy_split
    strategy = _rank_table(
        summary_frame,
        comparison_settings.ranking.strategy_metric,
        metric_prefix=strategy_prefix,
    )
    base_columns = [
        "rank",
        "run_label",
        "model_name",
        "model_group",
        "status",
        "reason",
        "degenerate_experiment",
        "task",
        "target_column",
        "lookback_steps",
        "best_epoch",
        "selected_loss",
        "checkpoint_selection_metric",
        "checkpoint_selection_effective_metric",
        "selected_threshold",
        "ranking_metric",
        "ranking_metric_value",
    ]
    validation_columns = base_columns + [
        "validation_pearson_corr",
        "validation_rmse",
        "validation_f1",
        "validation_roc_auc",
        "validation_avg_signal_return_bps",
        "validation_cumulative_signal_return_bps",
        "validation_signal_hit_rate",
        "validation_signal_count",
        "validation_top_quantile_avg_return_bps",
    ]
    test_columns = base_columns + [
        "test_pearson_corr",
        "test_rmse",
        "test_f1",
        "test_roc_auc",
        "test_avg_signal_return_bps",
        "test_cumulative_signal_return_bps",
        "test_signal_hit_rate",
        "test_signal_count",
        "test_top_quantile_avg_return_bps",
    ]
    strategy_columns = base_columns + [
        f"{strategy_prefix}_avg_signal_return_bps",
        f"{strategy_prefix}_cumulative_signal_return_bps",
        f"{strategy_prefix}_signal_hit_rate",
        f"{strategy_prefix}_signal_count",
        f"{strategy_prefix}_top_quantile_avg_return_bps",
        f"{strategy_prefix}_top_quantile_cumulative_return_bps",
    ]
    return (
        _narrow_columns(validation, validation_columns),
        _narrow_columns(test, test_columns),
        _narrow_columns(strategy, strategy_columns),
    )


def _table_to_markdown(frame: pd.DataFrame) -> str:
    return "_No rows._" if frame.empty else frame.to_markdown(index=False)


def _best_model_note(
    summary_frame: pd.DataFrame,
    comparison_settings: DeepLearningComparisonSettings,
) -> str:
    if summary_frame.empty:
        return "No comparison rows were produced."
    ranked = _rank_table(
        summary_frame,
        comparison_settings.ranking.test_metric,
        metric_prefix="test",
    )
    best_row = ranked.iloc[0]
    note = (
        "Current default best model under the configured test ranking metric "
        f"(`{comparison_settings.ranking.test_metric}`) is `{best_row['run_label']}` "
        f"with score `{best_row['ranking_metric_value']}`."
    )
    if bool(best_row.get("degenerate_experiment", False)):
        reason = best_row.get("reason") or best_row.get("degenerate_reason") or "degenerate experiment"
        note += f" This run is flagged as degenerate: {reason}."
    return note


def _write_bar_plot(
    frame: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    title: str,
    output_path: Path,
) -> str | None:
    if frame.empty or value_column not in frame.columns:
        return None
    plot_frame = frame[[label_column, value_column]].dropna()
    if plot_frame.empty:
        return None
    figure, axis = plt.subplots(figsize=(9, max(3, 1.6 + 0.6 * len(plot_frame))))
    axis.barh(plot_frame[label_column], plot_frame[value_column], color="#2E86AB")
    axis.set_title(title)
    axis.set_xlabel(value_column)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return str(output_path)


def _write_report(
    comparison_settings: DeepLearningComparisonSettings,
    output_dir: Path,
    runs: list[_ResolvedComparisonRun],
    summary_frame: pd.DataFrame,
    validation_leaderboard: pd.DataFrame,
    test_leaderboard: pd.DataFrame,
    strategy_leaderboard: pd.DataFrame,
    figure_paths: list[str],
) -> str | None:
    if not comparison_settings.output.write_markdown_report:
        return None
    report_path = output_dir / "comparison_report.md"
    run_lines = [
        f"- `{run.label}` -> `{run.settings.model.name}` using `{run.config_path}` "
        f"(reused_artifacts={run.artifact_reused})"
        for run in runs
    ]
    lines = [
        "# Deep-Learning Comparison Report",
        "",
        f"- Experiment: `{comparison_settings.experiment_name}`",
        f"- Description: {comparison_settings.description or 'N/A'}",
        f"- Compared runs: `{len(runs)}`",
        f"- Validation ranking metric: `{comparison_settings.ranking.validation_metric}`",
        f"- Test ranking metric: `{comparison_settings.ranking.test_metric}`",
        f"- Strategy ranking metric: `{comparison_settings.ranking.strategy_metric}` on `{comparison_settings.ranking.strategy_split}`",
        "",
        "## Runs",
        "",
        "\n".join(run_lines),
        "",
        "## Main Finding",
        "",
        _best_model_note(summary_frame, comparison_settings),
        "",
        "## Validation Leaderboard",
        "",
        _table_to_markdown(validation_leaderboard),
        "",
        "## Test Leaderboard",
        "",
        _table_to_markdown(test_leaderboard),
        "",
        "## Strategy-Oriented Leaderboard",
        "",
        _table_to_markdown(strategy_leaderboard),
        "",
        "## Notes",
        "",
        "- This Phase 2 comparison layer reuses the stable Phase 1 per-model artifact contract rather than inventing a new tracking system.",
        "- Runs are compared only after validating that provider, symbol, frequency, task, and target column match.",
        "- Artifact reuse is allowed so repeated comparison refreshes stay lightweight when model outputs already exist.",
        "- Comparison tables now preserve per-run `status`, `reason`, and `degenerate_experiment` so a high predictive score cannot silently hide a no-signal trading regime.",
    ]
    if figure_paths:
        lines.extend(
            [
                "",
                "## Figures",
                "",
                "\n".join(f"- `{Path(path).name}`" for path in figure_paths),
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def describe_deep_learning_comparison_job(
    config: DeepLearningComparisonSettings | dict[str, Any],
) -> str:
    """Return a short human-readable description of the comparison experiment."""
    settings = (
        config
        if isinstance(config, DeepLearningComparisonSettings)
        else DeepLearningComparisonSettings.model_validate(config)
    )
    enabled_runs = [run for run in settings.runs if run.enabled]
    return (
        "Deep-learning comparison experiment "
        f"'{settings.experiment_name}' with {len(enabled_runs)} enabled run(s); "
        f"validation metric={settings.ranking.validation_metric}, "
        f"test metric={settings.ranking.test_metric}."
    )


def run_deep_learning_comparison(
    settings: DeepLearningComparisonSettings,
) -> DeepLearningComparisonArtifacts:
    """Run or reuse multiple deep-learning experiments and aggregate outputs."""
    resolved_runs: list[_ResolvedComparisonRun] = []
    for run_spec in settings.runs:
        if not run_spec.enabled:
            continue
        try:
            resolved_runs.append(_ensure_run_result(run_spec, settings))
        except Exception:
            if settings.runner.fail_fast:
                raise
            continue

    _validate_run_compatibility(resolved_runs)
    reference_settings = resolved_runs[0].settings
    output_dir = _comparison_output_dir(settings, reference_settings)
    summary_frame = _build_summary_frame(settings, resolved_runs)
    validation_leaderboard, test_leaderboard, strategy_leaderboard = (
        _build_leaderboard_tables(summary_frame, settings)
    )

    comparison_summary_path = output_dir / "comparison_summary.parquet"
    summary_frame.to_parquet(comparison_summary_path, index=False)
    comparison_summary_csv_path: str | None = None
    if settings.output.write_csv:
        csv_path = output_dir / "comparison_summary.csv"
        summary_frame.to_csv(csv_path, index=False)
        comparison_summary_csv_path = str(csv_path)

    validation_leaderboard_path = output_dir / "validation_leaderboard.parquet"
    validation_leaderboard.to_parquet(validation_leaderboard_path, index=False)
    validation_leaderboard_csv_path: str | None = None
    if settings.output.write_csv:
        csv_path = output_dir / "validation_leaderboard.csv"
        validation_leaderboard.to_csv(csv_path, index=False)
        validation_leaderboard_csv_path = str(csv_path)

    test_leaderboard_path = output_dir / "test_leaderboard.parquet"
    test_leaderboard.to_parquet(test_leaderboard_path, index=False)
    test_leaderboard_csv_path: str | None = None
    if settings.output.write_csv:
        csv_path = output_dir / "test_leaderboard.csv"
        test_leaderboard.to_csv(csv_path, index=False)
        test_leaderboard_csv_path = str(csv_path)

    strategy_leaderboard_path = output_dir / "strategy_leaderboard.parquet"
    strategy_leaderboard.to_parquet(strategy_leaderboard_path, index=False)
    strategy_leaderboard_csv_path: str | None = None
    if settings.output.write_csv:
        csv_path = output_dir / "strategy_leaderboard.csv"
        strategy_leaderboard.to_csv(csv_path, index=False)
        strategy_leaderboard_csv_path = str(csv_path)

    figure_paths: list[str] = []
    if settings.output.write_plots:
        figures_dir = ensure_directory(output_dir / "figures")
        maybe_paths = [
            _write_bar_plot(
                validation_leaderboard,
                label_column="run_label",
                value_column="ranking_metric_value",
                title=f"Validation comparison: {settings.ranking.validation_metric}",
                output_path=figures_dir / "validation_metric_comparison.png",
            ),
            _write_bar_plot(
                test_leaderboard,
                label_column="run_label",
                value_column="ranking_metric_value",
                title=f"Test comparison: {settings.ranking.test_metric}",
                output_path=figures_dir / "test_metric_comparison.png",
            ),
            _write_bar_plot(
                strategy_leaderboard,
                label_column="run_label",
                value_column="ranking_metric_value",
                title=(
                    f"Strategy comparison ({settings.ranking.strategy_split}): "
                    f"{settings.ranking.strategy_metric}"
                ),
                output_path=figures_dir / "strategy_metric_comparison.png",
            ),
        ]
        figure_paths = [path for path in maybe_paths if path is not None]

    report_path = _write_report(
        settings,
        output_dir,
        resolved_runs,
        summary_frame,
        validation_leaderboard,
        test_leaderboard,
        strategy_leaderboard,
        figure_paths,
    )

    manifest_path = output_dir / "comparison_manifest.json"
    manifest_payload = {
        "experiment_name": settings.experiment_name,
        "description": settings.description,
        "run_count": len(resolved_runs),
        "validation_metric": settings.ranking.validation_metric,
        "test_metric": settings.ranking.test_metric,
        "strategy_metric": settings.ranking.strategy_metric,
        "strategy_split": settings.ranking.strategy_split,
        "best_model_note": _best_model_note(summary_frame, settings),
        "comparison_summary_path": str(comparison_summary_path),
        "comparison_summary_csv_path": comparison_summary_csv_path,
        "validation_leaderboard_path": str(validation_leaderboard_path),
        "validation_leaderboard_csv_path": validation_leaderboard_csv_path,
        "test_leaderboard_path": str(test_leaderboard_path),
        "test_leaderboard_csv_path": test_leaderboard_csv_path,
        "strategy_leaderboard_path": str(strategy_leaderboard_path),
        "strategy_leaderboard_csv_path": strategy_leaderboard_csv_path,
        "report_path": report_path,
        "figure_paths": figure_paths,
        "runs": [
            {
                "run_label": run.label,
                "model_name": run.settings.model.name,
                "model_group": _model_group(run.settings.model.name),
                "run_name": run.settings.output.run_name,
                "config_path": run.config_path,
                "resolved_config_path": run.resolved_config_path,
                "artifact_reused": run.artifact_reused,
                "status": run.manifest.get("status", "ok"),
                "reason": run.manifest.get("reason"),
                "degenerate_experiment": bool(run.manifest.get("degenerate_experiment", False)),
                "degenerate_stage": run.manifest.get("degenerate_stage"),
                "degenerate_reason": run.manifest.get("degenerate_reason"),
                "fallback_used": bool(run.manifest.get("fallback_used", False)),
                "fallback_reason": run.manifest.get("fallback_reason"),
                "manifest_path": str(_model_output_dir(run.settings) / "dl_manifest.json"),
                "report_path": run.manifest.get("report_path"),
            }
            for run in resolved_runs
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return DeepLearningComparisonArtifacts(
        output_dir=str(output_dir),
        comparison_summary_path=str(comparison_summary_path),
        comparison_summary_csv_path=comparison_summary_csv_path,
        validation_leaderboard_path=str(validation_leaderboard_path),
        validation_leaderboard_csv_path=validation_leaderboard_csv_path,
        test_leaderboard_path=str(test_leaderboard_path),
        test_leaderboard_csv_path=test_leaderboard_csv_path,
        strategy_leaderboard_path=str(strategy_leaderboard_path),
        strategy_leaderboard_csv_path=strategy_leaderboard_csv_path,
        report_path=report_path,
        manifest_path=str(manifest_path),
        figure_paths=figure_paths,
    )
