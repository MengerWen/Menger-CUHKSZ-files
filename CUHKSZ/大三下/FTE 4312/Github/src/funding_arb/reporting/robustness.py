"""Robustness-analysis experiments for the funding-rate arbitrage prototype."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from funding_arb.backtest.engine import BacktestArtifacts, run_backtest_pipeline
from funding_arb.config.loader import load_settings
from funding_arb.config.models import (
    BacktestSettings,
    BaselineSettings,
    DeepLearningSettings,
    RobustnessFamilySettings,
    RobustnessReportSettings,
    SignalSettings,
)
from funding_arb.models.baselines import run_baseline_pipeline
from funding_arb.models.deep_learning import run_deep_learning_pipeline
from funding_arb.signals.pipeline import run_signal_generation
from funding_arb.utils.paths import ensure_directory, repo_path

LOGGER = logging.getLogger(__name__)

PLOT_COLORS = {
    "rule_based": "#b45309",
    "baseline_ml": "#1d4ed8",
    "deep_learning": "#0f766e",
}


@dataclass(frozen=True)
class RobustnessReportArtifacts:
    """Files produced by the robustness-report command."""

    output_dir: str
    table_paths: list[str]
    figure_paths: list[str]
    summary_json_path: str | None
    markdown_report_path: str | None


def describe_robustness_job(config: RobustnessReportSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the robustness-report job."""
    settings = (
        config
        if isinstance(config, RobustnessReportSettings)
        else RobustnessReportSettings.model_validate(config)
    )
    enabled_families = [family.name for family in settings.families if family.enabled]
    return (
        f"Robustness reporting ready for {settings.input.symbol} on {settings.input.provider} at "
        f"{settings.input.frequency}, comparing families {enabled_families} and writing presentation-ready "
        f"artifacts under {settings.reporting.output_dir}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _load_json(path_text: str | Path | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frame(path_text: str | Path) -> pd.DataFrame:
    path = _resolve_path(path_text)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported tabular artifact format: {path.suffix}")
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "entry_timestamp" in frame.columns:
        frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    return frame


def _write_frame(frame: pd.DataFrame, path: Path, write_csv: bool) -> str:
    if write_csv or path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_json(path, orient="records", indent=2)
    return str(path)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_string(index=False)


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _family_label(family: RobustnessFamilySettings) -> str:
    return family.label or family.name.replace("_", " ").title()


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _strategy_detail_label(row: pd.Series) -> str:
    parts: list[str] = []
    source_subtype = _safe_text(row.get("source_subtype"))
    if source_subtype:
        parts.append(source_subtype)
    prediction_mode = _safe_text(row.get("prediction_mode"))
    if prediction_mode:
        parts.append(prediction_mode)
    calibration_method = _safe_text(row.get("calibration_method"))
    if calibration_method and calibration_method != "none":
        parts.append(f"cal={calibration_method}")
    signal_threshold = pd.to_numeric(pd.Series([row.get("signal_threshold")]), errors="coerce").iloc[0]
    if pd.notna(signal_threshold):
        parts.append(f"thr={float(signal_threshold):.4g}")
    selected_loss = _safe_text(row.get("selected_loss"))
    if selected_loss:
        parts.append(f"loss={selected_loss}")
    effective_metric = _safe_text(row.get("checkpoint_selection_effective_metric"))
    if effective_metric:
        parts.append(f"ckpt={effective_metric}")
    preprocessing_scaler = _safe_text(row.get("preprocessing_scaler"))
    if preprocessing_scaler:
        parts.append(f"scale={preprocessing_scaler}")
    fallback_used = _safe_bool(row.get("checkpoint_selection_fallback_used"))
    if fallback_used:
        parts.append("fallback")
    return " | ".join(parts) if parts else "n/a"


def _enabled_families(
    settings: RobustnessReportSettings,
) -> list[RobustnessFamilySettings]:
    families = [family for family in settings.families if family.enabled]
    if not families:
        raise ValueError(
            "Robustness report requires at least one enabled family in config.families."
        )
    return families


def _load_base_signal_settings(settings: RobustnessReportSettings) -> SignalSettings:
    loaded = load_settings(
        _resolve_path(settings.input.signal_config_path), SignalSettings
    )
    loaded.input.provider = settings.input.provider
    loaded.input.symbol = settings.input.symbol
    loaded.input.venue = settings.input.venue
    loaded.input.frequency = settings.input.frequency
    return loaded


def _load_base_backtest_settings(
    settings: RobustnessReportSettings,
) -> BacktestSettings:
    loaded = load_settings(
        _resolve_path(settings.input.backtest_config_path), BacktestSettings
    )
    loaded.input.provider = settings.input.provider
    loaded.input.symbol = settings.input.symbol
    loaded.input.venue = settings.input.venue
    loaded.input.frequency = settings.input.frequency
    return loaded


def _load_base_baseline_settings(
    settings: RobustnessReportSettings,
) -> BaselineSettings:
    loaded = load_settings(
        _resolve_path(settings.input.baseline_config_path), BaselineSettings
    )
    loaded.input.provider = settings.input.provider
    loaded.input.symbol = settings.input.symbol
    loaded.input.venue = settings.input.venue
    loaded.input.frequency = settings.input.frequency
    return loaded


def _load_base_dl_settings(settings: RobustnessReportSettings) -> DeepLearningSettings:
    loaded = load_settings(
        _resolve_path(settings.input.dl_config_path), DeepLearningSettings
    )
    loaded.input.provider = settings.input.provider
    loaded.input.symbol = settings.input.symbol
    loaded.input.venue = settings.input.venue
    loaded.input.frequency = settings.input.frequency
    return loaded


def _ensure_family_signals(
    settings: RobustnessReportSettings,
    family: RobustnessFamilySettings,
) -> tuple[str, str | None]:
    signal_path = (
        _resolve_path(family.signal_path) if family.signal_path is not None else None
    )
    manifest_path = (
        _resolve_path(family.signal_manifest_path)
        if family.signal_manifest_path is not None
        else None
    )
    if (
        signal_path is not None
        and signal_path.exists()
        and not family.regenerate_signal
    ):
        return str(signal_path), (
            str(manifest_path)
            if manifest_path is not None and manifest_path.exists()
            else None
        )

    LOGGER.info(
        "Generating signals for family '%s' from source '%s'.",
        family.name,
        family.source_name,
    )
    signal_settings = _load_base_signal_settings(settings).model_copy(deep=True)
    signal_settings.source.name = family.source_name
    signal_settings.source.model_names = list(family.strategy_names)
    artifacts = run_signal_generation(signal_settings)
    return artifacts.signals_path, artifacts.manifest_path


def _scenario_run_name(experiment: str, family: str, scenario_name: str) -> str:
    return _sanitize_name(f"{experiment}_{family}_{scenario_name}")


def _annotate_metrics(
    metrics: pd.DataFrame,
    *,
    experiment: str,
    family: RobustnessFamilySettings,
    scenario_name: str,
    scenario_order: int,
    run_name: str,
    scenario_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    annotated = metrics.copy()
    annotated.insert(0, "experiment", experiment)
    annotated.insert(1, "family_name", family.name)
    annotated.insert(2, "family_label", _family_label(family))
    annotated.insert(3, "family_source_name", family.source_name)
    annotated.insert(4, "scenario_name", scenario_name)
    annotated.insert(5, "scenario_order", int(scenario_order))
    annotated.insert(6, "run_name", run_name)
    if "source_subtype" in annotated.columns:
        annotated["strategy_detail_label"] = annotated.apply(
            _strategy_detail_label, axis=1
        )
    if scenario_params:
        for key, value in scenario_params.items():
            annotated[key] = value
    return annotated


def _run_backtest_experiment(
    base_backtest_settings: BacktestSettings,
    family: RobustnessFamilySettings,
    settings: RobustnessReportSettings,
    *,
    experiment: str,
    scenario_name: str,
    scenario_order: int,
    artifacts_root: Path,
    signal_path: str,
    signal_manifest_path: str | None,
    cost_overrides: dict[str, Any] | None = None,
    execution_overrides: dict[str, Any] | None = None,
    selection_overrides: dict[str, Any] | None = None,
    scenario_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, BacktestArtifacts]:
    run_name = _scenario_run_name(experiment, family.name, scenario_name)
    scenario_settings = base_backtest_settings.model_copy(deep=True)
    scenario_settings.input.signal_path = signal_path
    scenario_settings.input.signal_manifest_path = signal_manifest_path
    scenario_settings.selection.strategy_names = list(family.strategy_names)
    scenario_settings.selection.split_filter = list(settings.evaluation.split_filter)
    scenario_settings.reporting.primary_split = settings.evaluation.primary_split
    scenario_settings.reporting.output_dir = str(artifacts_root)
    scenario_settings.reporting.run_name = run_name
    scenario_settings.reporting.write_markdown_report = False
    scenario_settings.reporting.top_n_strategies_for_plots = max(
        settings.evaluation.top_n_strategies, 1
    )

    if cost_overrides:
        for key, value in cost_overrides.items():
            if value is not None:
                setattr(scenario_settings.costs, key, value)
    if execution_overrides:
        for key, value in execution_overrides.items():
            if value is not None:
                setattr(scenario_settings.execution, key, value)
    if selection_overrides:
        for key, value in selection_overrides.items():
            setattr(scenario_settings.selection, key, value)

    artifacts = run_backtest_pipeline(scenario_settings)
    metrics = _load_frame(artifacts.strategy_metrics_path)
    metrics = _annotate_metrics(
        metrics,
        experiment=experiment,
        family=family,
        scenario_name=scenario_name,
        scenario_order=scenario_order,
        run_name=run_name,
        scenario_params=scenario_params,
    )
    return metrics, artifacts


def _best_rows(
    frame: pd.DataFrame, ranking_metric: str, group_columns: list[str]
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if ranking_metric not in frame.columns:
        raise ValueError(
            f"Ranking metric '{ranking_metric}' is not available in robustness results."
        )
    ranking_frame = frame.copy()
    ranking_frame["_has_trades"] = (
        ranking_frame.get("trade_count", 0).fillna(0).astype(float) > 0.0
    )
    ordered = ranking_frame.sort_values(
        group_columns + ["_has_trades", ranking_metric, "trade_count", "strategy_name"],
        ascending=[True] * len(group_columns) + [False, False, False, True],
        na_position="last",
    )
    ordered = ordered.drop(columns="_has_trades")
    return (
        ordered.groupby(group_columns, as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def _resolve_feature_manifest_path(
    settings: RobustnessReportSettings, baseline_settings: BaselineSettings
) -> Path:
    if settings.input.feature_manifest_path is not None:
        return _resolve_path(settings.input.feature_manifest_path)
    supervised_manifest = _load_json(baseline_settings.input.manifest_path)
    if supervised_manifest is None:
        raise FileNotFoundError(
            "Could not infer feature manifest path because the supervised manifest is missing."
        )
    feature_manifest_path = supervised_manifest.get("input", {}).get(
        "feature_manifest_path"
    )
    if not feature_manifest_path:
        raise KeyError(
            "The supervised manifest does not record input.feature_manifest_path."
        )
    return _resolve_path(feature_manifest_path)


def _feature_group_columns(
    settings: RobustnessReportSettings, baseline_settings: BaselineSettings
) -> dict[str, list[str]]:
    feature_manifest = _load_json(
        _resolve_feature_manifest_path(settings, baseline_settings)
    )
    if feature_manifest is None:
        raise FileNotFoundError(
            "Feature manifest is required for feature-ablation experiments."
        )
    group_columns = feature_manifest.get("feature_group_columns")
    if not isinstance(group_columns, dict):
        raise KeyError("Feature manifest does not contain feature_group_columns.")
    return {
        str(name): [str(column) for column in columns]
        for name, columns in group_columns.items()
    }


def _ablation_columns(
    group_catalog: dict[str, list[str]],
    feature_groups: list[str],
    extra_columns: list[str],
) -> list[str]:
    aliases = {
        "interaction": "interaction_state",
        "state": "interaction_state",
        "activity": "liquidity",
    }
    columns: list[str] = []
    for group_name in feature_groups:
        resolved_name = aliases.get(group_name, group_name)
        columns.extend(group_catalog.get(resolved_name, []))
    columns.extend(extra_columns)
    deduped = sorted({column for column in columns if column})
    if not deduped:
        raise ValueError(
            "Feature ablation spec resolved to zero columns. Check feature_groups / exclude_columns."
        )
    return deduped


def _run_feature_ablation(
    settings: RobustnessReportSettings,
    base_backtest_settings: BacktestSettings,
    baseline_settings: BaselineSettings,
    dl_settings: DeepLearningSettings,
    group_catalog: dict[str, list[str]],
    artifacts_root: Path,
) -> pd.DataFrame:
    if not settings.feature_ablation.enabled or not settings.feature_ablation.groups:
        return pd.DataFrame()

    results: list[pd.DataFrame] = []
    signal_root = ensure_directory(
        _resolve_path(settings.feature_ablation.signal_output_dir)
    )
    backtest_root = ensure_directory(
        _resolve_path(settings.feature_ablation.backtest_output_dir)
    )

    for scenario_order, spec in enumerate(settings.feature_ablation.groups, start=1):
        excluded_columns = _ablation_columns(
            group_catalog, spec.feature_groups, spec.exclude_columns
        )
        scenario_name = spec.name
        scenario_params = {
            "ablation_name": scenario_name,
            "excluded_feature_groups": ",".join(spec.feature_groups),
            "excluded_feature_count": len(excluded_columns),
        }

        if spec.include_baseline_ml:
            scenario_baseline_settings = baseline_settings.model_copy(deep=True)
            scenario_baseline_settings.feature_selection.exclude_columns = sorted(
                {
                    *scenario_baseline_settings.feature_selection.exclude_columns,
                    *excluded_columns,
                }
            )
            scenario_baseline_settings.threshold_search.allow_degenerate_fallback = (
                settings.feature_ablation.allow_degenerate_fallback
            )
            scenario_baseline_settings.output.model_dir = str(
                artifacts_root / "models" / "baselines"
            )
            scenario_baseline_settings.output.run_name = f"{settings.feature_ablation.baseline_run_name_prefix}_{_sanitize_name(scenario_name)}"
            baseline_artifacts = run_baseline_pipeline(
                scenario_baseline_settings, train_models=True
            )

            signal_settings = _load_base_signal_settings(settings).model_copy(deep=True)
            signal_settings.source.name = "baseline-ml"
            signal_settings.output.output_dir = str(
                signal_root / _sanitize_name(scenario_name) / "baseline_ml"
            )
            signal_settings.input.baseline_predictions_path = (
                baseline_artifacts.predictions_path
            )
            signal_settings.input.baseline_manifest_path = (
                baseline_artifacts.manifest_path
            )
            signal_artifacts = run_signal_generation(signal_settings)

            family = RobustnessFamilySettings(
                name="baseline_ml",
                label="Simple ML Baseline",
                source_name="baseline-ml",
                signal_path=signal_artifacts.signals_path,
                signal_manifest_path=signal_artifacts.manifest_path,
            )
            metrics, _ = _run_backtest_experiment(
                base_backtest_settings,
                family,
                settings,
                experiment="feature_ablation",
                scenario_name=scenario_name,
                scenario_order=scenario_order,
                artifacts_root=backtest_root,
                signal_path=signal_artifacts.signals_path,
                signal_manifest_path=signal_artifacts.manifest_path,
                scenario_params={**scenario_params, "model_family_type": "baseline_ml"},
            )
            results.append(metrics)

        if spec.include_deep_learning:
            scenario_dl_settings = dl_settings.model_copy(deep=True)
            scenario_dl_settings.feature_selection.exclude_columns = sorted(
                {
                    *scenario_dl_settings.feature_selection.exclude_columns,
                    *excluded_columns,
                }
            )
            scenario_dl_settings.threshold_search.allow_degenerate_fallback = (
                settings.feature_ablation.allow_degenerate_fallback
            )
            scenario_dl_settings.training.allow_degenerate_fallback = (
                settings.feature_ablation.allow_degenerate_fallback
            )
            scenario_dl_settings.output.model_dir = str(
                artifacts_root / "models" / "dl"
            )
            scenario_dl_settings.output.run_name = f"{settings.feature_ablation.dl_run_name_prefix}_{_sanitize_name(scenario_name)}"
            dl_artifacts = run_deep_learning_pipeline(scenario_dl_settings)

            signal_settings = _load_base_signal_settings(settings).model_copy(deep=True)
            signal_settings.source.name = "dl"
            signal_settings.output.output_dir = str(
                signal_root / _sanitize_name(scenario_name) / "dl"
            )
            signal_settings.input.dl_predictions_path = dl_artifacts.predictions_path
            signal_settings.input.dl_manifest_path = dl_artifacts.manifest_path
            signal_artifacts = run_signal_generation(signal_settings)

            family = RobustnessFamilySettings(
                name="deep_learning",
                label="Deep Learning",
                source_name="dl",
                signal_path=signal_artifacts.signals_path,
                signal_manifest_path=signal_artifacts.manifest_path,
            )
            metrics, _ = _run_backtest_experiment(
                base_backtest_settings,
                family,
                settings,
                experiment="feature_ablation",
                scenario_name=scenario_name,
                scenario_order=scenario_order,
                artifacts_root=backtest_root,
                signal_path=signal_artifacts.signals_path,
                signal_manifest_path=signal_artifacts.manifest_path,
                scenario_params={
                    **scenario_params,
                    "model_family_type": "deep_learning",
                },
            )
            results.append(metrics)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _save_table(frame: pd.DataFrame, path: Path, write_csv: bool) -> str:
    return _write_frame(frame, path, write_csv)


def _plot_family_comparison(
    frame: pd.DataFrame, output_path: Path, *, ranking_metric: str, dpi: int
) -> str | None:
    if frame.empty:
        return None
    _apply_plot_style()
    plot_frame = frame.sort_values(ranking_metric, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [
        PLOT_COLORS.get(str(value), "#111827") for value in plot_frame["family_name"]
    ]
    ax.bar(plot_frame["family_label"], plot_frame[ranking_metric], color=colors)
    ax.set_title("Best Test-Period Strategy by Family")
    ax.set_ylabel(ranking_metric.replace("_", " ").title())
    ax.set_xlabel("Strategy family")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_cost_sensitivity(
    frame: pd.DataFrame, output_path: Path, *, ranking_metric: str, dpi: int
) -> str | None:
    if frame.empty:
        return None
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    for family_name, group in frame.groupby("family_name", sort=False):
        ordered = group.sort_values("scenario_order")
        ax.plot(
            ordered["scenario_order"],
            ordered[ranking_metric],
            marker="o",
            linewidth=2.0,
            label=ordered["family_label"].iloc[0],
            color=PLOT_COLORS.get(str(family_name), "#111827"),
        )
    tick_positions = sorted(frame["scenario_order"].unique().tolist())
    tick_labels = (
        frame.drop_duplicates("scenario_order")
        .sort_values("scenario_order")["scenario_name"]
        .tolist()
    )
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_title("Cost Sensitivity")
    ax.set_ylabel(ranking_metric.replace("_", " ").title())
    ax.set_xlabel("Cost scenario")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_holding_sensitivity(
    frame: pd.DataFrame, output_path: Path, *, ranking_metric: str, dpi: int
) -> str | None:
    if frame.empty:
        return None
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    for family_name, group in frame.groupby("family_name", sort=False):
        ordered = group.sort_values("holding_window_hours")
        ax.plot(
            ordered["holding_window_hours"],
            ordered[ranking_metric],
            marker="o",
            linewidth=2.0,
            label=ordered["family_label"].iloc[0],
            color=PLOT_COLORS.get(str(family_name), "#111827"),
        )
    ax.set_title("Holding-Window Sensitivity")
    ax.set_ylabel(ranking_metric.replace("_", " ").title())
    ax.set_xlabel("Holding window (hours)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_threshold_sensitivity(
    frame: pd.DataFrame, output_path: Path, *, ranking_metric: str, dpi: int
) -> str | None:
    if frame.empty:
        return None
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    for strategy_name, group in frame.groupby("strategy_name", sort=False):
        ordered = group.sort_values("min_signal_score")
        ax.plot(
            ordered["min_signal_score"],
            ordered[ranking_metric],
            marker="o",
            linewidth=2.0,
            label=strategy_name,
        )
    ax.set_title("Rule-Threshold Sensitivity")
    ax.set_ylabel(ranking_metric.replace("_", " ").title())
    ax.set_xlabel("Minimum signal score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_feature_ablation(
    frame: pd.DataFrame, output_path: Path, *, ranking_metric: str, dpi: int
) -> str | None:
    if frame.empty:
        return None
    _apply_plot_style()
    pivot = frame.pivot(
        index="scenario_name", columns="family_label", values=ranking_metric
    ).sort_index()
    if pivot.empty:
        return None
    ax = pivot.plot(kind="bar", figsize=(12, 5), color=["#1d4ed8", "#0f766e"])
    ax.set_title("Feature-Ablation Sensitivity")
    ax.set_ylabel(ranking_metric.replace("_", " ").title())
    ax.set_xlabel("Ablation scenario")
    ax.tick_params(axis="x", rotation=20)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _build_summary_json(
    family_comparison_best: pd.DataFrame,
    cost_best: pd.DataFrame,
    holding_best: pd.DataFrame,
    threshold_detail: pd.DataFrame,
    ablation_best: pd.DataFrame,
    ranking_metric: str,
) -> dict[str, Any]:
    metadata_columns = [
        "strategy_name",
        "source_subtype",
        "strategy_detail_label",
        "prediction_mode",
        "calibration_method",
        "signal_threshold",
        "threshold_objective",
        "feature_importance_method",
        "checkpoint_selection_metric",
        "best_checkpoint_metric_value",
        "checkpoint_selection_effective_metric",
        "best_checkpoint_effective_metric_value",
        "checkpoint_selection_fallback_used",
        "selected_loss",
        "regression_loss",
        "use_balanced_classification_loss",
        "preprocessing_scaler",
        "winsorize_lower_quantile",
        "winsorize_upper_quantile",
    ]
    summary: dict[str, Any] = {
        "ranking_metric": ranking_metric,
        "family_comparison": family_comparison_best.to_dict(orient="records"),
    }
    if not cost_best.empty:
        fragility = (
            cost_best.groupby("family_name")[ranking_metric]
            .agg(["max", "min"])
            .assign(metric_range=lambda frame: frame["max"] - frame["min"])
            .reset_index()
            .sort_values("metric_range")
        )
        summary["cost_sensitivity_range"] = fragility.to_dict(orient="records")
    if not holding_best.empty:
        best_holds = _best_rows(holding_best, ranking_metric, ["family_name"])
        summary["best_holding_window_by_family"] = best_holds[
            [
                column
                for column in [
                    "family_name",
                    "family_label",
                    "scenario_name",
                    "holding_window_hours",
                    *metadata_columns,
                    ranking_metric,
                ]
                if column in best_holds.columns
            ]
        ].to_dict(orient="records")
    if not threshold_detail.empty:
        threshold_best = _best_rows(threshold_detail, ranking_metric, ["strategy_name"])
        summary["best_rule_threshold_by_strategy"] = threshold_best[
            [
                column
                for column in [
                    "strategy_name",
                    "scenario_name",
                    "min_signal_score",
                    *metadata_columns,
                    ranking_metric,
                ]
                if column in threshold_best.columns
            ]
        ].to_dict(orient="records")
    if not ablation_best.empty:
        worst_ablation = ablation_best.sort_values(ranking_metric).head(5)
        summary["most_damaging_ablation_cases"] = worst_ablation[
            [
                column
                for column in [
                    "family_name",
                    "family_label",
                    "scenario_name",
                    "strategy_name",
                    *metadata_columns,
                    ranking_metric,
                ]
                if column in worst_ablation.columns
            ]
        ].to_dict(orient="records")
    return summary


def _build_markdown_report(
    settings: RobustnessReportSettings,
    family_comparison_best: pd.DataFrame,
    cost_best: pd.DataFrame,
    holding_best: pd.DataFrame,
    threshold_detail: pd.DataFrame,
    ablation_best: pd.DataFrame,
    figure_paths: list[str],
) -> str:
    ranking_metric = settings.evaluation.ranking_metric
    best_family = (
        family_comparison_best.sort_values(ranking_metric, ascending=False).iloc[0][
            "family_label"
        ]
        if not family_comparison_best.empty
        else "n/a"
    )
    figure_markdown = "\n".join(
        f"![{Path(path).stem}](figures/{Path(path).name})" for path in figure_paths
    )
    return f"""# Robustness Report

## Overview

- Symbol: `{settings.input.symbol}`
- Provider: `{settings.input.provider}`
- Venue: `{settings.input.venue}`
- Frequency: `{settings.input.frequency}`
- Evaluation split(s): `{settings.evaluation.split_filter}`
- Primary backtest split: `{settings.evaluation.primary_split}`
- Ranking metric: `{ranking_metric}`
- Best family under the base test-period configuration: `{best_family}`

## Method Notes

- Cost and holding-window sweeps reuse the same standardized signals and the same backtest engine, so the accounting logic stays identical to the main strategy evaluation.
- Scenario leaderboards use the upgraded backtest contract: primary metrics come from mark-to-market equity on the configured primary split, and no-trade strategies are not allowed to outrank traded strategies merely because their return is zero.
- Rule-threshold sensitivity is implemented by tightening or relaxing `min_signal_score` in the standardized rule-based signal layer. This measures robustness to stronger or weaker entry confidence, not a full redefinition of the raw heuristic itself.
- Feature ablation retrains only predictive baselines and the deep-learning model, regenerates their signals, and reruns the same backtest logic. Rule-based heuristics are excluded from ablation because they do not consume the engineered feature matrix.
- The comparison tables preserve model-family metadata such as `source_subtype`, `prediction_mode`, `calibration_method`, `signal_threshold`, `threshold_objective`, checkpoint-selection fields, and loss/preprocessing settings, so upgraded baseline and deep-learning runs can be compared without collapsing everything into one generic ML bucket.

## Family Comparison

{_dataframe_to_markdown(family_comparison_best.round(6))}

## Cost Sensitivity

{_dataframe_to_markdown(cost_best.round(6))}

## Holding-Window Sensitivity

{_dataframe_to_markdown(holding_best.round(6))}

## Rule Threshold Sensitivity

{_dataframe_to_markdown(threshold_detail.round(6))}

## Feature Ablation

{_dataframe_to_markdown(ablation_best.round(6))}

## Figures

{figure_markdown}
"""


def run_robustness_report(
    settings: RobustnessReportSettings,
) -> RobustnessReportArtifacts:
    """Run reusable robustness experiments across costs, holding rules, thresholds, and feature sets."""
    families = _enabled_families(settings)
    base_backtest_settings = _load_base_backtest_settings(settings)
    baseline_settings = _load_base_baseline_settings(settings)
    dl_settings = _load_base_dl_settings(settings)

    output_root = ensure_directory(
        _resolve_path(settings.reporting.output_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
    )
    tables_dir = ensure_directory(output_root / "tables")
    figures_dir = ensure_directory(output_root / "figures")
    artifacts_root = ensure_directory(
        repo_path(
            "data",
            "artifacts",
            "robustness",
            settings.input.provider,
            settings.input.symbol.lower(),
            settings.input.frequency,
        )
    )

    family_signal_paths: dict[str, tuple[str, str | None]] = {}
    for family in families:
        family_signal_paths[family.name] = _ensure_family_signals(settings, family)

    family_results: list[pd.DataFrame] = []
    for family in families:
        signal_path, signal_manifest_path = family_signal_paths[family.name]
        metrics, _ = _run_backtest_experiment(
            base_backtest_settings,
            family,
            settings,
            experiment="family_comparison",
            scenario_name="base",
            scenario_order=1,
            artifacts_root=artifacts_root / "family_comparison",
            signal_path=signal_path,
            signal_manifest_path=signal_manifest_path,
            scenario_params={"scenario_label": "base"},
        )
        family_results.append(metrics)
    family_comparison_detail = (
        pd.concat(family_results, ignore_index=True)
        if family_results
        else pd.DataFrame()
    )
    family_comparison_best = _best_rows(
        family_comparison_detail,
        settings.evaluation.ranking_metric,
        ["family_name", "scenario_name"],
    )

    cost_results: list[pd.DataFrame] = []
    if settings.cost_sensitivity.enabled:
        for scenario_order, scenario in enumerate(
            settings.cost_sensitivity.scenarios, start=1
        ):
            for family in families:
                signal_path, signal_manifest_path = family_signal_paths[family.name]
                metrics, _ = _run_backtest_experiment(
                    base_backtest_settings,
                    family,
                    settings,
                    experiment="cost_sensitivity",
                    scenario_name=scenario.name,
                    scenario_order=scenario_order,
                    artifacts_root=artifacts_root / "cost_sensitivity",
                    signal_path=signal_path,
                    signal_manifest_path=signal_manifest_path,
                    cost_overrides=scenario.model_dump(exclude={"name"}),
                    scenario_params=scenario.model_dump(),
                )
                cost_results.append(metrics)
    cost_detail = (
        pd.concat(cost_results, ignore_index=True) if cost_results else pd.DataFrame()
    )
    cost_best = (
        _best_rows(
            cost_detail,
            settings.evaluation.ranking_metric,
            ["family_name", "scenario_name"],
        )
        if not cost_detail.empty
        else pd.DataFrame()
    )

    holding_results: list[pd.DataFrame] = []
    if settings.holding_sensitivity.enabled:
        for scenario_order, scenario in enumerate(
            settings.holding_sensitivity.scenarios, start=1
        ):
            max_holding = (
                scenario.maximum_holding_hours
                if scenario.maximum_holding_hours is not None
                else scenario.holding_window_hours
            )
            execution_overrides = {
                "holding_window_hours": scenario.holding_window_hours,
                "maximum_holding_hours": max_holding,
            }
            for family in families:
                signal_path, signal_manifest_path = family_signal_paths[family.name]
                metrics, _ = _run_backtest_experiment(
                    base_backtest_settings,
                    family,
                    settings,
                    experiment="holding_window_sensitivity",
                    scenario_name=scenario.name,
                    scenario_order=scenario_order,
                    artifacts_root=artifacts_root / "holding_window_sensitivity",
                    signal_path=signal_path,
                    signal_manifest_path=signal_manifest_path,
                    execution_overrides=execution_overrides,
                    scenario_params={
                        "name": scenario.name,
                        "holding_window_hours": scenario.holding_window_hours,
                        "maximum_holding_hours": max_holding,
                    },
                )
                holding_results.append(metrics)
    holding_detail = (
        pd.concat(holding_results, ignore_index=True)
        if holding_results
        else pd.DataFrame()
    )
    holding_best = (
        _best_rows(
            holding_detail,
            settings.evaluation.ranking_metric,
            ["family_name", "scenario_name"],
        )
        if not holding_detail.empty
        else pd.DataFrame()
    )

    threshold_detail = pd.DataFrame()
    if (
        settings.threshold_sensitivity.enabled
        and settings.threshold_sensitivity.scenarios
    ):
        threshold_family = next(
            (
                family
                for family in families
                if family.name == settings.threshold_sensitivity.family_name
            ),
            None,
        )
        if threshold_family is None:
            raise ValueError(
                f"Threshold sensitivity requested family '{settings.threshold_sensitivity.family_name}', "
                "but that family is not enabled in config.families."
            )
        signal_path, signal_manifest_path = family_signal_paths[threshold_family.name]
        threshold_results: list[pd.DataFrame] = []
        for scenario_order, scenario in enumerate(
            settings.threshold_sensitivity.scenarios, start=1
        ):
            metrics, _ = _run_backtest_experiment(
                base_backtest_settings,
                threshold_family,
                settings,
                experiment="rule_threshold_sensitivity",
                scenario_name=scenario.name,
                scenario_order=scenario_order,
                artifacts_root=artifacts_root / "rule_threshold_sensitivity",
                signal_path=signal_path,
                signal_manifest_path=signal_manifest_path,
                selection_overrides={
                    "min_signal_score": scenario.min_signal_score,
                    "min_confidence": scenario.min_confidence,
                    "min_expected_return_bps": scenario.min_expected_return_bps,
                },
                scenario_params=scenario.model_dump(),
            )
            threshold_results.append(metrics)
        threshold_detail = (
            pd.concat(threshold_results, ignore_index=True)
            if threshold_results
            else pd.DataFrame()
        )

    ablation_detail = _run_feature_ablation(
        settings,
        base_backtest_settings,
        baseline_settings,
        dl_settings,
        _feature_group_columns(settings, baseline_settings),
        artifacts_root / "feature_ablation",
    )
    ablation_best = (
        _best_rows(
            ablation_detail,
            settings.evaluation.ranking_metric,
            ["family_name", "scenario_name"],
        )
        if not ablation_detail.empty
        else pd.DataFrame()
    )

    table_paths = [
        _save_table(
            family_comparison_detail,
            tables_dir / "family_comparison_detail.csv",
            settings.reporting.write_csv,
        ),
        _save_table(
            family_comparison_best,
            tables_dir / "family_comparison_best.csv",
            settings.reporting.write_csv,
        ),
        _save_table(
            cost_detail,
            tables_dir / "cost_sensitivity_detail.csv",
            settings.reporting.write_csv,
        ),
        _save_table(
            cost_best,
            tables_dir / "cost_sensitivity_best.csv",
            settings.reporting.write_csv,
        ),
        _save_table(
            holding_detail,
            tables_dir / "holding_window_sensitivity_detail.csv",
            settings.reporting.write_csv,
        ),
        _save_table(
            holding_best,
            tables_dir / "holding_window_sensitivity_best.csv",
            settings.reporting.write_csv,
        ),
        _save_table(
            threshold_detail,
            tables_dir / "rule_threshold_sensitivity_detail.csv",
            settings.reporting.write_csv,
        ),
        _save_table(
            ablation_detail,
            tables_dir / "feature_ablation_detail.csv",
            settings.reporting.write_csv,
        ),
        _save_table(
            ablation_best,
            tables_dir / "feature_ablation_best.csv",
            settings.reporting.write_csv,
        ),
    ]

    figure_paths: list[str] = []
    ranking_metric = settings.evaluation.ranking_metric
    for figure_path in [
        _plot_family_comparison(
            family_comparison_best,
            figures_dir / f"family_comparison.{settings.reporting.figure_format}",
            ranking_metric=ranking_metric,
            dpi=settings.reporting.dpi,
        ),
        _plot_cost_sensitivity(
            cost_best,
            figures_dir / f"cost_sensitivity.{settings.reporting.figure_format}",
            ranking_metric=ranking_metric,
            dpi=settings.reporting.dpi,
        ),
        _plot_holding_sensitivity(
            holding_best,
            figures_dir
            / f"holding_window_sensitivity.{settings.reporting.figure_format}",
            ranking_metric=ranking_metric,
            dpi=settings.reporting.dpi,
        ),
        _plot_threshold_sensitivity(
            threshold_detail,
            figures_dir
            / f"rule_threshold_sensitivity.{settings.reporting.figure_format}",
            ranking_metric=ranking_metric,
            dpi=settings.reporting.dpi,
        ),
        _plot_feature_ablation(
            ablation_best,
            figures_dir / f"feature_ablation.{settings.reporting.figure_format}",
            ranking_metric=ranking_metric,
            dpi=settings.reporting.dpi,
        ),
    ]:
        if figure_path is not None:
            figure_paths.append(figure_path)

    summary_json_path: str | None = None
    if settings.reporting.write_json_summary:
        summary = _build_summary_json(
            family_comparison_best,
            cost_best,
            holding_best,
            threshold_detail,
            ablation_best,
            settings.evaluation.ranking_metric,
        )
        summary_path = output_root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        summary_json_path = str(summary_path)

    markdown_report_path: str | None = None
    if settings.reporting.write_markdown:
        report_path = output_root / "report.md"
        report_path.write_text(
            _build_markdown_report(
                settings,
                family_comparison_best,
                cost_best,
                holding_best,
                threshold_detail,
                ablation_best,
                figure_paths,
            ),
            encoding="utf-8",
        )
        markdown_report_path = str(report_path)

    return RobustnessReportArtifacts(
        output_dir=str(output_root),
        table_paths=table_paths,
        figure_paths=figure_paths,
        summary_json_path=summary_json_path,
        markdown_report_path=markdown_report_path,
    )
