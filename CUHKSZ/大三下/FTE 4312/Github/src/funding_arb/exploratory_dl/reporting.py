"""Reporting pipeline for the exploratory deep-learning showcase track."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funding_arb.config.models import ExploratoryDLReportSettings
from funding_arb.exploratory_dl.signals import _core_prediction_frame, _run_frame
from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class ExploratoryDLReportArtifacts:
    """Files produced by exploratory showcase reporting."""

    output_dir: str
    markdown_report_path: str | None
    summary_json_path: str | None
    full_leaderboard_path: str
    showcase_leaderboard_path: str
    frontend_public_dir: str | None
    figure_paths: list[str]


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_path(*path.parts)


def _load_json(path_text: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_path(path_text).read_text(encoding="utf-8"))


def _load_optional_json(path_text: str | Path | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_table(path_text: str | Path) -> pd.DataFrame:
    path = _resolve_path(path_text)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _load_optional_table(path_text: str | Path | None) -> pd.DataFrame | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    return _load_table(path)


def _load_table_collection(paths: list[str]) -> pd.DataFrame | None:
    frames = [frame for frame in (_load_optional_table(path) for path in paths) if frame is not None and not frame.empty]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


def _output_dir(settings: ExploratoryDLReportSettings) -> Path:
    snapshot = _load_json(settings.input.strict_demo_snapshot_path)
    meta = snapshot["meta"]
    return ensure_directory(
        _resolve_path(settings.output.output_dir)
        / str(meta["venue"])
        / str(meta["symbol"]).lower()
        / str(meta["frequency"])
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(_json_ready(frame.to_dict(orient="records")), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return str(path)


def describe_exploratory_report_job(
    config: ExploratoryDLReportSettings | dict[str, Any]
) -> str:
    settings = (
        config
        if isinstance(config, ExploratoryDLReportSettings)
        else ExploratoryDLReportSettings.model_validate(config)
    )
    return (
        "Exploratory DL showcase reporting ready, combining strict snapshot diagnostics with "
        f"exploratory artifacts and writing outputs under {settings.output.output_dir}."
    )


def _merge_leaderboard(
    backtest_leaderboard: pd.DataFrame,
    strategy_catalog: pd.DataFrame,
) -> pd.DataFrame:
    catalog = strategy_catalog[
        [
            "strategy_name",
            "model_name",
            "run_name",
            "target_type",
            "task",
            "signal_rule",
            "signal_rule_type",
            "selection_reason",
            "status",
            "reason",
        ]
    ].rename(
        columns={
            "task": "signal_task",
            "status": "signal_status",
            "reason": "signal_reason",
        }
    )
    merged = backtest_leaderboard.merge(catalog, on="strategy_name", how="left")
    if "task" not in merged.columns and "signal_task" in merged.columns:
        merged["task"] = merged["signal_task"]
    merged["reason"] = merged["diagnostic_reason"].where(
        merged["diagnostic_reason"].notna(), merged["signal_reason"]
    )
    ordered_columns = [
        "strategy_name",
        "model_name",
        "run_name",
        "target_type",
        "signal_rule",
        "signal_rule_type",
        "source",
        "source_subtype",
        "task",
        "evaluation_split",
        "status",
        "reason",
        "trade_count",
        "cumulative_return",
        "mark_to_market_max_drawdown",
        "sharpe_ratio",
        "total_net_pnl_usd",
        "selection_reason",
        "signal_status",
        "signal_reason",
    ]
    tail_columns = [column for column in merged.columns if column not in ordered_columns]
    return merged[ordered_columns + tail_columns].copy()


def _showcase_leaderboard(full_leaderboard: pd.DataFrame) -> pd.DataFrame:
    if full_leaderboard.empty:
        return full_leaderboard.copy()
    ranked = full_leaderboard.copy()
    ranked["_has_trades"] = ranked["trade_count"].fillna(0).astype(float) > 0.0
    ranked = ranked.sort_values(
        ["_has_trades", "cumulative_return", "sharpe_ratio"],
        ascending=[False, False, False],
        kind="stable",
        na_position="last",
    )
    showcase = ranked.loc[
        ranked["_has_trades"] | ranked["status"].eq("completed")
    ].drop(columns="_has_trades")
    if showcase.empty:
        showcase = ranked.drop(columns="_has_trades")
    return showcase.reset_index(drop=True)


def _prediction_diagnostics(
    signal_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    distribution_rows: list[dict[str, Any]] = []
    quantile_tables: dict[str, pd.DataFrame] = {}
    calibration_tables: dict[str, pd.DataFrame] = {}
    for run_spec in signal_manifest["input"]["runs"]:
        if not run_spec.get("enabled", True):
            continue
        run_name = str(run_spec["name"])
        frame, _manifest = _run_frame(
            type("RunSpec", (), run_spec)()  # type: ignore[misc]
        )
        core = _core_prediction_frame(frame, type("RunSpec", (), run_spec)())  # type: ignore[misc]
        for split_name, split_frame in core.groupby("split", sort=True):
            distribution_rows.append(
                {
                    "run_name": run_name,
                    "target_type": run_spec.get("target_type"),
                    "task": split_frame["task"].iloc[0],
                    "split": split_name,
                    "row_count": int(len(split_frame)),
                    "signed_score_mean": _json_ready(split_frame["signed_score"].mean()),
                    "signed_score_std": _json_ready(split_frame["signed_score"].std(ddof=0)),
                    "signed_score_min": _json_ready(split_frame["signed_score"].min()),
                    "signed_score_max": _json_ready(split_frame["signed_score"].max()),
                    "absolute_score_90pct": _json_ready(split_frame["absolute_score"].quantile(0.9)),
                    "actual_directional_mean_bps": _json_ready(split_frame["actual_directional_return_bps"].mean()),
                    "predicted_short_rate": _json_ready(
                        (split_frame["predicted_direction"] == "short_perp_long_spot").mean()
                    ),
                }
            )

        test_frame = core.loc[core["split"] == "test"].copy()
        test_frame = test_frame.dropna(subset=["absolute_score", "actual_directional_return_bps"])
        if not test_frame.empty:
            bucket_count = min(10, max(2, int(test_frame["absolute_score"].nunique())))
            test_frame["absolute_score_quantile"] = pd.qcut(
                test_frame["absolute_score"].rank(method="first"),
                q=bucket_count,
                labels=False,
                duplicates="drop",
            )
            quantile_table = (
                test_frame.groupby("absolute_score_quantile", dropna=False)
                .agg(
                    row_count=("timestamp", "size"),
                    avg_absolute_score=("absolute_score", "mean"),
                    avg_directional_return_bps=("actual_directional_return_bps", "mean"),
                    cumulative_directional_return_bps=("actual_directional_return_bps", "sum"),
                    short_signal_rate=("predicted_class", "mean"),
                )
                .reset_index()
            )
            quantile_table.insert(0, "run_name", run_name)
            quantile_tables[run_name] = quantile_table

            calibration_frame = test_frame.copy()
            calibration_frame["predicted_bucket"] = pd.qcut(
                calibration_frame["signed_score"].rank(method="first"),
                q=min(10, max(2, int(calibration_frame["signed_score"].nunique()))),
                labels=False,
                duplicates="drop",
            )
            calibration_table = (
                calibration_frame.groupby("predicted_bucket", dropna=False)
                .agg(
                    row_count=("timestamp", "size"),
                    avg_predicted_signed_score=("signed_score", "mean"),
                    avg_actual_signed_return_bps=("actual_return_bps", "mean"),
                    avg_actual_directional_return_bps=("actual_directional_return_bps", "mean"),
                )
                .reset_index()
            )
            calibration_table.insert(0, "run_name", run_name)
            calibration_tables[run_name] = calibration_table

    return pd.DataFrame(distribution_rows), quantile_tables, calibration_tables


def _top_quantile_summary(quantile_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, table in quantile_tables.items():
        if table.empty:
            continue
        top = table.sort_values("absolute_score_quantile", ascending=False).head(1).iloc[0]
        rows.append(
            {
                "run_name": run_name,
                "top_bucket_row_count": int(top["row_count"]),
                "top_bucket_avg_directional_return_bps": _json_ready(top["avg_directional_return_bps"]),
                "top_bucket_cumulative_directional_return_bps": _json_ready(top["cumulative_directional_return_bps"]),
            }
        )
    return pd.DataFrame(rows)


def _direction_summaries(signals: pd.DataFrame, trade_log: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_summary = (
        signals.loc[signals["should_trade"] == 1]
        .groupby(["strategy_name", "split", "suggested_direction"], dropna=False)
        .size()
        .reset_index(name="signal_count")
        .sort_values(["strategy_name", "split", "suggested_direction"])
        .reset_index(drop=True)
    )
    if trade_log is None or trade_log.empty:
        return signal_summary, pd.DataFrame(
            columns=["direction", "trade_count", "total_net_pnl_usd", "average_net_return_bps"]
        )
    trade_summary = (
        trade_log.groupby("direction", dropna=False)
        .agg(
            trade_count=("strategy_name", "size"),
            total_net_pnl_usd=("net_pnl_usd", "sum"),
            average_net_return_bps=("net_return_bps", "mean"),
        )
        .reset_index()
    )
    return signal_summary, trade_summary


def _strict_model_summary(strict_snapshot: dict[str, Any]) -> dict[str, Any]:
    baseline = strict_snapshot["models"]["baseline_best"]
    deep_learning = strict_snapshot["models"]["deep_learning_best"]
    backtest = strict_snapshot["backtest"]
    return {
        "best_baseline_model": baseline.get("model_name"),
        "best_baseline_metric": baseline.get("pearson_corr"),
        "best_baseline_signal_count": baseline.get("signal_count"),
        "best_deep_learning_model": deep_learning.get("model_name"),
        "best_deep_learning_metric": deep_learning.get(
            "ranking_metric_value", deep_learning.get("pearson_corr")
        ),
        "best_deep_learning_signal_count": deep_learning.get(
            "test_signal_count", deep_learning.get("signal_count")
        ),
        "strict_backtest_best_strategy": backtest["best_strategy"].get("strategy_name"),
        "strict_backtest_verdict": backtest["summary"].get(
            "best_strategy_status", backtest["best_strategy"].get("status")
        ),
        "strict_backtest_trade_count": backtest["best_strategy"].get("trade_count"),
    }


def _comparison_table(
    strict_comparison: pd.DataFrame | None,
    exploratory_comparison: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for track, frame in [
        ("strict", strict_comparison),
        ("exploratory", exploratory_comparison),
    ]:
        if frame is None or frame.empty:
            continue
        if "split" in frame.columns:
            frame = frame.loc[frame["split"] == "test"].copy()
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "track": track,
                        "model_name": row.get("model_name"),
                        "task": row.get("task"),
                        "target_column": row.get("target_column"),
                        "status": row.get("status"),
                        "test_pearson_corr": row.get("pearson_corr"),
                        "test_rmse": row.get("rmse"),
                        "test_signal_count": row.get("signal_count"),
                        "test_f1": row.get("f1"),
                    }
                )
            continue
        for _, row in frame.iterrows():
            rows.append(
                {
                    "track": track,
                    "model_name": row.get("model_name", row.get("run_label")),
                    "task": row.get("task"),
                    "target_column": row.get("target_column"),
                    "status": row.get("status"),
                    "test_pearson_corr": row.get("test_pearson_corr", row.get("pearson_corr")),
                    "test_rmse": row.get("test_rmse", row.get("rmse")),
                    "test_signal_count": row.get("test_signal_count", row.get("signal_count")),
                    "test_f1": row.get("test_f1", row.get("f1")),
                }
            )
    return pd.DataFrame(rows)


def _plot_prediction_distribution(
    run_name: str,
    run_table: pd.DataFrame,
    output_path: Path,
) -> str | None:
    if run_table.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        run_table["signed_score"].dropna(),
        bins=40,
        color="#0f766e",
        alpha=0.8,
    )
    ax.set_title(f"Prediction Distribution: {run_name}")
    ax.set_xlabel("Signed prediction score")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_quantile_analysis(run_name: str, table: pd.DataFrame, output_path: Path) -> str | None:
    if table.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        table["absolute_score_quantile"].astype(str),
        table["avg_directional_return_bps"],
        color="#1d4ed8",
    )
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.set_title(f"Quantile Return Analysis: {run_name}")
    ax.set_xlabel("Absolute-score quantile bucket")
    ax.set_ylabel("Avg directional return (bps)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_actual_vs_predicted(
    run_name: str, table: pd.DataFrame, output_path: Path
) -> str | None:
    if table.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        table["predicted_bucket"].astype(str),
        table["avg_predicted_signed_score"],
        marker="o",
        label="Avg predicted score",
        color="#0f766e",
    )
    ax.plot(
        table["predicted_bucket"].astype(str),
        table["avg_actual_signed_return_bps"],
        marker="o",
        label="Avg actual signed return",
        color="#b45309",
    )
    ax.axhline(0.0, color="#111827", linewidth=1)
    ax.set_title(f"Actual vs Predicted Buckets: {run_name}")
    ax.set_xlabel("Predicted-score bucket")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _table_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_string(index=False)


def run_exploratory_dl_report(
    settings: ExploratoryDLReportSettings,
) -> ExploratoryDLReportArtifacts:
    """Create a standalone showcase report and frontend-ready exploratory JSON files."""
    output_dir = _output_dir(settings)
    strict_snapshot = _load_json(settings.input.strict_demo_snapshot_path)
    signal_manifest = _load_json(settings.input.exploratory_signals_manifest_path)
    strategy_catalog = _load_table(signal_manifest["strategy_catalog_path"])
    signals = _load_table(settings.input.exploratory_signals_path)
    backtest_manifest = _load_json(settings.input.exploratory_backtest_manifest_path)
    backtest_leaderboard = _load_table(settings.input.exploratory_backtest_leaderboard_path)
    trade_log = _load_optional_table(settings.input.exploratory_trade_log_path)
    strict_final_summary = _load_optional_json(settings.input.strict_final_report_summary_path)
    strict_comparison = _load_optional_table(settings.input.strict_comparison_summary_path)
    exploratory_comparison = _load_table_collection(
        [
            settings.input.exploratory_comparison_summary_path,
            *settings.input.exploratory_extra_comparison_summary_paths,
        ]
    )

    full_leaderboard = _merge_leaderboard(backtest_leaderboard, strategy_catalog)
    showcase_leaderboard = _showcase_leaderboard(full_leaderboard)
    distribution_table, quantile_tables, calibration_tables = _prediction_diagnostics(signal_manifest)
    top_quantile_summary = _top_quantile_summary(quantile_tables)
    signal_direction_summary, trade_direction_summary = _direction_summaries(signals, trade_log)
    comparison_table = _comparison_table(strict_comparison, exploratory_comparison)

    figures_dir = ensure_directory(output_dir / "figures")
    best_showcase_row = showcase_leaderboard.iloc[0].to_dict() if not showcase_leaderboard.empty else {}
    best_run_name = str(best_showcase_row.get("run_name", "")) if best_showcase_row else ""
    figure_paths: list[str] = []
    if best_run_name:
        run_spec = next(
            (run for run in signal_manifest["input"]["runs"] if run.get("name") == best_run_name),
            None,
        )
        if run_spec is not None:
            run_frame, _ = _run_frame(type("RunSpec", (), run_spec)())  # type: ignore[misc]
            run_core = _core_prediction_frame(run_frame, type("RunSpec", (), run_spec)())  # type: ignore[misc]
            run_core = run_core.loc[run_core["split"] == "test"].copy()
            for candidate in [
                _plot_prediction_distribution(
                    best_run_name,
                    run_core,
                    figures_dir / f"{best_run_name}_prediction_distribution.png",
                ),
                _plot_quantile_analysis(
                    best_run_name,
                    quantile_tables.get(best_run_name, pd.DataFrame()),
                    figures_dir / f"{best_run_name}_quantile_analysis.png",
                ),
                _plot_actual_vs_predicted(
                    best_run_name,
                    calibration_tables.get(best_run_name, pd.DataFrame()),
                    figures_dir / f"{best_run_name}_actual_vs_predicted.png",
                ),
            ]:
                if candidate is not None:
                    figure_paths.append(candidate)

    full_leaderboard_path = _write_frame(full_leaderboard, output_dir / "exploratory_full_leaderboard.csv")
    _write_frame(full_leaderboard, output_dir / "exploratory_full_leaderboard.json")
    showcase_leaderboard_path = _write_frame(
        showcase_leaderboard, output_dir / "exploratory_showcase_leaderboard.csv"
    )
    showcase_leaderboard_json_path = _write_frame(
        showcase_leaderboard, output_dir / "exploratory_showcase_leaderboard.json"
    )
    distribution_json_path = _write_frame(
        distribution_table, output_dir / "exploratory_prediction_distribution.json"
    )
    quantile_summary_frame = (
        pd.concat(quantile_tables.values(), ignore_index=True)
        if quantile_tables
        else pd.DataFrame()
    )
    quantile_json_path = _write_frame(
        quantile_summary_frame, output_dir / "exploratory_quantile_analysis.json"
    )
    _write_frame(top_quantile_summary, output_dir / "exploratory_top_quantile_summary.json")
    _write_frame(signal_direction_summary, output_dir / "exploratory_signal_direction_summary.csv")
    _write_frame(trade_direction_summary, output_dir / "exploratory_trade_direction_summary.csv")
    _write_frame(comparison_table, output_dir / "exploratory_model_target_comparison.csv")

    strict_summary = _strict_model_summary(strict_snapshot)
    public_figure_assets = [
        {
            "label": Path(path).stem,
            "image": f"demo/assets/{Path(path).name}",
            "file_name": Path(path).name,
        }
        for path in figure_paths
    ]

    summary_payload = {
        "strict_summary": strict_summary,
        "strict_final_summary": strict_final_summary,
        "exploratory_summary": {
            "strategy_count": int(full_leaderboard["strategy_name"].nunique()) if not full_leaderboard.empty else 0,
            "nonzero_trade_strategy_count": int((full_leaderboard["trade_count"].fillna(0) > 0).sum()) if not full_leaderboard.empty else 0,
            "best_showcase_row": _json_ready(best_showcase_row),
            "full_leaderboard_path": full_leaderboard_path,
            "showcase_leaderboard_path": showcase_leaderboard_path,
            "prediction_distribution_path": distribution_json_path,
            "quantile_analysis_path": quantile_json_path,
            "figure_assets": public_figure_assets,
        },
        "disclaimer": (
            "Exploratory DL results are supplementary showcase results designed to demonstrate model learning behavior, "
            "ranking ability, and alternative opportunity definitions. They do not replace the strict post-cost primary conclusion."
        ),
        "figure_paths": figure_paths,
    }

    markdown_lines = [
        "# Exploratory DL Showcase Report",
        "",
        "Exploratory DL results are supplementary showcase results designed to demonstrate model learning behavior, ranking ability, and alternative opportunity definitions. They do not replace the strict post-cost primary conclusion.",
        "",
        "## Strict Context",
        "",
        f"- Strict best baseline: `{strict_summary['best_baseline_model']}` with metric `{strict_summary['best_baseline_metric']}` and signal count `{strict_summary['best_baseline_signal_count']}`",
        f"- Strict best deep learning: `{strict_summary['best_deep_learning_model']}` with metric `{strict_summary['best_deep_learning_metric']}` and signal count `{strict_summary['best_deep_learning_signal_count']}`",
        f"- Strict backtest verdict: `{strict_summary['strict_backtest_verdict']}` from `{strict_summary['strict_backtest_best_strategy']}`",
        "",
        "## What Changed In The Exploratory Track",
        "",
        "- Exploratory models use independent showcase targets such as gross opportunity and direction-aware opportunity labels.",
        "- Signal generation uses ranking-based and support-aware threshold rules instead of only `predicted_value >= 0`.",
        "- A dedicated exploratory backtest evaluates only exploratory DL signals and keeps strict outputs untouched.",
        "",
        "## Showcase Leaderboard",
        "",
        _table_to_markdown(
            showcase_leaderboard[
                [
                    "strategy_name",
                    "model_name",
                    "target_type",
                    "signal_rule",
                    "evaluation_split",
                    "trade_count",
                    "cumulative_return",
                    "mark_to_market_max_drawdown",
                    "sharpe_ratio",
                    "total_net_pnl_usd",
                    "status",
                    "reason",
                ]
            ].head(12)
        ),
        "",
        "## Full Exploratory Leaderboard",
        "",
        _table_to_markdown(
            full_leaderboard[
                [
                    "strategy_name",
                    "model_name",
                    "target_type",
                    "signal_rule",
                    "evaluation_split",
                    "trade_count",
                    "cumulative_return",
                    "mark_to_market_max_drawdown",
                    "sharpe_ratio",
                    "total_net_pnl_usd",
                    "status",
                    "reason",
                ]
            ]
        ),
        "",
        "## Model And Target Comparison",
        "",
        _table_to_markdown(comparison_table),
        "",
        "## Direction Summary",
        "",
        _table_to_markdown(trade_direction_summary),
        "",
        "## Top-Quantile Diagnostic Summary",
        "",
        _table_to_markdown(top_quantile_summary),
        "",
        "## Files",
        "",
        f"- Full leaderboard: `{full_leaderboard_path}`",
        f"- Showcase leaderboard: `{showcase_leaderboard_json_path}`",
        f"- Prediction distribution: `{distribution_json_path}`",
        f"- Quantile analysis: `{quantile_json_path}`",
    ]

    markdown_report_path: str | None = None
    if settings.output.write_markdown:
        markdown_path = output_dir / "exploratory_dl_report.md"
        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        markdown_report_path = str(markdown_path)

    summary_json_path: str | None = None
    if settings.output.write_json_summary:
        summary_path = output_dir / "exploratory_dl_summary.json"
        summary_path.write_text(
            json.dumps(_json_ready(summary_payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary_json_path = str(summary_path)

    frontend_public_dir: str | None = None
    if settings.output.copy_to_frontend_public:
        public_dir = ensure_directory(_resolve_path(settings.output.frontend_public_dir))
        frontend_public_dir = str(public_dir)
        copy_specs = {
            "exploratory_dl_summary.json": summary_json_path,
            "exploratory_dl_leaderboard.json": showcase_leaderboard_json_path,
            "exploratory_prediction_distribution.json": distribution_json_path,
            "exploratory_quantile_analysis.json": quantile_json_path,
        }
        for target_name, source_path in copy_specs.items():
            if source_path is None:
                continue
            shutil.copy2(_resolve_path(source_path), public_dir / target_name)
        public_assets_dir = ensure_directory(public_dir / "assets")
        for figure_path in figure_paths:
            figure = _resolve_path(figure_path)
            shutil.copy2(figure, public_assets_dir / figure.name)

    return ExploratoryDLReportArtifacts(
        output_dir=str(output_dir),
        markdown_report_path=markdown_report_path,
        summary_json_path=summary_json_path,
        full_leaderboard_path=full_leaderboard_path,
        showcase_leaderboard_path=showcase_leaderboard_path,
        frontend_public_dir=frontend_public_dir,
        figure_paths=figure_paths,
    )
