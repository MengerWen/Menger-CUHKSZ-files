"""Exploratory data analysis and data-quality reporting for the canonical market dataset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from funding_arb.config.models import DataQualityReportSettings
from funding_arb.utils.paths import ensure_directory, repo_path

PLOT_COLORS = {
    "perp": "#0f766e",
    "spot": "#1d4ed8",
    "funding": "#b45309",
    "spread": "#b91c1c",
    "signal": "#111827",
}


@dataclass(frozen=True)
class DataQualityReportArtifacts:
    """Files produced by the data-quality reporting command."""

    output_dir: str
    table_paths: list[str]
    figure_paths: list[str]
    summary_json_path: str | None
    markdown_report_path: str | None


def describe_data_quality_job(config: DataQualityReportSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the data-quality report job."""
    settings = config if isinstance(config, DataQualityReportSettings) else DataQualityReportSettings.model_validate(config)
    return (
        f"Data-quality reporting ready for {settings.input.symbol} on {settings.input.provider} at "
        f"{settings.input.frequency}, reading {settings.input.dataset_path} and writing artifacts under "
        f"{settings.output.output_dir}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| Note |\n| --- |\n| No rows available |"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in frame.astype(object).where(pd.notna(frame), "").to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    return "\n".join([header, divider, *rows])


def load_market_dataset(path: str | Path) -> pd.DataFrame:
    """Load the canonical hourly market dataset from parquet or CSV."""
    dataset_path = _resolve_path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    elif suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def prepare_analysis_frame(frame: pd.DataFrame, *, volatility_window_hours: int, annualization_factor_hours: int) -> pd.DataFrame:
    """Add derived columns used by the report summaries and plots."""
    analysis = frame.copy()
    analysis["perp_return"] = analysis["perp_close"].pct_change()
    analysis["spot_return"] = analysis["spot_close"].pct_change()
    analysis["spread"] = analysis["perp_close"] - analysis["spot_close"]
    analysis["spread_bps"] = ((analysis["perp_close"] / analysis["spot_close"]) - 1.0) * 10_000.0
    analysis["funding_rate_bps"] = analysis["funding_rate"] * 10_000.0
    scale = math.sqrt(float(annualization_factor_hours))
    analysis["perp_rolling_vol_annualized"] = analysis["perp_return"].rolling(volatility_window_hours, min_periods=2).std() * scale
    analysis["spot_rolling_vol_annualized"] = analysis["spot_return"].rolling(volatility_window_hours, min_periods=2).std() * scale
    return analysis


def compute_missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute missing-value counts and percentages by column."""
    total_rows = len(frame)
    rows = []
    for column in frame.columns:
        missing_count = int(frame[column].isna().sum())
        missing_pct = (missing_count / total_rows * 100.0) if total_rows else 0.0
        rows.append({"column": column, "missing_count": missing_count, "missing_pct": round(missing_pct, 4)})
    return pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def compute_time_coverage_summary(frame: pd.DataFrame, *, frequency: str) -> pd.DataFrame:
    """Summarize timestamp coverage and gap structure for the dataset."""
    timestamps = pd.DatetimeIndex(frame["timestamp"]).sort_values()
    expected_delta = pd.Timedelta(to_offset(frequency))
    diffs = pd.Series(timestamps[1:] - timestamps[:-1]) if len(timestamps) > 1 else pd.Series(dtype="timedelta64[ns]")
    gap_hours = (diffs.dt.total_seconds() / 3600.0) if not diffs.empty else pd.Series(dtype=float)
    expected_index = pd.date_range(start=timestamps[0], end=timestamps[-1], freq=frequency, tz=timestamps.tz) if len(timestamps) else pd.DatetimeIndex([])
    missing_inside = int(len(expected_index) - timestamps.nunique()) if len(expected_index) else 0
    non_standard_gaps = gap_hours[gap_hours > (expected_delta.total_seconds() / 3600.0)] if not gap_hours.empty else pd.Series(dtype=float)
    summary = {
        "start_timestamp": timestamps[0].isoformat() if len(timestamps) else "",
        "end_timestamp": timestamps[-1].isoformat() if len(timestamps) else "",
        "actual_rows": int(len(frame)),
        "unique_timestamps": int(timestamps.nunique()),
        "expected_rows_in_observed_range": int(len(expected_index)),
        "coverage_ratio": round((timestamps.nunique() / len(expected_index)) if len(expected_index) else 0.0, 6),
        "duplicate_timestamps": int(frame["timestamp"].duplicated().sum()),
        "missing_hours_inside_observed_range": missing_inside,
        "non_standard_gap_count": int(len(non_standard_gaps)),
        "max_gap_hours": round(float(non_standard_gaps.max()), 4) if len(non_standard_gaps) else 1.0,
        "median_gap_hours": round(float(gap_hours.median()), 4) if len(gap_hours) else 0.0,
        "funding_event_rows": int(frame.get("funding_event", pd.Series(dtype=int)).sum()) if "funding_event" in frame.columns else 0,
    }
    return pd.DataFrame([summary])


def compute_distribution_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute distribution summaries for selected numeric columns."""
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        rows.append(
            {
                "column": column,
                "count": int(series.count()),
                "mean": float(series.mean()) if not series.empty else np.nan,
                "std": float(series.std()) if not series.empty else np.nan,
                "min": float(series.min()) if not series.empty else np.nan,
                "p01": float(series.quantile(0.01)) if not series.empty else np.nan,
                "p05": float(series.quantile(0.05)) if not series.empty else np.nan,
                "median": float(series.median()) if not series.empty else np.nan,
                "p95": float(series.quantile(0.95)) if not series.empty else np.nan,
                "p99": float(series.quantile(0.99)) if not series.empty else np.nan,
                "max": float(series.max()) if not series.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_correlation_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute a simple correlation matrix for key variables."""
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.DataFrame()
    return frame[available].corr().round(4)


def summarize_key_findings(
    analysis: pd.DataFrame,
    missingness: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, Any]:
    """Create concise headline metrics for markdown and JSON summaries."""
    funding_event_mask = analysis["funding_event"].eq(1) if "funding_event" in analysis.columns else analysis["funding_rate_bps"].ne(0)
    funding_bps = analysis.loc[funding_event_mask, "funding_rate_bps"].dropna()
    spread_bps = analysis["spread_bps"].dropna()
    perp_vol = analysis["perp_rolling_vol_annualized"].dropna()
    spot_vol = analysis["spot_rolling_vol_annualized"].dropna()
    missing_cols = missingness[missingness["missing_count"] > 0][["column", "missing_count", "missing_pct"]].head(5)
    coverage_row = coverage.iloc[0].to_dict()
    return {
        "coverage": coverage_row,
        "funding_event_count": int(funding_bps.count()),
        "funding_mean_bps": round(float(funding_bps.mean()), 6) if not funding_bps.empty else None,
        "funding_std_bps": round(float(funding_bps.std()), 6) if not funding_bps.empty else None,
        "funding_min_bps": round(float(funding_bps.min()), 6) if not funding_bps.empty else None,
        "funding_max_bps": round(float(funding_bps.max()), 6) if not funding_bps.empty else None,
        "positive_funding_event_share": round(float((funding_bps > 0).mean()), 6) if not funding_bps.empty else None,
        "spread_mean_bps": round(float(spread_bps.mean()), 6) if not spread_bps.empty else None,
        "abs_spread_p95_bps": round(float(spread_bps.abs().quantile(0.95)), 6) if not spread_bps.empty else None,
        "mean_perp_annualized_vol": round(float(perp_vol.mean()), 6) if not perp_vol.empty else None,
        "mean_spot_annualized_vol": round(float(spot_vol.mean()), 6) if not spot_vol.empty else None,
        "top_missing_columns": missing_cols.to_dict(orient="records"),
    }


def _save_table(frame: pd.DataFrame, path: Path, write_csv: bool) -> str:
    if write_csv:
        frame.to_csv(path, index=False)
    else:
        path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    return str(path)


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _plot_funding_rate(analysis: pd.DataFrame, output_path: Path, *, smoothing_window: int, dpi: int) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(analysis["timestamp"], analysis["funding_rate_bps"], color=PLOT_COLORS["funding"], linewidth=0.8, alpha=0.45, label="Hourly funding (bps)")
    ax.plot(
        analysis["timestamp"],
        analysis["funding_rate_bps"].rolling(smoothing_window, min_periods=1).mean(),
        color=PLOT_COLORS["signal"],
        linewidth=1.5,
        label=f"{smoothing_window}h rolling mean",
    )
    ax.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
    ax.set_title("Funding Rate Time Series")
    ax.set_ylabel("Funding rate (bps)")
    ax.set_xlabel("Timestamp")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_spread(analysis: pd.DataFrame, output_path: Path, *, smoothing_window: int, dpi: int) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(analysis["timestamp"], analysis["spread_bps"], color=PLOT_COLORS["spread"], linewidth=0.8, alpha=0.4, label="Perp - spot spread (bps)")
    ax.plot(
        analysis["timestamp"],
        analysis["spread_bps"].rolling(smoothing_window, min_periods=1).mean(),
        color=PLOT_COLORS["signal"],
        linewidth=1.5,
        label=f"{smoothing_window}h rolling mean",
    )
    ax.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
    ax.set_title("Perpetual vs Spot Spread")
    ax.set_ylabel("Spread (bps)")
    ax.set_xlabel("Timestamp")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_volatility(analysis: pd.DataFrame, output_path: Path, *, dpi: int) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(analysis["timestamp"], analysis["perp_rolling_vol_annualized"] * 100.0, color=PLOT_COLORS["perp"], linewidth=1.2, label="Perp annualized volatility")
    ax.plot(analysis["timestamp"], analysis["spot_rolling_vol_annualized"] * 100.0, color=PLOT_COLORS["spot"], linewidth=1.2, label="Spot annualized volatility")
    ax.set_title("Rolling Return Volatility")
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_xlabel("Timestamp")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_correlation_heatmap(correlation: pd.DataFrame, output_path: Path, *, dpi: int) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = correlation.to_numpy()
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    ax.set_title("Correlation Summary")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="#111827", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _load_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = _resolve_path(path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _build_markdown_report(
    settings: DataQualityReportSettings,
    coverage: pd.DataFrame,
    missingness: pd.DataFrame,
    distribution: pd.DataFrame,
    correlation: pd.DataFrame,
    key_findings: dict[str, Any],
    figure_paths: list[str],
    manifest: dict[str, Any] | None,
) -> str:
    top_missing = missingness[missingness["missing_count"] > 0].head(8)
    key_distribution = distribution[distribution["column"].isin(["perp_close", "spot_close", "funding_rate_bps", "spread_bps", "perp_volume", "spot_volume"])]
    manifest_lines = ""
    if manifest is not None:
        manifest_lines = (
            f"- Canonical rows: `{manifest.get('canonical_row_count', 'n/a')}`\n"
            f"- Raw source rows: `{manifest.get('row_counts', {})}`\n"
        )

    figure_markdown = "\n".join(
        f"![{Path(path).stem}](figures/{Path(path).name})" for path in figure_paths
    )
    return f"""# Data Quality Report

## Overview

- Symbol: `{settings.input.symbol}`
- Provider: `{settings.input.provider}`
- Venue: `{settings.input.venue}`
- Frequency: `{settings.input.frequency}`
- Source dataset: `{settings.input.dataset_path}`
- Report output root: `{settings.output.output_dir}`
{manifest_lines}
## Time Coverage

{_dataframe_to_markdown(coverage.round(6))}

## Missingness Summary

{_dataframe_to_markdown(top_missing.round(4))}

## Distribution Summary

{_dataframe_to_markdown(key_distribution.round(6))}

## Correlation Summary

{_dataframe_to_markdown(correlation.reset_index().rename(columns={'index': 'feature'}).round(4))}

## Key Findings

- Funding events observed: `{key_findings.get('funding_event_count')}`
- Average realized funding rate: `{key_findings.get('funding_mean_bps')}` bps
- Funding-rate standard deviation: `{key_findings.get('funding_std_bps')}` bps
- Funding-rate range: `{key_findings.get('funding_min_bps')}` to `{key_findings.get('funding_max_bps')}` bps
- Share of positive funding events: `{key_findings.get('positive_funding_event_share')}`
- Average perp-vs-spot spread: `{key_findings.get('spread_mean_bps')}` bps
- 95th percentile absolute spread: `{key_findings.get('abs_spread_p95_bps')}` bps
- Mean annualized perp volatility: `{key_findings.get('mean_perp_annualized_vol')}`
- Mean annualized spot volatility: `{key_findings.get('mean_spot_annualized_vol')}`

## Figures

{figure_markdown}
"""


def run_data_quality_report(settings: DataQualityReportSettings) -> DataQualityReportArtifacts:
    """Generate presentation-ready exploratory summaries and plots for the canonical dataset."""
    dataset = load_market_dataset(settings.input.dataset_path)
    manifest = _load_manifest(settings.input.manifest_path)
    analysis = prepare_analysis_frame(
        dataset,
        volatility_window_hours=settings.plots.rolling_volatility_window_hours,
        annualization_factor_hours=settings.plots.annualization_factor_hours,
    )

    missingness = compute_missingness_summary(analysis)
    coverage = compute_time_coverage_summary(analysis, frequency=settings.input.frequency)
    distribution = compute_distribution_summary(
        analysis,
        columns=[
            "perp_close",
            "spot_close",
            "funding_rate",
            "funding_rate_bps",
            "perp_volume",
            "spot_volume",
            "spread",
            "spread_bps",
            "perp_return",
            "spot_return",
            "perp_rolling_vol_annualized",
            "spot_rolling_vol_annualized",
        ],
    )
    correlation = compute_correlation_summary(
        analysis,
        columns=["perp_return", "spot_return", "funding_rate_bps", "spread_bps", "perp_volume", "spot_volume"],
    )
    key_findings = summarize_key_findings(analysis, missingness, coverage)

    output_root = ensure_directory(_resolve_path(settings.output.output_dir) / settings.input.provider / settings.input.symbol.lower() / settings.input.frequency)
    tables_dir = ensure_directory(output_root / "tables")
    figures_dir = ensure_directory(output_root / "figures")

    table_paths = [
        _save_table(missingness, tables_dir / "missingness_summary.csv", settings.output.write_csv),
        _save_table(coverage, tables_dir / "time_coverage_summary.csv", settings.output.write_csv),
        _save_table(distribution, tables_dir / "distribution_summary.csv", settings.output.write_csv),
        _save_table(correlation.reset_index().rename(columns={"index": "feature"}), tables_dir / "correlation_summary.csv", settings.output.write_csv),
    ]

    figure_extension = settings.plots.figure_format.lower().lstrip(".")
    figure_paths = [
        _plot_funding_rate(
            analysis,
            figures_dir / f"funding_rate_time_series.{figure_extension}",
            smoothing_window=settings.plots.funding_smoothing_window_hours,
            dpi=settings.plots.dpi,
        ),
        _plot_spread(
            analysis,
            figures_dir / f"perp_vs_spot_spread.{figure_extension}",
            smoothing_window=settings.plots.spread_smoothing_window_hours,
            dpi=settings.plots.dpi,
        ),
        _plot_volatility(
            analysis,
            figures_dir / f"return_volatility.{figure_extension}",
            dpi=settings.plots.dpi,
        ),
    ]
    if not correlation.empty:
        figure_paths.append(_plot_correlation_heatmap(correlation, figures_dir / f"correlation_heatmap.{figure_extension}", dpi=settings.plots.dpi))

    summary_json_path: str | None = None
    if settings.output.write_json_summary:
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(key_findings, indent=2, default=str), encoding="utf-8")
        summary_json_path = str(summary_path)

    markdown_report_path: str | None = None
    if settings.output.write_markdown:
        report_path = output_root / "report.md"
        report_path.write_text(
            _build_markdown_report(settings, coverage, missingness, distribution, correlation, key_findings, figure_paths, manifest),
            encoding="utf-8",
        )
        markdown_report_path = str(report_path)

    return DataQualityReportArtifacts(
        output_dir=str(output_root),
        table_paths=table_paths,
        figure_paths=figure_paths,
        summary_json_path=summary_json_path,
        markdown_report_path=markdown_report_path,
    )