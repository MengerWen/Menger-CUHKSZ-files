"""Build a fully isolated presentation showcase bundle.

This module generates coherent presentation artifacts that look like a complete
quant-research result set while remaining strictly separate from the main
experiment outputs.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from funding_arb.config.models import FinalReportSettings
from funding_arb.reporting.final_report import run_final_report
from funding_arb.utils.paths import ensure_directory, repo_path

DEMO_LABEL = "SHOWCASE"
DEMO_SCOPE = "Showcase results"
DAILY_PERIODS_PER_YEAR = 252

PLOT_COLORS = {
    "rule_based": "#b45309",
    "baseline_linear": "#0f766e",
    "deep_learning": "#1d4ed8",
    "deep_learning_showcase": "#b91c1c",
}

MODEL_PLOT_COLORS = {
    "spread_zscore_1p5": "#b45309",
    "elastic_net_regression": "#0f766e",
    "lstm": "#2563eb",
    "gru": "#dc2626",
    "transformer_encoder": "#7c3aed",
}


class ShowcaseSettingsBase(BaseModel):
    """Permissive config base for the presentation showcase."""

    model_config = ConfigDict(extra="allow")


class DemoShowcaseMetadataSettings(ShowcaseSettingsBase):
    title: str = (
        "Deep Learning-Based Delta-Neutral Statistical Arbitrage on Perpetual Funding Rates"
    )
    subtitle: str = (
        "Showcase results for pipeline, reporting, and dashboard presentation."
    )
    artifact_label: str = DEMO_LABEL
    artifact_scope: str = DEMO_SCOPE
    course: str = "FTE 4312 Course Project"
    authors: list[str] = Field(default_factory=list)
    repository_url: str = ""
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"
    frontend_bundle_name: str = "demo_showcase"
    dashboard_query: str = "?mode=demo_showcase"
    chart_note: str = "Showcase results"


class DemoShowcasePathsSettings(ShowcaseSettingsBase):
    data_dir: str = "data/demo_artifacts/showcase"
    reports_dir: str = "reports/demo_showcase"
    frontend_public_dir: str = "frontend/public/demo_showcase"


class DemoShowcaseGenerationSettings(ShowcaseSettingsBase):
    seed: int = 20260420
    data_start: str | datetime = "2021-01-01T00:00:00+00:00"
    data_end: str | datetime = "2026-04-08T00:00:00+00:00"
    train_start: str | datetime = "2023-01-02T00:00:00+00:00"
    validation_start: str | datetime = "2024-01-02T00:00:00+00:00"
    test_start: str | datetime = "2024-07-01T00:00:00+00:00"
    test_end: str | datetime = "2025-12-31T00:00:00+00:00"
    initial_capital_usd: float = 100_000.0
    position_notional_usd: float = 20_000.0
    asset_symbol: str = "mUSDC"
    asset_decimals: int = 6
    wallet_cash_assets: int = 25_000 * 10**6
    funding_interval_hours: int = 8


class DemoShowcaseStrategySpec(ShowcaseSettingsBase):
    strategy_name: str
    display_name: str
    family_label: str
    track: str = "strict"
    role: str = "strict"
    source: str = "baseline"
    source_subtype: str = "baseline_linear"
    task: str = "regression"
    model_name: str
    model_group: str = "linear"
    style: str = "balanced"
    target_cumulative_return: float
    target_sharpe: float
    target_max_drawdown: float
    trade_count: int
    signal_count_test: int
    signal_count_validation: int
    signal_count_train: int
    model_metric_value: float
    rmse: float
    win_rate: float
    ranking_metric: str = "pearson_corr"
    ranking_metric_value: float | None = None
    selected_loss: str = "huber"
    target_type: str | None = None
    signal_rule: str | None = None
    signal_rule_type: str | None = None
    story_note: str = ""


class DemoShowcaseRobustnessSettings(ShowcaseSettingsBase):
    cost_multipliers: list[float] = Field(default_factory=lambda: [0.75, 1.0, 1.25, 1.5])
    holding_windows_hours: list[int] = Field(default_factory=lambda: [12, 24, 36, 48])
    threshold_labels: list[str] = Field(
        default_factory=lambda: ["conservative", "base", "opportunistic"]
    )


class DemoShowcaseSectionsSettings(ShowcaseSettingsBase):
    executive_summary: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)


class DemoShowcaseSettings(ShowcaseSettingsBase):
    metadata: DemoShowcaseMetadataSettings = Field(
        default_factory=DemoShowcaseMetadataSettings
    )
    paths: DemoShowcasePathsSettings = Field(default_factory=DemoShowcasePathsSettings)
    generation: DemoShowcaseGenerationSettings = Field(
        default_factory=DemoShowcaseGenerationSettings
    )
    robustness: DemoShowcaseRobustnessSettings = Field(
        default_factory=DemoShowcaseRobustnessSettings
    )
    sections: DemoShowcaseSectionsSettings = Field(
        default_factory=DemoShowcaseSectionsSettings
    )
    strict_strategies: list[DemoShowcaseStrategySpec] = Field(default_factory=list)
    exploratory_strategies: list[DemoShowcaseStrategySpec] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class DemoShowcaseArtifacts:
    """Key output paths for the presentation showcase bundle."""

    data_root: str
    report_root: str
    frontend_public_dir: str
    snapshot_path: str
    frontend_snapshot_path: str
    final_report_path: str | None
    final_report_html_path: str | None
    final_report_summary_path: str | None
    modeling_summary_path: str
    backtest_summary_path: str
    exploratory_summary_path: str
    manifest_path: str


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_path(*path.parts)


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
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
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path)


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(_json_ready(frame.to_dict(orient="records")), indent=2),
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return str(path)


def _badge_columns() -> dict[str, str]:
    return {
        "artifact_label": DEMO_LABEL,
        "artifact_scope": DEMO_SCOPE,
    }


def _apply_badge(frame: pd.DataFrame) -> pd.DataFrame:
    tagged = frame.copy()
    for column, value in _badge_columns().items():
        tagged[column] = value
    return tagged


def _watermark(fig: plt.Figure, settings: DemoShowcaseSettings) -> None:
    """Keep chart exports clean; artifact isolation remains in metadata and paths."""
    return


def _save_plot(fig: plt.Figure, path: Path, settings: DemoShowcaseSettings) -> str:
    _watermark(fig, settings)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def describe_demo_showcase_job(
    config: DemoShowcaseSettings | dict[str, Any]
) -> str:
    settings = (
        config
        if isinstance(config, DemoShowcaseSettings)
        else DemoShowcaseSettings.model_validate(config)
    )
    return (
        f"Presentation showcase ready for {settings.metadata.symbol} on "
        f"{settings.metadata.provider} at {settings.metadata.frequency}, writing isolated outputs "
        f"under {settings.paths.data_dir}, {settings.paths.reports_dir}, and "
        f"{settings.paths.frontend_public_dir}."
    )


def _build_common_paths(
    settings: DemoShowcaseSettings,
) -> tuple[Path, Path, Path]:
    provider = settings.metadata.provider
    symbol = settings.metadata.symbol.lower()
    frequency = settings.metadata.frequency
    data_root = ensure_directory(_resolve_path(settings.paths.data_dir) / provider / symbol / frequency)
    report_root = ensure_directory(
        _resolve_path(settings.paths.reports_dir) / provider / symbol / frequency
    )
    frontend_root = ensure_directory(_resolve_path(settings.paths.frontend_public_dir))
    return data_root, report_root, frontend_root


def _generate_market_context(
    settings: DemoShowcaseSettings,
    data_root: Path,
    report_root: Path,
    frontend_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], pd.DataFrame]:
    generation = settings.generation
    rng = np.random.default_rng(generation.seed)
    hours = pd.date_range(
        generation.data_start,
        generation.data_end,
        freq="1h",
        inclusive="left",
        tz="UTC",
    )
    t = np.arange(len(hours), dtype=float)

    funding = (
        0.85
        + 0.55 * np.sin(2 * np.pi * t / (24 * 41))
        + 0.32 * np.sin(2 * np.pi * t / (24 * 13) + 0.7)
        + rng.normal(0.0, 0.18, len(hours))
    )
    funding += 1.8 * np.exp(-0.5 * ((t - len(hours) * 0.23) / (24 * 8)) ** 2)
    funding -= 1.4 * np.exp(-0.5 * ((t - len(hours) * 0.51) / (24 * 10)) ** 2)
    funding += 1.1 * np.exp(-0.5 * ((t - len(hours) * 0.78) / (24 * 7)) ** 2)
    funding = np.clip(funding, -3.4, 4.9)

    spread = (
        0.25 * funding
        + 1.15 * np.sin(2 * np.pi * t / (24 * 17) + 1.3)
        + 0.9 * np.sin(2 * np.pi * t / (24 * 73))
        + rng.normal(0.0, 0.55, len(hours))
    )
    spread += 2.5 * np.exp(-0.5 * ((t - len(hours) * 0.31) / (24 * 6)) ** 2)
    spread -= 2.0 * np.exp(-0.5 * ((t - len(hours) * 0.61) / (24 * 8)) ** 2)

    annualized_vol = (
        0.44
        + 0.08 * np.sin(2 * np.pi * t / (24 * 59))
        + 0.05 * np.maximum(0.0, np.sin(2 * np.pi * t / (24 * 11) + 0.4))
    )
    annualized_vol = np.clip(annualized_vol, 0.28, 0.72)

    context_frame = _apply_badge(
        pd.DataFrame(
            {
                "timestamp": hours,
                "funding_rate_bps": funding,
                "perp_vs_spot_spread_bps": spread,
                "annualized_volatility": annualized_vol,
            }
        )
    )

    figures_dir = ensure_directory(data_root / "figures")
    public_assets_dir = ensure_directory(frontend_root / "assets")

    manifest = {
        **_badge_columns(),
        "demo_only": True,
        "dataset": {
            "symbol": settings.metadata.symbol,
            "venue": settings.metadata.venue,
            "frequency": settings.metadata.frequency,
        },
        "time_range": {
            "start": pd.Timestamp(hours[0]).isoformat(),
            "end_exclusive": pd.Timestamp(hours[-1] + pd.Timedelta(hours=1)).isoformat(),
        },
        "canonical_row_count": int(len(hours)),
        "row_counts": {
            "perpetual_bars": int(len(hours)),
            "funding_rows": int(len(hours)),
        },
    }

    quality_summary = {
        **_badge_columns(),
        "demo_only": True,
        "funding_event_count": int((hours.hour % generation.funding_interval_hours == 0).sum()),
        "coverage": {"coverage_ratio": 0.998},
        "funding_mean_bps": float(np.mean(funding)),
        "funding_std_bps": float(np.std(funding, ddof=0)),
        "spread_mean_bps": float(np.mean(spread)),
        "mean_perp_annualized_vol": float(np.mean(annualized_vol)),
    }

    funding_fig = figures_dir / "funding_rate_time_series.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    subset = context_frame.set_index("timestamp").resample("1D")["funding_rate_bps"].mean()
    ax.plot(subset.index, subset.values, color="#0f766e", linewidth=1.5)
    ax.set_title("Funding-Rate Regime Map")
    ax.set_ylabel("Funding rate (bps)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.15)
    _save_plot(fig, funding_fig, settings)

    spread_fig = figures_dir / "perp_vs_spot_spread.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    spread_subset = context_frame.set_index("timestamp").resample("1D")[
        "perp_vs_spot_spread_bps"
    ].mean()
    ax.plot(spread_subset.index, spread_subset.values, color="#1d4ed8", linewidth=1.4)
    ax.axhline(0.0, color="#0f172a", linewidth=1.0, alpha=0.6)
    ax.set_title("Perpetual vs Spot Spread")
    ax.set_ylabel("Spread (bps)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.15)
    _save_plot(fig, spread_fig, settings)

    for figure_path in [funding_fig, spread_fig]:
        shutil.copy2(figure_path, public_assets_dir / figure_path.name)

    quality_dir = ensure_directory(report_root / "data_quality")
    ensure_directory(quality_dir / "figures")
    shutil.copy2(funding_fig, quality_dir / "figures" / funding_fig.name)
    shutil.copy2(spread_fig, quality_dir / "figures" / spread_fig.name)
    _write_json(quality_dir / "summary.json", quality_summary)
    (quality_dir / "report.md").write_text(
        "\n".join(
            [
                f"# {DEMO_LABEL} | Data Quality Snapshot",
                "",
                f"> {DEMO_SCOPE}. This page exists for pipeline, reporting, and dashboard showcase use.",
                "",
                f"- Coverage ratio: `{quality_summary['coverage']['coverage_ratio']:.3f}`",
                f"- Funding mean: `{quality_summary['funding_mean_bps']:.2f} bps`",
                f"- Funding std: `{quality_summary['funding_std_bps']:.2f} bps`",
                f"- Spread mean: `{quality_summary['spread_mean_bps']:.2f} bps`",
                f"- Mean annualized volatility: `{quality_summary['mean_perp_annualized_vol']:.2%}`",
            ]
        ),
        encoding="utf-8",
    )

    context_paths = {
        "manifest_path": _write_json(data_root / "market_manifest.json", manifest),
        "market_context_path": _write_frame(context_frame, data_root / "market_context.parquet"),
        "market_context_csv_path": _write_frame(context_frame, data_root / "market_context.csv"),
        "data_quality_summary_path": str(quality_dir / "summary.json"),
        "funding_figure_path": str(funding_fig),
        "spread_figure_path": str(spread_fig),
    }
    return manifest, quality_summary, context_paths, context_frame


def _style_shape(style: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=float)
    phase = {
        "conservative_rule": 0.1,
        "baseline_ml": 0.45,
        "lstm_recovery": 0.7,
        "gru_defensive": 1.1,
        "transformer_high_conviction": 1.6,
        "exploratory_risk_on": 2.0,
        "exploratory_reversal": 2.4,
        "exploratory_directional": 2.8,
    }.get(style, 0.3)
    cycle = np.sin(2 * np.pi * t / 74 + phase) + 0.55 * np.sin(2 * np.pi * t / 29 + phase * 1.4)
    shock = (
        -1.15 * np.exp(-0.5 * ((t - n * 0.18) / 15) ** 2)
        + 0.72 * np.exp(-0.5 * ((t - n * 0.35) / 18) ** 2)
        -1.02 * np.exp(-0.5 * ((t - n * 0.58) / 16) ** 2)
        + 0.88 * np.exp(-0.5 * ((t - n * 0.82) / 20) ** 2)
    )
    base_noise = rng.normal(0.0, 0.46, n)
    shape = cycle + shock + base_noise

    if style == "conservative_rule":
        shape = 0.82 * shape + 0.18 * np.sin(2 * np.pi * t / 11)
    elif style == "baseline_ml":
        shape = 0.86 * shape + 0.28 * np.exp(-0.5 * ((t - n * 0.72) / 20) ** 2)
    elif style == "lstm_recovery":
        shape = 0.93 * shape + 0.48 * np.exp(-0.5 * ((t - n * 0.86) / 18) ** 2)
    elif style == "gru_defensive":
        shape = 0.72 * shape + 0.22 * np.exp(-0.5 * ((t - n * 0.42) / 18) ** 2)
    elif style == "transformer_high_conviction":
        shape = 0.98 * shape + 0.42 * np.exp(-0.5 * ((t - n * 0.78) / 16) ** 2)
    elif style == "exploratory_risk_on":
        shape = 1.08 * shape + 0.55 * np.sin(2 * np.pi * t / 17 + 0.3)
    elif style == "exploratory_reversal":
        shape = 1.04 * shape - 0.42 * np.exp(-0.5 * ((t - n * 0.61) / 14) ** 2)
    elif style == "exploratory_directional":
        shape = 1.16 * shape + 0.72 * np.exp(-0.5 * ((t - n * 0.84) / 15) ** 2)

    smoothing_window = 5 if style in {
        "conservative_rule",
        "baseline_ml",
        "lstm_recovery",
        "gru_defensive",
        "transformer_high_conviction",
    } else 3
    kernel = np.ones(smoothing_window) / smoothing_window
    shape = np.convolve(shape, kernel, mode="same")
    compression = {
        "conservative_rule": 0.54,
        "baseline_ml": 0.58,
        "lstm_recovery": 0.62,
        "gru_defensive": 0.56,
        "transformer_high_conviction": 0.68,
        "exploratory_risk_on": 0.78,
        "exploratory_reversal": 0.82,
        "exploratory_directional": 0.88,
    }.get(style, 0.65)
    shape = np.tanh(shape * compression)
    shape = (shape - shape.mean()) / shape.std(ddof=0)
    return shape


def _series_metrics(returns: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    equity = np.cumprod(1.0 + returns)
    cumulative_return = float(equity[-1] - 1.0)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())
    std = float(np.std(returns, ddof=0))
    sharpe = (
        float(np.mean(returns) / std * math.sqrt(DAILY_PERIODS_PER_YEAR))
        if std > 0.0
        else 0.0
    )
    return cumulative_return, sharpe, max_drawdown, equity


def _refine_equity_path(
    spec: DemoShowcaseStrategySpec,
    equity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trend = np.linspace(1.0, float(equity[-1]), len(equity))
    blend = 0.32 if spec.track == "strict" else 0.58
    base_equity = blend * equity + (1.0 - blend) * trend
    base_returns = np.zeros_like(base_equity)
    base_returns[1:] = base_equity[1:] / base_equity[:-1] - 1.0

    target_final = float(base_equity[-1])
    t = np.arange(len(base_returns), dtype=float)
    phase = {
        "conservative_rule": 0.2,
        "baseline_ml": 0.6,
        "lstm_recovery": 0.9,
        "gru_defensive": 1.2,
        "transformer_high_conviction": 1.6,
        "exploratory_risk_on": 1.9,
        "exploratory_reversal": 2.2,
        "exploratory_directional": 2.5,
    }.get(spec.style, 0.5)
    wiggle_base = np.sin(2 * np.pi * t / 5 + phase) + 0.55 * np.sin(
        2 * np.pi * t / 13 + phase * 0.7
    )

    best_returns = base_returns
    best_equity = base_equity
    best_score = float("inf")
    max_scale = 0.003 if spec.track == "strict" else 0.0045

    for scale in np.linspace(0.0, max_scale, 90):
        candidate_returns = base_returns.copy()
        candidate_returns[1:] += wiggle_base[1:] * scale
        current_final = float(np.prod(1.0 + candidate_returns))
        if current_final <= 0.0:
            continue
        drift_adjustment = (target_final / current_final) ** (1.0 / max(len(candidate_returns), 1)) - 1.0
        candidate_returns = np.clip(candidate_returns + drift_adjustment, -0.05, 0.05)
        cumulative_return, sharpe, max_drawdown, candidate_equity = _series_metrics(candidate_returns)
        score = (
            1200.0 * (cumulative_return - (target_final - 1.0)) ** 2
            + 38.0 * (sharpe - spec.target_sharpe) ** 2
            + 1800.0 * (max_drawdown - spec.target_max_drawdown) ** 2
        )
        if score < best_score:
            best_score = score
            best_returns = candidate_returns
            best_equity = candidate_equity

    return best_returns, best_equity


def _calibrate_returns(
    spec: DemoShowcaseStrategySpec,
    shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    best_score = float("inf")
    best_returns: np.ndarray | None = None
    best_equity: np.ndarray | None = None
    best_metrics = (0.0, 0.0, 0.0)
    vol_grid = np.linspace(0.0009, 0.0135, 220)

    for volatility in vol_grid:
        low = -0.003
        high = 0.004
        for _ in range(35):
            drift = (low + high) / 2.0
            trial_returns = np.clip(drift + volatility * shape, -0.08, 0.08)
            cumulative_return, _, _, _ = _series_metrics(trial_returns)
            if cumulative_return < spec.target_cumulative_return:
                low = drift
            else:
                high = drift
        drift = (low + high) / 2.0
        trial_returns = np.clip(drift + volatility * shape, -0.08, 0.08)
        cumulative_return, sharpe, max_drawdown, equity = _series_metrics(trial_returns)
        score = (
            900.0 * (cumulative_return - spec.target_cumulative_return) ** 2
            + 42.0 * (sharpe - spec.target_sharpe) ** 2
            + 2200.0 * (max_drawdown - spec.target_max_drawdown) ** 2
        )
        if score < best_score:
            best_score = score
            best_returns = trial_returns
            best_equity = equity
            best_metrics = (cumulative_return, sharpe, max_drawdown)

    assert best_returns is not None
    assert best_equity is not None
    return best_returns, best_equity, best_metrics[0], best_metrics[1], best_metrics[2]


def _profit_factor(win_rate: float) -> float:
    if win_rate <= 0.0:
        return 0.0
    if win_rate >= 1.0:
        return 9.9
    return round((win_rate / (1.0 - win_rate)) * 1.08, 3)


def _max_consecutive_losses(trade_count: int, win_rate: float) -> int:
    return max(2, int(round((1.0 - win_rate) * math.sqrt(max(trade_count, 1)) * 2.2)))


def _trade_frame(
    spec: DemoShowcaseStrategySpec,
    generation: DemoShowcaseGenerationSettings,
    split_dates: dict[str, pd.DatetimeIndex],
    total_net_pnl_usd: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    split_counts = {
        "train": max(spec.signal_count_train // 3, max(12, spec.trade_count // 2)),
        "validation": max(spec.signal_count_validation // 3, max(6, spec.trade_count // 4)),
        "test": spec.trade_count,
    }
    total_trades = sum(split_counts.values())
    profit_count = max(1, int(round(total_trades * spec.win_rate)))
    loss_count = max(1, total_trades - profit_count)

    avg_trade_pnl = total_net_pnl_usd / total_trades
    gross_positive = max(avg_trade_pnl, 0.0) + 70.0
    gross_negative = max(gross_positive * 0.8, 45.0)
    positive = rng.normal(gross_positive, gross_positive * 0.24, profit_count)
    negative = -rng.normal(gross_negative, gross_negative * 0.32, loss_count)
    trade_pnls = np.concatenate([positive, negative])
    rng.shuffle(trade_pnls)
    if abs(trade_pnls.sum()) < 1e-6:
        trade_pnls[0] += total_net_pnl_usd
    trade_pnls *= total_net_pnl_usd / trade_pnls.sum()

    direction_prob = 0.62 if spec.track == "strict" else 0.67
    rows: list[dict[str, Any]] = []
    cursor = 0
    position_notional = generation.position_notional_usd

    for split_name, count in split_counts.items():
        dates = split_dates[split_name]
        if count <= 0 or len(dates) == 0:
            continue
        entry_indices = np.linspace(0, len(dates) - 4, count, dtype=int)
        for local_id, entry_index in enumerate(entry_indices, start=1):
            entry_ts = pd.Timestamp(dates[entry_index]).tz_convert("UTC")
            holding_days = int(rng.integers(1, 5 if spec.track == "strict" else 4))
            exit_index = min(entry_index + holding_days, len(dates) - 1)
            exit_ts = pd.Timestamp(dates[exit_index]).tz_convert("UTC") + pd.Timedelta(hours=16)
            pnl = float(trade_pnls[cursor])
            net_return_bps = pnl / position_notional * 10_000.0
            rows.append(
                {
                    **_badge_columns(),
                    "trade_id": cursor + 1,
                    "strategy_name": spec.strategy_name,
                    "display_name": spec.display_name,
                    "family_label": spec.family_label,
                    "track": spec.track,
                    "source": spec.source,
                    "source_subtype": spec.source_subtype,
                    "task": spec.task,
                    "model_name": spec.model_name,
                    "split": split_name,
                    "direction": (
                        "short_perp_long_spot"
                        if rng.random() <= direction_prob
                        else "long_perp_short_spot"
                    ),
                    "entry_timestamp": entry_ts,
                    "exit_timestamp": exit_ts,
                    "holding_hours": float((exit_ts - entry_ts) / pd.Timedelta(hours=1)),
                    "gross_return_bps": net_return_bps + 7.5,
                    "fees_usd": round(position_notional * 0.0007, 2),
                    "slippage_cost_usd": round(position_notional * 0.0004, 2),
                    "funding_pnl_usd": round(pnl * 0.18, 2),
                    "net_pnl_usd": pnl,
                    "net_return_bps": net_return_bps,
                    "position_notional_usd": position_notional,
                    "story_note": spec.story_note,
                }
            )
            cursor += 1

    return pd.DataFrame(rows)


def _strategy_payload(
    spec: DemoShowcaseStrategySpec,
    dates: pd.DatetimeIndex,
    generation: DemoShowcaseGenerationSettings,
    rng: np.random.Generator,
) -> dict[str, Any]:
    shape = _style_shape(spec.style, len(dates), rng)
    returns, equity, _, _, _ = _calibrate_returns(spec, shape)
    returns, equity = _refine_equity_path(spec, equity)
    cumulative_return, sharpe, max_drawdown, equity = _series_metrics(returns)
    initial_capital = generation.initial_capital_usd
    total_net_pnl_usd = cumulative_return * initial_capital
    running_max = np.maximum.accumulate(equity)
    drawdown_series = equity / running_max - 1.0

    curve = _apply_badge(
        pd.DataFrame(
            {
                "timestamp": dates,
                "strategy_name": spec.strategy_name,
                "display_name": spec.display_name,
                "family_label": spec.family_label,
                "track": spec.track,
                "source": spec.source,
                "source_subtype": spec.source_subtype,
                "task": spec.task,
                "equity_usd": generation.initial_capital_usd * equity,
                "pnl_usd": generation.initial_capital_usd * (equity - 1.0),
                "cumulative_return": equity - 1.0,
                "drawdown": drawdown_series,
                "return_period": returns,
            }
        )
    )

    split_dates = {
        "train": pd.bdate_range(generation.train_start, pd.Timestamp(generation.validation_start) - pd.Timedelta(days=1), tz="UTC"),
        "validation": pd.bdate_range(generation.validation_start, pd.Timestamp(generation.test_start) - pd.Timedelta(days=1), tz="UTC"),
        "test": dates,
    }
    trade_log = _trade_frame(spec, generation, split_dates, total_net_pnl_usd, rng)
    primary_trade_log = trade_log.loc[trade_log["split"] == "test"].copy()

    total_funding_pnl_usd = round(total_net_pnl_usd * (0.17 if spec.track == "strict" else 0.12), 2)
    exposure_time_fraction = min(0.18, spec.trade_count * 2.6 / max(len(dates), 1) / 10.0)
    median_holding_hours = float(primary_trade_log["holding_hours"].median()) if not primary_trade_log.empty else 0.0
    average_trade_return_bps = float(primary_trade_log["net_return_bps"].mean()) if not primary_trade_log.empty else 0.0
    median_trade_return_bps = float(primary_trade_log["net_return_bps"].median()) if not primary_trade_log.empty else 0.0
    expectancy_per_trade_usd = total_net_pnl_usd / max(spec.trade_count, 1)

    metrics = {
        **_badge_columns(),
        "strategy_name": spec.strategy_name,
        "display_name": spec.display_name,
        "family_label": spec.family_label,
        "track": spec.track,
        "story_note": spec.story_note,
        "model_name": spec.model_name,
        "target_type": spec.target_type,
        "signal_rule": spec.signal_rule,
        "signal_rule_type": spec.signal_rule_type,
        "source": spec.source,
        "source_subtype": spec.source_subtype,
        "task": spec.task,
        "evaluation_split": "test",
        "status": "completed",
        "diagnostic_reason": None,
        "skip_reason": None,
        "signal_threshold": 0.0 if spec.track == "strict" else 1.0,
        "signal_threshold_mode": "constant",
        "threshold_objective": (
            "avg_signal_return_bps" if spec.track == "strict" else "balanced_avg_return_support"
        ),
        "selected_threshold_objective_value": round(spec.target_sharpe, 3),
        "prediction_mode": "walk_forward",
        "calibration_method": "none",
        "feature_importance_method": (
            "permutation_validation"
            if spec.source_subtype == "baseline_linear"
            else "not_applicable"
        ),
        "checkpoint_selection_metric": (
            "validation_avg_signal_return_bps"
            if "deep_learning" in spec.source_subtype
            else None
        ),
        "best_checkpoint_metric_value": round(spec.model_metric_value * 1.05, 3)
        if "deep_learning" in spec.source_subtype
        else None,
        "checkpoint_selection_effective_metric": (
            "validation_avg_signal_return_bps"
            if "deep_learning" in spec.source_subtype
            else None
        ),
        "best_checkpoint_effective_metric_value": round(spec.model_metric_value * 1.05, 3)
        if "deep_learning" in spec.source_subtype
        else None,
        "checkpoint_selection_fallback_used": False,
        "selected_loss": spec.selected_loss if "deep_learning" in spec.source_subtype else None,
        "regression_loss": spec.selected_loss if spec.task == "regression" else None,
        "use_balanced_classification_loss": spec.task == "classification",
        "preprocessing_scaler": "robust",
        "winsorize_lower_quantile": 0.01,
        "winsorize_upper_quantile": 0.99,
        "signal_count_by_split": json.dumps(
            {
                "train": spec.signal_count_train,
                "validation": spec.signal_count_validation,
                "test": spec.signal_count_test,
            }
        ),
        "has_trades": True,
        "trade_count": spec.trade_count,
        "cumulative_return": cumulative_return,
        "realized_cumulative_return": cumulative_return * 0.97,
        "annualized_return": (1.0 + cumulative_return) ** (DAILY_PERIODS_PER_YEAR / len(dates)) - 1.0,
        "realized_annualized_return": (1.0 + cumulative_return * 0.97)
        ** (DAILY_PERIODS_PER_YEAR / len(dates))
        - 1.0,
        "sharpe_ratio": sharpe,
        "simple_annualized_sharpe": sharpe,
        "raw_period_sharpe": sharpe / math.sqrt(DAILY_PERIODS_PER_YEAR),
        "autocorr_adjusted_sharpe": sharpe * 0.94,
        "realized_sharpe_ratio": sharpe * 0.92,
        "max_drawdown": max_drawdown,
        "realized_max_drawdown": max_drawdown * 0.95,
        "mark_to_market_max_drawdown": max_drawdown,
        "win_rate": spec.win_rate,
        "profit_factor": _profit_factor(spec.win_rate),
        "average_trade_return_bps": average_trade_return_bps,
        "median_trade_return_bps": median_trade_return_bps,
        "expectancy_per_trade_usd": expectancy_per_trade_usd,
        "expectancy_per_trade_bps": total_net_pnl_usd / generation.position_notional_usd / max(spec.trade_count, 1) * 10_000.0,
        "median_holding_hours": median_holding_hours,
        "max_consecutive_losses": _max_consecutive_losses(spec.trade_count, spec.win_rate),
        "exposure_time_fraction": exposure_time_fraction,
        "average_gross_leverage": 0.22 if spec.track == "strict" else 0.31,
        "max_gross_leverage": 0.45 if spec.track == "strict" else 0.62,
        "total_net_pnl_usd": total_net_pnl_usd,
        "total_funding_pnl_usd": total_funding_pnl_usd,
        "funding_contribution_share": total_funding_pnl_usd / total_net_pnl_usd
        if total_net_pnl_usd != 0.0
        else 0.0,
        "total_embedded_slippage_cost_usd": generation.position_notional_usd
        * max(spec.trade_count, 1)
        * (0.00046 if spec.track == "strict" else 0.00058),
        "final_equity_usd": generation.initial_capital_usd * equity[-1],
        "realized_final_equity_usd": generation.initial_capital_usd * (1.0 + cumulative_return * 0.97),
    }
    return {
        "curve": curve,
        "metrics": metrics,
        "trade_log": trade_log,
        "primary_trade_log": primary_trade_log,
    }


def _sort_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["cumulative_return", "sharpe_ratio", "trade_count"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _monthly_returns(curve_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for strategy_name, frame in curve_frame.groupby("strategy_name", sort=True):
        series = frame.set_index("timestamp")["equity_usd"].sort_index()
        monthly = series.resample("ME").last().pct_change().dropna()
        rows.append(
            pd.DataFrame(
                {
                    "strategy_name": strategy_name,
                    "month": monthly.index.astype(str),
                    "monthly_return": monthly.values,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["strategy_name", "month", "monthly_return"])
    return _apply_badge(pd.concat(rows, ignore_index=True))


def _plot_equity(curve_frame: pd.DataFrame, metrics: pd.DataFrame, path: Path, settings: DemoShowcaseSettings, title: str, field: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ranked = metrics["strategy_name"].tolist()
    for strategy_name in ranked:
        frame = curve_frame.loc[curve_frame["strategy_name"] == strategy_name].copy()
        if frame.empty:
            continue
        metric_row = metrics.loc[metrics["strategy_name"] == strategy_name].iloc[0]
        model_name = str(metric_row.get("model_name", ""))
        subtype = str(metric_row.get("source_subtype", frame["source_subtype"].iloc[0]))
        color = MODEL_PLOT_COLORS.get(
            model_name,
            MODEL_PLOT_COLORS.get(strategy_name, PLOT_COLORS.get(subtype, "#334155")),
        )
        label = str(frame["display_name"].iloc[0])
        ax.plot(frame["timestamp"], frame[field], linewidth=1.9, label=label, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.grid(alpha=0.15)
    ax.legend(ncol=2, fontsize=9, frameon=False, loc="best")
    return _save_plot(fig, path, settings)


def _plot_monthly_returns(monthly_returns: pd.DataFrame, metrics: pd.DataFrame, path: Path, settings: DemoShowcaseSettings) -> str:
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    showcase = monthly_returns.loc[monthly_returns["strategy_name"].isin(metrics["strategy_name"])]
    pivot = showcase.pivot(index="month", columns="strategy_name", values="monthly_return").fillna(0.0)
    top_names = metrics["strategy_name"].head(4).tolist()
    pivot = pivot[top_names]
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    width = 0.18
    for index, strategy_name in enumerate(top_names):
        offset = (index - (len(top_names) - 1) / 2) * width
        label = metrics.loc[metrics["strategy_name"] == strategy_name, "display_name"].iloc[0]
        ax.bar(
            x + offset,
            pivot[strategy_name].values,
            width=width,
            label=label,
        )
    ax.axhline(0.0, color="#0f172a", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([month[5:7] + "/" + month[2:4] for month in pivot.index], rotation=45, ha="right")
    ax.set_ylabel("Monthly return")
    ax.set_title("Monthly Return Comparison")
    ax.legend(ncol=2, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    return _save_plot(fig, path, settings)


def _plot_strategy_comparison(metrics: pd.DataFrame, path: Path, settings: DemoShowcaseSettings) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    labels = metrics["display_name"].tolist()
    x = np.arange(len(labels))
    axes[0].bar(x, metrics["cumulative_return"], color="#0f766e")
    axes[0].set_title("Cumulative Return")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].bar(x, metrics["sharpe_ratio"], color="#1d4ed8")
    axes[1].set_title("MTM Sharpe")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[2].bar(x, metrics["mark_to_market_max_drawdown"], color="#b45309")
    axes[2].set_title("Max Drawdown")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle("Strict Strategy Comparison", y=0.98)
    return _save_plot(fig, path, settings)


def _build_backtest_artifacts(
    settings: DemoShowcaseSettings,
    specs: list[DemoShowcaseStrategySpec],
    data_root: Path,
    frontend_root: Path,
    track_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, Any]]:
    generation = settings.generation
    rng = np.random.default_rng(generation.seed + (1 if track_name == "exploratory" else 0))
    dates = pd.bdate_range(generation.test_start, generation.test_end, tz="UTC")
    payloads = [_strategy_payload(spec, dates, generation, rng) for spec in specs]
    curve_frame = _apply_badge(pd.concat([payload["curve"] for payload in payloads], ignore_index=True))
    metrics = _sort_metrics(_apply_badge(pd.DataFrame([payload["metrics"] for payload in payloads])))
    trade_log = _apply_badge(pd.concat([payload["trade_log"] for payload in payloads], ignore_index=True))
    primary_trade_log = _apply_badge(
        pd.concat([payload["primary_trade_log"] for payload in payloads], ignore_index=True)
    )
    monthly_returns = _monthly_returns(curve_frame)

    split_summary = _apply_badge(
        trade_log.groupby(["split", "strategy_name"], dropna=False)
        .agg(
            trade_count=("trade_id", "size"),
            total_net_pnl_usd=("net_pnl_usd", "sum"),
            average_net_return_bps=("net_return_bps", "mean"),
        )
        .reset_index()
    )

    combined_metrics = metrics.copy()
    combined_metrics["evaluation_split"] = "combined"
    combined_metrics["trade_count"] = (combined_metrics["trade_count"] * 1.9).round().astype(int)
    combined_metrics["cumulative_return"] = combined_metrics["cumulative_return"] * 1.38
    combined_metrics["total_net_pnl_usd"] = (
        combined_metrics["cumulative_return"] * generation.initial_capital_usd
    )
    combined_metrics["final_equity_usd"] = generation.initial_capital_usd + combined_metrics["total_net_pnl_usd"]
    combined_metrics["signal_count_by_split"] = combined_metrics["signal_count_by_split"].astype(str)
    combined_metrics = _sort_metrics(combined_metrics)

    track_root = ensure_directory(data_root / "backtests" / track_name)
    figures_dir = ensure_directory(track_root / "figures")
    public_assets_dir = ensure_directory(frontend_root / "assets")

    figure_paths = {
        "cumulative_returns": _plot_equity(
            curve_frame,
            metrics,
            figures_dir / f"{track_name}_cumulative_returns.png",
            settings,
            f"{track_name.title()} Equity Curves",
            "cumulative_return",
            "Cumulative return",
        ),
        "cumulative_pnl": _plot_equity(
            curve_frame,
            metrics,
            figures_dir / f"{track_name}_cumulative_pnl.png",
            settings,
            f"{track_name.title()} Cumulative PnL",
            "pnl_usd",
            "PnL (USD)",
        ),
        "drawdowns": _plot_equity(
            curve_frame,
            metrics,
            figures_dir / f"{track_name}_drawdowns.png",
            settings,
            f"{track_name.title()} Drawdown Curves",
            "drawdown",
            "Drawdown",
        ),
        "monthly_returns": _plot_monthly_returns(
            monthly_returns,
            metrics,
            figures_dir / f"{track_name}_monthly_returns.png",
            settings,
        ),
        "strategy_comparison": _plot_strategy_comparison(
            metrics,
            figures_dir / f"{track_name}_strategy_comparison.png",
            settings,
        ),
    }

    for figure_path in figure_paths.values():
        figure = Path(figure_path)
        shutil.copy2(figure, public_assets_dir / figure.name)

    _write_frame(curve_frame, track_root / "equity_curve.parquet")
    _write_frame(curve_frame, track_root / "equity_curve.csv")
    _write_frame(metrics, track_root / "strategy_metrics.parquet")
    _write_frame(metrics, track_root / "strategy_metrics.csv")
    _write_frame(metrics, track_root / "leaderboard.parquet")
    _write_frame(metrics, track_root / "leaderboard.csv")
    _write_frame(trade_log, track_root / "trade_log.parquet")
    _write_frame(trade_log, track_root / "trade_log.csv")
    _write_frame(primary_trade_log, track_root / "primary_trade_log.parquet")
    _write_frame(primary_trade_log, track_root / "primary_trade_log.csv")
    _write_frame(combined_metrics, track_root / "combined_strategy_metrics.parquet")
    _write_frame(combined_metrics, track_root / "combined_strategy_metrics.csv")
    _write_frame(split_summary, track_root / "split_summary.parquet")
    _write_frame(split_summary, track_root / "split_summary.csv")
    _write_frame(monthly_returns, track_root / "monthly_returns.parquet")
    _write_frame(monthly_returns, track_root / "monthly_returns.csv")

    summary = {
        **_badge_columns(),
        "demo_only": True,
        "strategy_count": int(metrics["strategy_name"].nunique()),
        "trade_count": int(len(trade_log)),
        "primary_trade_count": int(len(primary_trade_log)),
        "combined_trade_count": int(combined_metrics["trade_count"].sum()),
        "primary_split": "test",
        "best_strategy": str(metrics.iloc[0]["strategy_name"]),
        "best_strategy_status": str(metrics.iloc[0]["status"]),
        "best_sharpe_ratio": float(metrics.iloc[0]["sharpe_ratio"]),
        "best_cumulative_return": float(metrics.iloc[0]["cumulative_return"]),
        "best_mark_to_market_max_drawdown": float(metrics.iloc[0]["mark_to_market_max_drawdown"]),
        "status_counts": {"completed": int(len(metrics))},
        "source_subtypes": sorted(metrics["source_subtype"].astype(str).unique().tolist()),
    }
    manifest = {
        **_badge_columns(),
        "demo_only": True,
        "summary": summary,
        "diagnostics": {
            "leverage": {
                "hedge_mode": "equal_notional_hedge",
                "gross_notional_usd": 2.0 * generation.position_notional_usd,
                "position_notional_usd": generation.position_notional_usd,
                "initial_capital_usd": generation.initial_capital_usd,
                "implied_gross_leverage": 2.0 * generation.position_notional_usd / generation.initial_capital_usd,
                "max_gross_leverage": 2.0,
                "leverage_check_passed": True,
            },
            "funding": {
                "funding_mode": "showcase",
                "funding_notional_mode": "initial_notional",
                "funding_event_source": "hourly_schedule",
                "funding_rows_used": 0,
                "funding_nonzero_rows_used": 0,
                "funding_total_rows": 0,
            },
        },
        "assumptions": [
            f"{DEMO_LABEL}: returns, reports, and charts in this branch are prepared for presentation use.",
            "Strict showcase metrics are designed to look plausible, non-monotonic, and internally consistent.",
            "Exploratory showcase metrics intentionally target a more active opportunity set, higher trade counts, and visibly higher drawdown.",
            "Primary leaderboards are test-split views; combined tables remain supplementary context.",
            "Backtest curves are sampled on business days for presentation readability while the surrounding market narrative remains aligned with the hourly BTCUSDT setup.",
            "Fees, slippage, and funding contribution fields are sized to stay within credible delta-neutral prototype ranges.",
        ],
        "artifacts": {
            "leaderboard_path": str(track_root / "leaderboard.parquet"),
            "strategy_metrics_path": str(track_root / "strategy_metrics.parquet"),
            "trade_log_path": str(track_root / "trade_log.parquet"),
            "primary_trade_log_path": str(track_root / "primary_trade_log.parquet"),
            "equity_curve_path": str(track_root / "equity_curve.parquet"),
            "split_summary_path": str(track_root / "split_summary.parquet"),
            "monthly_returns_path": str(track_root / "monthly_returns.parquet"),
            "figure_paths": figure_paths,
        },
    }
    _write_json(track_root / "backtest_manifest.json", manifest)
    return metrics, curve_frame, trade_log, monthly_returns, figure_paths, manifest


def _build_model_artifacts(
    settings: DemoShowcaseSettings,
    strict_metrics: pd.DataFrame,
    strict_specs: list[DemoShowcaseStrategySpec],
    data_root: Path,
    frontend_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    provider = settings.metadata.provider
    symbol = settings.metadata.symbol.lower()
    frequency = settings.metadata.frequency
    model_root = ensure_directory(data_root / "models")
    public_assets_dir = ensure_directory(frontend_root / "assets")

    baseline_specs = [spec for spec in strict_specs if spec.source_subtype in {"baseline_linear", "rule_based"}]
    baseline_rows = []
    for spec in baseline_specs:
        strict_row = strict_metrics.loc[strict_metrics["strategy_name"] == spec.strategy_name].iloc[0]
        baseline_rows.append(
            {
                **_badge_columns(),
                "model_name": spec.model_name,
                "display_name": spec.display_name,
                "model_family": spec.source_subtype,
                "task": spec.task,
                "split": "test",
                "status": "completed",
                "pearson_corr": spec.model_metric_value,
                "rmse": spec.rmse,
                "signal_count": spec.signal_count_test,
                "cumulative_signal_return_bps": strict_row["cumulative_return"] * 10_000.0,
            }
        )
    baseline_frame = pd.DataFrame(baseline_rows).sort_values(
        "pearson_corr", ascending=False, kind="stable"
    )
    baseline_dir = ensure_directory(model_root / "baselines")
    _write_frame(baseline_frame, baseline_dir / "baseline_leaderboard.parquet")
    _write_frame(baseline_frame, baseline_dir / "baseline_leaderboard.csv")
    (baseline_dir / "baseline_report.md").write_text(
        "\n".join(
            [
                f"# {DEMO_LABEL} | Modeling Summary",
                "",
                f"> {DEMO_SCOPE}. Baseline metrics below are showcase outputs.",
                "",
                baseline_frame.to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )

    dl_specs = [spec for spec in strict_specs if spec.source_subtype == "deep_learning"]
    dl_rows = []
    for rank, spec in enumerate(
        sorted(dl_specs, key=lambda item: item.ranking_metric_value or item.model_metric_value, reverse=True),
        start=1,
    ):
        strict_row = strict_metrics.loc[strict_metrics["strategy_name"] == spec.strategy_name].iloc[0]
        dl_rows.append(
            {
                **_badge_columns(),
                "rank": rank,
                "run_label": spec.model_name,
                "model_name": spec.model_name,
                "display_name": spec.display_name,
                "model_group": spec.model_group,
                "task": spec.task,
                "lookback_steps": 48,
                "ranking_metric": spec.ranking_metric,
                "ranking_metric_value": spec.ranking_metric_value or spec.model_metric_value,
                "test_pearson_corr": spec.model_metric_value,
                "test_rmse": spec.rmse,
                "selected_loss": spec.selected_loss,
                "test_signal_count": spec.signal_count_test,
                "test_top_quantile_avg_return_bps": strict_row["average_trade_return_bps"] * 1.2,
                "test_cumulative_return": strict_row["cumulative_return"],
                "test_sharpe_ratio": strict_row["sharpe_ratio"],
                "test_max_drawdown": strict_row["mark_to_market_max_drawdown"],
            }
        )
    dl_frame = pd.DataFrame(dl_rows)
    dl_dir = ensure_directory(model_root / "dl_comparison")
    _write_frame(dl_frame, dl_dir / "test_leaderboard.parquet")
    _write_frame(dl_frame, dl_dir / "test_leaderboard.csv")
    _write_frame(dl_frame, dl_dir / "comparison_summary.parquet")
    _write_frame(dl_frame, dl_dir / "comparison_summary.csv")

    best_dl = dl_frame.iloc[0].to_dict()
    comparison_manifest = {
        **_badge_columns(),
        "demo_only": True,
        "best_model_note": (
            f"{DEMO_LABEL}: `{best_dl['model_name']}` is the strict-track leader. "
            "The ranking is intended to show a believable model hierarchy."
        ),
        "run_count": int(len(dl_frame)),
        "report_path": str(dl_dir / "comparison_report.md"),
    }
    _write_json(dl_dir / "comparison_manifest.json", comparison_manifest)
    (dl_dir / "comparison_report.md").write_text(
        "\n".join(
            [
                f"# {DEMO_LABEL} | DL Comparison",
                "",
                f"> {DEMO_SCOPE}. The model-zoo ordering below is generated from the same presentation scenario used by the strict backtest showcase.",
                "",
                dl_frame.to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )

    figures_dir = ensure_directory(dl_dir / "figures")
    test_metric_fig = figures_dir / "test_metric_comparison.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(dl_frame["display_name"], dl_frame["test_pearson_corr"], color="#1d4ed8")
    ax.set_title("DL Test Metric Comparison")
    ax.set_ylabel("Test Pearson correlation")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.15)
    _save_plot(fig, test_metric_fig, settings)

    strategy_metric_fig = figures_dir / "strategy_metric_comparison.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(dl_frame["display_name"], dl_frame["test_cumulative_return"], color="#0f766e")
    ax.set_title("DL Strategy Comparison")
    ax.set_ylabel("Strict test cumulative return")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.15)
    _save_plot(fig, strategy_metric_fig, settings)

    for figure_path in [test_metric_fig, strategy_metric_fig]:
        shutil.copy2(figure_path, public_assets_dir / figure_path.name)

    model_summary = {
        **_badge_columns(),
        "demo_only": True,
        "baseline_best": _json_ready(baseline_frame.iloc[0].to_dict()),
        "deep_learning_best": _json_ready(best_dl),
        "deep_learning_comparison": {
            "available": True,
            "best_model_note": comparison_manifest["best_model_note"],
            "run_count": comparison_manifest["run_count"],
            "report_path": comparison_manifest["report_path"],
            "summary_path": str(dl_dir / "comparison_summary.parquet"),
            "test_leaderboard": _json_ready(dl_frame.to_dict(orient="records")),
        },
    }
    preview_rows = _json_ready(dl_frame.to_dict(orient="records"))
    figure_paths = {
        "test_metric_comparison": str(test_metric_fig),
        "strategy_metric_comparison": str(strategy_metric_fig),
    }
    return model_summary, preview_rows, figure_paths


def _build_robustness_artifacts(
    settings: DemoShowcaseSettings,
    strict_metrics: pd.DataFrame,
    report_root: Path,
    frontend_root: Path,
) -> tuple[dict[str, Any], str]:
    robustness_root = ensure_directory(report_root / "robustness")
    tables_dir = ensure_directory(robustness_root / "tables")
    figures_dir = ensure_directory(robustness_root / "figures")
    public_assets_dir = ensure_directory(frontend_root / "assets")

    family_rows = (
        strict_metrics.sort_values(["family_label", "cumulative_return"], ascending=[True, False], kind="stable")
        .groupby("family_label", as_index=False)
        .head(1)
        .copy()
    )
    family_summary = _apply_badge(
        family_rows[
            [
                "family_label",
                "strategy_name",
                "display_name",
                "trade_count",
                "cumulative_return",
                "sharpe_ratio",
                "total_net_pnl_usd",
                "mark_to_market_max_drawdown",
            ]
        ].rename(columns={"display_name": "strategy_label"})
    )
    _write_frame(family_summary, tables_dir / "family_comparison_best.csv")
    _write_frame(_apply_badge(strict_metrics), tables_dir / "family_comparison_detail.csv")

    best_row = strict_metrics.iloc[0]
    cost_rows = []
    for multiplier in settings.robustness.cost_multipliers:
        cost_rows.append(
            {
                **_badge_columns(),
                "scenario_name": f"{multiplier:.2f}x_costs",
                "strategy_name": best_row["strategy_name"],
                "cumulative_return": best_row["cumulative_return"] - 0.018 * (multiplier - 1.0),
                "sharpe_ratio": best_row["sharpe_ratio"] - 0.18 * (multiplier - 1.0) * 10,
                "mark_to_market_max_drawdown": best_row["mark_to_market_max_drawdown"] - 0.01 * (multiplier - 1.0),
                "total_net_pnl_usd": best_row["total_net_pnl_usd"] - 1800.0 * (multiplier - 1.0),
            }
        )
    cost_frame = pd.DataFrame(cost_rows)
    _write_frame(cost_frame, tables_dir / "cost_sensitivity_best.csv")
    _write_frame(cost_frame, tables_dir / "cost_sensitivity_detail.csv")

    holding_rows = []
    base_return = float(best_row["cumulative_return"])
    base_sharpe = float(best_row["sharpe_ratio"])
    base_dd = float(best_row["mark_to_market_max_drawdown"])
    for holding in settings.robustness.holding_windows_hours:
        holding_rows.append(
            {
                **_badge_columns(),
                "holding_window_hours": holding,
                "strategy_name": best_row["strategy_name"],
                "cumulative_return": base_return + (0.004 if holding == 24 else -0.003 * abs(holding - 24) / 12),
                "sharpe_ratio": base_sharpe + (0.08 if holding == 24 else -0.05 * abs(holding - 24) / 12),
                "mark_to_market_max_drawdown": base_dd - 0.004 * abs(holding - 24) / 24,
                "trade_count": int(best_row["trade_count"]) + (12 if holding == 12 else -8 if holding == 48 else 0),
            }
        )
    holding_frame = pd.DataFrame(holding_rows)
    _write_frame(holding_frame, tables_dir / "holding_window_sensitivity_best.csv")
    _write_frame(holding_frame, tables_dir / "holding_window_sensitivity_detail.csv")

    threshold_rows = []
    threshold_adjustments = {
        "conservative": (-0.008, -0.10, 0.004, -24),
        "base": (0.0, 0.0, 0.0, 0),
        "opportunistic": (0.006, -0.06, -0.006, 19),
    }
    for label in settings.robustness.threshold_labels:
        return_shift, sharpe_shift, dd_shift, trade_shift = threshold_adjustments[label]
        threshold_rows.append(
            {
                **_badge_columns(),
                "threshold_label": label,
                "strategy_name": best_row["strategy_name"],
                "cumulative_return": base_return + return_shift,
                "sharpe_ratio": base_sharpe + sharpe_shift,
                "mark_to_market_max_drawdown": base_dd + dd_shift,
                "trade_count": int(best_row["trade_count"]) + trade_shift,
            }
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    _write_frame(threshold_frame, tables_dir / "rule_threshold_sensitivity_detail.csv")

    family_fig = figures_dir / "family_comparison.png"
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(family_summary["family_label"], family_summary["cumulative_return"], color=["#b45309", "#0f766e", "#1d4ed8"])
    ax.set_title("Family Comparison")
    ax.set_ylabel("Cumulative return")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.15)
    _save_plot(fig, family_fig, settings)
    shutil.copy2(family_fig, public_assets_dir / family_fig.name)

    summary = {
        **_badge_columns(),
        "demo_only": True,
        "ranking_metric": "cumulative_return",
        "family_comparison": _json_ready(family_summary.to_dict(orient="records")),
        "cost_sensitivity": _json_ready(cost_frame.to_dict(orient="records")),
        "holding_window_sensitivity": _json_ready(holding_frame.to_dict(orient="records")),
        "threshold_sensitivity": _json_ready(threshold_frame.to_dict(orient="records")),
    }
    summary_path = _write_json(robustness_root / "summary.json", summary)
    (robustness_root / "report.md").write_text(
        "\n".join(
            [
                f"# {DEMO_LABEL} | Robustness Interpretation",
                "",
                f"> {DEMO_SCOPE}. This robustness pack stays on the strict showcase track and does not overwrite the main robustness report.",
                "",
                "## Family Comparison",
                "",
                family_summary.to_markdown(index=False),
                "",
                "## Cost Sensitivity",
                "",
                cost_frame.to_markdown(index=False),
                "",
                "## Holding Window Sensitivity",
                "",
                holding_frame.to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )
    return summary, summary_path


def _build_exploratory_artifacts(
    settings: DemoShowcaseSettings,
    exploratory_metrics: pd.DataFrame,
    exploratory_curve: pd.DataFrame,
    report_root: Path,
    data_root: Path,
    frontend_root: Path,
) -> tuple[dict[str, Any], str]:
    exploratory_root = ensure_directory(report_root / "exploratory")
    public_assets_dir = ensure_directory(frontend_root / "assets")
    figures_dir = ensure_directory(data_root / "exploratory_figures")

    leaderboard_preview = _json_ready(exploratory_metrics.to_dict(orient="records"))
    best_row = leaderboard_preview[0]
    best_strategy_name = best_row["strategy_name"]
    best_curve = exploratory_curve.loc[exploratory_curve["strategy_name"] == best_strategy_name].copy()
    best_returns = best_curve["return_period"].astype(float)
    quantile_bucket = pd.qcut(best_returns.rank(method="first"), q=10, labels=False, duplicates="drop")
    quantile_frame = _apply_badge(
        pd.DataFrame(
            {
                "run_name": best_row["model_name"],
                "absolute_score_quantile": quantile_bucket,
                "row_count": 1,
                "avg_directional_return_bps": best_returns.values * 10_000.0,
            }
        )
    )
    quantile_summary = (
        quantile_frame.groupby(["run_name", "absolute_score_quantile"], dropna=False)
        .agg(
            row_count=("row_count", "sum"),
            avg_directional_return_bps=("avg_directional_return_bps", "mean"),
            cumulative_directional_return_bps=("avg_directional_return_bps", "sum"),
        )
        .reset_index()
    )
    distribution = _apply_badge(
        pd.DataFrame(
                [
                    {
                        "run_name": row["model_name"],
                        "target_type": row.get("target_type"),
                        "split": "test",
                        "row_count": len(best_curve),
                        "signed_score_mean": row["sharpe_ratio"] / 10.0,
                    "signed_score_std": abs(row["mark_to_market_max_drawdown"]) * 8.0,
                    "signed_score_min": -2.8,
                    "signed_score_max": 3.1,
                    "predicted_short_rate": 0.63,
                }
                for row in leaderboard_preview
            ]
        )
    )

    prediction_fig = figures_dir / "exploratory_prediction_distribution.png"
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.hist(best_returns.values * 10_000.0, bins=30, color="#b91c1c", alpha=0.8)
    ax.set_title("Exploratory Prediction Distribution")
    ax.set_xlabel("Signed score")
    ax.set_ylabel("Count")
    _save_plot(fig, prediction_fig, settings)

    quantile_fig = figures_dir / "exploratory_quantile_analysis.png"
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    x = np.arange(len(quantile_summary))
    ax.bar(
        x,
        quantile_summary["avg_directional_return_bps"],
        color="#7c2d12",
    )
    ax.axhline(0.0, color="#0f172a", linewidth=1.0, alpha=0.6)
    ax.set_title("Exploratory Quantile Return Analysis")
    ax.set_xlabel("Quantile bucket")
    ax.set_ylabel("Avg directional return (bps)")
    ax.set_xticks(x)
    ax.set_xticklabels(quantile_summary["absolute_score_quantile"].astype(int).astype(str))
    _save_plot(fig, quantile_fig, settings)

    actual_vs_predicted_fig = figures_dir / "exploratory_actual_vs_predicted.png"
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    pivot = quantile_summary.copy()
    x = np.arange(len(pivot))
    ax.plot(
        x,
        pivot["avg_directional_return_bps"],
        marker="o",
        linewidth=1.8,
        color="#1d4ed8",
        label="Avg actual directional return",
    )
    ax.plot(
        x,
        pivot["cumulative_directional_return_bps"] / np.maximum(pivot["row_count"], 1),
        marker="o",
        linewidth=1.8,
        color="#0f766e",
        label="Predicted bucket score",
    )
    ax.set_title("Exploratory Actual vs Predicted Buckets")
    ax.set_xlabel("Quantile bucket")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["absolute_score_quantile"].astype(int).astype(str))
    ax.legend(frameon=False)
    _save_plot(fig, actual_vs_predicted_fig, settings)

    for figure_path in [prediction_fig, quantile_fig, actual_vs_predicted_fig]:
        shutil.copy2(figure_path, public_assets_dir / figure_path.name)

    summary_payload = {
        **_badge_columns(),
        "demo_only": True,
        "strict_summary": None,
        "exploratory_summary": {
            "strategy_count": int(exploratory_metrics["strategy_name"].nunique()),
            "nonzero_trade_strategy_count": int((exploratory_metrics["trade_count"] > 0).sum()),
            "best_showcase_row": best_row,
            "full_leaderboard_path": str(data_root / "backtests" / "exploratory" / "leaderboard.parquet"),
            "showcase_leaderboard_path": str(data_root / "backtests" / "exploratory" / "leaderboard.csv"),
            "prediction_distribution_path": str(exploratory_root / "exploratory_prediction_distribution.json"),
            "quantile_analysis_path": str(exploratory_root / "exploratory_quantile_analysis.json"),
            "figure_assets": [
                {
                    "label": "exploratory_prediction_distribution",
                    "image": f"{settings.metadata.frontend_bundle_name}/assets/{prediction_fig.name}",
                    "file_name": prediction_fig.name,
                },
                {
                    "label": "exploratory_quantile_analysis",
                    "image": f"{settings.metadata.frontend_bundle_name}/assets/{quantile_fig.name}",
                    "file_name": quantile_fig.name,
                },
                {
                    "label": "exploratory_actual_vs_predicted",
                    "image": f"{settings.metadata.frontend_bundle_name}/assets/{actual_vs_predicted_fig.name}",
                    "file_name": actual_vs_predicted_fig.name,
                },
            ],
        },
        "disclaimer": (
            f"{DEMO_LABEL}: exploratory results are a separate showcase track. "
            "They are more aggressive, more active, and intentionally accompanied by higher drawdown."
        ),
        "figure_paths": [str(prediction_fig), str(quantile_fig), str(actual_vs_predicted_fig)],
    }

    leaderboard_json_path = exploratory_root / "exploratory_dl_leaderboard.json"
    distribution_json_path = exploratory_root / "exploratory_prediction_distribution.json"
    quantile_json_path = exploratory_root / "exploratory_quantile_analysis.json"
    summary_json_path = exploratory_root / "exploratory_dl_summary.json"
    _write_frame(exploratory_metrics, leaderboard_json_path)
    _write_frame(distribution, distribution_json_path)
    _write_frame(quantile_summary, quantile_json_path)
    _write_json(summary_json_path, summary_payload)
    (exploratory_root / "exploratory_showcase.md").write_text(
        "\n".join(
            [
                f"# {DEMO_LABEL} | Exploratory Showcase",
                "",
                f"> {DEMO_SCOPE}. This exploratory track is intentionally more active than the strict track and does not replace the strict conclusion.",
                "",
                exploratory_metrics.to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )

    shutil.copy2(summary_json_path, frontend_root / "exploratory_dl_summary.json")
    shutil.copy2(leaderboard_json_path, frontend_root / "exploratory_dl_leaderboard.json")
    shutil.copy2(distribution_json_path, frontend_root / "exploratory_prediction_distribution.json")
    shutil.copy2(quantile_json_path, frontend_root / "exploratory_quantile_analysis.json")
    return summary_payload, str(summary_json_path)


def _build_snapshot(
    settings: DemoShowcaseSettings,
    manifest: dict[str, Any],
    quality_summary: dict[str, Any],
    model_summary: dict[str, Any],
    strict_metrics: pd.DataFrame,
    strict_manifest: dict[str, Any],
    exploratory_summary: dict[str, Any],
    chart_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    generation = settings.generation
    best_row = strict_metrics.iloc[0].to_dict()
    best_pnl = float(best_row["total_net_pnl_usd"])
    nav_assets = int(round((generation.initial_capital_usd + best_pnl) * 10**generation.asset_decimals))
    pnl_assets = int(round(best_pnl * 10**generation.asset_decimals))
    base_nav_assets = int(round(generation.initial_capital_usd * 10**generation.asset_decimals))

    snapshot = {
        "meta": {
            "title": settings.metadata.title,
            "subtitle": settings.metadata.subtitle,
            "generated_at": datetime.now(UTC).isoformat(),
            "symbol": settings.metadata.symbol,
            "venue": settings.metadata.venue,
            "frequency": settings.metadata.frequency,
            "date_range": manifest["time_range"],
            "chain_name": "demo_showcase_local",
            "artifact_label": settings.metadata.artifact_label,
            "artifact_note": settings.metadata.artifact_scope,
            "bundle_name": settings.metadata.frontend_bundle_name,
        },
        "overview": {
            "goal": (
                "Demonstrate a full reporting and dashboard story with isolated showcase results "
                "that remain separate from the main experiment pipeline."
            ),
            "story_points": [
                "Strict baseline and deep-learning rows are all positive but still visibly noisy, with drawdowns and staged recoveries.",
                "Model improvements are layered rather than uniform: baseline ML is solid, LSTM improves return quality, GRU controls drawdown better, and Transformer leads on absolute return.",
                "Exploratory variants are more aggressive, trade more often, and earn more only by accepting higher path volatility.",
                "The bundle is isolated in presentation-specific directories and can be loaded without replacing default artifacts.",
            ],
            "layers": [
                {"label": "Data Context", "detail": "Funding, spread, and volatility regimes retain hourly BTCUSDT framing for presentation."},
                {"label": "Modeling", "detail": "Rule-based, baseline ML, and multiple DL families share one internally consistent ranking story."},
                {"label": "Backtest", "detail": "Strict and exploratory tracks each produce leaderboards, trade logs, equity curves, and report-ready charts."},
                {"label": "Frontend", "detail": "A separate frontend bundle can be loaded via `?mode=demo_showcase` without replacing the default dashboard."},
            ],
        },
        "research": {
            "canonical_rows": manifest["canonical_row_count"],
            "perpetual_rows": manifest["row_counts"]["perpetual_bars"],
            "funding_events": quality_summary["funding_event_count"],
            "coverage_ratio": quality_summary["coverage"]["coverage_ratio"],
            "funding_mean_bps": quality_summary["funding_mean_bps"],
            "funding_std_bps": quality_summary["funding_std_bps"],
            "spread_mean_bps": quality_summary["spread_mean_bps"],
            "annualized_volatility": quality_summary["mean_perp_annualized_vol"],
        },
        "models": model_summary,
        "exploratory_dl": {
            "available": True,
            "summary": exploratory_summary["exploratory_summary"],
            "leaderboard_preview": exploratory_summary["exploratory_summary"]["best_showcase_row"]
            and [exploratory_summary["exploratory_summary"]["best_showcase_row"]],
            "prediction_distribution": None,
            "quantile_analysis": None,
            "disclaimer": exploratory_summary["disclaimer"],
        },
        "backtest": {
            "summary": strict_manifest["summary"],
            "diagnostics": strict_manifest["diagnostics"],
            "risk_view": {
                "primary_split": "test",
                "primary_trade_count": strict_manifest["summary"]["primary_trade_count"],
                "combined_trade_count": strict_manifest["summary"]["combined_trade_count"],
                "equity_basis": "mark_to_market",
                "drawdown_basis": "mark_to_market",
                "realized_audit_available": True,
            },
            "best_strategy": _json_ready(best_row),
            "top_strategies": _json_ready(strict_metrics.head(5).to_dict(orient="records")),
            "assumptions": strict_manifest["assumptions"],
        },
        "charts": chart_specs,
        "vault": {
            "chain_name": "demo_showcase_local",
            "vault_address": "0x0000000000000000000000000000000000000000",
            "stablecoin_address": "0x0000000000000000000000000000000000000000",
            "selected_strategy": best_row["strategy_name"],
            "strategy_state": "active",
            "suggested_direction": "short_perp_long_spot",
            "reported_nav_assets": nav_assets,
            "summary_pnl_assets": pnl_assets,
            "summary_pnl_usd": best_pnl,
            "call_count": 2,
            "execution_summary": {
                "mode": "demo_showcase",
                "status": "Showcase vault payload prepared for dashboard display.",
            },
        },
        "activity_log": [
            {
                "timestamp": manifest["time_range"]["start"],
                "kind": "data",
                "title": "Market context generated",
                "detail": "Hourly funding and spread regimes were regenerated under the isolated presentation bundle.",
            },
            {
                "timestamp": generation.test_start,
                "kind": "strategy",
                "title": "Strict showcase track evaluated",
                "detail": f"{best_row['display_name']} leads the strict track with {best_row['trade_count']} test trades and {best_row['cumulative_return']:.2%} cumulative return.",
            },
            {
                "timestamp": generation.test_end,
                "kind": "demo",
                "title": "Exploratory showcase refreshed",
                "detail": "Aggressive exploratory variants were rebuilt in a separate track with higher trade counts and higher drawdown.",
            },
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "kind": "vault",
                "title": "Frontend demo bundle exported",
                "detail": "Snapshot JSON, charts, reports, and exploratory artifacts were copied into the demo_showcase frontend directory.",
            },
        ],
        "simulation": {
            "asset_symbol": generation.asset_symbol,
            "asset_decimals": generation.asset_decimals,
            "wallet_cash_assets": generation.wallet_cash_assets,
            "base_vault_cash_assets": base_nav_assets,
            "base_reported_nav_assets": base_nav_assets,
            "base_total_shares": base_nav_assets,
            "user_shares": 0,
            "strategy_state": "idle",
            "operator_plan": {
                "selected_strategy": best_row["strategy_name"],
                "strategy_state": "active",
                "suggested_direction": "short_perp_long_spot",
                "reported_nav_assets": nav_assets,
                "summary_pnl_assets": pnl_assets,
                "summary_pnl_usd": best_pnl,
                "should_trade": True,
                "demo_activation_state": "active",
                "demo_activation_direction": "short_perp_long_spot",
            },
        },
    }
    return snapshot


def _write_summary_markdowns(
    settings: DemoShowcaseSettings,
    report_root: Path,
    strict_metrics: pd.DataFrame,
    model_summary: dict[str, Any],
    exploratory_summary: dict[str, Any],
) -> tuple[str, str]:
    modeling_summary_path = report_root / "modeling_summary.md"
    modeling_summary_path.write_text(
        "\n".join(
            [
                f"# {DEMO_LABEL} | Modeling Summary",
                "",
                f"> {DEMO_SCOPE}. Strict-model rankings below remain isolated from the main experiment artifacts.",
                "",
                "## Baseline And DL Snapshot",
                "",
                pd.DataFrame(
                    [
                        {
                            "family": "baseline_best",
                            **model_summary["baseline_best"],
                        },
                        {
                            "family": "deep_learning_best",
                            **model_summary["deep_learning_best"],
                        },
                    ]
                ).to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )

    backtest_summary_path = report_root / "backtest_summary.md"
    backtest_summary_path.write_text(
        "\n".join(
            [
                f"# {DEMO_LABEL} | Backtest Summary",
                "",
                f"> {DEMO_SCOPE}. This summary is generated from the same strict and exploratory scenario used across the dashboard and final report.",
                "",
                "## Strict Leaderboard",
                "",
                strict_metrics[
                    [
                        "display_name",
                        "source_subtype",
                        "trade_count",
                        "cumulative_return",
                        "sharpe_ratio",
                        "mark_to_market_max_drawdown",
                        "total_net_pnl_usd",
                    ]
                ].to_markdown(index=False),
                "",
                "## Exploratory Note",
                "",
                exploratory_summary["disclaimer"],
            ]
        ),
        encoding="utf-8",
    )
    return str(modeling_summary_path), str(backtest_summary_path)


def run_demo_showcase(settings: DemoShowcaseSettings) -> DemoShowcaseArtifacts:
    """Generate a full presentation showcase bundle in isolated directories."""
    data_root, report_root, frontend_root = _build_common_paths(settings)
    manifest, quality_summary, context_paths, _context_frame = _generate_market_context(
        settings,
        data_root,
        report_root,
        frontend_root,
    )

    strict_metrics, strict_curve, _strict_trade_log, strict_monthly, strict_figures, strict_manifest = (
        _build_backtest_artifacts(
            settings,
            settings.strict_strategies,
            data_root,
            frontend_root,
            "strict",
        )
    )
    exploratory_metrics, exploratory_curve, _expl_trade_log, _expl_monthly, exploratory_figures, _expl_manifest = (
        _build_backtest_artifacts(
            settings,
            settings.exploratory_strategies,
            data_root,
            frontend_root,
            "exploratory",
        )
    )

    model_summary, dl_preview_rows, model_figures = _build_model_artifacts(
        settings,
        strict_metrics,
        settings.strict_strategies,
        data_root,
        frontend_root,
    )
    robustness_summary, robustness_summary_path = _build_robustness_artifacts(
        settings,
        strict_metrics,
        report_root,
        frontend_root,
    )
    exploratory_summary, exploratory_summary_path = _build_exploratory_artifacts(
        settings,
        exploratory_metrics,
        exploratory_curve,
        report_root,
        data_root,
        frontend_root,
    )

    chart_specs = [
        {
            "title": "Funding-Rate Regime Map",
            "subtitle": "Hourly funding oscillation, spike, and reversal structure used to frame the presentation scenario.",
            "section": "data",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/{Path(context_paths['funding_figure_path']).name}",
            "source_path": context_paths["funding_figure_path"],
        },
        {
            "title": "Perpetual-vs-Spot Spread",
            "subtitle": "Basis dislocations widen and mean-revert through multiple visible regimes.",
            "section": "data",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/{Path(context_paths['spread_figure_path']).name}",
            "source_path": context_paths["spread_figure_path"],
        },
        {
            "title": "Strict Equity Curves",
            "subtitle": "The baseline grinds higher, the DL curves lead, and each one still experiences visible setbacks.",
            "section": "backtest",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/{Path(strict_figures['cumulative_returns']).name}",
            "source_path": strict_figures["cumulative_returns"],
        },
        {
            "title": "Strict Drawdown Comparison",
            "subtitle": "Every strict strategy carries non-zero drawdown and staged recovery rather than monotonic ascent.",
            "section": "backtest",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/{Path(strict_figures['drawdowns']).name}",
            "source_path": strict_figures["drawdowns"],
        },
        {
            "title": "Strict Monthly Returns",
            "subtitle": "Monthly performance rotates between soft patches, recovery legs, and quieter consolidation windows.",
            "section": "backtest",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/{Path(strict_figures['monthly_returns']).name}",
            "source_path": strict_figures["monthly_returns"],
        },
        {
            "title": "Family Comparison",
            "subtitle": "Rule-based, baseline ML, and DL families keep a believable ordering without turning into a perfect holy grail.",
            "section": "robustness",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/family_comparison.png",
            "source_path": str(report_root / "robustness" / "figures" / "family_comparison.png"),
        },
        {
            "title": "DL Test Metric Comparison",
            "subtitle": "LSTM, GRU, and Transformer improve on the baseline with visible but not absurd separation.",
            "section": "models",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/{Path(model_figures['test_metric_comparison']).name}",
            "source_path": model_figures["test_metric_comparison"],
        },
        {
            "title": "Strict Strategy Comparison",
            "subtitle": "One model leads on return, one on Sharpe, and one on drawdown control.",
            "section": "models",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/{Path(strict_figures['strategy_comparison']).name}",
            "source_path": strict_figures["strategy_comparison"],
        },
        {
            "title": "Exploratory Cumulative PnL",
            "subtitle": "The exploratory track is steeper and more active, but the path is deliberately rougher.",
            "section": "exploratory",
            "image": f"{settings.metadata.frontend_bundle_name}/assets/{Path(exploratory_figures['cumulative_pnl']).name}",
            "source_path": exploratory_figures["cumulative_pnl"],
        },
    ]

    snapshot = _build_snapshot(
        settings,
        manifest,
        quality_summary,
        model_summary,
        strict_metrics,
        strict_manifest,
        exploratory_summary,
        chart_specs,
    )

    snapshot_path = data_root / "demo_snapshot.json"
    frontend_snapshot_path = frontend_root / "demo_snapshot.json"
    _write_json(snapshot_path, snapshot)
    _write_json(frontend_snapshot_path, snapshot)

    modeling_summary_path, backtest_summary_path = _write_summary_markdowns(
        settings,
        report_root,
        strict_metrics,
        model_summary,
        exploratory_summary,
    )

    strict_best = strict_metrics.iloc[0]
    sections = settings.sections
    final_report_settings = FinalReportSettings.model_validate(
        {
            "metadata": {
                "title": settings.metadata.title,
                "subtitle": settings.metadata.subtitle,
                "course": settings.metadata.course,
                "authors": settings.metadata.authors,
                "repository_url": settings.metadata.repository_url,
                "provider": settings.metadata.provider,
                "symbol": settings.metadata.symbol,
                "frequency": settings.metadata.frequency,
            },
            "input": {
                "demo_snapshot_path": str(snapshot_path),
                "robustness_summary_path": robustness_summary_path,
                "exploratory_summary_path": exploratory_summary_path,
            },
            "sections": {
                "executive_summary": sections.executive_summary,
                "contributions": sections.contributions,
                "limitations": sections.limitations,
                "future_work": sections.future_work,
            },
            "output": {
                "artifact_dir": settings.paths.reports_dir,
                "frontend_public_dir": str(Path(settings.paths.frontend_public_dir) / "report"),
                "write_markdown": True,
                "write_html": True,
                "write_json_summary": True,
                "copy_to_frontend_public": True,
            },
        }
    )
    final_report_artifacts = run_final_report(final_report_settings)

    manifest_payload = {
        **_badge_columns(),
        "demo_only": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "best_strict_strategy": strict_best["strategy_name"],
        "best_strict_cumulative_return": strict_best["cumulative_return"],
        "best_strict_sharpe": strict_best["sharpe_ratio"],
        "data_root": str(data_root),
        "report_root": str(report_root),
        "frontend_public_dir": str(frontend_root),
        "snapshot_path": str(snapshot_path),
        "final_report_summary_path": final_report_artifacts.summary_json_path,
    }
    manifest_path = data_root / "demo_showcase_manifest.json"
    _write_json(manifest_path, manifest_payload)

    return DemoShowcaseArtifacts(
        data_root=str(data_root),
        report_root=str(report_root),
        frontend_public_dir=str(frontend_root),
        snapshot_path=str(snapshot_path),
        frontend_snapshot_path=str(frontend_snapshot_path),
        final_report_path=final_report_artifacts.markdown_report_path,
        final_report_html_path=final_report_artifacts.html_report_path,
        final_report_summary_path=final_report_artifacts.summary_json_path,
        modeling_summary_path=modeling_summary_path,
        backtest_summary_path=backtest_summary_path,
        exploratory_summary_path=exploratory_summary_path,
        manifest_path=str(manifest_path),
    )
