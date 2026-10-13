"""Explicit delta-neutral prototype backtesting engine for perpetual funding-rate arbitrage."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funding_arb.config.models import BacktestSettings
from funding_arb.evaluation.metrics import (
    calculate_autocorrelation_adjusted_sharpe,
    calculate_max_consecutive_losses,
    calculate_max_drawdown,
    calculate_period_sharpe_ratio,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_total_return,
)
from funding_arb.utils.degeneracy import signal_split_diagnostics
from funding_arb.utils.paths import ensure_directory, repo_path

PLOT_COLORS = [
    "#0f766e",
    "#1d4ed8",
    "#b45309",
    "#b91c1c",
    "#7c3aed",
    "#0891b2",
    "#65a30d",
]
PERIODS_PER_YEAR_1H = 24 * 365
OPTIONAL_SIGNAL_DEFAULTS = {
    "signal_threshold": np.nan,
    "threshold_objective": None,
    "selected_threshold_objective_value": np.nan,
    "prediction_mode": None,
    "calibration_method": None,
    "feature_importance_method": None,
    "selected_hyperparameters_json": "{}",
    "checkpoint_selection_metric": None,
    "best_checkpoint_metric_value": np.nan,
    "checkpoint_selection_effective_metric": None,
    "best_checkpoint_effective_metric_value": np.nan,
    "checkpoint_selection_fallback_used": False,
    "selected_loss": None,
    "regression_loss": None,
    "use_balanced_classification_loss": False,
    "preprocessing_scaler": None,
    "winsorize_lower_quantile": np.nan,
    "winsorize_upper_quantile": np.nan,
}


@dataclass(frozen=True)
class BacktestArtifacts:
    """Files produced by the backtest pipeline."""

    output_dir: str
    trade_log_path: str
    trade_log_csv_path: str | None
    primary_trade_log_path: str
    primary_trade_log_csv_path: str | None
    equity_curve_path: str
    equity_curve_csv_path: str | None
    strategy_metrics_path: str
    strategy_metrics_csv_path: str | None
    combined_strategy_metrics_path: str | None
    combined_strategy_metrics_csv_path: str | None
    split_summary_path: str
    split_summary_csv_path: str | None
    leaderboard_path: str
    leaderboard_csv_path: str | None
    report_path: str | None
    figure_paths: list[str]
    manifest_path: str


@dataclass(frozen=True)
class OpenPosition:
    """Single delta-neutral position in the prototype backtester."""

    strategy_name: str
    source: str
    source_subtype: str
    task: str
    signal_timestamp: pd.Timestamp
    signal_split: str
    signal_score: float | None
    predicted_class: float | None
    expected_return_bps: float | None
    signal_threshold: float | None
    threshold_objective: str | None
    selected_threshold_objective_value: float | None
    prediction_mode: str | None
    calibration_method: str | None
    feature_importance_method: str | None
    selected_hyperparameters_json: str
    checkpoint_selection_metric: str | None
    best_checkpoint_metric_value: float | None
    checkpoint_selection_effective_metric: str | None
    best_checkpoint_effective_metric_value: float | None
    checkpoint_selection_fallback_used: bool | None
    selected_loss: str | None
    regression_loss: str | None
    use_balanced_classification_loss: bool | None
    preprocessing_scaler: str | None
    winsorize_lower_quantile: float | None
    winsorize_upper_quantile: float | None
    confidence: float | None
    metadata_json: str
    direction: str
    entry_signal_index: int
    entry_market_index: int
    entry_timestamp: pd.Timestamp
    position_notional_usd: float
    perp_entry_price_raw: float
    spot_entry_price_raw: float


def describe_backtest_job(config: BacktestSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the backtest job."""
    settings = config if isinstance(config, BacktestSettings) else BacktestSettings.model_validate(config)
    strategy_count = "all strategies" if not settings.selection.strategy_names else f"{len(settings.selection.strategy_names)} strategy filters"
    return (
        f"Backtest ready for {settings.input.symbol} on {settings.input.provider} at {settings.input.frequency}, "
        f"reading {settings.input.signal_path} and {settings.input.market_dataset_path}. "
        f"It will simulate {strategy_count} with holding window {settings.execution.holding_window_hours}h, "
        f"maximum holding {settings.execution.maximum_holding_hours}h, and write artifacts under "
        f"{settings.reporting.output_dir}/{settings.input.provider}/{settings.input.symbol.lower()}/"
        f"{settings.input.frequency}/{settings.reporting.run_name}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)


def _load_frame(path_text: str | Path) -> pd.DataFrame:
    path = _resolve_path(path_text)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_signal_table(settings: BacktestSettings) -> pd.DataFrame:
    frame = _load_frame(settings.input.signal_path)
    required = {
        "timestamp",
        "asset",
        "source",
        "source_subtype",
        "strategy_name",
        "task",
        "signal_score",
        "predicted_class",
        "expected_return_bps",
        "suggested_direction",
        "confidence",
        "should_trade",
        "split",
        "metadata_json",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Signal artifact is missing required columns: {', '.join(missing)}")
    for column, default in OPTIONAL_SIGNAL_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["strategy_name", "timestamp"]).reset_index(drop=True)
    split_filter = set(settings.selection.split_filter)
    if split_filter:
        frame = frame[frame["split"].isin(split_filter)].copy()
    if settings.selection.strategy_names:
        allowed = set(settings.selection.strategy_names)
        frame = frame[frame["strategy_name"].isin(allowed)].copy()
    if frame.empty:
        raise ValueError("No signal rows remain after applying backtest filters.")
    return frame


def _load_market_table(settings: BacktestSettings) -> pd.DataFrame:
    frame = _load_frame(settings.input.market_dataset_path)
    required = {
        "timestamp",
        "symbol",
        "venue",
        "frequency",
        "perp_open",
        "perp_close",
        "spot_open",
        "spot_close",
        "funding_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Market dataset is missing required columns: {', '.join(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted) or math.isinf(converted):
        return None
    return converted


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
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
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _constant_signal_value(values: pd.Series) -> Any:
    non_null = values.dropna()
    if non_null.empty:
        return None
    unique_values = pd.unique(non_null.astype(object))
    if len(unique_values) == 1:
        return unique_values[0]
    return "mixed"


def _signal_threshold_metadata(values: pd.Series) -> tuple[float | None, str]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None, "missing"
    rounded = pd.unique(numeric.round(10))
    if len(rounded) == 1:
        return float(rounded[0]), "constant"
    return None, "varying"


def _constant_numeric_value(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    rounded = pd.unique(numeric.round(10))
    if len(rounded) == 1:
        return float(rounded[0])
    return None


def _strategy_signal_metadata(strategy_frame: pd.DataFrame) -> dict[str, Any]:
    signal_threshold, signal_threshold_mode = _signal_threshold_metadata(strategy_frame["signal_threshold"])
    split_diagnostics = signal_split_diagnostics(
        strategy_frame,
        signal_column="should_trade",
        split_names=strategy_frame["split"].dropna().astype(str).unique().tolist(),
    )
    signal_count_by_split = {
        split_name: int(diagnostics.get("signal_count", 0))
        for split_name, diagnostics in split_diagnostics.items()
    }
    flagged = [
        (split_name, diagnostics)
        for split_name, diagnostics in split_diagnostics.items()
        if diagnostics.get("status") != "ok"
    ]
    return {
        "signal_threshold": signal_threshold,
        "signal_threshold_mode": signal_threshold_mode,
        "signal_count_by_split": signal_count_by_split,
        "signal_status": (
            "ok"
            if not flagged
            else "no_tradable_signals"
        ),
        "signal_status_reason": (
            None
            if not flagged
            else "; ".join(
                f"{split_name}: {diagnostics.get('reason') or diagnostics.get('status')}"
                for split_name, diagnostics in flagged
            )
        ),
        "threshold_objective": _safe_text(_constant_signal_value(strategy_frame["threshold_objective"])),
        "selected_threshold_objective_value": _constant_numeric_value(strategy_frame["selected_threshold_objective_value"]),
        "prediction_mode": _safe_text(_constant_signal_value(strategy_frame["prediction_mode"])),
        "calibration_method": _safe_text(_constant_signal_value(strategy_frame["calibration_method"])),
        "feature_importance_method": _safe_text(_constant_signal_value(strategy_frame["feature_importance_method"])),
        "selected_hyperparameters_json": _safe_text(_constant_signal_value(strategy_frame["selected_hyperparameters_json"])),
        "checkpoint_selection_metric": _safe_text(_constant_signal_value(strategy_frame["checkpoint_selection_metric"])),
        "best_checkpoint_metric_value": _constant_numeric_value(strategy_frame["best_checkpoint_metric_value"]),
        "checkpoint_selection_effective_metric": _safe_text(_constant_signal_value(strategy_frame["checkpoint_selection_effective_metric"])),
        "best_checkpoint_effective_metric_value": _constant_numeric_value(strategy_frame["best_checkpoint_effective_metric_value"]),
        "checkpoint_selection_fallback_used": _safe_bool(_constant_signal_value(strategy_frame["checkpoint_selection_fallback_used"])),
        "selected_loss": _safe_text(_constant_signal_value(strategy_frame["selected_loss"])),
        "regression_loss": _safe_text(_constant_signal_value(strategy_frame["regression_loss"])),
        "use_balanced_classification_loss": _safe_bool(_constant_signal_value(strategy_frame["use_balanced_classification_loss"])),
        "preprocessing_scaler": _safe_text(_constant_signal_value(strategy_frame["preprocessing_scaler"])),
        "winsorize_lower_quantile": _constant_numeric_value(strategy_frame["winsorize_lower_quantile"]),
        "winsorize_upper_quantile": _constant_numeric_value(strategy_frame["winsorize_upper_quantile"]),
    }


def _price_column(prefix: str, field: str) -> str:
    return f"{prefix}_{field}"


def _direction_signs(direction: str) -> tuple[int, int]:
    if direction == "short_perp_long_spot":
        return -1, 1
    if direction == "long_perp_short_spot":
        return 1, -1
    raise ValueError(f"Unsupported direction: {direction}")


def _effective_price(raw_price: float, position_sign: int, *, is_entry: bool, slippage_bps: float) -> float:
    slippage = float(slippage_bps) / 10_000.0
    if position_sign > 0:
        multiplier = 1.0 + slippage if is_entry else 1.0 - slippage
    else:
        multiplier = 1.0 - slippage if is_entry else 1.0 + slippage
    return float(raw_price) * multiplier


def _funding_rate_sum(cumulative_rates: pd.Series, start_index: int, end_index: int) -> float:
    """Return cumulative funding rate from start_index through end_index - 1."""
    if end_index <= start_index:
        return 0.0
    end_value = float(cumulative_rates.iloc[end_index - 1])
    start_value = float(cumulative_rates.iloc[start_index - 1]) if start_index > 0 else 0.0
    return end_value - start_value


def _funding_event_mask(market: pd.DataFrame, settings: BacktestSettings) -> tuple[pd.Series, str]:
    """Return rows used for funding accrual under the configured mode.

    `prototype_bar_sum` preserves the original prototype behavior by summing
    every aligned funding-rate row. `event_aware` uses an explicit event marker
    when the dataset has one, otherwise it falls back to UTC hour modulo the
    configured funding interval. That fallback matches Binance-style 8-hour
    funding events better than blindly summing every hourly row.
    """
    mode = settings.execution.funding_mode
    if mode == "prototype_bar_sum":
        return pd.Series(True, index=market.index), "all_aligned_rows"
    for marker in ("funding_event", "is_funding_event"):
        if marker in market.columns:
            return market[marker].fillna(False).astype(bool), marker
    timestamps = pd.to_datetime(market["timestamp"], utc=True)
    interval = int(settings.execution.funding_interval_hours)
    return pd.Series((timestamps.dt.hour % interval) == 0, index=market.index), "utc_hour_mod_interval_fallback"


def _funding_rate_series(market: pd.DataFrame, settings: BacktestSettings) -> pd.Series:
    rates = pd.to_numeric(market["funding_rate"], errors="coerce").fillna(0.0)
    mask, _ = _funding_event_mask(market, settings)
    return rates.where(mask, 0.0)


def _funding_mode_diagnostics(market: pd.DataFrame, settings: BacktestSettings) -> dict[str, Any]:
    rates = pd.to_numeric(market["funding_rate"], errors="coerce").fillna(0.0)
    mask, source = _funding_event_mask(market, settings)
    rows_used = int(mask.sum())
    return {
        "funding_mode": settings.execution.funding_mode,
        "funding_notional_mode": settings.execution.funding_notional_mode,
        "funding_event_source": source,
        "funding_rows_used": rows_used,
        "funding_nonzero_rows_used": int(((rates != 0.0) & mask).sum()),
        "funding_total_rows": int(len(market)),
    }


def _funding_rows_for_period(
    market: pd.DataFrame,
    start_index: int,
    end_index: int,
    settings: BacktestSettings,
) -> pd.DataFrame:
    if end_index <= start_index:
        return market.iloc[0:0].copy()
    rates = _funding_rate_series(market, settings)
    mask, _ = _funding_event_mask(market, settings)
    period = market.iloc[start_index:end_index].copy()
    period["funding_rate_for_backtest"] = rates.iloc[start_index:end_index].to_numpy(dtype=float)
    period["funding_event_for_backtest"] = mask.iloc[start_index:end_index].to_numpy(dtype=bool)
    return period


def _funding_pnl_for_period(
    *,
    direction: str,
    position_notional_usd: float,
    perp_entry_price_raw: float,
    market: pd.DataFrame,
    start_index: int,
    end_index: int,
    settings: BacktestSettings,
) -> dict[str, float]:
    """Calculate funding PnL under the configured funding/notional assumption."""
    period = _funding_rows_for_period(market, start_index, end_index, settings)
    if period.empty:
        return {
            "funding_rate_sum": 0.0,
            "funding_pnl_usd": 0.0,
            "funding_rows_used": 0.0,
            "funding_nonzero_rows_used": 0.0,
        }
    rates = pd.to_numeric(period["funding_rate_for_backtest"], errors="coerce").fillna(0.0)
    perp_sign, _ = _direction_signs(direction)
    funding_rate_sum = float(rates.sum())
    if settings.execution.funding_notional_mode == "initial_notional":
        funding_notional = pd.Series(float(position_notional_usd), index=period.index)
    else:
        entry_effective = _effective_price(
            float(perp_entry_price_raw),
            perp_sign,
            is_entry=True,
            slippage_bps=settings.costs.slippage_bps,
        )
        quantity = float(position_notional_usd) / entry_effective
        funding_notional = quantity * pd.to_numeric(period["perp_close"], errors="coerce").ffill().fillna(float(perp_entry_price_raw))
    funding_pnl_usd = (-perp_sign) * float((rates * funding_notional).sum())
    return {
        "funding_rate_sum": funding_rate_sum,
        "funding_pnl_usd": float(funding_pnl_usd),
        "funding_rows_used": float(period["funding_event_for_backtest"].sum()),
        "funding_nonzero_rows_used": float((rates != 0.0).sum()),
    }


def _gross_notional_usd(settings: BacktestSettings) -> float:
    if settings.execution.hedge_mode != "equal_notional_hedge":
        raise ValueError("Only equal_notional_hedge is implemented in the prototype backtester.")
    return 2.0 * float(settings.portfolio.position_notional) * int(settings.portfolio.max_open_positions)


def _leverage_diagnostics(settings: BacktestSettings) -> dict[str, Any]:
    gross_notional = _gross_notional_usd(settings)
    initial_capital = float(settings.portfolio.initial_capital)
    implied_gross_leverage = gross_notional / initial_capital
    max_gross_leverage = float(settings.portfolio.max_gross_leverage)
    diagnostics = {
        "hedge_mode": settings.execution.hedge_mode,
        "gross_notional_usd": float(gross_notional),
        "position_notional_usd": float(settings.portfolio.position_notional),
        "initial_capital_usd": initial_capital,
        "implied_gross_leverage": float(implied_gross_leverage),
        "max_gross_leverage": max_gross_leverage,
        "leverage_check_mode": settings.portfolio.leverage_check_mode,
        "leverage_check_passed": bool(implied_gross_leverage <= max_gross_leverage),
    }
    if implied_gross_leverage > max_gross_leverage and settings.portfolio.leverage_check_mode != "off":
        message = (
            f"Configured gross leverage {implied_gross_leverage:.2f}x exceeds "
            f"max_gross_leverage {max_gross_leverage:.2f}x."
        )
        if settings.portfolio.leverage_check_mode == "fail":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return diagnostics


def calculate_trade_pnl(
    *,
    direction: str,
    position_notional_usd: float,
    perp_entry_price_raw: float,
    perp_exit_price_raw: float,
    spot_entry_price_raw: float,
    spot_exit_price_raw: float,
    funding_rate_sum: float,
    taker_fee_bps: float,
    slippage_bps: float,
    gas_cost_usd: float,
    other_friction_bps: float,
    funding_pnl_usd_override: float | None = None,
) -> dict[str, float]:
    """Calculate explicit trade-level PnL for a delta-neutral position."""
    if position_notional_usd <= 0:
        raise ValueError("position_notional_usd must be positive.")
    prices = [perp_entry_price_raw, perp_exit_price_raw, spot_entry_price_raw, spot_exit_price_raw]
    if any(float(price) <= 0 for price in prices):
        raise ValueError("All trade prices must be strictly positive.")

    perp_sign, hedge_sign = _direction_signs(direction)
    perp_entry_price_effective = _effective_price(perp_entry_price_raw, perp_sign, is_entry=True, slippage_bps=slippage_bps)
    perp_exit_price_effective = _effective_price(perp_exit_price_raw, perp_sign, is_entry=False, slippage_bps=slippage_bps)
    spot_entry_price_effective = _effective_price(spot_entry_price_raw, hedge_sign, is_entry=True, slippage_bps=slippage_bps)
    spot_exit_price_effective = _effective_price(spot_exit_price_raw, hedge_sign, is_entry=False, slippage_bps=slippage_bps)

    perp_quantity = float(position_notional_usd) / perp_entry_price_effective
    spot_quantity = float(position_notional_usd) / spot_entry_price_effective

    perp_leg_pnl_usd = perp_quantity * (perp_exit_price_effective - perp_entry_price_effective)
    if perp_sign < 0:
        perp_leg_pnl_usd *= -1.0
    spot_leg_pnl_usd = spot_quantity * (spot_exit_price_effective - spot_entry_price_effective)
    if hedge_sign < 0:
        spot_leg_pnl_usd *= -1.0

    funding_pnl_usd = (
        float(funding_pnl_usd_override)
        if funding_pnl_usd_override is not None
        else (-perp_sign) * float(funding_rate_sum) * float(position_notional_usd)
    )
    trading_fees_usd = 4.0 * float(position_notional_usd) * (float(taker_fee_bps) / 10_000.0)
    gas_cost_usd = float(gas_cost_usd)
    other_friction_usd = float(position_notional_usd) * (float(other_friction_bps) / 10_000.0)
    turnover_usd = 4.0 * float(position_notional_usd)
    embedded_slippage_cost_usd = turnover_usd * (float(slippage_bps) / 10_000.0)

    gross_pnl_usd = perp_leg_pnl_usd + spot_leg_pnl_usd + funding_pnl_usd
    net_pnl_usd = gross_pnl_usd - trading_fees_usd - gas_cost_usd - other_friction_usd
    gross_return_bps = gross_pnl_usd / float(position_notional_usd) * 10_000.0
    net_return_bps = net_pnl_usd / float(position_notional_usd) * 10_000.0

    return {
        "perp_entry_price_effective": float(perp_entry_price_effective),
        "perp_exit_price_effective": float(perp_exit_price_effective),
        "spot_entry_price_effective": float(spot_entry_price_effective),
        "spot_exit_price_effective": float(spot_exit_price_effective),
        "perp_quantity": float(perp_quantity),
        "spot_quantity": float(spot_quantity),
        "perp_leg_pnl_usd": float(perp_leg_pnl_usd),
        "spot_leg_pnl_usd": float(spot_leg_pnl_usd),
        "funding_pnl_usd": float(funding_pnl_usd),
        "gross_pnl_usd": float(gross_pnl_usd),
        "trading_fees_usd": float(trading_fees_usd),
        "gas_cost_usd": float(gas_cost_usd),
        "other_friction_usd": float(other_friction_usd),
        "estimated_slippage_cost_usd": float(embedded_slippage_cost_usd),
        "embedded_slippage_cost_usd": float(embedded_slippage_cost_usd),
        "net_pnl_usd": float(net_pnl_usd),
        "gross_return_bps": float(gross_return_bps),
        "net_return_bps": float(net_return_bps),
        "turnover_usd": float(turnover_usd),
    }


def _entry_conditions_met(row: pd.Series, settings: BacktestSettings) -> bool:
    direction = str(row.get("suggested_direction", "flat"))
    configured_direction = str(settings.selection.direction)
    if configured_direction not in {"any", "both"} and direction != configured_direction:
        return False
    if settings.selection.require_should_trade and int(row.get("should_trade", 0)) != 1:
        return False
    if settings.selection.min_signal_score is not None:
        score = _safe_float(row.get("signal_score"))
        if score is None or score < float(settings.selection.min_signal_score):
            return False
    if settings.selection.min_confidence is not None:
        confidence = _safe_float(row.get("confidence"))
        if confidence is None or confidence < float(settings.selection.min_confidence):
            return False
    if settings.selection.min_expected_return_bps is not None:
        expected_return = _safe_float(row.get("expected_return_bps"))
        if expected_return is None or expected_return < float(settings.selection.min_expected_return_bps):
            return False
    return True


def _open_position(row: pd.Series, market: pd.DataFrame, entry_market_index: int, settings: BacktestSettings) -> OpenPosition:
    execution_price_field = settings.execution.execution_price_field
    perp_entry_price_raw = float(market.iloc[entry_market_index][_price_column("perp", execution_price_field)])
    spot_entry_price_raw = float(market.iloc[entry_market_index][_price_column("spot", execution_price_field)])
    return OpenPosition(
        strategy_name=str(row["strategy_name"]),
        source=str(row["source"]),
        source_subtype=str(row["source_subtype"]),
        task=str(row["task"]),
        signal_timestamp=pd.Timestamp(row["timestamp"]),
        signal_split=str(row["split"]),
        signal_score=_safe_float(row.get("signal_score")),
        predicted_class=_safe_float(row.get("predicted_class")),
        expected_return_bps=_safe_float(row.get("expected_return_bps")),
        signal_threshold=_safe_float(row.get("signal_threshold")),
        threshold_objective=_safe_text(row.get("threshold_objective")),
        selected_threshold_objective_value=_safe_float(row.get("selected_threshold_objective_value")),
        prediction_mode=_safe_text(row.get("prediction_mode")),
        calibration_method=_safe_text(row.get("calibration_method")),
        feature_importance_method=_safe_text(row.get("feature_importance_method")),
        selected_hyperparameters_json=_safe_text(row.get("selected_hyperparameters_json")) or "{}",
        checkpoint_selection_metric=_safe_text(row.get("checkpoint_selection_metric")),
        best_checkpoint_metric_value=_safe_float(row.get("best_checkpoint_metric_value")),
        checkpoint_selection_effective_metric=_safe_text(row.get("checkpoint_selection_effective_metric")),
        best_checkpoint_effective_metric_value=_safe_float(row.get("best_checkpoint_effective_metric_value")),
        checkpoint_selection_fallback_used=_safe_bool(row.get("checkpoint_selection_fallback_used")),
        selected_loss=_safe_text(row.get("selected_loss")),
        regression_loss=_safe_text(row.get("regression_loss")),
        use_balanced_classification_loss=_safe_bool(row.get("use_balanced_classification_loss")),
        preprocessing_scaler=_safe_text(row.get("preprocessing_scaler")),
        winsorize_lower_quantile=_safe_float(row.get("winsorize_lower_quantile")),
        winsorize_upper_quantile=_safe_float(row.get("winsorize_upper_quantile")),
        confidence=_safe_float(row.get("confidence")),
        metadata_json=str(row.get("metadata_json", "{}")),
        direction=str(row["suggested_direction"]),
        entry_signal_index=int(row["market_index"]),
        entry_market_index=int(entry_market_index),
        entry_timestamp=pd.Timestamp(market.iloc[entry_market_index]["timestamp"]),
        position_notional_usd=float(settings.portfolio.position_notional),
        perp_entry_price_raw=perp_entry_price_raw,
        spot_entry_price_raw=spot_entry_price_raw,
    )


def _mark_trade_return_bps(
    position: OpenPosition,
    market: pd.DataFrame,
    current_market_index: int,
    settings: BacktestSettings,
) -> float:
    funding = _funding_pnl_for_period(
        direction=position.direction,
        position_notional_usd=position.position_notional_usd,
        perp_entry_price_raw=position.perp_entry_price_raw,
        market=market,
        start_index=position.entry_market_index,
        end_index=current_market_index + 1,
        settings=settings,
    )
    pnl = calculate_trade_pnl(
        direction=position.direction,
        position_notional_usd=position.position_notional_usd,
        perp_entry_price_raw=position.perp_entry_price_raw,
        perp_exit_price_raw=float(market.iloc[current_market_index]["perp_close"]),
        spot_entry_price_raw=position.spot_entry_price_raw,
        spot_exit_price_raw=float(market.iloc[current_market_index]["spot_close"]),
        funding_rate_sum=funding["funding_rate_sum"],
        taker_fee_bps=settings.costs.taker_fee_bps,
        slippage_bps=settings.costs.slippage_bps,
        gas_cost_usd=settings.costs.gas_cost_usd,
        other_friction_bps=settings.costs.other_friction_bps,
        funding_pnl_usd_override=funding["funding_pnl_usd"],
    )
    return float(pnl["net_return_bps"])


def _close_position(
    position: OpenPosition,
    *,
    exit_market_index: int,
    exit_reason: str,
    observed_market_index: int,
    market: pd.DataFrame,
    settings: BacktestSettings,
) -> dict[str, Any]:
    execution_price_field = settings.execution.execution_price_field
    perp_exit_price_raw = float(market.iloc[exit_market_index][_price_column("perp", execution_price_field)])
    spot_exit_price_raw = float(market.iloc[exit_market_index][_price_column("spot", execution_price_field)])
    funding = _funding_pnl_for_period(
        direction=position.direction,
        position_notional_usd=position.position_notional_usd,
        perp_entry_price_raw=position.perp_entry_price_raw,
        market=market,
        start_index=position.entry_market_index,
        end_index=exit_market_index,
        settings=settings,
    )
    pnl = calculate_trade_pnl(
        direction=position.direction,
        position_notional_usd=position.position_notional_usd,
        perp_entry_price_raw=position.perp_entry_price_raw,
        perp_exit_price_raw=perp_exit_price_raw,
        spot_entry_price_raw=position.spot_entry_price_raw,
        spot_exit_price_raw=spot_exit_price_raw,
        funding_rate_sum=funding["funding_rate_sum"],
        taker_fee_bps=settings.costs.taker_fee_bps,
        slippage_bps=settings.costs.slippage_bps,
        gas_cost_usd=settings.costs.gas_cost_usd,
        other_friction_bps=settings.costs.other_friction_bps,
        funding_pnl_usd_override=funding["funding_pnl_usd"],
    )
    return {
        "strategy_name": position.strategy_name,
        "source": position.source,
        "source_subtype": position.source_subtype,
        "task": position.task,
        "signal_timestamp": position.signal_timestamp,
        "signal_split": position.signal_split,
        "entry_timestamp": position.entry_timestamp,
        "exit_timestamp": pd.Timestamp(market.iloc[exit_market_index]["timestamp"]),
        "entry_signal_index": int(position.entry_signal_index),
        "entry_market_index": int(position.entry_market_index),
        "observed_exit_signal_timestamp": pd.Timestamp(market.iloc[observed_market_index]["timestamp"]),
        "observed_exit_signal_index": int(observed_market_index),
        "exit_market_index": int(exit_market_index),
        "holding_hours": int(exit_market_index - position.entry_market_index),
        "entry_reason": "signal_entry",
        "exit_reason": exit_reason,
        "direction": position.direction,
        "position_notional_usd": float(position.position_notional_usd),
        "signal_score_at_entry": position.signal_score,
        "predicted_class_at_entry": position.predicted_class,
        "expected_return_bps_at_entry": position.expected_return_bps,
        "signal_threshold_at_entry": position.signal_threshold,
        "threshold_objective_at_entry": position.threshold_objective,
        "selected_threshold_objective_value_at_entry": position.selected_threshold_objective_value,
        "prediction_mode_at_entry": position.prediction_mode,
        "calibration_method_at_entry": position.calibration_method,
        "feature_importance_method_at_entry": position.feature_importance_method,
        "selected_hyperparameters_json_at_entry": position.selected_hyperparameters_json,
        "checkpoint_selection_metric_at_entry": position.checkpoint_selection_metric,
        "best_checkpoint_metric_value_at_entry": position.best_checkpoint_metric_value,
        "checkpoint_selection_effective_metric_at_entry": position.checkpoint_selection_effective_metric,
        "best_checkpoint_effective_metric_value_at_entry": position.best_checkpoint_effective_metric_value,
        "checkpoint_selection_fallback_used_at_entry": position.checkpoint_selection_fallback_used,
        "selected_loss_at_entry": position.selected_loss,
        "regression_loss_at_entry": position.regression_loss,
        "use_balanced_classification_loss_at_entry": position.use_balanced_classification_loss,
        "preprocessing_scaler_at_entry": position.preprocessing_scaler,
        "winsorize_lower_quantile_at_entry": position.winsorize_lower_quantile,
        "winsorize_upper_quantile_at_entry": position.winsorize_upper_quantile,
        "confidence_at_entry": position.confidence,
        "metadata_json_at_entry": position.metadata_json,
        "perp_entry_price_raw": float(position.perp_entry_price_raw),
        "perp_exit_price_raw": float(perp_exit_price_raw),
        "spot_entry_price_raw": float(position.spot_entry_price_raw),
        "spot_exit_price_raw": float(spot_exit_price_raw),
        "funding_rate_sum": float(funding["funding_rate_sum"]),
        "funding_rows_used": int(funding["funding_rows_used"]),
        "funding_nonzero_rows_used": int(funding["funding_nonzero_rows_used"]),
        "hedge_mode": settings.execution.hedge_mode,
        "funding_mode": settings.execution.funding_mode,
        "funding_notional_mode": settings.execution.funding_notional_mode,
        "stop_observation_mode": settings.execution.stop_observation_mode,
        "stop_execution_mode": settings.execution.stop_execution_mode,
        **pnl,
    }


def _simulate_strategy(
    strategy_frame: pd.DataFrame,
    market: pd.DataFrame,
    timestamp_to_market_index: dict[pd.Timestamp, int],
    settings: BacktestSettings,
) -> pd.DataFrame:
    if strategy_frame.empty:
        return pd.DataFrame()
    if strategy_frame["timestamp"].duplicated().any():
        raise ValueError(f"Strategy {strategy_frame['strategy_name'].iloc[0]} has duplicate timestamps in the signal table.")

    delay = int(settings.execution.entry_delay_bars)
    holding_hours = int(settings.execution.holding_window_hours)
    maximum_holding_hours = int(settings.execution.maximum_holding_hours)
    stop_loss_bps = settings.execution.stop_loss_bps
    take_profit_bps = settings.execution.take_profit_bps

    trades: list[dict[str, Any]] = []
    open_position: OpenPosition | None = None

    for _, row in strategy_frame.iterrows():
        current_market_index = timestamp_to_market_index.get(pd.Timestamp(row["timestamp"]))
        if current_market_index is None:
            continue

        exited_on_this_signal = False
        if open_position is not None:
            exit_candidates: list[tuple[int, int, str]] = []
            marked_return_bps = _mark_trade_return_bps(open_position, market, current_market_index, settings)

            if stop_loss_bps is not None and marked_return_bps <= -abs(float(stop_loss_bps)):
                exit_candidates.append((current_market_index + delay, 0, "stop_loss"))
            if take_profit_bps is not None and marked_return_bps >= abs(float(take_profit_bps)):
                exit_candidates.append((current_market_index + delay, 1, "take_profit"))

            planned_exit_market_index = open_position.entry_market_index + holding_hours
            if current_market_index + delay >= planned_exit_market_index:
                exit_candidates.append((planned_exit_market_index, 2, "holding_window"))

            maximum_exit_market_index = open_position.entry_market_index + maximum_holding_hours
            if current_market_index + delay >= maximum_exit_market_index:
                exit_candidates.append((maximum_exit_market_index, 3, "maximum_holding"))

            if settings.execution.exit_on_signal_off and not _entry_conditions_met(row, settings):
                exit_candidates.append((current_market_index + delay, 4, "signal_off"))

            valid_candidates = [candidate for candidate in exit_candidates if open_position.entry_market_index < candidate[0] < len(market)]
            if valid_candidates:
                exit_market_index, _, exit_reason = sorted(valid_candidates, key=lambda item: (item[0], item[1]))[0]
                trades.append(
                    _close_position(
                        open_position,
                        exit_market_index=exit_market_index,
                        exit_reason=exit_reason,
                        observed_market_index=current_market_index,
                        market=market,
                        settings=settings,
                    )
                )
                open_position = None
                exited_on_this_signal = True

        if exited_on_this_signal:
            continue

        if open_position is None and _entry_conditions_met(row, settings):
            entry_market_index = current_market_index + delay
            if entry_market_index >= len(market) - 1:
                continue
            open_position = _open_position(row, market, entry_market_index, settings)

    if open_position is not None:
        final_exit_market_index = len(market) - 1
        if final_exit_market_index > open_position.entry_market_index:
            trades.append(
                _close_position(
                    open_position,
                    exit_market_index=final_exit_market_index,
                    exit_reason="end_of_data",
                    observed_market_index=final_exit_market_index,
                    market=market,
                    settings=settings,
                )
            )

    trade_log = pd.DataFrame(trades)
    if trade_log.empty:
        return pd.DataFrame(
            columns=[
                "strategy_name",
                "source",
                "source_subtype",
                "task",
                "signal_timestamp",
                "signal_split",
                "entry_timestamp",
                "exit_timestamp",
                "entry_signal_index",
                "entry_market_index",
                "observed_exit_signal_timestamp",
                "observed_exit_signal_index",
                "exit_market_index",
                "holding_hours",
                "entry_reason",
                "exit_reason",
                "direction",
                "position_notional_usd",
                "signal_score_at_entry",
                "predicted_class_at_entry",
                "expected_return_bps_at_entry",
                "confidence_at_entry",
                "metadata_json_at_entry",
                "perp_entry_price_raw",
                "perp_exit_price_raw",
                "spot_entry_price_raw",
                "spot_exit_price_raw",
                "funding_rate_sum",
                "funding_rows_used",
                "funding_nonzero_rows_used",
                "hedge_mode",
                "funding_mode",
                "funding_notional_mode",
                "stop_observation_mode",
                "stop_execution_mode",
                "perp_entry_price_effective",
                "perp_exit_price_effective",
                "spot_entry_price_effective",
                "spot_exit_price_effective",
                "perp_quantity",
                "spot_quantity",
                "perp_leg_pnl_usd",
                "spot_leg_pnl_usd",
                "funding_pnl_usd",
                "gross_pnl_usd",
                "trading_fees_usd",
                "gas_cost_usd",
                "other_friction_usd",
                "estimated_slippage_cost_usd",
                "embedded_slippage_cost_usd",
                "net_pnl_usd",
                "gross_return_bps",
                "net_return_bps",
                "turnover_usd",
            ]
        )
    return trade_log.sort_values(["entry_timestamp", "exit_timestamp"]).reset_index(drop=True)


def build_realized_equity_curve(
    market_timestamps: pd.Series,
    trade_log: pd.DataFrame,
    *,
    initial_capital: float,
    strategy_name: str,
) -> pd.DataFrame:
    """Build a realized-PnL equity curve from closed trades only."""
    timestamps = pd.to_datetime(market_timestamps, utc=True)
    pnl_events = pd.Series(0.0, index=range(len(timestamps)), dtype="float64")
    if not trade_log.empty:
        for _, trade in trade_log.iterrows():
            pnl_events.iloc[int(trade["exit_market_index"])] += float(trade["net_pnl_usd"])
    equity = float(initial_capital) + pnl_events.cumsum()
    curve = pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_name": strategy_name,
            "realized_pnl_event_usd": pnl_events.to_numpy(dtype=float),
            "equity_usd": equity.to_numpy(dtype=float),
        }
    )
    curve["period_return"] = curve["equity_usd"].pct_change().fillna(0.0)
    curve["cumulative_return"] = curve["equity_usd"] / float(initial_capital) - 1.0
    curve["drawdown"] = curve["equity_usd"] / curve["equity_usd"].cummax() - 1.0
    return curve


def _mark_open_trade_pnl(
    trade: pd.Series,
    market: pd.DataFrame,
    current_market_index: int,
    settings: BacktestSettings,
) -> dict[str, float]:
    """Conservatively mark one open trade using current close prices."""
    funding = _funding_pnl_for_period(
        direction=str(trade["direction"]),
        position_notional_usd=float(trade["position_notional_usd"]),
        perp_entry_price_raw=float(trade["perp_entry_price_raw"]),
        market=market,
        start_index=int(trade["entry_market_index"]),
        end_index=current_market_index + 1,
        settings=settings,
    )
    return calculate_trade_pnl(
        direction=str(trade["direction"]),
        position_notional_usd=float(trade["position_notional_usd"]),
        perp_entry_price_raw=float(trade["perp_entry_price_raw"]),
        perp_exit_price_raw=float(market.iloc[current_market_index]["perp_close"]),
        spot_entry_price_raw=float(trade["spot_entry_price_raw"]),
        spot_exit_price_raw=float(market.iloc[current_market_index]["spot_close"]),
        funding_rate_sum=funding["funding_rate_sum"],
        taker_fee_bps=settings.costs.taker_fee_bps,
        slippage_bps=settings.costs.slippage_bps,
        gas_cost_usd=settings.costs.gas_cost_usd,
        other_friction_bps=settings.costs.other_friction_bps,
        funding_pnl_usd_override=funding["funding_pnl_usd"],
    )


def build_mark_to_market_equity_curve(
    market: pd.DataFrame,
    trade_log: pd.DataFrame,
    *,
    initial_capital: float,
    strategy_name: str,
    settings: BacktestSettings,
    curve_scope: str,
    start_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a mark-to-market equity curve while retaining realized audit columns."""
    timestamps = pd.to_datetime(market["timestamp"], utc=True)
    realized_events = pd.Series(0.0, index=range(len(timestamps)), dtype="float64")
    unrealized = pd.Series(0.0, index=range(len(timestamps)), dtype="float64")
    open_counts = pd.Series(0, index=range(len(timestamps)), dtype="int64")
    gross_exposure = pd.Series(0.0, index=range(len(timestamps)), dtype="float64")

    if not trade_log.empty:
        for _, trade in trade_log.iterrows():
            exit_index = int(trade["exit_market_index"])
            entry_index = int(trade["entry_market_index"])
            if 0 <= exit_index < len(realized_events):
                realized_events.iloc[exit_index] += float(trade["net_pnl_usd"])
            # Mark open positions from entry bar through the bar before realized exit.
            for market_index in range(max(entry_index, 0), min(exit_index, len(market))):
                mark = _mark_open_trade_pnl(trade, market, market_index, settings)
                unrealized.iloc[market_index] += float(mark["net_pnl_usd"])
                open_counts.iloc[market_index] += 1
                gross_exposure.iloc[market_index] += 2.0 * float(trade["position_notional_usd"])

    realized_equity = float(initial_capital) + realized_events.cumsum()
    mtm_equity = realized_equity + unrealized
    curve = pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_name": strategy_name,
            "curve_scope": curve_scope,
            "realized_pnl_event_usd": realized_events.to_numpy(dtype=float),
            "realized_equity_usd": realized_equity.to_numpy(dtype=float),
            "unrealized_pnl_usd": unrealized.to_numpy(dtype=float),
            "mark_to_market_equity_usd": mtm_equity.to_numpy(dtype=float),
            "open_position_count": open_counts.to_numpy(dtype=int),
            "gross_exposure_notional_usd": gross_exposure.to_numpy(dtype=float),
        }
    )
    curve["equity_usd"] = curve["mark_to_market_equity_usd"]
    curve["realized_period_return"] = curve["realized_equity_usd"].pct_change().fillna(0.0)
    curve["mark_to_market_period_return"] = curve["mark_to_market_equity_usd"].pct_change().fillna(0.0)
    curve["period_return"] = curve["mark_to_market_period_return"]
    curve["realized_cumulative_return"] = curve["realized_equity_usd"] / float(initial_capital) - 1.0
    curve["mark_to_market_cumulative_return"] = curve["mark_to_market_equity_usd"] / float(initial_capital) - 1.0
    curve["cumulative_return"] = curve["mark_to_market_cumulative_return"]
    curve["realized_drawdown"] = curve["realized_equity_usd"] / curve["realized_equity_usd"].cummax() - 1.0
    curve["mark_to_market_drawdown"] = (
        curve["mark_to_market_equity_usd"] / curve["mark_to_market_equity_usd"].cummax() - 1.0
    )
    curve["drawdown"] = curve["mark_to_market_drawdown"]
    curve["implied_gross_leverage"] = curve["gross_exposure_notional_usd"] / float(initial_capital)
    if start_timestamp is not None:
        start = pd.Timestamp(start_timestamp)
        curve = curve.loc[curve["timestamp"] >= start].reset_index(drop=True)
        for column in ("realized_period_return", "mark_to_market_period_return", "period_return"):
            if not curve.empty:
                curve.loc[curve.index[0], column] = 0.0
    return curve


def _filter_trades_for_split(trade_log: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "combined" or trade_log.empty or "signal_split" not in trade_log.columns:
        return trade_log.copy()
    return trade_log.loc[trade_log["signal_split"] == split].copy()


def _primary_split_start(strategy_frame: pd.DataFrame, split: str) -> pd.Timestamp | None:
    if split == "combined":
        return None
    split_rows = strategy_frame.loc[strategy_frame["split"] == split]
    if split_rows.empty:
        return None
    return pd.Timestamp(split_rows["timestamp"].min())


def _annualized_return(equity_curve: pd.DataFrame, initial_capital: float, *, equity_column: str = "equity_usd") -> float:
    if equity_curve.empty:
        return 0.0
    final_equity = float(equity_curve[equity_column].iloc[-1])
    periods = max(len(equity_curve) - 1, 0)
    if periods <= 0 or initial_capital <= 0 or final_equity <= 0:
        return 0.0
    total_growth = final_equity / float(initial_capital)
    return float(total_growth ** (PERIODS_PER_YEAR_1H / periods) - 1.0)


def summarize_strategy_backtest(
    *,
    strategy_name: str,
    source: str,
    source_subtype: str,
    task: str,
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    initial_capital: float,
    strategy_metadata: dict[str, Any] | None = None,
    evaluation_split: str = "combined",
) -> dict[str, Any]:
    """Summarize a backtest into report-friendly, strategy-aware metrics."""
    trade_count = int(len(trade_log))
    no_trades = trade_count == 0
    nan_metric = float("nan")
    win_rate = float((trade_log["net_pnl_usd"] > 0).mean()) if trade_count else nan_metric
    average_trade_return_bps = float(trade_log["net_return_bps"].mean()) if trade_count else nan_metric
    median_trade_return_bps = float(trade_log["net_return_bps"].median()) if trade_count else nan_metric
    average_trade_pnl_usd = float(trade_log["net_pnl_usd"].mean()) if trade_count else nan_metric
    expectancy_per_trade_usd = average_trade_pnl_usd
    expectancy_per_trade_bps = average_trade_return_bps
    total_turnover_usd = float(trade_log["turnover_usd"].sum()) if trade_count else 0.0
    total_fees_usd = float(trade_log["trading_fees_usd"].sum()) if trade_count else 0.0
    total_gas_cost_usd = float(trade_log["gas_cost_usd"].sum()) if trade_count else 0.0
    total_other_friction_usd = float(trade_log["other_friction_usd"].sum()) if trade_count else 0.0
    slippage_column = (
        "embedded_slippage_cost_usd"
        if "embedded_slippage_cost_usd" in trade_log.columns
        else "estimated_slippage_cost_usd"
        if "estimated_slippage_cost_usd" in trade_log.columns
        else None
    )
    total_embedded_slippage_cost_usd = float(trade_log[slippage_column].sum()) if trade_count and slippage_column else 0.0
    total_funding_pnl_usd = float(trade_log["funding_pnl_usd"].sum()) if trade_count else 0.0
    total_gross_pnl_usd = float(trade_log["gross_pnl_usd"].sum()) if trade_count else 0.0
    total_net_pnl_usd = float(trade_log["net_pnl_usd"].sum()) if trade_count else 0.0
    average_holding_hours = float(trade_log["holding_hours"].mean()) if trade_count else nan_metric
    median_holding_hours = float(trade_log["holding_hours"].median()) if trade_count else nan_metric
    profit_factor = calculate_profit_factor(trade_log["net_pnl_usd"]) if trade_count else nan_metric
    max_consecutive_losses = calculate_max_consecutive_losses(trade_log["net_pnl_usd"]) if trade_count else 0
    funding_contribution_share = (
        float(total_funding_pnl_usd / abs(total_net_pnl_usd)) if not math.isclose(total_net_pnl_usd, 0.0) else 0.0
    )

    mtm_equity_column = "mark_to_market_equity_usd" if "mark_to_market_equity_usd" in equity_curve.columns else "equity_usd"
    realized_equity_column = "realized_equity_usd" if "realized_equity_usd" in equity_curve.columns else "equity_usd"
    mtm_return_column = "mark_to_market_period_return" if "mark_to_market_period_return" in equity_curve.columns else "period_return"
    realized_return_column = "realized_period_return" if "realized_period_return" in equity_curve.columns else "period_return"

    final_equity = float(equity_curve[mtm_equity_column].iloc[-1]) if not equity_curve.empty else float(initial_capital)
    realized_final_equity = (
        float(equity_curve[realized_equity_column].iloc[-1]) if not equity_curve.empty else float(initial_capital)
    )
    total_return = calculate_total_return(equity_curve[mtm_return_column]) if not equity_curve.empty else 0.0
    realized_total_return = calculate_total_return(equity_curve[realized_return_column]) if not equity_curve.empty else 0.0
    annualized_return = _annualized_return(equity_curve, initial_capital, equity_column=mtm_equity_column)
    realized_annualized_return = _annualized_return(equity_curve, initial_capital, equity_column=realized_equity_column)
    sharpe_ratio = (
        calculate_sharpe_ratio(equity_curve[mtm_return_column], periods_per_year=PERIODS_PER_YEAR_1H)
        if not equity_curve.empty and not no_trades
        else nan_metric
    )
    raw_period_sharpe = (
        calculate_period_sharpe_ratio(equity_curve[mtm_return_column])
        if not equity_curve.empty and not no_trades
        else nan_metric
    )
    autocorr_adjusted_sharpe = (
        calculate_autocorrelation_adjusted_sharpe(equity_curve[mtm_return_column], periods_per_year=PERIODS_PER_YEAR_1H)
        if not equity_curve.empty and not no_trades
        else nan_metric
    )
    realized_sharpe_ratio = (
        calculate_sharpe_ratio(equity_curve[realized_return_column], periods_per_year=PERIODS_PER_YEAR_1H)
        if not equity_curve.empty and not no_trades
        else nan_metric
    )
    mark_to_market_max_drawdown = (
        calculate_max_drawdown(equity_curve[mtm_equity_column])
        if not equity_curve.empty and not no_trades
        else nan_metric
    )
    realized_max_drawdown = (
        calculate_max_drawdown(equity_curve[realized_equity_column])
        if not equity_curve.empty and not no_trades
        else nan_metric
    )
    max_drawdown = mark_to_market_max_drawdown
    exposure_time_fraction = (
        float((equity_curve["open_position_count"] > 0).mean())
        if not equity_curve.empty and "open_position_count" in equity_curve.columns
        else 0.0
    )
    average_gross_leverage = (
        float(equity_curve["implied_gross_leverage"].mean())
        if not equity_curve.empty and "implied_gross_leverage" in equity_curve.columns
        else 0.0
    )
    max_gross_leverage = (
        float(equity_curve["implied_gross_leverage"].max())
        if not equity_curve.empty and "implied_gross_leverage" in equity_curve.columns
        else 0.0
    )
    active_signal_count = int(trade_log["entry_timestamp"].count())
    metadata = strategy_metadata or {}
    signal_count_by_split = metadata.get("signal_count_by_split", {}) or {}
    evaluation_signal_count = signal_count_by_split.get(evaluation_split)
    status = "completed"
    diagnostic_reason = None
    if no_trades:
        if evaluation_signal_count == 0:
            status = "no_tradable_signals"
            diagnostic_reason = metadata.get("signal_status_reason") or (
                f"signal_count == 0 for evaluation split '{evaluation_split}'."
            )
        else:
            status = "no_executed_trades"
            diagnostic_reason = (
                f"No closed trades were executed for evaluation split '{evaluation_split}'."
            )
    return {
        "strategy_name": strategy_name,
        "source": source,
        "source_subtype": source_subtype,
        "task": task,
        "evaluation_split": evaluation_split,
        "status": status,
        "diagnostic_reason": diagnostic_reason,
        "skip_reason": diagnostic_reason if no_trades else None,
        "signal_threshold": metadata.get("signal_threshold"),
        "signal_threshold_mode": metadata.get("signal_threshold_mode"),
        "threshold_objective": metadata.get("threshold_objective"),
        "selected_threshold_objective_value": metadata.get("selected_threshold_objective_value"),
        "prediction_mode": metadata.get("prediction_mode"),
        "calibration_method": metadata.get("calibration_method"),
        "feature_importance_method": metadata.get("feature_importance_method"),
        "selected_hyperparameters_json": metadata.get("selected_hyperparameters_json"),
        "checkpoint_selection_metric": metadata.get("checkpoint_selection_metric"),
        "best_checkpoint_metric_value": metadata.get("best_checkpoint_metric_value"),
        "checkpoint_selection_effective_metric": metadata.get("checkpoint_selection_effective_metric"),
        "best_checkpoint_effective_metric_value": metadata.get("best_checkpoint_effective_metric_value"),
        "checkpoint_selection_fallback_used": metadata.get("checkpoint_selection_fallback_used"),
        "selected_loss": metadata.get("selected_loss"),
        "regression_loss": metadata.get("regression_loss"),
        "use_balanced_classification_loss": metadata.get("use_balanced_classification_loss"),
        "preprocessing_scaler": metadata.get("preprocessing_scaler"),
        "winsorize_lower_quantile": metadata.get("winsorize_lower_quantile"),
        "winsorize_upper_quantile": metadata.get("winsorize_upper_quantile"),
        "signal_count_by_split": json.dumps(
            metadata.get("signal_count_by_split", {}),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "trade_count": trade_count,
        "active_position_count": active_signal_count,
        "cumulative_return": float(total_return),
        "realized_cumulative_return": float(realized_total_return),
        "annualized_return": float(annualized_return),
        "realized_annualized_return": float(realized_annualized_return),
        "sharpe_ratio": float(sharpe_ratio),
        "simple_annualized_sharpe": float(sharpe_ratio),
        "raw_period_sharpe": float(raw_period_sharpe),
        "autocorr_adjusted_sharpe": float(autocorr_adjusted_sharpe),
        "realized_sharpe_ratio": float(realized_sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "realized_max_drawdown": float(realized_max_drawdown),
        "mark_to_market_max_drawdown": float(mark_to_market_max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "average_trade_return_bps": float(average_trade_return_bps),
        "median_trade_return_bps": float(median_trade_return_bps),
        "average_trade_pnl_usd": float(average_trade_pnl_usd),
        "expectancy_per_trade_usd": float(expectancy_per_trade_usd),
        "expectancy_per_trade_bps": float(expectancy_per_trade_bps),
        "average_holding_hours": float(average_holding_hours),
        "median_holding_hours": float(median_holding_hours),
        "max_consecutive_losses": int(max_consecutive_losses),
        "exposure_time_fraction": float(exposure_time_fraction),
        "average_gross_leverage": float(average_gross_leverage),
        "max_gross_leverage": float(max_gross_leverage),
        "total_turnover_usd": float(total_turnover_usd),
        "total_fees_usd": float(total_fees_usd),
        "total_gas_cost_usd": float(total_gas_cost_usd),
        "total_other_friction_usd": float(total_other_friction_usd),
        "total_embedded_slippage_cost_usd": float(total_embedded_slippage_cost_usd),
        "total_funding_pnl_usd": float(total_funding_pnl_usd),
        "funding_contribution_share": float(funding_contribution_share),
        "total_gross_pnl_usd": float(total_gross_pnl_usd),
        "total_net_pnl_usd": float(total_net_pnl_usd),
        "final_equity_usd": float(final_equity),
        "realized_final_equity_usd": float(realized_final_equity),
    }


def _split_trade_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "strategy_name",
        "signal_split",
        "source_subtype",
        "prediction_mode",
        "calibration_method",
        "checkpoint_selection_effective_metric",
        "selected_loss",
        "trade_count",
        "win_rate",
        "profit_factor",
        "average_trade_return_bps",
        "median_trade_return_bps",
        "average_holding_hours",
        "median_holding_hours",
        "max_consecutive_losses",
        "total_turnover_usd",
        "total_funding_pnl_usd",
        "total_net_pnl_usd",
    ]
    if trade_log.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (strategy_name, signal_split), group in trade_log.groupby(["strategy_name", "signal_split"], dropna=False):
        rows.append(
            {
                "strategy_name": strategy_name,
                "signal_split": signal_split,
                "source_subtype": _safe_text(_constant_signal_value(group["source_subtype"])),
                "prediction_mode": _safe_text(_constant_signal_value(group["prediction_mode_at_entry"])),
                "calibration_method": _safe_text(_constant_signal_value(group["calibration_method_at_entry"])),
                "checkpoint_selection_effective_metric": _safe_text(
                    _constant_signal_value(group["checkpoint_selection_effective_metric_at_entry"])
                ),
                "selected_loss": _safe_text(_constant_signal_value(group["selected_loss_at_entry"])),
                "trade_count": int(len(group)),
                "win_rate": float((group["net_pnl_usd"] > 0).mean()) if len(group) else 0.0,
                "profit_factor": float(calculate_profit_factor(group["net_pnl_usd"])) if len(group) else 0.0,
                "average_trade_return_bps": float(group["net_return_bps"].mean()) if len(group) else 0.0,
                "median_trade_return_bps": float(group["net_return_bps"].median()) if len(group) else 0.0,
                "average_holding_hours": float(group["holding_hours"].mean()) if len(group) else 0.0,
                "median_holding_hours": float(group["holding_hours"].median()) if len(group) else 0.0,
                "max_consecutive_losses": calculate_max_consecutive_losses(group["net_pnl_usd"]) if len(group) else 0,
                "total_turnover_usd": float(group["turnover_usd"].sum()),
                "total_funding_pnl_usd": float(group["funding_pnl_usd"].sum()),
                "total_net_pnl_usd": float(group["net_pnl_usd"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy_name", "signal_split"]).reset_index(drop=True)


def _sort_strategy_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank traded strategies ahead of no-trade placeholders, then by risk metrics."""
    if metrics.empty:
        return metrics
    ranked = metrics.copy()
    ranked["has_trades"] = ranked["trade_count"].astype(float) > 0.0
    return ranked.sort_values(
        ["has_trades", "sharpe_ratio", "cumulative_return"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _resolve_output_dir(settings: BacktestSettings) -> Path:
    return ensure_directory(
        _resolve_path(settings.reporting.output_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
        / settings.reporting.run_name
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return str(path)


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


def _apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _plot_cumulative_returns(equity_curve: pd.DataFrame, leaderboard: pd.DataFrame, output_path: Path, *, top_n: int, dpi: int) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    top = leaderboard.head(max(top_n, 1))
    for color_index, strategy_name in enumerate(top["strategy_name"].tolist()):
        curve = equity_curve[equity_curve["strategy_name"] == strategy_name]
        if curve.empty:
            continue
        ax.plot(
            curve["timestamp"],
            curve["cumulative_return"] * 100.0,
            label=strategy_name,
            linewidth=1.8,
            color=PLOT_COLORS[color_index % len(PLOT_COLORS)],
        )
    ax.set_title("Mark-to-Market Cumulative Return by Strategy")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xlabel("Timestamp")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_drawdowns(equity_curve: pd.DataFrame, leaderboard: pd.DataFrame, output_path: Path, *, top_n: int, dpi: int) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    top = leaderboard.head(max(top_n, 1))
    for color_index, strategy_name in enumerate(top["strategy_name"].tolist()):
        curve = equity_curve[equity_curve["strategy_name"] == strategy_name]
        if curve.empty:
            continue
        ax.plot(
            curve["timestamp"],
            curve["drawdown"] * 100.0,
            label=strategy_name,
            linewidth=1.5,
            color=PLOT_COLORS[color_index % len(PLOT_COLORS)],
        )
    ax.set_title("Mark-to-Market Drawdown by Strategy")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Timestamp")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_trade_return_boxplot(
    trade_log: pd.DataFrame,
    leaderboard: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int,
    dpi: int,
    title: str = "Trade Return Distribution",
) -> str:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    top_names = leaderboard.head(max(top_n, 1))["strategy_name"].tolist()
    if trade_log.empty or "strategy_name" not in trade_log.columns or "net_return_bps" not in trade_log.columns:
        ax.text(0.5, 0.5, "No trades available", ha="center", va="center")
        ax.set_axis_off()
    else:
        filtered = trade_log[trade_log["strategy_name"].isin(top_names)].copy()
        if filtered.empty:
            ax.text(0.5, 0.5, "No trades available", ha="center", va="center")
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            return str(output_path)
        ordered_groups = [filtered.loc[filtered["strategy_name"] == name, "net_return_bps"].astype(float).to_numpy() for name in top_names]
        ax.boxplot(ordered_groups, tick_labels=top_names, patch_artist=True)
        ax.set_ylabel("Net trade return (bps)")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _load_manifest(path_text: str | Path | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_markdown_report(
    settings: BacktestSettings,
    strategy_metrics: pd.DataFrame,
    split_summary: pd.DataFrame,
    figure_paths: list[str],
    signal_manifest: dict[str, Any] | None,
    market_manifest: dict[str, Any] | None,
    leverage_diagnostics: dict[str, Any],
    funding_diagnostics: dict[str, Any],
) -> str:
    assumptions = [
        f"Signals are observed at timestamp `t` and executed after `{settings.execution.entry_delay_bars}` bar(s) using `{settings.execution.execution_price_field}` prices.",
        f"Direction is fixed to `{settings.selection.direction}` for this run.",
        f"Hedge mode is `{settings.execution.hedge_mode}`; the implemented prototype uses equal USD notional on perp and spot legs.",
        f"Delta-neutral abstraction uses fixed per-leg notional of `{settings.portfolio.position_notional}` USD.",
        f"Primary leaderboard split is `{settings.reporting.primary_split}`; combined/in-sample behavior should be treated as secondary diagnostics.",
        "Primary risk metrics use mark-to-market equity. Realized-only equity remains in the artifact for auditability.",
        f"Funding mode is `{settings.execution.funding_mode}` with `{settings.execution.funding_notional_mode}` funding notional.",
        "PnL includes explicit perp-leg, spot-leg, and funding components.",
        f"Trading fees use taker fee `{settings.costs.taker_fee_bps}` bps on all four round-trip transactions.",
        f"Slippage is modeled through adverse execution prices using `{settings.costs.slippage_bps}` bps. `embedded_slippage_cost_usd` is a diagnostic, not a second deduction.",
        f"Gas cost is `{settings.costs.gas_cost_usd}` USD per closed trade.",
        f"Stop logic is `{settings.execution.stop_observation_mode}` and `{settings.execution.stop_execution_mode}`, not intrabar execution.",
        "Simple annualized Sharpe uses square-root scaling and may be distorted by sparse, serially correlated trading returns.",
        "When a strategy has zero executable trades on the evaluation split, Sharpe and drawdown fields are reported as NaN because they are not meaningful for a flat path.",
    ]
    figure_markdown = "\n".join(f"![{Path(path).stem}](figures/{Path(path).name})" for path in figure_paths)
    manifest_lines = ""
    if signal_manifest is not None:
        manifest_lines += f"- Signal rows: `{signal_manifest.get('summary', {}).get('row_count', 'n/a')}`\n"
        manifest_lines += f"- Active signal count: `{signal_manifest.get('summary', {}).get('active_signal_count', 'n/a')}`\n"
        manifest_lines += f"- Signal status: `{signal_manifest.get('status', 'n/a')}`\n"
        manifest_lines += f"- Signal reason: `{signal_manifest.get('reason', 'n/a')}`\n"
        manifest_lines += f"- Signal prediction modes: `{signal_manifest.get('summary', {}).get('prediction_modes', [])}`\n"
        manifest_lines += f"- Signal calibration methods: `{signal_manifest.get('summary', {}).get('calibration_methods', [])}`\n"
        manifest_lines += f"- Signal checkpoint metrics: `{signal_manifest.get('summary', {}).get('checkpoint_selection_metrics', [])}`\n"
        manifest_lines += f"- Signal effective checkpoint metrics: `{signal_manifest.get('summary', {}).get('checkpoint_selection_effective_metrics', [])}`\n"
        manifest_lines += f"- Signal selected losses: `{signal_manifest.get('summary', {}).get('selected_losses', [])}`\n"
        manifest_lines += f"- Signal preprocessing scalers: `{signal_manifest.get('summary', {}).get('preprocessing_scalers', [])}`\n"
    if market_manifest is not None:
        manifest_lines += f"- Canonical market rows: `{market_manifest.get('canonical_row_count', 'n/a')}`\n"
    diagnostics_lines = "\n".join(
        [
            f"- Implied gross leverage: `{leverage_diagnostics.get('implied_gross_leverage', 'n/a')}`",
            f"- Max gross leverage guard: `{leverage_diagnostics.get('max_gross_leverage', 'n/a')}`",
            f"- Leverage check passed: `{leverage_diagnostics.get('leverage_check_passed', 'n/a')}`",
            f"- Funding event source: `{funding_diagnostics.get('funding_event_source', 'n/a')}`",
            f"- Funding rows used: `{funding_diagnostics.get('funding_rows_used', 'n/a')}`",
            f"- Nonzero funding rows used: `{funding_diagnostics.get('funding_nonzero_rows_used', 'n/a')}`",
        ]
    )

    top_metrics = (
        strategy_metrics[
            [
                "strategy_name",
                "source_subtype",
                "evaluation_split",
                "status",
                "diagnostic_reason",
                "prediction_mode",
                "calibration_method",
                "checkpoint_selection_effective_metric",
                "selected_loss",
                "preprocessing_scaler",
                "signal_threshold",
                "has_trades",
                "trade_count",
                "cumulative_return",
                "annualized_return",
                "sharpe_ratio",
                "raw_period_sharpe",
                "autocorr_adjusted_sharpe",
                "max_drawdown",
                "realized_max_drawdown",
                "mark_to_market_max_drawdown",
                "win_rate",
                "profit_factor",
                "average_trade_return_bps",
                "median_trade_return_bps",
                "exposure_time_fraction",
                "total_net_pnl_usd",
            ]
        ].copy()
        if not strategy_metrics.empty
        else pd.DataFrame()
    )
    primary_trade_count = int(strategy_metrics["trade_count"].sum()) if "trade_count" in strategy_metrics.columns else 0
    primary_warning = (
        f"\n> Note: the primary split `{settings.reporting.primary_split}` produced no closed trades for the current signals. "
        "Use `split_summary` and `combined_strategy_metrics` as secondary diagnostics, not as the primary out-of-sample result.\n"
        if primary_trade_count == 0 and settings.reporting.primary_split != "combined"
        else ""
    )

    return f"""# Backtest Report

## Overview

- Symbol: `{settings.input.symbol}`
- Provider: `{settings.input.provider}`
- Frequency: `{settings.input.frequency}`
- Signal artifact: `{settings.input.signal_path}`
- Market dataset: `{settings.input.market_dataset_path}`
- Output root: `{settings.reporting.output_dir}`
{manifest_lines}
## Simplifying Assumptions

""" + "\n".join(f"- {line}" for line in assumptions) + f"""

## Run Diagnostics

{diagnostics_lines}

## Equity And Risk Notes

- `mark_to_market_equity_usd` marks open positions on every bar and is the source for primary drawdown and Sharpe metrics.
- `realized_equity_usd` only changes when trades close. It is useful for audit trails, but can understate intratrade risk.
- `embedded_slippage_cost_usd` is the preferred diagnostic implied by adverse execution prices; legacy `estimated_slippage_cost_usd` is retained only as an alias.
{primary_warning}

## Strategy Summary

{_dataframe_to_markdown(top_metrics.round(6))}

## Split Summary

{_dataframe_to_markdown(split_summary.round(6))}

## Figures

{figure_markdown}
"""


def run_backtest_pipeline(settings: BacktestSettings) -> BacktestArtifacts:
    """Run the explicit delta-neutral prototype backtest."""
    if int(settings.portfolio.max_open_positions) != 1:
        raise ValueError("The first backtest version supports max_open_positions = 1 only.")
    if settings.execution.allow_partial_exit:
        raise ValueError("The first backtest version does not support partial exits.")
    if settings.execution.maximum_holding_hours < settings.execution.holding_window_hours:
        raise ValueError("maximum_holding_hours must be greater than or equal to holding_window_hours.")
    leverage_diagnostics = _leverage_diagnostics(settings)

    signals = _load_signal_table(settings)
    market = _load_market_table(settings)
    timestamp_to_market_index = {
        pd.Timestamp(timestamp): index for index, timestamp in enumerate(pd.to_datetime(market["timestamp"], utc=True))
    }
    signals = signals[signals["timestamp"].isin(timestamp_to_market_index)].copy()
    if signals.empty:
        raise ValueError("No signal timestamps overlap with the market dataset.")

    selected_market = market.loc[market["timestamp"] >= signals["timestamp"].min()].reset_index(drop=True)
    selected_index_map = {pd.Timestamp(timestamp): index for index, timestamp in enumerate(pd.to_datetime(selected_market["timestamp"], utc=True))}
    signals = signals[signals["timestamp"].isin(selected_index_map)].copy()
    signals["market_index"] = signals["timestamp"].map(selected_index_map).astype(int)
    funding_diagnostics = _funding_mode_diagnostics(selected_market, settings)

    trade_logs: list[pd.DataFrame] = []
    primary_trade_logs: list[pd.DataFrame] = []
    equity_curves: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    combined_metric_rows: list[dict[str, Any]] = []

    for strategy_name, strategy_frame in signals.groupby("strategy_name", sort=True):
        strategy_frame = strategy_frame.sort_values("timestamp").reset_index(drop=True)
        strategy_metadata = _strategy_signal_metadata(strategy_frame)
        trade_log = _simulate_strategy(strategy_frame, selected_market, selected_index_map, settings)
        if not trade_log.empty:
            trade_log["trade_id"] = np.arange(1, len(trade_log) + 1, dtype=int)
        trade_logs.append(trade_log)

        strategy_source = str(strategy_frame["source"].iloc[0])
        strategy_subtype = str(strategy_frame["source_subtype"].iloc[0])
        strategy_task = str(strategy_frame["task"].iloc[0])
        primary_trade_log = _filter_trades_for_split(trade_log, settings.reporting.primary_split)
        primary_trade_logs.append(primary_trade_log)
        primary_start = _primary_split_start(strategy_frame, settings.reporting.primary_split)
        equity_curve = build_mark_to_market_equity_curve(
            selected_market,
            primary_trade_log,
            initial_capital=settings.portfolio.initial_capital,
            strategy_name=strategy_name,
            settings=settings,
            curve_scope=settings.reporting.primary_split,
            start_timestamp=primary_start,
        )
        equity_curve["source"] = strategy_source
        equity_curve["source_subtype"] = strategy_subtype
        equity_curves.append(equity_curve)
        metric_rows.append(
            summarize_strategy_backtest(
                strategy_name=strategy_name,
                source=strategy_source,
                source_subtype=strategy_subtype,
                task=strategy_task,
                equity_curve=equity_curve,
                trade_log=primary_trade_log,
                initial_capital=settings.portfolio.initial_capital,
                strategy_metadata=strategy_metadata,
                evaluation_split=settings.reporting.primary_split,
            )
        )
        if settings.reporting.include_combined_summary and settings.reporting.primary_split != "combined":
            combined_equity_curve = build_mark_to_market_equity_curve(
                selected_market,
                trade_log,
                initial_capital=settings.portfolio.initial_capital,
                strategy_name=strategy_name,
                settings=settings,
                curve_scope="combined",
            )
            combined_metric_rows.append(
                summarize_strategy_backtest(
                    strategy_name=strategy_name,
                    source=strategy_source,
                    source_subtype=strategy_subtype,
                    task=strategy_task,
                    equity_curve=combined_equity_curve,
                    trade_log=trade_log,
                    initial_capital=settings.portfolio.initial_capital,
                    strategy_metadata=strategy_metadata,
                    evaluation_split="combined",
                )
            )

    non_empty_trade_logs = [frame for frame in trade_logs if not frame.empty]
    if non_empty_trade_logs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            trade_log_frame = pd.concat(non_empty_trade_logs, ignore_index=True)
    else:
        trade_log_frame = pd.DataFrame()
    non_empty_primary_trade_logs = [frame for frame in primary_trade_logs if not frame.empty]
    if non_empty_primary_trade_logs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            primary_trade_log_frame = pd.concat(non_empty_primary_trade_logs, ignore_index=True)
    elif not trade_log_frame.empty:
        primary_trade_log_frame = trade_log_frame.iloc[0:0].copy()
    else:
        primary_trade_log_frame = pd.DataFrame()
    equity_curve_frame = pd.concat(equity_curves, ignore_index=True) if equity_curves else pd.DataFrame()
    strategy_metrics = (
        _sort_strategy_metrics(pd.DataFrame(metric_rows))
        if metric_rows
        else pd.DataFrame()
    )
    combined_strategy_metrics = (
        _sort_strategy_metrics(pd.DataFrame(combined_metric_rows))
        if combined_metric_rows
        else pd.DataFrame()
    )
    split_summary = _split_trade_summary(trade_log_frame)
    leaderboard = (
        strategy_metrics[
            [
                "strategy_name",
                "source",
                "source_subtype",
                "task",
                "evaluation_split",
                "status",
                "diagnostic_reason",
                "skip_reason",
                "signal_threshold",
                "signal_threshold_mode",
                "threshold_objective",
                "selected_threshold_objective_value",
                "prediction_mode",
                "calibration_method",
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
                "signal_count_by_split",
                "has_trades",
                "trade_count",
                "cumulative_return",
                "realized_cumulative_return",
                "annualized_return",
                "realized_annualized_return",
                "sharpe_ratio",
                "simple_annualized_sharpe",
                "raw_period_sharpe",
                "autocorr_adjusted_sharpe",
                "realized_sharpe_ratio",
                "max_drawdown",
                "realized_max_drawdown",
                "mark_to_market_max_drawdown",
                "win_rate",
                "profit_factor",
                "average_trade_return_bps",
                "median_trade_return_bps",
                "expectancy_per_trade_usd",
                "expectancy_per_trade_bps",
                "median_holding_hours",
                "max_consecutive_losses",
                "exposure_time_fraction",
                "average_gross_leverage",
                "max_gross_leverage",
                "total_net_pnl_usd",
                "total_funding_pnl_usd",
                "funding_contribution_share",
                "total_embedded_slippage_cost_usd",
                "final_equity_usd",
                "realized_final_equity_usd",
            ]
        ].copy()
        if not strategy_metrics.empty
        else pd.DataFrame()
    )

    output_dir = _resolve_output_dir(settings)
    figures_dir = ensure_directory(output_dir / "figures")

    trade_log_path = _write_frame(trade_log_frame, output_dir / "trade_log.parquet")
    primary_trade_log_path = _write_frame(primary_trade_log_frame, output_dir / "primary_trade_log.parquet")
    equity_curve_path = _write_frame(equity_curve_frame, output_dir / "equity_curve.parquet")
    strategy_metrics_path = _write_frame(strategy_metrics, output_dir / "strategy_metrics.parquet")
    split_summary_path = _write_frame(split_summary, output_dir / "split_summary.parquet")
    leaderboard_path = _write_frame(leaderboard, output_dir / "leaderboard.parquet")
    combined_strategy_metrics_path = (
        _write_frame(combined_strategy_metrics, output_dir / "combined_strategy_metrics.parquet")
        if not combined_strategy_metrics.empty
        else None
    )

    trade_log_csv_path = _write_frame(trade_log_frame, output_dir / "trade_log.csv") if settings.reporting.write_csv else None
    primary_trade_log_csv_path = (
        _write_frame(primary_trade_log_frame, output_dir / "primary_trade_log.csv")
        if settings.reporting.write_csv
        else None
    )
    equity_curve_csv_path = _write_frame(equity_curve_frame, output_dir / "equity_curve.csv") if settings.reporting.write_csv else None
    strategy_metrics_csv_path = _write_frame(strategy_metrics, output_dir / "strategy_metrics.csv") if settings.reporting.write_csv else None
    split_summary_csv_path = _write_frame(split_summary, output_dir / "split_summary.csv") if settings.reporting.write_csv else None
    leaderboard_csv_path = _write_frame(leaderboard, output_dir / "leaderboard.csv") if settings.reporting.write_csv else None
    combined_strategy_metrics_csv_path = (
        _write_frame(combined_strategy_metrics, output_dir / "combined_strategy_metrics.csv")
        if settings.reporting.write_csv and not combined_strategy_metrics.empty
        else None
    )

    figure_paths: list[str] = []
    if not equity_curve_frame.empty and not leaderboard.empty:
        figure_paths.append(
            _plot_cumulative_returns(
                equity_curve_frame,
                leaderboard,
                figures_dir / f"cumulative_returns.{settings.reporting.figure_format}",
                top_n=settings.reporting.top_n_strategies_for_plots,
                dpi=settings.reporting.dpi,
            )
        )
        figure_paths.append(
            _plot_drawdowns(
                equity_curve_frame,
                leaderboard,
                figures_dir / f"drawdowns.{settings.reporting.figure_format}",
                top_n=settings.reporting.top_n_strategies_for_plots,
                dpi=settings.reporting.dpi,
            )
        )
    figure_paths.append(
        _plot_trade_return_boxplot(
            primary_trade_log_frame,
            leaderboard,
            figures_dir / f"trade_return_boxplot.{settings.reporting.figure_format}",
            top_n=settings.reporting.top_n_strategies_for_plots,
            dpi=settings.reporting.dpi,
            title=f"{settings.reporting.primary_split.title()}-Split Trade Return Distribution",
        )
    )

    signal_manifest = _load_manifest(settings.input.signal_manifest_path)
    market_manifest = _load_manifest(settings.input.market_manifest_path)
    report_path: str | None = None
    if settings.reporting.write_markdown_report:
        report_text = _build_markdown_report(
            settings,
            strategy_metrics,
            split_summary,
            figure_paths,
            signal_manifest,
            market_manifest,
            leverage_diagnostics,
            funding_diagnostics,
        )
        report_file = output_dir / "backtest_report.md"
        report_file.write_text(report_text, encoding="utf-8")
        report_path = str(report_file)

    manifest = {
        "input": settings.input.model_dump(),
        "selection": settings.selection.model_dump(),
        "portfolio": settings.portfolio.model_dump(),
        "costs": settings.costs.model_dump(),
        "execution": settings.execution.model_dump(),
        "reporting": settings.reporting.model_dump(),
        "summary": {
            "strategy_count": int(strategy_metrics["strategy_name"].nunique()) if not strategy_metrics.empty else 0,
            "trade_count": int(len(trade_log_frame)),
            "primary_trade_log_rows": int(len(primary_trade_log_frame)),
            "primary_trade_count": int(strategy_metrics["trade_count"].sum()) if not strategy_metrics.empty else 0,
            "combined_trade_count": int(combined_strategy_metrics["trade_count"].sum()) if not combined_strategy_metrics.empty else int(len(trade_log_frame)),
            "primary_split": settings.reporting.primary_split,
            "best_strategy": None if leaderboard.empty else str(leaderboard.iloc[0]["strategy_name"]),
            "best_strategy_status": None if leaderboard.empty else _safe_text(leaderboard.iloc[0]["status"]),
            "best_sharpe_ratio": None if leaderboard.empty else _safe_float(leaderboard.iloc[0]["sharpe_ratio"]),
            "best_cumulative_return": None if leaderboard.empty else _safe_float(leaderboard.iloc[0]["cumulative_return"]),
            "best_mark_to_market_max_drawdown": None
            if leaderboard.empty
            else _safe_float(leaderboard.iloc[0]["mark_to_market_max_drawdown"]),
            "best_realized_max_drawdown": None if leaderboard.empty else _safe_float(leaderboard.iloc[0]["realized_max_drawdown"]),
            "status_counts": {}
            if strategy_metrics.empty
            else {
                str(key): int(value)
                for key, value in strategy_metrics["status"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .to_dict()
                .items()
            },
            "prediction_modes": []
            if strategy_metrics.empty
            else sorted(strategy_metrics["prediction_mode"].dropna().astype(str).unique().tolist()),
            "calibration_methods": []
            if strategy_metrics.empty
            else sorted(strategy_metrics["calibration_method"].dropna().astype(str).unique().tolist()),
            "checkpoint_selection_metrics": []
            if strategy_metrics.empty
            else sorted(strategy_metrics["checkpoint_selection_metric"].dropna().astype(str).unique().tolist()),
            "checkpoint_selection_effective_metrics": []
            if strategy_metrics.empty
            else sorted(strategy_metrics["checkpoint_selection_effective_metric"].dropna().astype(str).unique().tolist()),
            "selected_losses": []
            if strategy_metrics.empty
            else sorted(strategy_metrics["selected_loss"].dropna().astype(str).unique().tolist()),
            "preprocessing_scalers": []
            if strategy_metrics.empty
            else sorted(strategy_metrics["preprocessing_scaler"].dropna().astype(str).unique().tolist()),
            "source_subtypes": []
            if strategy_metrics.empty
            else sorted(strategy_metrics["source_subtype"].dropna().astype(str).unique().tolist()),
        },
        "diagnostics": {
            "leverage": leverage_diagnostics,
            "funding": funding_diagnostics,
        },
        "signal_manifest": signal_manifest,
        "market_manifest": market_manifest,
        "artifacts": {
            "trade_log_path": trade_log_path,
            "primary_trade_log_path": primary_trade_log_path,
            "equity_curve_path": equity_curve_path,
            "strategy_metrics_path": strategy_metrics_path,
            "combined_strategy_metrics_path": combined_strategy_metrics_path,
            "combined_strategy_metrics_csv_path": combined_strategy_metrics_csv_path,
            "split_summary_path": split_summary_path,
            "leaderboard_path": leaderboard_path,
            "report_path": report_path,
            "figure_paths": figure_paths,
        },
        "assumptions": [
            "Single-asset, delta-neutral prototype with at most one open position per strategy.",
            "Signals at timestamp t are executed after entry_delay_bars using the configured execution price field.",
            "Primary leaderboard and strategy metrics use the configured reporting.primary_split, which defaults to test.",
            "Primary equity, drawdown, and Sharpe metrics use mark-to-market equity; realized-only columns are retained for audit.",
            f"Funding PnL uses funding_mode={settings.execution.funding_mode} and funding_notional_mode={settings.execution.funding_notional_mode}.",
            f"Hedge mode is {settings.execution.hedge_mode}; current implementation uses equal USD notional on the perp and spot legs.",
            "Trading fees use taker_fee_bps on all four round-trip leg transactions.",
            "Slippage is modeled by adverse execution prices; embedded_slippage_cost_usd is the preferred diagnostic and is not deducted twice.",
            f"Stop logic is {settings.execution.stop_observation_mode} and {settings.execution.stop_execution_mode}.",
            "Simple annualized Sharpe uses square-root scaling and should be interpreted cautiously for sparse or serially correlated returns.",
            "Strategies with no executable trades keep cumulative return at zero but report Sharpe and drawdown as NaN because those risk metrics are not meaningful without positions.",
        ],
    }
    manifest_path = output_dir / "backtest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return BacktestArtifacts(
        output_dir=str(output_dir),
        trade_log_path=trade_log_path,
        trade_log_csv_path=trade_log_csv_path,
        primary_trade_log_path=primary_trade_log_path,
        primary_trade_log_csv_path=primary_trade_log_csv_path,
        equity_curve_path=equity_curve_path,
        equity_curve_csv_path=equity_curve_csv_path,
        strategy_metrics_path=strategy_metrics_path,
        strategy_metrics_csv_path=strategy_metrics_csv_path,
        combined_strategy_metrics_path=combined_strategy_metrics_path,
        combined_strategy_metrics_csv_path=combined_strategy_metrics_csv_path,
        split_summary_path=split_summary_path,
        split_summary_csv_path=split_summary_csv_path,
        leaderboard_path=leaderboard_path,
        leaderboard_csv_path=leaderboard_csv_path,
        report_path=report_path,
        figure_paths=figure_paths,
        manifest_path=str(manifest_path),
    )
