"""Leakage-safe helpers for post-cost label generation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from funding_arb.config.models import LabelCostSettings, LabelPipelineSettings, LabelTargetSettings, ModelSplitSettings


def describe_labeling_assumption(config: dict[str, Any]) -> str:
    """Summarize current label-generation assumptions for feature or label configs."""
    if "target" in config:
        target_cfg = config.get("target", {})
        horizons = target_cfg.get("holding_windows_hours", [])
        direction = target_cfg.get("direction", "unknown")
        edge = target_cfg.get("min_expected_edge_bps", "unknown")
        return (
            f"Labels target horizons {horizons} hours for {direction}, with tradeable edge threshold "
            f"of {edge} bps after estimated costs."
        )
    label_cfg = config.get("labels", {})
    horizon = label_cfg.get("forward_horizon_hours", "unknown")
    edge = label_cfg.get("min_expected_edge_bps", "unknown")
    return (
        f"Labels will target a {horizon}-hour horizon with minimum expected edge "
        f"of {edge} bps after costs."
    )


def forward_window_sum(series: pd.Series, start_offset: int, window: int) -> pd.Series:
    """Sum a forward-looking window using explicit offsets without leaking farther-ahead values."""
    shifted = series.shift(-start_offset)
    return shifted.rolling(window=window, min_periods=window).sum().shift(-(window - 1))


def _parse_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def assign_time_series_split(timestamps: pd.Series, split: ModelSplitSettings) -> pd.Series:
    """Assign train, validation, or test splits using chronological boundaries only."""
    train_end = _parse_timestamp(split.train_end)
    validation_end = _parse_timestamp(split.validation_end)
    test_end = _parse_timestamp(split.test_end)
    labels = pd.Series("excluded", index=timestamps.index, dtype="object")
    labels.loc[timestamps <= train_end] = "train"
    labels.loc[(timestamps > train_end) & (timestamps <= validation_end)] = "validation"
    labels.loc[(timestamps > validation_end) & (timestamps <= test_end)] = "test"
    return labels


def _position_signs(direction: str) -> tuple[int, int]:
    if direction == "short_perp_long_spot":
        return -1, 1
    if direction == "long_perp_short_spot":
        return 1, -1
    raise ValueError(f"Unsupported label direction: {direction}")


def _execution_price_column(prefix: str, price_field: str) -> str:
    return f"{prefix}_{price_field}"


def _trade_cost_bps(costs: LabelCostSettings, horizon_hours: int, hedge_sign: int) -> float:
    trading_cost_bps = 4.0 * (costs.taker_fee_bps + costs.slippage_bps)
    gas_cost_bps = (costs.gas_cost_usd / costs.position_notional_usd) * 10_000.0 if costs.position_notional_usd > 0 else 0.0
    borrow_cost_bps = costs.borrow_cost_bps_per_hour * float(horizon_hours) if hedge_sign < 0 else 0.0
    return trading_cost_bps + gas_cost_bps + costs.other_friction_bps + borrow_cost_bps


def build_label_table(
    market_frame: pd.DataFrame,
    target: LabelTargetSettings,
    costs: LabelCostSettings,
) -> pd.DataFrame:
    """Build explicit classification and regression targets from future delta-neutral trade outcomes."""
    label_table = pd.DataFrame({"timestamp": pd.to_datetime(market_frame["timestamp"], utc=True)})
    perp_sign, hedge_sign = _position_signs(target.direction)
    delay = int(target.execution_delay_bars)

    perp_price_column = _execution_price_column("perp", target.execution_price_field)
    spot_price_column = _execution_price_column("spot", target.execution_price_field)
    if perp_price_column not in market_frame.columns or spot_price_column not in market_frame.columns:
        raise ValueError(
            f"Market dataset must contain {perp_price_column} and {spot_price_column} for label generation."
        )

    perp_prices = pd.to_numeric(market_frame[perp_price_column], errors="coerce")
    spot_prices = pd.to_numeric(market_frame[spot_price_column], errors="coerce")
    funding_bps = pd.to_numeric(market_frame["funding_rate"], errors="coerce") * 10_000.0

    horizons = sorted({*target.holding_windows_hours, target.primary_horizon_hours})
    for horizon in horizons:
        entry_perp = perp_prices.shift(-delay)
        exit_perp = perp_prices.shift(-(delay + horizon))
        entry_spot = spot_prices.shift(-delay)
        exit_spot = spot_prices.shift(-(delay + horizon))

        perp_leg_return_bps = perp_sign * ((exit_perp / entry_perp) - 1.0) * 10_000.0
        spot_leg_return_bps = hedge_sign * ((exit_spot / entry_spot) - 1.0) * 10_000.0
        funding_return_bps = (-perp_sign) * forward_window_sum(funding_bps, start_offset=delay, window=horizon)

        gross_return_bps = perp_leg_return_bps + spot_leg_return_bps + funding_return_bps
        estimated_cost_bps = _trade_cost_bps(costs, horizon, hedge_sign)
        net_return_bps = gross_return_bps - estimated_cost_bps if target.use_post_cost_target else gross_return_bps

        valid = entry_perp.notna() & exit_perp.notna() & entry_spot.notna() & exit_spot.notna() & funding_return_bps.notna()
        positive_target = pd.Series(pd.NA, index=label_table.index, dtype="Int64")
        tradeable_target = pd.Series(pd.NA, index=label_table.index, dtype="Int64")
        positive_target.loc[valid] = (net_return_bps.loc[valid] > target.positive_return_threshold_bps).astype(int)
        tradeable_target.loc[valid] = (net_return_bps.loc[valid] > target.min_expected_edge_bps).astype(int)

        prefix = f"{horizon}h"
        label_table[f"target_future_perp_leg_return_bps_{prefix}"] = perp_leg_return_bps
        label_table[f"target_future_spot_leg_return_bps_{prefix}"] = spot_leg_return_bps
        label_table[f"target_future_funding_return_bps_{prefix}"] = funding_return_bps
        label_table[f"target_future_gross_return_bps_{prefix}"] = gross_return_bps
        label_table[f"target_estimated_cost_bps_{prefix}"] = float(estimated_cost_bps)
        label_table[f"target_future_net_return_bps_{prefix}"] = net_return_bps
        label_table[f"target_is_profitable_{prefix}"] = positive_target
        label_table[f"target_is_tradeable_{prefix}"] = tradeable_target

    return label_table