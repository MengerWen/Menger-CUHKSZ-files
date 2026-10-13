"""Grouped feature builders for the perpetual funding-rate arbitrage dataset."""

from __future__ import annotations

import pandas as pd

from funding_arb.config.models import FeatureSetSettings
from funding_arb.features.transforms import (
    realized_volatility,
    relative_to_rolling_mean,
    rolling_mean,
    rolling_positive_share,
    rolling_regime_indicator,
    rolling_std,
    rolling_zscore,
    shock_score,
    sign_indicator,
    sign_reversal_indicator,
    threshold_indicator,
)


def _unique_windows(settings: FeatureSetSettings, *additional: int) -> list[int]:
    windows = sorted({*settings.rolling_windows, *additional})
    return [window for window in windows if window > 0]


def build_funding_features(frame: pd.DataFrame, settings: FeatureSetSettings) -> pd.DataFrame:
    """Create funding-driven features from the canonical hourly dataset."""
    features = pd.DataFrame(index=frame.index)
    funding_rate = pd.to_numeric(frame["funding_rate"], errors="coerce")
    funding_bps = funding_rate * 10_000.0

    features["funding_rate_raw"] = funding_rate
    features["funding_rate_bps"] = funding_bps
    features["funding_annualized_proxy"] = funding_rate * (24.0 / float(settings.funding_interval_hours)) * 365.0
    features["funding_sign"] = sign_indicator(funding_rate)
    features["funding_sign_reversal"] = sign_reversal_indicator(funding_rate)
    features["funding_event_flag"] = pd.to_numeric(frame.get("funding_event", 0), errors="coerce").fillna(0.0)

    for window in _unique_windows(settings, settings.funding_mean_window, settings.zscore_window):
        features[f"funding_mean_{window}h"] = rolling_mean(funding_bps, window)
        features[f"funding_std_{window}h"] = rolling_std(funding_bps, window)
        features[f"funding_zscore_{window}h"] = rolling_zscore(funding_bps, window)
        features[f"funding_positive_share_{window}h"] = rolling_positive_share(funding_bps, window)

    return features


def build_basis_features(frame: pd.DataFrame, settings: FeatureSetSettings) -> pd.DataFrame:
    """Create basis and spread features from perp-versus-spot dislocations."""
    features = pd.DataFrame(index=frame.index)
    perp_close = pd.to_numeric(frame["perp_close"], errors="coerce")
    spot_close = pd.to_numeric(frame["spot_close"], errors="coerce")
    spread_usd = perp_close - spot_close
    spread_bps = ((perp_close / spot_close) - 1.0) * 10_000.0

    features["spread_usd"] = spread_usd
    features["spread_bps"] = spread_bps
    features["spread_change_1h"] = spread_bps.diff(1)
    features[f"spread_change_{settings.basis_mean_window}h"] = spread_bps.diff(settings.basis_mean_window)

    for window in _unique_windows(settings, settings.basis_mean_window, settings.zscore_window):
        features[f"spread_mean_{window}h"] = rolling_mean(spread_bps, window)
        features[f"spread_std_{window}h"] = rolling_std(spread_bps, window)
        features[f"spread_zscore_{window}h"] = rolling_zscore(spread_bps, window)
        features[f"spread_deviation_{window}h"] = spread_bps - features[f"spread_mean_{window}h"]
        features[f"spread_reversion_signal_{window}h"] = -features[f"spread_zscore_{window}h"]

    return features


def build_volatility_features(frame: pd.DataFrame, settings: FeatureSetSettings) -> pd.DataFrame:
    """Create volatility and shock features from hourly returns."""
    features = pd.DataFrame(index=frame.index)
    perp_close = pd.to_numeric(frame["perp_close"], errors="coerce")
    spot_close = pd.to_numeric(frame["spot_close"], errors="coerce")
    perp_return = perp_close.pct_change()
    spot_return = spot_close.pct_change()

    features["perp_return_1h"] = perp_return
    features["spot_return_1h"] = spot_return
    features["perp_abs_return_1h"] = perp_return.abs()
    features["spot_abs_return_1h"] = spot_return.abs()

    for window in _unique_windows(settings, settings.volatility_window):
        features[f"perp_realized_vol_{window}h"] = realized_volatility(
            perp_return,
            window,
            settings.annualization_factor_hours,
        )
        features[f"spot_realized_vol_{window}h"] = realized_volatility(
            spot_return,
            window,
            settings.annualization_factor_hours,
        )

    features[f"perp_return_shock_{settings.shock_window}h"] = shock_score(perp_return, settings.shock_window)
    features[f"spot_return_shock_{settings.shock_window}h"] = shock_score(spot_return, settings.shock_window)
    return features


def build_liquidity_features(frame: pd.DataFrame, settings: FeatureSetSettings) -> pd.DataFrame:
    """Create liquidity and activity features from volume and optional open interest."""
    features = pd.DataFrame(index=frame.index)
    perp_volume = pd.to_numeric(frame["perp_volume"], errors="coerce")
    spot_volume = pd.to_numeric(frame["spot_volume"], errors="coerce")
    perp_close = pd.to_numeric(frame["perp_close"], errors="coerce")
    spot_close = pd.to_numeric(frame["spot_close"], errors="coerce")

    features["perp_volume_raw"] = perp_volume
    features["spot_volume_raw"] = spot_volume
    features["perp_dollar_volume_raw"] = perp_volume * perp_close
    features["spot_dollar_volume_raw"] = spot_volume * spot_close
    features["perp_volume_change_1h"] = perp_volume.pct_change()
    features["spot_volume_change_1h"] = spot_volume.pct_change()

    for window in _unique_windows(settings, settings.liquidity_window):
        features[f"perp_volume_ratio_{window}h"] = relative_to_rolling_mean(perp_volume, window)
        features[f"spot_volume_ratio_{window}h"] = relative_to_rolling_mean(spot_volume, window)

    if "open_interest" in frame.columns:
        open_interest = pd.to_numeric(frame["open_interest"], errors="coerce")
        if open_interest.notna().any():
            features["open_interest_raw"] = open_interest
            features["open_interest_change_1h"] = open_interest.pct_change()
            for window in _unique_windows(settings, settings.liquidity_window):
                features[f"open_interest_ratio_{window}h"] = relative_to_rolling_mean(open_interest, window)
                features[f"open_interest_zscore_{window}h"] = rolling_zscore(open_interest, window)

    return features


def build_interaction_state_features(
    frame: pd.DataFrame,
    settings: FeatureSetSettings,
    funding_features: pd.DataFrame,
    basis_features: pd.DataFrame,
    volatility_features: pd.DataFrame,
    liquidity_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create interaction terms and simple regime indicators."""
    features = pd.DataFrame(index=frame.index)
    funding_bps = funding_features["funding_rate_bps"]
    spread_bps = basis_features["spread_bps"]
    perp_vol_col = f"perp_realized_vol_{settings.volatility_window}h"
    spread_z_col = f"spread_zscore_{settings.zscore_window}h"
    funding_mean_col = f"funding_mean_{settings.funding_mean_window}h"
    shock_col = f"perp_return_shock_{settings.shock_window}h"

    features[f"funding_x_perp_vol_{settings.volatility_window}h"] = funding_bps * volatility_features[perp_vol_col]
    features["funding_x_spread_bps"] = funding_bps * spread_bps
    features[f"spread_x_perp_vol_{settings.volatility_window}h"] = spread_bps * volatility_features[perp_vol_col]

    features["positive_funding_regime"] = threshold_indicator(funding_features[funding_mean_col], 0.0)
    features["high_vol_regime"] = rolling_regime_indicator(volatility_features[perp_vol_col], settings.regime_window)
    features["wide_spread_regime"] = threshold_indicator(basis_features[spread_z_col].abs(), 1.0)
    features["shock_regime"] = threshold_indicator(volatility_features[shock_col], 2.0)

    if "open_interest_raw" in liquidity_features.columns:
        features["funding_x_open_interest"] = funding_bps * liquidity_features["open_interest_raw"]

    return features