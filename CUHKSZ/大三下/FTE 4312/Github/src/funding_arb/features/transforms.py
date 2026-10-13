"""Reusable feature transforms for rolling statistics and regime flags."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two aligned series, returning NaN where the denominator is zero."""
    denominator = denominator.replace(0.0, np.nan)
    return numerator / denominator


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a leakage-safe rolling mean using only current and past observations."""
    return series.rolling(window=window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Compute a leakage-safe rolling standard deviation using only current and past observations."""
    return series.rolling(window=window, min_periods=window).std(ddof=0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score without using future observations."""
    mean = rolling_mean(series, window)
    std = rolling_std(series, window).replace(0.0, np.nan)
    return (series - mean) / std


def sign_indicator(series: pd.Series) -> pd.Series:
    """Return the sign of each value as -1, 0, or 1."""
    return pd.Series(np.sign(series.fillna(0.0)), index=series.index, dtype=float)


def sign_reversal_indicator(series: pd.Series) -> pd.Series:
    """Return 1 when the sign flips between consecutive non-zero observations."""
    current = sign_indicator(series)
    previous = current.shift(1)
    return (((current != 0.0) & (previous != 0.0) & (current != previous)).astype(int)).astype(float)


def realized_volatility(returns: pd.Series, window: int, annualization_factor_hours: int) -> pd.Series:
    """Compute annualized realized volatility from hourly returns."""
    rolling = returns.rolling(window=window, min_periods=window).std(ddof=0)
    return rolling * np.sqrt(float(annualization_factor_hours))


def shock_score(returns: pd.Series, window: int) -> pd.Series:
    """Scale absolute returns by their rolling standard deviation to detect short-term shocks."""
    volatility = rolling_std(returns, window).replace(0.0, np.nan)
    return returns.abs() / volatility


def rolling_positive_share(series: pd.Series, window: int) -> pd.Series:
    """Compute the rolling share of strictly positive observations."""
    positive = (series > 0.0).astype(float)
    return positive.rolling(window=window, min_periods=window).mean()


def relative_to_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Scale a series by its rolling mean, useful for activity and liquidity features."""
    return safe_divide(series, rolling_mean(series, window))


def rolling_regime_indicator(series: pd.Series, regime_window: int) -> pd.Series:
    """Flag whether a series is above its rolling median regime benchmark."""
    median = series.rolling(window=regime_window, min_periods=regime_window).median()
    indicator = pd.Series(np.nan, index=series.index, dtype=float)
    valid = median.notna() & series.notna()
    indicator.loc[valid] = (series.loc[valid] > median.loc[valid]).astype(float)
    return indicator


def threshold_indicator(series: pd.Series, threshold: float) -> pd.Series:
    """Flag whether a series exceeds a fixed threshold, preserving NaN where unavailable."""
    indicator = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.notna()
    indicator.loc[valid] = (series.loc[valid] > threshold).astype(float)
    return indicator