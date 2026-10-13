"""Small, reusable performance metrics for research and backtesting.

The helpers in this module intentionally stay simple and dependency-light.
They are not a replacement for a full performance analytics library, but they
make the project's assumptions explicit and keep calculations consistent across
modeling, signal evaluation, and backtesting reports.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd



def _to_series(values: Iterable[float]) -> pd.Series:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        raise ValueError("Metric input cannot be empty.")
    return series



def calculate_total_return(returns: Iterable[float]) -> float:
    """Calculate compounded total return from periodic simple returns."""
    series = _to_series(returns)
    return float((1.0 + series).prod() - 1.0)



def calculate_period_sharpe_ratio(
    returns: Iterable[float],
    risk_free_rate_per_period: float = 0.0,
) -> float:
    """Calculate a raw per-period Sharpe ratio without annualization."""
    series = _to_series(returns)
    excess = series - risk_free_rate_per_period
    volatility = float(excess.std(ddof=0))
    if math.isclose(volatility, 0.0):
        return 0.0
    return float(excess.mean() / volatility)


def calculate_sharpe_ratio(
    returns: Iterable[float],
    periods_per_year: int = 365 * 24,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate a simple square-root-scaled annualized Sharpe ratio.

    This is the common classroom/research approximation. It assumes periodic
    returns are close enough to IID for square-root scaling to be meaningful,
    which is often imperfect for sparse trading strategies.
    """
    period_sharpe = calculate_period_sharpe_ratio(
        returns,
        risk_free_rate_per_period=risk_free_rate / periods_per_year,
    )
    annualization = math.sqrt(periods_per_year)
    return float(period_sharpe * annualization)


def calculate_autocorrelation_adjusted_sharpe(
    returns: Iterable[float],
    periods_per_year: int = 365 * 24,
    risk_free_rate: float = 0.0,
    max_lag: int = 24,
) -> float:
    """Calculate a caveat-friendly Sharpe adjusted for serial correlation.

    The adjustment is inspired by Lo-style autocorrelation corrections. It is
    deliberately conservative and lightweight: positive serial correlation
    increases the denominator and therefore lowers the reported Sharpe.
    """
    series = _to_series(returns).astype(float)
    if len(series) < 3:
        return calculate_sharpe_ratio(series, periods_per_year, risk_free_rate)
    excess = series - (risk_free_rate / periods_per_year)
    volatility = float(excess.std(ddof=0))
    if math.isclose(volatility, 0.0):
        return 0.0
    usable_lag = min(max_lag, len(excess) - 1)
    if usable_lag <= 0:
        return calculate_sharpe_ratio(series, periods_per_year, risk_free_rate)
    autocorr_sum = 0.0
    for lag in range(1, usable_lag + 1):
        if len(excess) - lag < 2:
            continue
        autocorr = excess.autocorr(lag=lag)
        if pd.notna(autocorr):
            weight = 1.0 - lag / (usable_lag + 1.0)
            autocorr_sum += weight * float(autocorr)
    adjustment = 1.0 + 2.0 * autocorr_sum
    if adjustment <= 0.0 or math.isclose(adjustment, 0.0):
        adjustment = 1.0
    simple_sharpe = calculate_sharpe_ratio(series, periods_per_year, risk_free_rate)
    return float(simple_sharpe / math.sqrt(adjustment))



def calculate_max_drawdown(equity_curve: Iterable[float]) -> float:
    """Calculate maximum drawdown from a sequence of equity values."""
    series = _to_series(equity_curve)
    running_max = series.cummax()
    drawdown = (series / running_max) - 1.0
    return float(np.min(drawdown))


def calculate_profit_factor(pnl_values: Iterable[float]) -> float:
    """Calculate gross profit divided by gross loss for trade PnL values."""
    series = _to_series(pnl_values)
    gross_profit = float(series[series > 0.0].sum())
    gross_loss = abs(float(series[series < 0.0].sum()))
    if math.isclose(gross_loss, 0.0):
        return math.inf if gross_profit > 0.0 else 0.0
    return float(gross_profit / gross_loss)


def calculate_max_consecutive_losses(pnl_values: Iterable[float]) -> int:
    """Return the longest streak of strictly negative trade PnL values."""
    series = _to_series(pnl_values)
    current = 0
    longest = 0
    for value in series:
        if float(value) < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)
