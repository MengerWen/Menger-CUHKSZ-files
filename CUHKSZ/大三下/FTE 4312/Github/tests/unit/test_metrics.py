from __future__ import annotations

import math

from funding_arb.evaluation.metrics import (
    calculate_autocorrelation_adjusted_sharpe,
    calculate_max_consecutive_losses,
    calculate_max_drawdown,
    calculate_period_sharpe_ratio,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_total_return,
)



def test_calculate_total_return_compounds() -> None:
    result = calculate_total_return([0.10, -0.05])
    assert math.isclose(result, 0.045, rel_tol=1e-9)



def test_calculate_sharpe_ratio_returns_zero_for_flat_returns() -> None:
    result = calculate_sharpe_ratio([0.0, 0.0, 0.0], periods_per_year=252)
    assert result == 0.0


def test_period_and_adjusted_sharpe_are_available() -> None:
    returns = [0.01, -0.005, 0.003, 0.002, -0.001, 0.004]

    assert calculate_period_sharpe_ratio(returns) != 0.0
    assert math.isfinite(calculate_autocorrelation_adjusted_sharpe(returns, periods_per_year=252, max_lag=2))



def test_calculate_max_drawdown_tracks_peak_to_trough() -> None:
    result = calculate_max_drawdown([100.0, 110.0, 90.0, 95.0])
    assert math.isclose(result, -0.1818181818, rel_tol=1e-6)


def test_trade_quality_helpers() -> None:
    pnl = [100.0, -20.0, -10.0, 50.0, -5.0]

    assert math.isclose(calculate_profit_factor(pnl), 150.0 / 35.0, rel_tol=1e-9)
    assert calculate_max_consecutive_losses(pnl) == 2
