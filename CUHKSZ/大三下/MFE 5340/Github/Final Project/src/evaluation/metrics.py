"""Evaluation metrics with pandas, numpy, scikit-learn, and linearmodels."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.metrics import r2_score


PRIMARY_METRICS: tuple[str, ...] = (
    "oos_r2",
    "rank_ic",
    "pricing_error",
    "long_short_return",
    "long_only_return",
    "sharpe_ratio",
    "max_drawdown",
    "turnover",
)


def metric_catalog() -> pd.DataFrame:
    """Return the planned metric catalog as a pandas DataFrame."""

    return pd.DataFrame(
        [
            {"metric": "oos_r2", "stage": "prediction", "backend": "sklearn", "description": "Out-of-sample R2 for return forecasts"},
            {"metric": "rank_ic", "stage": "prediction", "backend": "pandas", "description": "Monthly Spearman rank correlation"},
            {"metric": "pricing_error", "stage": "asset_pricing", "backend": "linearmodels", "description": "Cross-sectional/panel pricing error"},
            {"metric": "long_short_return", "stage": "portfolio", "backend": "pandas", "description": "Long-short realized portfolio return"},
            {"metric": "long_only_return", "stage": "portfolio", "backend": "pandas", "description": "Long-only realized portfolio return"},
            {"metric": "sharpe_ratio", "stage": "portfolio", "backend": "numpy", "description": "Annualized Sharpe ratio"},
            {"metric": "max_drawdown", "stage": "portfolio", "backend": "pandas", "description": "Maximum drawdown"},
            {"metric": "turnover", "stage": "portfolio", "backend": "pandas", "description": "Portfolio turnover"},
        ]
    )


def as_series(values: Iterable[float] | pd.Series, name: str | None = None) -> pd.Series:
    """Convert numeric inputs to a clean pandas Series."""

    series = values.copy() if isinstance(values, pd.Series) else pd.Series(list(values), dtype="float64")
    if name is not None:
        series.name = name
    return pd.to_numeric(series, errors="coerce").dropna()


def annualized_sharpe(monthly_returns: Iterable[float] | pd.Series) -> float:
    """Compute annualized Sharpe ratio from monthly returns with pandas/numpy."""

    returns = as_series(monthly_returns, name="monthly_return")
    if returns.size < 2:
        raise ValueError("at least two monthly returns are required")
    volatility = returns.std(ddof=1)
    if np.isclose(volatility, 0.0):
        raise ValueError("return volatility is zero")
    return float(np.sqrt(12.0) * returns.mean() / volatility)


def out_of_sample_r2(y_true: Iterable[float] | pd.Series, y_pred: Iterable[float] | pd.Series) -> float:
    """Compute OOS R2 using scikit-learn after pandas alignment."""

    frame = pd.concat(
        [as_series(y_true, "y_true"), as_series(y_pred, "y_pred")],
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise ValueError("aligned y_true/y_pred data is empty")
    return float(r2_score(frame["y_true"], frame["y_pred"]))


def rank_ic_by_month(frame: pd.DataFrame, date_col: str, actual_col: str, forecast_col: str) -> pd.Series:
    """Compute Spearman rank IC for each month with pandas."""

    required = [date_col, actual_col, forecast_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return (
        frame[required]
        .dropna()
        .groupby(date_col, sort=True)
        .apply(lambda block: block[actual_col].corr(block[forecast_col], method="spearman"), include_groups=False)
        .rename("rank_ic")
    )


def max_drawdown(monthly_returns: Iterable[float] | pd.Series) -> float:
    """Compute maximum drawdown from a monthly return series."""

    returns = as_series(monthly_returns, name="monthly_return")
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def panel_pricing_regression(
    frame: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str = "stock_id",
    time_col: str = "date",
    entity_effects: bool = True,
    time_effects: bool = True,
) -> pd.Series:
    """Fit a simple PanelOLS pricing diagnostic with linearmodels.

    This helper is for diagnostics and robustness checks, not the final IPCA
    implementation.
    """

    required = [entity_col, time_col, y_col, *x_cols]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    panel = frame[required].dropna().copy()
    if panel.empty:
        raise ValueError("panel regression data is empty")
    panel[time_col] = pd.to_datetime(panel[time_col])
    panel = panel.set_index([entity_col, time_col]).sort_index()
    y = panel[y_col]
    x = panel[x_cols]
    result = PanelOLS(y, x, entity_effects=entity_effects, time_effects=time_effects, check_rank=False).fit(cov_type="clustered", cluster_entity=True)
    return pd.Series(
        {
            "nobs": result.nobs,
            "rsquared": result.rsquared,
            "rsquared_within": result.rsquared_within,
            "loglik": result.loglik,
        },
        name="panel_pricing_regression",
    )
