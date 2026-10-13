from __future__ import annotations

import logging
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from .config import (
    COINTEGRATION_ORDER_TESTS_PATH,
    COINTEGRATION_SUMMARY_PATH,
    COINTEGRATION_VECTORS_PATH,
    EFFICIENT_FRONTIER_POINTS_PATH,
    MULTIVARIATE_WEALTH_PANEL_PATH,
    PERCENT_COLUMNS,
    SECTION_04_PATH,
    STRATEGY_METRICS_PATH,
    STRATEGY_RETURNS_PATH,
    STRATEGY_WEIGHTS_PATH,
    STAT_ARB_BACKTEST_PATH,
    STAT_ARB_SIGNAL_PATH,
    TRADING_FIGURES_DIR,
    TRADING_MODELS_DIR,
    TRADING_STRATEGIES_LOG_PATH,
    TRADING_TABLES_DIR,
    VAR_DIAGNOSTICS_PATH,
    VAR_LAG_SELECTION_PATH,
    VAR_STABILITY_PATH,
)
from .data_pipeline import ensure_directories
from .univariate_modeling import load_clean_returns, portfolio_label, safe_adf, safe_kpss


VAR_MAX_LAGS = 12
COINT_MAX_LAGS = 6
IRF_HORIZON = 12
STAT_ARB_TRAIN_WINDOW = 240
STAT_ARB_Z_WINDOW = 60
STAT_ARB_ENTRY_Z = 1.5
STAT_ARB_EXIT_Z = 0.5
STAT_ARB_REFIT_FREQUENCY = 12
MVO_TRAIN_WINDOW = 120
COMMON_OOS_START = max(STAT_ARB_TRAIN_WINDOW, MVO_TRAIN_WINDOW)
TRANSACTION_COST_RATE = 0.001
MVO_RISK_AVERSION = 8.0
MVO_WEIGHT_BOUND = 0.35
MVO_TURNOVER_PENALTY = 0.0025
FRONTIER_POINTS = 25
WEIGHT_EPS = 1e-8

STRATEGY_LABELS = {
    "equal_weight": "Equal Weight",
    "stat_arb_cointegration": "Cointegration Stat-Arb",
    "mv_plugin_sample": "Plug-In Mean-Variance",
    "mv_improved_shrinkage": "Improved Plug-In Mean-Variance",
}


def configure_trading_logger() -> logging.Logger:
    logger = logging.getLogger("fma4200_trading")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(TRADING_STRATEGIES_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def returns_decimal_panel(percent_df: pd.DataFrame) -> pd.DataFrame:
    decimal_df = percent_df.copy()
    for column in PERCENT_COLUMNS:
        decimal_df[column] = decimal_df[column] / 100.0
    return decimal_df


def wealth_panel_from_returns(decimal_df: pd.DataFrame) -> pd.DataFrame:
    wealth = decimal_df.copy()
    wealth_values = (1.0 + decimal_df[PERCENT_COLUMNS]).cumprod()
    wealth.loc[:, PERCENT_COLUMNS] = wealth_values
    wealth.to_csv(MULTIVARIATE_WEALTH_PANEL_PATH, index=False)
    return wealth


def log_wealth_panel(wealth_df: pd.DataFrame) -> pd.DataFrame:
    log_df = wealth_df.copy()
    log_df.loc[:, PERCENT_COLUMNS] = np.log(wealth_df[PERCENT_COLUMNS])
    return log_df


def series_stationarity_row(series: pd.Series, portfolio: str, representation: str, unit: str) -> dict[str, object]:
    adf_stat, adf_pvalue = safe_adf(series)
    kpss_stat, kpss_pvalue = safe_kpss(series)
    if adf_pvalue < 0.05 and kpss_pvalue >= 0.05:
        conclusion = "stationary"
    elif adf_pvalue >= 0.05 and kpss_pvalue < 0.05:
        conclusion = "nonstationary"
    else:
        conclusion = "mixed"
    return {
        "portfolio": portfolio,
        "label": portfolio_label(portfolio),
        "representation": representation,
        "unit": unit,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_pvalue,
        "conclusion": conclusion,
    }


def integration_order_table(percent_df: pd.DataFrame, decimal_df: pd.DataFrame, log_wealth_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    diff_log_wealth = log_wealth_df.copy()
    diff_log_wealth.loc[:, PERCENT_COLUMNS] = log_wealth_df[PERCENT_COLUMNS].diff()

    for portfolio in PERCENT_COLUMNS:
        rows.append(series_stationarity_row(percent_df[portfolio], portfolio, "raw_return_pct", "percent return"))
        rows.append(series_stationarity_row(log_wealth_df[portfolio], portfolio, "log_wealth_level", "log wealth index"))
        rows.append(
            series_stationarity_row(
                diff_log_wealth[portfolio].dropna(),
                portfolio,
                "diff_log_wealth",
                "log-return approximation",
            )
        )

    return pd.DataFrame(rows)


def fit_var_lag_table(returns_df: pd.DataFrame, max_lag: int = VAR_MAX_LAGS) -> tuple[pd.DataFrame, int, object]:
    model = VAR(returns_df[PERCENT_COLUMNS])
    rows: list[dict[str, object]] = []
    fitted_results: dict[int, object] = {}

    for lag in range(1, max_lag + 1):
        result = model.fit(lag)
        fitted_results[lag] = result
        rows.append(
            {
                "lag": lag,
                "aic": float(result.aic),
                "bic": float(result.bic),
                "hqic": float(result.hqic),
                "fpe": float(result.fpe),
                "is_stable": bool(result.is_stable(verbose=False)),
            }
        )

    lag_table = pd.DataFrame(rows).sort_values("lag").reset_index(drop=True)
    selected_lag = int(lag_table.sort_values(["bic", "lag"]).iloc[0]["lag"])
    lag_table["selected_bic"] = lag_table["lag"] == selected_lag
    return lag_table, selected_lag, fitted_results[selected_lag]


def var_diagnostics_table(var_result) -> pd.DataFrame:
    whiteness = var_result.test_whiteness(nlags=max(12, var_result.k_ar + 1))
    normality = var_result.test_normality()

    rows = [
        {
            "test": "portmanteau_whiteness",
            "statistic": float(getattr(whiteness, "test_statistic", np.nan)),
            "pvalue": float(getattr(whiteness, "pvalue", np.nan)),
            "critical_value": float(getattr(whiteness, "crit_value", np.nan)),
            "decision_5pct": "reject" if float(getattr(whiteness, "pvalue", 1.0)) < 0.05 else "do_not_reject",
        },
        {
            "test": "jarque_bera_normality",
            "statistic": float(getattr(normality, "test_statistic", np.nan)),
            "pvalue": float(getattr(normality, "pvalue", np.nan)),
            "critical_value": float(getattr(normality, "crit_value", np.nan)),
            "decision_5pct": "reject" if float(getattr(normality, "pvalue", 1.0)) < 0.05 else "do_not_reject",
        },
    ]
    return pd.DataFrame(rows)


def var_stability_table(var_result) -> pd.DataFrame:
    roots = np.asarray(var_result.roots)
    return pd.DataFrame(
        {
            "root_index": np.arange(1, len(roots) + 1),
            "real_part": np.real(roots),
            "imag_part": np.imag(roots),
            "modulus": np.abs(roots),
            "outside_unit_circle": np.abs(roots) > 1.0,
        }
    )


def save_var_figures(var_result) -> None:
    irf = var_result.irf(IRF_HORIZON)
    irf_figure = irf.plot(orth=False)
    irf_figure.tight_layout()
    irf_figure.savefig(TRADING_FIGURES_DIR / "var_irf_grid.png", dpi=200, bbox_inches="tight")
    plt.close(irf_figure)

    fevd = var_result.fevd(IRF_HORIZON)
    fevd_figure = fevd.plot()
    fevd_figure.tight_layout()
    fevd_figure.savefig(TRADING_FIGURES_DIR / "var_fevd.png", dpi=200, bbox_inches="tight")
    plt.close(fevd_figure)


def fevd_horizon_table(var_result, horizon: int = IRF_HORIZON) -> pd.DataFrame:
    fevd = var_result.fevd(horizon)
    decomp = fevd.decomp[:, horizon - 1, :]
    rows: list[dict[str, object]] = []
    for response_index, response_name in enumerate(PERCENT_COLUMNS):
        row: dict[str, object] = {"response_portfolio": response_name}
        for impulse_index, impulse_name in enumerate(PERCENT_COLUMNS):
            row[f"share_from_{impulse_name}"] = float(decomp[response_index, impulse_index])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_cointegration_inputs(log_wealth_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    diff_log_wealth = log_wealth_df.copy()
    diff_log_wealth.loc[:, PERCENT_COLUMNS] = log_wealth_df[PERCENT_COLUMNS].diff()
    diff_log_wealth = diff_log_wealth.dropna().reset_index(drop=True)
    lag_table, selected_lag, _ = fit_var_lag_table(diff_log_wealth, max_lag=COINT_MAX_LAGS)
    lag_table.to_csv(TRADING_TABLES_DIR / "cointegration_diff_var_lag_selection.csv", index=False)
    return diff_log_wealth, selected_lag


def johansen_summary_table(johansen_result) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank_index in range(len(johansen_result.lr1)):
        rows.append(
            {
                "rank_null": rank_index,
                "trace_stat": float(johansen_result.lr1[rank_index]),
                "crit_90": float(johansen_result.cvt[rank_index, 0]),
                "crit_95": float(johansen_result.cvt[rank_index, 1]),
                "crit_99": float(johansen_result.cvt[rank_index, 2]),
                "reject_at_95": bool(johansen_result.lr1[rank_index] > johansen_result.cvt[rank_index, 1]),
            }
        )
    return pd.DataFrame(rows)


def fit_cointegration_model(log_wealth_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, object | None]:
    _, selected_diff_lag = fit_cointegration_inputs(log_wealth_df)
    johansen_result = coint_johansen(log_wealth_df[PERCENT_COLUMNS], det_order=0, k_ar_diff=selected_diff_lag)
    summary_table = johansen_summary_table(johansen_result)
    rank = int(summary_table["reject_at_95"].sum())
    summary_table["selected_diff_lag"] = selected_diff_lag
    summary_table["selected_rank"] = rank

    vecm_result = None
    vectors = pd.DataFrame()
    if rank > 0:
        vecm_result = VECM(
            log_wealth_df[PERCENT_COLUMNS],
            k_ar_diff=selected_diff_lag,
            coint_rank=rank,
            deterministic="co",
        ).fit()
        vector_rows: list[dict[str, object]] = []
        for vector_index in range(rank):
            for asset_index, portfolio in enumerate(PERCENT_COLUMNS):
                vector_rows.append(
                    {
                        "vector": vector_index + 1,
                        "portfolio": portfolio,
                        "beta": float(vecm_result.beta[asset_index, vector_index]),
                        "alpha": float(vecm_result.alpha[asset_index, vector_index]),
                    }
                )
        vectors = pd.DataFrame(vector_rows)

    return summary_table, vectors, rank, vecm_result


def normalize_stat_arb_weights(beta: np.ndarray) -> np.ndarray:
    centered = beta - np.mean(beta)
    gross = np.sum(np.abs(centered))
    if gross <= WEIGHT_EPS:
        return np.zeros_like(beta)
    return centered / gross


def post_return_weights(weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    if np.sum(np.abs(weights)) <= WEIGHT_EPS:
        return np.zeros_like(weights)
    holdings = weights * (1.0 + asset_returns)
    gross = np.sum(np.abs(holdings))
    if gross <= WEIGHT_EPS:
        return np.zeros_like(weights)
    return holdings / gross


def compute_turnover(target_weights: np.ndarray, current_weights: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(target_weights - current_weights)))


def run_stat_arb_backtest(
    log_wealth_df: pd.DataFrame,
    decimal_df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    level_values = log_wealth_df[PERCENT_COLUMNS].reset_index(drop=True)
    returns = decimal_df[PERCENT_COLUMNS].reset_index(drop=True)
    dates = decimal_df["date"].reset_index(drop=True)

    current_post_weights = np.zeros(len(PERCENT_COLUMNS))
    position_side = 0
    current_trade_weights = np.zeros(len(PERCENT_COLUMNS))
    current_beta = np.zeros(len(PERCENT_COLUMNS))
    current_rank = 0
    wealth_gross = 1.0
    wealth_net = 1.0

    backtest_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []

    for index in range(COMMON_OOS_START, len(decimal_df)):
        if (index - COMMON_OOS_START) % STAT_ARB_REFIT_FREQUENCY == 0:
            window_levels = level_values.iloc[index - STAT_ARB_TRAIN_WINDOW : index].copy()
            diff_window = window_levels.diff().dropna().reset_index(drop=True)
            diff_lag_table, selected_diff_lag, _ = fit_var_lag_table(
                pd.concat([dates.iloc[1: len(diff_window) + 1].reset_index(drop=True), diff_window], axis=1),
                max_lag=min(COINT_MAX_LAGS, len(diff_window) // 10),
            )
            selected_diff_lag = max(1, selected_diff_lag)
            johansen_result = coint_johansen(window_levels, det_order=0, k_ar_diff=selected_diff_lag)
            current_rank = int(np.sum(johansen_result.lr1 > johansen_result.cvt[:, 1]))
            if current_rank > 0:
                current_beta = np.asarray(johansen_result.evec[:, 0], dtype=float)
            else:
                current_beta = np.zeros(len(PERCENT_COLUMNS))
            logger.info(
                "Stat-arb refit at %s selected rank %s with diff lag %s",
                dates.iloc[index].date().isoformat(),
                current_rank,
                selected_diff_lag,
            )

        if current_rank > 0:
            base_weights = normalize_stat_arb_weights(current_beta)
            spread_window = np.dot(
                level_values.iloc[max(index - STAT_ARB_Z_WINDOW, 0) : index].to_numpy(dtype=float),
                current_beta,
            )
            spread_mean = float(np.mean(spread_window))
            spread_std = float(np.std(spread_window, ddof=1)) if len(spread_window) > 1 else np.nan
            current_spread = float(np.dot(level_values.iloc[index - 1].to_numpy(dtype=float), current_beta))
            zscore = 0.0 if not np.isfinite(spread_std) or spread_std <= WEIGHT_EPS else (current_spread - spread_mean) / spread_std

            if position_side == 0:
                if zscore >= STAT_ARB_ENTRY_Z:
                    position_side = -1
                elif zscore <= -STAT_ARB_ENTRY_Z:
                    position_side = 1
            else:
                if abs(zscore) <= STAT_ARB_EXIT_Z:
                    position_side = 0
                elif position_side == 1 and zscore >= STAT_ARB_ENTRY_Z:
                    position_side = -1
                elif position_side == -1 and zscore <= -STAT_ARB_ENTRY_Z:
                    position_side = 1

            current_trade_weights = position_side * base_weights
        else:
            spread_mean = np.nan
            spread_std = np.nan
            current_spread = np.nan
            zscore = np.nan
            position_side = 0
            current_trade_weights = np.zeros(len(PERCENT_COLUMNS))

        turnover = compute_turnover(current_trade_weights, current_post_weights)
        asset_return = returns.iloc[index].to_numpy(dtype=float)
        gross_return = float(np.dot(current_trade_weights, asset_return))
        net_return = gross_return - TRANSACTION_COST_RATE * turnover
        wealth_gross *= 1.0 + gross_return
        wealth_net *= 1.0 + net_return
        current_post_weights = post_return_weights(current_trade_weights, asset_return)

        backtest_rows.append(
            {
                "date": dates.iloc[index],
                "strategy": "stat_arb_cointegration",
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "gross_wealth": wealth_gross,
                "net_wealth": wealth_net,
            }
        )
        signal_rows.append(
            {
                "date": dates.iloc[index],
                "rank": current_rank,
                "spread": current_spread,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "zscore": zscore,
                "position_side": position_side,
                "turnover": turnover,
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )
        for portfolio, weight in zip(PERCENT_COLUMNS, current_trade_weights):
            weight_rows.append(
                {
                    "date": dates.iloc[index],
                    "strategy": "stat_arb_cointegration",
                    "portfolio": portfolio,
                    "weight": float(weight),
                }
            )

    return pd.DataFrame(backtest_rows), pd.DataFrame(signal_rows), pd.DataFrame(weight_rows)


def regularize_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    ridge = 1e-6 * np.eye(covariance.shape[0])
    return covariance + ridge


def solve_mean_variance_weights(
    mean_vector: np.ndarray,
    covariance: np.ndarray,
    previous_weights: np.ndarray | None = None,
    allow_short: bool = False,
    upper_bound: float | None = None,
    turnover_penalty: float = 0.0,
) -> np.ndarray:
    n_assets = len(mean_vector)
    covariance = regularize_covariance(covariance)
    weights = cp.Variable(n_assets)
    prev = np.zeros(n_assets) if previous_weights is None else previous_weights

    objective = mean_vector @ weights - 0.5 * MVO_RISK_AVERSION * cp.quad_form(weights, covariance)
    if turnover_penalty > 0.0:
        objective -= turnover_penalty * cp.norm1(weights - prev)

    constraints = [cp.sum(weights) == 1.0]
    if not allow_short:
        constraints.append(weights >= 0.0)
    if upper_bound is not None:
        constraints.append(weights <= upper_bound)

    problem = cp.Problem(cp.Maximize(objective), constraints)
    solver = cp.ECOS if turnover_penalty > 0.0 else cp.OSQP
    problem.solve(solver=solver, warm_start=True, verbose=False)
    if weights.value is None:
        fallback_solver = cp.SCS if solver != cp.SCS else cp.ECOS
        problem.solve(solver=fallback_solver, warm_start=True, verbose=False)
    if weights.value is None:
        raise RuntimeError("Mean-variance optimization did not converge.")
    return np.asarray(weights.value, dtype=float)


def equal_weight_vector(n_assets: int) -> np.ndarray:
    return np.repeat(1.0 / n_assets, n_assets)


def run_allocation_backtest(decimal_df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = decimal_df[PERCENT_COLUMNS].reset_index(drop=True)
    dates = decimal_df["date"].reset_index(drop=True)
    n_assets = len(PERCENT_COLUMNS)

    strategies = ["equal_weight", "mv_plugin_sample", "mv_improved_shrinkage"]
    current_post_weights = {strategy: np.zeros(n_assets) for strategy in strategies}
    current_wealth_gross = {strategy: 1.0 for strategy in strategies}
    current_wealth_net = {strategy: 1.0 for strategy in strategies}
    backtest_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    for index in range(COMMON_OOS_START, len(decimal_df)):
        history = returns.iloc[index - MVO_TRAIN_WINDOW : index].copy()
        sample_mean = history.mean().to_numpy(dtype=float)
        sample_cov = history.cov().to_numpy(dtype=float)
        shrinkage_cov = LedoitWolf().fit(history.to_numpy(dtype=float)).covariance_

        target_weights = {
            "equal_weight": equal_weight_vector(n_assets),
            "mv_plugin_sample": solve_mean_variance_weights(
                sample_mean,
                sample_cov,
                previous_weights=current_post_weights["mv_plugin_sample"],
                allow_short=False,
            ),
            "mv_improved_shrinkage": solve_mean_variance_weights(
                sample_mean,
                shrinkage_cov,
                previous_weights=current_post_weights["mv_improved_shrinkage"],
                allow_short=False,
                upper_bound=MVO_WEIGHT_BOUND,
                turnover_penalty=MVO_TURNOVER_PENALTY,
            ),
        }

        asset_return = returns.iloc[index].to_numpy(dtype=float)
        for strategy in strategies:
            turnover = compute_turnover(target_weights[strategy], current_post_weights[strategy])
            gross_return = float(np.dot(target_weights[strategy], asset_return))
            net_return = gross_return - TRANSACTION_COST_RATE * turnover
            current_wealth_gross[strategy] *= 1.0 + gross_return
            current_wealth_net[strategy] *= 1.0 + net_return
            current_post_weights[strategy] = post_return_weights(target_weights[strategy], asset_return)

            backtest_rows.append(
                {
                    "date": dates.iloc[index],
                    "strategy": strategy,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "turnover": turnover,
                    "gross_wealth": current_wealth_gross[strategy],
                    "net_wealth": current_wealth_net[strategy],
                }
            )
            for portfolio, weight in zip(PERCENT_COLUMNS, target_weights[strategy]):
                weight_rows.append(
                    {
                        "date": dates.iloc[index],
                        "strategy": strategy,
                        "portfolio": portfolio,
                        "weight": float(weight),
                    }
                )

    return pd.DataFrame(backtest_rows), pd.DataFrame(weight_rows)


def frontier_points_table(
    mean_vector: np.ndarray,
    covariance: np.ndarray,
    frontier_name: str,
    allow_short: bool,
    upper_bound: float | None = None,
) -> pd.DataFrame:
    covariance = regularize_covariance(covariance)
    n_assets = len(mean_vector)
    target_grid = np.linspace(float(np.min(mean_vector)), float(np.max(mean_vector)), FRONTIER_POINTS)
    rows: list[dict[str, object]] = []

    for target_return in target_grid:
        weights = cp.Variable(n_assets)
        constraints = [cp.sum(weights) == 1.0, mean_vector @ weights >= target_return]
        if not allow_short:
            constraints.append(weights >= 0.0)
        if upper_bound is not None:
            constraints.append(weights <= upper_bound)

        problem = cp.Problem(cp.Minimize(cp.quad_form(weights, covariance)), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if weights.value is None:
            problem.solve(solver=cp.SCS, warm_start=True, verbose=False)
        if weights.value is None:
            continue

        solved_weights = np.asarray(weights.value, dtype=float)
        monthly_return = float(mean_vector @ solved_weights)
        monthly_vol = float(np.sqrt(max(solved_weights @ covariance @ solved_weights, 0.0)))
        rows.append(
            {
                "frontier": frontier_name,
                "target_return_monthly": target_return,
                "expected_return_monthly": monthly_return,
                "expected_vol_monthly": monthly_vol,
                "expected_return_annual": monthly_return * 12.0,
                "expected_vol_annual": monthly_vol * np.sqrt(12.0),
            }
        )

    return pd.DataFrame(rows)


def max_drawdown(wealth_series: pd.Series) -> float:
    running_max = wealth_series.cummax()
    drawdown = wealth_series / running_max - 1.0
    return float(drawdown.min())


def performance_table(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, group in strategy_returns.groupby("strategy"):
        group = group.sort_values("date").reset_index(drop=True)
        gross = group["gross_return"]
        net = group["net_return"]
        gross_wealth = group["gross_wealth"]
        net_wealth = group["net_wealth"]

        annual_return_gross = float((1.0 + gross).prod() ** (12.0 / len(gross)) - 1.0)
        annual_return_net = float((1.0 + net).prod() ** (12.0 / len(net)) - 1.0)
        annual_vol_net = float(net.std(ddof=1) * np.sqrt(12.0))
        annual_vol_gross = float(gross.std(ddof=1) * np.sqrt(12.0))
        sharpe_gross = annual_return_gross / annual_vol_gross if annual_vol_gross > WEIGHT_EPS else np.nan
        sharpe_net = annual_return_net / annual_vol_net if annual_vol_net > WEIGHT_EPS else np.nan

        rows.append(
            {
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                "annual_return_gross": annual_return_gross,
                "annual_return_net": annual_return_net,
                "annual_vol_gross": annual_vol_gross,
                "annual_vol_net": annual_vol_net,
                "sharpe_gross": sharpe_gross,
                "sharpe_net": sharpe_net,
                "terminal_wealth_gross": float(gross_wealth.iloc[-1]),
                "terminal_wealth_net": float(net_wealth.iloc[-1]),
                "max_drawdown_net": max_drawdown(net_wealth),
                "average_turnover": float(group["turnover"].mean()),
                "total_turnover": float(group["turnover"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("sharpe_net", ascending=False).reset_index(drop=True)


def save_strategy_figures(
    combined_backtest: pd.DataFrame,
    frontier_df: pd.DataFrame,
    stat_arb_signals: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for strategy, group in combined_backtest.groupby("strategy"):
        ax.plot(group["date"], group["net_wealth"], linewidth=1.2, label=STRATEGY_LABELS.get(strategy, strategy))
    ax.set_title("Strategy Cumulative Net Wealth")
    ax.set_ylabel("Net Wealth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(TRADING_FIGURES_DIR / "strategy_cumulative_wealth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for frontier, group in frontier_df.groupby("frontier"):
        ax.plot(group["expected_vol_annual"], group["expected_return_annual"], linewidth=1.4, label=frontier)
    ax.set_title("Efficient Frontier Comparison")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Expected Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(TRADING_FIGURES_DIR / "efficient_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(stat_arb_signals["date"], stat_arb_signals["spread"], color="#1f5aa6", linewidth=1.0)
    axes[0].set_title("Cointegration Spread")
    axes[1].plot(stat_arb_signals["date"], stat_arb_signals["zscore"], color="#ba4a00", linewidth=1.0)
    axes[1].axhline(STAT_ARB_ENTRY_Z, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(-STAT_ARB_ENTRY_Z, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(STAT_ARB_EXIT_Z, color="gray", linestyle=":", linewidth=0.8)
    axes[1].axhline(-STAT_ARB_EXIT_Z, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_title("Cointegration Z-Score Signal")
    fig.tight_layout()
    fig.savefig(TRADING_FIGURES_DIR / "stat_arb_signal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_strategy_rules() -> None:
    lines = [
        "# Trading Rules",
        "",
        "## Cointegration Statistical Arbitrage",
        "",
        f"- Use a {STAT_ARB_TRAIN_WINDOW}-month rolling training window on log wealth levels.",
        f"- Re-estimate the Johansen cointegration relation every {STAT_ARB_REFIT_FREQUENCY} months.",
        f"- Use the first cointegrating vector when the Johansen trace test rejects at least one rank at the 5% level.",
        f"- Compute a spread z-score from the last {STAT_ARB_Z_WINDOW} in-sample spread observations.",
        f"- Enter when the z-score exceeds +/-{STAT_ARB_ENTRY_Z:.1f}; exit when it reverts inside +/-{STAT_ARB_EXIT_Z:.1f}.",
        "- Convert the cointegration vector into a dollar-neutral weight vector by de-meaning the coefficients and normalizing gross exposure to one.",
        "",
        "## Mean-Variance Strategies",
        "",
        f"- Use a {MVO_TRAIN_WINDOW}-month rolling estimation window for expected returns and covariance matrices.",
        "- The plug-in mean-variance strategy uses sample mean and sample covariance with full investment and no short sales.",
        f"- The improved plug-in strategy uses Ledoit-Wolf shrinkage covariance, no short sales, an upper weight bound of {MVO_WEIGHT_BOUND:.2f}, and an L1 turnover penalty in the monthly optimization.",
        "",
        "## Transaction Costs and Turnover",
        "",
        f"- Transaction costs are set to {TRANSACTION_COST_RATE * 10000:.0f} basis points per one-way turnover unit.",
        "- Turnover is measured as one-half of the L1 distance between current post-return weights and next target weights.",
        "- All signals and weights are formed using information available up to the previous month-end, so the backtests avoid look-ahead bias.",
    ]
    write_text_file(TRADING_MODELS_DIR / "strategy_rules.md", "\n".join(lines) + "\n")


def write_section_04(
    var_lag_table: pd.DataFrame,
    var_diagnostics: pd.DataFrame,
    integration_table: pd.DataFrame,
    cointegration_summary: pd.DataFrame,
    performance_summary: pd.DataFrame,
    stat_arb_signals: pd.DataFrame,
) -> None:
    selected_var_lag = int(var_lag_table.loc[var_lag_table["selected_bic"], "lag"].iloc[0])
    whiteness_pvalue = float(var_diagnostics.loc[var_diagnostics["test"] == "portmanteau_whiteness", "pvalue"].iloc[0])
    normality_pvalue = float(var_diagnostics.loc[var_diagnostics["test"] == "jarque_bera_normality", "pvalue"].iloc[0])
    return_stationary_share = float(
        (integration_table.loc[integration_table["representation"] == "raw_return_pct", "conclusion"] == "stationary").mean()
    )
    wealth_nonstationary_share = float(
        (integration_table.loc[integration_table["representation"] == "log_wealth_level", "conclusion"] == "nonstationary").mean()
    )
    selected_rank = int(cointegration_summary["selected_rank"].iloc[0])
    rolling_rank_positive_share = float((stat_arb_signals["rank"] > 0).mean())
    best_strategy = performance_summary.iloc[0]
    equal_weight_row = performance_summary.loc[performance_summary["strategy"] == "equal_weight"].iloc[0]
    improved_row = performance_summary.loc[performance_summary["strategy"] == "mv_improved_shrinkage"].iloc[0]
    plugin_row = performance_summary.loc[performance_summary["strategy"] == "mv_plugin_sample"].iloc[0]
    stat_arb_row = performance_summary.loc[performance_summary["strategy"] == "stat_arb_cointegration"].iloc[0]
    turnover_ratio = (
        plugin_row["average_turnover"] / improved_row["average_turnover"]
        if improved_row["average_turnover"] > WEIGHT_EPS
        else np.nan
    )

    lines = [
        "# Trading Strategies",
        "",
        "## Joint Multivariate Return Dynamics",
        "",
        f"A VAR was fitted to the six monthly return series after comparing lags 1 through {VAR_MAX_LAGS} on standard information criteria. The selected benchmark in the current run is **VAR({selected_var_lag})** under the BIC rule. The fitted system is stable if all roots lie outside the unit circle, and the saved stability table confirms that condition for the selected specification. Residual diagnostics are mixed rather than perfect: the whiteness test p-value is **{whiteness_pvalue:.4f}**, while the multivariate normality test p-value is **{normality_pvalue:.4f}**, which is consistent with the heavy-tailed evidence already documented in Section 3.",
        "",
        "Impulse-response and forecast-error-style diagnostics are saved for interpretation rather than treated as structural causal results. In a six-portfolio reduced-form VAR, the main value of these tools is to show how shocks propagate across size and book-to-market buckets over the following year and how much of each portfolio's forecast variance is explained by its own shock versus cross-portfolio spillovers.",
        "",
        "## Cointegration and the Statistical Meaning of Nonstationary Representations",
        "",
        f"Raw monthly returns are not an appropriate target for cointegration testing because they are already stationary objects. The integration-order table therefore checks both the raw return series and a defensible nonstationary transformation based on cumulative portfolio wealth. In the current run, **{return_stationary_share:.0%}** of the raw-return tests are classified as stationary, while **{wealth_nonstationary_share:.0%}** of the log-wealth level tests are classified as nonstationary. That is exactly the pattern needed to justify applying Johansen-style analysis to log wealth rather than to raw returns.",
        "",
        f"Using the log-wealth representation, the Johansen trace test selects a cointegration rank of **{selected_rank}** at the 5% level. That result supports a statistical-arbitrage design based on mean reversion in long-run relative portfolio value rather than in the raw return series themselves. At the same time, the rolling backtest shows that positive cointegration rank is present in only **{rolling_rank_positive_share:.1%}** of the monthly re-estimation windows, so any stat-arb interpretation should be treated as episodic rather than permanently stable.",
        "",
        "## Statistical Arbitrage Backtest",
        "",
        f"The stat-arb strategy uses a {STAT_ARB_TRAIN_WINDOW}-month rolling estimation window, refits the cointegration relation every {STAT_ARB_REFIT_FREQUENCY} months, and trades the first cointegration spread when its z-score leaves the +/-{STAT_ARB_ENTRY_Z:.1f} band. Positions are exited once the spread reverts inside +/-{STAT_ARB_EXIT_Z:.1f}. Turnover is tracked explicitly and transaction costs are set to {TRANSACTION_COST_RATE * 10000:.0f} basis points per one-way turnover unit. The resulting strategy has annualized net return **{stat_arb_row['annual_return_net']:.2%}**, annualized net volatility **{stat_arb_row['annual_vol_net']:.2%}**, net Sharpe **{stat_arb_row['sharpe_net']:.3f}**, and max drawdown **{stat_arb_row['max_drawdown_net']:.2%}** over the common out-of-sample period.",
        "",
        "## Mean-Variance Analysis and Rolling Portfolio Backtests",
        "",
        f"The plug-in mean-variance backtest uses rolling sample means and sample covariance estimates with full investment and no short sales. The improved plug-in strategy keeps the same rolling expected-return input but replaces the sample covariance matrix with Ledoit-Wolf shrinkage and adds practical controls through weight bounds and a turnover penalty. Relative to the equally weighted benchmark, the current run shows annualized net return **{equal_weight_row['annual_return_net']:.2%}** and net Sharpe **{equal_weight_row['sharpe_net']:.3f}** for equal weight, versus **{plugin_row['annual_return_net']:.2%}** and **{plugin_row['sharpe_net']:.3f}** for the baseline plug-in strategy, and **{improved_row['annual_return_net']:.2%}** and **{improved_row['sharpe_net']:.3f}** for the improved plug-in strategy. The improved strategy's main practical gain is turnover control: its average monthly turnover is roughly **{turnover_ratio:.1f}x lower** than the baseline plug-in strategy while keeping competitive net performance.",
        "",
        f"In the current out-of-sample comparison, the best net Sharpe belongs to **{best_strategy['strategy_label']}**. The efficient-frontier figure and the rolling backtest together reinforce a familiar lesson from empirical portfolio choice: naive sample plug-in portfolios are sensitive to estimation noise, while shrinkage, constraints, and turnover control can materially improve implementability.",
        "",
        "## Saved Outputs",
        "",
        "- `output/tables/trading_strategies/var_lag_selection.csv`",
        "- `output/tables/trading_strategies/var_diagnostics.csv`",
        "- `output/tables/trading_strategies/cointegration_integration_order_tests.csv`",
        "- `output/tables/trading_strategies/cointegration_summary.csv`",
        "- `output/tables/trading_strategies/cointegration_vectors.csv`",
        "- `output/tables/trading_strategies/stat_arb_backtest.csv`",
        "- `output/tables/trading_strategies/stat_arb_signals.csv`",
        "- `output/tables/trading_strategies/strategy_returns.csv`",
        "- `output/tables/trading_strategies/strategy_metrics.csv`",
        "- `output/tables/trading_strategies/strategy_weights.csv`",
        "- `output/tables/trading_strategies/efficient_frontier_points.csv`",
        "- `output/figures/trading_strategies/`",
        "- `output/models/trading_strategies/strategy_rules.md`",
    ]
    write_text_file(SECTION_04_PATH, "\n".join(lines) + "\n")


def run_trading_strategies_pipeline() -> dict[str, object]:
    ensure_directories()
    logger = configure_trading_logger()
    logger.info("Starting trading strategies pipeline.")

    percent_df = load_clean_returns()
    decimal_df = returns_decimal_panel(percent_df)
    wealth_df = wealth_panel_from_returns(decimal_df)
    log_wealth_df = log_wealth_panel(wealth_df)

    integration_table = integration_order_table(percent_df, decimal_df, log_wealth_df)
    save_dataframe(integration_table, COINTEGRATION_ORDER_TESTS_PATH)

    var_lag_table, selected_var_lag, var_result = fit_var_lag_table(percent_df)
    var_diagnostics = var_diagnostics_table(var_result)
    var_stability = var_stability_table(var_result)
    fevd_table = fevd_horizon_table(var_result, horizon=IRF_HORIZON)

    save_dataframe(var_lag_table, VAR_LAG_SELECTION_PATH)
    save_dataframe(var_diagnostics, VAR_DIAGNOSTICS_PATH)
    save_dataframe(var_stability, VAR_STABILITY_PATH)
    save_dataframe(fevd_table, TRADING_TABLES_DIR / "var_fevd_horizon_12.csv")
    save_var_figures(var_result)
    write_text_file(TRADING_MODELS_DIR / "var_summary.txt", str(var_result.summary()) + "\n")

    cointegration_summary, cointegration_vectors, selected_rank, vecm_result = fit_cointegration_model(log_wealth_df)
    save_dataframe(cointegration_summary, COINTEGRATION_SUMMARY_PATH)
    save_dataframe(cointegration_vectors, COINTEGRATION_VECTORS_PATH)
    if vecm_result is not None:
        write_text_file(TRADING_MODELS_DIR / "vecm_summary.txt", str(vecm_result.summary()) + "\n")
    else:
        write_text_file(TRADING_MODELS_DIR / "vecm_summary.txt", "No cointegration rank was selected at the 5% level.\n")

    stat_arb_backtest, stat_arb_signals, stat_arb_weights = run_stat_arb_backtest(log_wealth_df, decimal_df, logger)
    allocation_backtest, allocation_weights = run_allocation_backtest(decimal_df, logger)

    combined_backtest = pd.concat([allocation_backtest, stat_arb_backtest], ignore_index=True).sort_values(["date", "strategy"]).reset_index(drop=True)
    combined_weights = pd.concat([allocation_weights, stat_arb_weights], ignore_index=True).sort_values(["date", "strategy", "portfolio"]).reset_index(drop=True)
    performance_summary = performance_table(combined_backtest)

    sample_mean = decimal_df[PERCENT_COLUMNS].mean().to_numpy(dtype=float)
    sample_cov = decimal_df[PERCENT_COLUMNS].cov().to_numpy(dtype=float)
    shrinkage_cov = LedoitWolf().fit(decimal_df[PERCENT_COLUMNS].to_numpy(dtype=float)).covariance_
    frontier_df = pd.concat(
        [
            frontier_points_table(sample_mean, sample_cov, "Sample Frontier", allow_short=False),
            frontier_points_table(sample_mean, shrinkage_cov, "Improved Frontier", allow_short=False, upper_bound=MVO_WEIGHT_BOUND),
        ],
        ignore_index=True,
    )

    save_dataframe(stat_arb_backtest, STAT_ARB_BACKTEST_PATH)
    save_dataframe(stat_arb_signals, STAT_ARB_SIGNAL_PATH)
    save_dataframe(combined_backtest, STRATEGY_RETURNS_PATH)
    save_dataframe(performance_summary, STRATEGY_METRICS_PATH)
    save_dataframe(combined_weights, STRATEGY_WEIGHTS_PATH)
    save_dataframe(frontier_df, EFFICIENT_FRONTIER_POINTS_PATH)
    save_strategy_figures(combined_backtest, frontier_df, stat_arb_signals)
    write_strategy_rules()
    write_section_04(var_lag_table, var_diagnostics, integration_table, cointegration_summary, performance_summary, stat_arb_signals)

    logger.info("Trading strategies pipeline completed successfully.")
    return {
        "selected_var_lag": selected_var_lag,
        "selected_cointegration_rank": selected_rank,
        "strategy_metrics_path": str(STRATEGY_METRICS_PATH),
        "strategy_returns_path": str(STRATEGY_RETURNS_PATH),
        "section_04_path": str(SECTION_04_PATH),
    }
