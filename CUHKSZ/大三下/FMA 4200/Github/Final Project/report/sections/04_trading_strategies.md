# Trading Strategies

## Joint Multivariate Return Dynamics

A VAR was fitted to the six monthly return series after comparing lags 1 through 12 on standard information criteria. The selected benchmark in the current run is **VAR(1)** under the BIC rule. The fitted system is stable if all roots lie outside the unit circle, and the saved stability table confirms that condition for the selected specification. Residual diagnostics are mixed rather than perfect: the whiteness test p-value is **0.0000**, while the multivariate normality test p-value is **0.0000**, which is consistent with the heavy-tailed evidence already documented in Section 3.

Impulse-response and forecast-error-style diagnostics are saved for interpretation rather than treated as structural causal results. In a six-portfolio reduced-form VAR, the main value of these tools is to show how shocks propagate across size and book-to-market buckets over the following year and how much of each portfolio's forecast variance is explained by its own shock versus cross-portfolio spillovers.

## Cointegration and the Statistical Meaning of Nonstationary Representations

Raw monthly returns are not an appropriate target for cointegration testing because they are already stationary objects. The integration-order table therefore checks both the raw return series and a defensible nonstationary transformation based on cumulative portfolio wealth. In the current run, **100%** of the raw-return tests are classified as stationary, while **100%** of the log-wealth level tests are classified as nonstationary. That is exactly the pattern needed to justify applying Johansen-style analysis to log wealth rather than to raw returns.

Using the log-wealth representation, the Johansen trace test selects a cointegration rank of **1** at the 5% level. That result supports a statistical-arbitrage design based on mean reversion in long-run relative portfolio value rather than in the raw return series themselves. At the same time, the rolling backtest shows that positive cointegration rank is present in only **38.4%** of the monthly re-estimation windows, so any stat-arb interpretation should be treated as episodic rather than permanently stable.

## Statistical Arbitrage Backtest

The stat-arb strategy uses a 240-month rolling estimation window, refits the cointegration relation every 12 months, and trades the first cointegration spread when its z-score leaves the +/-1.5 band. Positions are exited once the spread reverts inside +/-0.5. Turnover is tracked explicitly and transaction costs are set to 10 basis points per one-way turnover unit. The resulting strategy has annualized net return **-0.11%**, annualized net volatility **1.42%**, net Sharpe **-0.077**, and max drawdown **-18.83%** over the common out-of-sample period.

## Mean-Variance Analysis and Rolling Portfolio Backtests

The plug-in mean-variance backtest uses rolling sample means and sample covariance estimates with full investment and no short sales. The improved plug-in strategy keeps the same rolling expected-return input but replaces the sample covariance matrix with Ledoit-Wolf shrinkage and adds practical controls through weight bounds and a turnover penalty. Relative to the equally weighted benchmark, the current run shows annualized net return **12.24%** and net Sharpe **0.743** for equal weight, versus **13.42%** and **0.873** for the baseline plug-in strategy, and **12.63%** and **0.815** for the improved plug-in strategy. The improved strategy's main practical gain is turnover control: its average monthly turnover is roughly **12.8x lower** than the baseline plug-in strategy while keeping competitive net performance.

In the current out-of-sample comparison, the best net Sharpe belongs to **Plug-In Mean-Variance**. The efficient-frontier figure and the rolling backtest together reinforce a familiar lesson from empirical portfolio choice: naive sample plug-in portfolios are sensitive to estimation noise, while shrinkage, constraints, and turnover control can materially improve implementability.

## Saved Outputs

- `output/tables/trading_strategies/var_lag_selection.csv`
- `output/tables/trading_strategies/var_diagnostics.csv`
- `output/tables/trading_strategies/cointegration_integration_order_tests.csv`
- `output/tables/trading_strategies/cointegration_summary.csv`
- `output/tables/trading_strategies/cointegration_vectors.csv`
- `output/tables/trading_strategies/stat_arb_backtest.csv`
- `output/tables/trading_strategies/stat_arb_signals.csv`
- `output/tables/trading_strategies/strategy_returns.csv`
- `output/tables/trading_strategies/strategy_metrics.csv`
- `output/tables/trading_strategies/strategy_weights.csv`
- `output/tables/trading_strategies/efficient_frontier_points.csv`
- `output/figures/trading_strategies/`
- `output/models/trading_strategies/strategy_rules.md`
