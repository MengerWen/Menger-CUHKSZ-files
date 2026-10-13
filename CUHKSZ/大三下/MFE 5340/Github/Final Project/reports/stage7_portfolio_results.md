# Stage 7 Portfolio Results

## Scope

Stage 7 converts the saved OOS model predictions into investable monthly portfolio tests on the default `features500` main specification. The main comparison uses the common stock-date intersection across all models, so the portfolio results are as comparable as possible.

The default backtest uses month-`t` model scores to form portfolios at the end of month `t`, then applies realized month `t+1` returns from the cleaned Stage 2 panel. Value weights use `mcap_t`, which comes from the project market-cap data. Raw `BLACKLIST.pkl` and `UNTRADABLE.pkl` are aggregated to signal-month flags and can be used as additional formation filters.

## Core Result

At the current default settings, graph-enhanced exposure estimation does not yet produce a dominant investable winner across both long-short and long-only portfolios.

The strongest long-short performer at the main transaction-cost setting (`10` bps) is:

- `graph_conditional_latent_factor (equal)` by annualized return
- `ipca_style (equal)` by Sharpe ratio

The strongest long-only performer at the same setting is:

- `graph_conditional_latent_factor (equal)` by annualized return
- `graph_conditional_latent_factor (value)` by Sharpe ratio

## Graph Model Interpretation

The graph model should be read against the strongest characteristic-only alternatives, not against a naive predictor alone.

Examples from the saved summary table at `10` bps:

- Graph long-short equal-weight annualized return: `0.0732`
- Graph long-short equal-weight Sharpe: `0.6536`
- Graph long-only value-weight annualized return: `0.2098`
- Graph long-only value-weight Sharpe: `1.1794`

If the graph model outperforms, the current evidence should be interpreted mainly as improved **sorting quality**, because Stage 6 already suggested that graph structure helps rank-order stocks more than it improves raw OOS prediction fit. If the graph model does not dominate the portfolio table, that means the current graph beta function has not yet translated its ranking gains into a consistent investable edge after weighting, turnover, and transaction costs.

## Long-Short vs Long-Only

Long-short portfolios isolate cross-sectional sorting ability most directly. Long-only portfolios are more sensitive to whether the top-ranked names are robust enough to survive realistic weighting and costs.

In this stage, the right question is not only “which model has the highest return,” but also:

- whether the return comes with better Sharpe or just higher volatility,
- whether value weighting changes the ranking of models,
- whether turnover erodes the raw signal,
- and whether the graph model helps more in long-short ranking portfolios than in long-only implementations.

## Figures

- `outputs/portfolio/stage7_plots/stage7_cumulative_long_only_equal_10bps.png`
- `outputs/portfolio/stage7_plots/stage7_cumulative_long_only_value_10bps.png`
- `outputs/portfolio/stage7_plots/stage7_cumulative_long_short_equal_10bps.png`
- `outputs/portfolio/stage7_plots/stage7_cumulative_long_short_value_10bps.png`
- `outputs/portfolio/stage7_plots/stage7_summary_bars_10bps.png`


## Strongest Remaining Limitations

1. The current portfolio test still relies on a short 24-month default OOS span.
2. The common stock-date intersection makes the comparison fairer, but it also narrows the investable universe.
3. The graph model is still a first-pass GCN/GAT latent-factor implementation rather than a stronger structural no-arbitrage system.
4. Turnover is computed with a simple one-way weight-change approximation and does not model post-return drift before rebalancing.
5. Transaction costs are only sensitivity adjustments, not a full microstructure model.
6. Because Stage 2 already filtered the main panel for feasibility, the raw blacklist/untradable overlays may have limited incremental effect in the default run.
7. There is still no broader-spec `features/` comparison in this stage.
