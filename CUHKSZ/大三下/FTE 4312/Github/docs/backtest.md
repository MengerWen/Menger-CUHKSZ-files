# Backtesting Engine

## Purpose

The backtesting module evaluates standardized strategy signals for a single-asset, delta-neutral perpetual funding-rate arbitrage prototype. It is intentionally narrower than a generic portfolio simulator: the goal is to make the course-project strategy accounting explicit, reproducible, and hard to overstate.

The engine answers one practical question:

> Given hourly model or rule-based signals, does a simple delta-neutral execution rule produce useful post-cost performance after fees, slippage, gas, funding effects, and holding-period constraints?

## Scope

The current version supports:

- one asset at a time
- one open position at a time per strategy
- standardized signals from the signal layer
- fixed-notional delta-neutral positions
- explicit perp, spot, funding, fee, gas, and friction PnL
- realized-only and mark-to-market equity curves
- test-split-first leaderboards by default
- strategy metrics, split summaries, trade logs, plots, markdown reports, and manifests

It compares each `strategy_name` independently. If one signal artifact contains several rule, baseline, or deep-learning strategies, each strategy receives its own simulated equity curve and metrics row.

## Inputs

Default inputs:

- signals: `data/artifacts/signals/binance/btcusdt/1h/baseline/signals.parquet`
- market data: `data/processed/binance/btcusdt/1h/hourly_market_data.parquet`
- config: `configs/backtests/default.yaml`

Required signal fields include:

- `timestamp`
- `asset`
- `source`
- `source_subtype`
- `strategy_name`
- `task`
- `signal_score`
- `predicted_class`
- `expected_return_bps`
- `suggested_direction`
- `confidence`
- `should_trade`
- `split`
- `metadata_json`

Optional model metadata fields such as selected threshold, calibration method, checkpoint-selection metric, and preprocessing settings are preserved when present.

Required market fields include:

- `timestamp`
- `symbol`
- `venue`
- `frequency`
- `perp_open`, `perp_close`
- `spot_open`, `spot_close`
- `funding_rate`

## Position Model

The default direction is `short_perp_long_spot`.

The implemented hedge mode is:

- `equal_notional_hedge`

For a configured `position_notional` of `10,000 USD`, each trade uses:

- `10,000 USD` notional on the perpetual leg
- `10,000 USD` notional on the spot hedge leg
- `20,000 USD` gross exposure

Other hedge modes are reserved in config for future work:

- `equal_quantity_hedge`
- `contract_multiplier_adjusted_hedge`

They are not implemented yet because this prototype does not model exchange-specific contract multipliers or quantity-level hedge rebalancing.

## Entry And Exit Logic

Entry logic:

1. observe signal at timestamp `t`
2. require `should_trade = 1` unless disabled in config
3. require `suggested_direction` to match config direction
4. optionally require minimum signal score, confidence, or expected return
5. execute after `entry_delay_bars` using `execution_price_field`

With the default config, a signal observed at hour `t` executes on the next bar open. This preserves the no-lookahead convention used by the label and signal pipelines.

Exit logic:

- `signal_off`: close when the signal no longer satisfies entry conditions
- `holding_window`: close at the configured soft holding horizon
- `maximum_holding`: hard time cap
- `stop_loss` / `take_profit`: optional stop conditions
- `end_of_data`: close any remaining open position at the final market row

Stop approximation:

- `stop_observation_mode = bar_close_observed`
- `stop_execution_mode = next_bar_executed`

This means stops are observed on hourly bar-close marks and executed on the configured next execution bar. The engine does not model intrabar stop fills.

## PnL Logic

Trade PnL is decomposed into:

- perpetual leg PnL
- spot hedge leg PnL
- funding PnL
- trading fees
- gas cost
- other friction
- embedded slippage diagnostic

Slippage is applied through adverse effective execution prices. The preferred output column is `embedded_slippage_cost_usd`, which is a diagnostic approximation of the embedded price impact and is not deducted a second time. The older `estimated_slippage_cost_usd` column is retained as a compatibility alias.

## Funding Assumptions

Funding behavior is configurable:

- `funding_mode = prototype_bar_sum`: original prototype behavior; sum every aligned `funding_rate` row between entry and exit.
- `funding_mode = event_aware`: use explicit `funding_event` / `is_funding_event` markers if present; otherwise use UTC hour modulo `funding_interval_hours` as a Binance-style fallback.

Funding notional is configurable:

- `funding_notional_mode = initial_notional`: apply funding to fixed entry notional.
- `funding_notional_mode = dynamic_position_value`: approximate funding using current perp close value of the original position quantity.

The default stays `prototype_bar_sum` plus `initial_notional` for backward compatibility and simple auditability. Reports and manifests record the chosen mode and funding rows used.

## Equity Curves

The output `equity_curve.parquet` includes both audit and risk views:

- `realized_equity_usd`: changes only when trades close.
- `mark_to_market_equity_usd`: marks open positions on each market bar using current close prices.
- `equity_usd`: alias for the mark-to-market equity used by primary risk metrics.
- `realized_drawdown`: drawdown from realized-only equity.
- `mark_to_market_drawdown`: drawdown from mark-to-market equity.
- `drawdown`: alias for mark-to-market drawdown.

Primary drawdown, Sharpe, annualized return, and cumulative return use the mark-to-market curve by default. Realized-only columns remain available because they are easier to audit against closed trades, but they can understate intratrade risk.

## Split-Aware Evaluation

The config field `reporting.primary_split` controls the main leaderboard. The default is:

```yaml
reporting:
  primary_split: test
```

This means:

- `strategy_metrics.parquet` and `leaderboard.parquet` are test-split primary outputs.
- `primary_trade_log.parquet` contains only trades from the configured primary split.
- `split_summary.parquet` keeps train, validation, and test trade summaries separate.
- `combined_strategy_metrics.parquet` is written as a secondary diagnostic when enabled.

This prevents the main leaderboard from silently mixing in-sample and out-of-sample trades.
The config model also validates that `reporting.primary_split` is present in `selection.split_filter` unless the primary split is explicitly set to `combined`.

## Capital And Leverage Checks

The engine reports implied gross leverage:

```text
gross exposure = 2 * position_notional * max_open_positions
implied gross leverage = gross exposure / initial_capital
```

Config fields:

```yaml
portfolio:
  max_gross_leverage: 2.0
  leverage_check_mode: warn
```

`leverage_check_mode` can be:

- `off`
- `warn`
- `fail`

For course-project clarity, the default warns rather than failing, but submission/demo configs should avoid unrealistic leverage.

## Metrics

Primary strategy metrics include:

- cumulative return
- annualized return
- simple annualized Sharpe
- raw-period Sharpe
- autocorrelation-adjusted Sharpe diagnostic
- mark-to-market max drawdown
- realized-only max drawdown
- win rate
- profit factor
- average and median trade return
- expectancy per trade
- average and median holding hours
- max consecutive losses
- exposure time fraction
- average and maximum gross leverage
- turnover
- fee, gas, other friction, funding, and embedded slippage diagnostics
- funding contribution share

Sharpe caveat:

The main Sharpe is a simple square-root-scaled annualized Sharpe from the mark-to-market return stream. This is useful for comparison, but sparse and serially correlated strategy returns can make annualization optimistic. Treat it as a compact diagnostic rather than proof of live-trading quality.

No-trade caveat:

If a strategy has no executable trades on the evaluation split, the backtest now records:

- `status = no_tradable_signals` when the upstream signal layer produced no tradeable signals
- `status = no_executed_trades` when signals existed but execution logic never opened a position
- `diagnostic_reason` and `skip_reason`

In those cases, Sharpe and drawdown style risk metrics are written as `NaN`, not `0`, because `0` can falsely imply a meaningful flat-risk outcome when nothing actually traded.

## Outputs

The backtester writes:

- `trade_log.parquet` / `trade_log.csv`
- `primary_trade_log.parquet` / `primary_trade_log.csv`
- `equity_curve.parquet` / `equity_curve.csv`
- `strategy_metrics.parquet` / `strategy_metrics.csv`
- `combined_strategy_metrics.parquet` / `combined_strategy_metrics.csv`
- `split_summary.parquet` / `split_summary.csv`
- `leaderboard.parquet` / `leaderboard.csv`
- `backtest_report.md`
- `backtest_manifest.json`
- figures under `figures/`

Strategy metric and leaderboard tables now also preserve `status`, `diagnostic_reason`, `skip_reason`, and split-level signal counts so zero-trade rows remain interpretable downstream.

Plots:

- mark-to-market cumulative return by strategy
- mark-to-market drawdown by strategy
- primary-split trade-return distribution boxplot

## Command

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main backtest --config configs/backtests/default.yaml
```

## Main Files

- engine: `src/funding_arb/backtest/engine.py`
- metrics helpers: `src/funding_arb/evaluation/metrics.py`
- config model: `src/funding_arb/config/models.py`
- default config: `configs/backtests/default.yaml`
- CLI wrapper: `scripts/backtests/run_backtest.py`
- tests: `tests/unit/test_backtest_engine.py`, `tests/unit/test_metrics.py`

## Limitations

The engine is still a research backtester, not a production execution simulator.

Known simplifications:

- single-asset only
- one open position per strategy
- no partial exits
- no order book, latency, liquidation, margin, or borrow-cost model
- no intrabar stop execution
- event-aware funding uses cleaned dataset markers when available and a simple UTC-hour fallback otherwise
- no dynamic hedge rebalancing beyond the optional dynamic funding notional approximation

These limitations should be disclosed in the final presentation and interpreted as prototype boundaries, not hidden assumptions.
