# Backtesting Module

## Status

The backtesting module is implemented and has been upgraded from a first-pass realized-PnL simulator into a more conservative research backtester.

Primary implementation:

- `src/funding_arb/backtest/engine.py`

Primary documentation:

- `docs/backtest.md`

## What It Does

- consumes standardized signals from the signal layer
- simulates one delta-neutral position at a time per strategy
- applies explicit fee, slippage, gas, and funding logic
- writes trade logs, mark-to-market equity curves, realized-only audit columns, metrics, plots, and a markdown report
- writes a separate `primary_trade_log` so plots and primary metrics stay aligned with the configured evaluation split
- ranks strategies on the configured primary split, which defaults to `test`
- records funding mode, hedge mode, leverage diagnostics, and stop-logic assumptions in the manifest

## Current Design Choice

The primary equity curve is now **mark-to-market**. Open positions are marked on every bar using current market prices, while realized-only equity is retained for auditability.

This is intentionally more conservative:

- primary drawdown and Sharpe reflect intratrade risk
- realized-only drawdown remains available as a secondary diagnostic
- the main leaderboard defaults to out-of-sample `test` trades instead of silently mixing train/validation/test
- slippage remains embedded in effective prices and is not deducted twice
- no-trade rows now carry explicit `status` / `diagnostic_reason`, and no-trade Sharpe or drawdown metrics are written as `NaN` rather than misleading `0`

## Outputs

Default output root:

- `data/artifacts/backtests/<provider>/<symbol>/<frequency>/<run_name>/`

Key outputs:

- `trade_log.parquet`
- `primary_trade_log.parquet`
- `equity_curve.parquet`
- `strategy_metrics.parquet`
- `combined_strategy_metrics.parquet`
- `split_summary.parquet`
- `leaderboard.parquet`
- `backtest_report.md`
- `backtest_manifest.json`

## Command

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main backtest --config configs/backtests/default.yaml
```
