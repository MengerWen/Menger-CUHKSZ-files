# Mock Off-Chain To On-Chain Integration

This document explains the prototype bridge between the Python research stack and the Solidity vault.

The goal is not to build a production oracle network. The goal is to show, clearly and locally, how an off-chain strategy system can produce a simplified vault update that is submitted by a trusted operator account.

## What This Module Does

The integration layer:

- reads generated strategy artifacts from the Python side
- selects one strategy snapshot to represent the current off-chain state
- converts that snapshot into a mock vault update payload
- optionally calls the vault contract with `updateStrategyState`, `updateNav`, or `updatePnl`
- writes JSON and markdown artifacts so the flow is easy to inspect in demos and slides

The default mode is `dry-run`. In dry-run mode, the module does not broadcast any transaction. It only generates the exact payload and calldata that would be sent on-chain.

## What This Module Does Not Do

This module does not:

- provide a decentralized oracle network
- verify strategy results trustlessly
- implement message relaying, signatures, committees, or consensus
- run as a persistent backend service
- claim production security

This is a course-project operator/oracle-style demo flow.

## Current Inputs

The default config uses:

- standardized signals: `data/artifacts/signals/binance/btcusdt/1h/baseline/signals.parquet`
- backtest leaderboard: `data/artifacts/backtests/binance/btcusdt/1h/baseline_signals_default/leaderboard.parquet`
- backtest manifest: `data/artifacts/backtests/binance/btcusdt/1h/baseline_signals_default/backtest_manifest.json`

By default the module:

1. ranks strategies using `total_net_pnl_usd`
2. prefers leaderboard rows with real primary-split trades when `prefer_traded_strategy: true`
3. selects the top-ranked strategy unless `strategy_name` is specified
4. chooses the latest signal row from the preferred split order
5. converts the signal state into a vault `StrategyState`
6. converts leaderboard PnL into mock stablecoin asset units
7. computes hashes for signal, metadata, and report payloads

## Backtest-Aware Selection

The upgraded backtest leaderboard is test-split-primary by default and includes a `has_trades` field. This matters for the operator flow: a no-trade row can have zero drawdown and zero return, but it should not automatically outrank a traded strategy when the purpose is to demonstrate a strategy state update.

The integration config therefore includes:

```yaml
selection:
  ranking_metric: total_net_pnl_usd
  ranking_ascending: false
  prefer_traded_strategy: true
```

If `prefer_traded_strategy` is enabled, rows with `has_trades = true` or `trade_count > 0` are considered before no-trade rows. This preserves a useful demo behavior while still keeping the backtest leaderboard honest about out-of-sample performance.

## Interface Assumptions

### Off-chain assumptions

- the signal layer already writes normalized `signals.parquet`
- the backtest layer already writes a primary-split leaderboard
- the primary leaderboard uses mark-to-market risk metrics and keeps realized-only fields for auditability
- `total_net_pnl_usd` is treated as the simplest demo proxy for off-chain strategy performance
- no-trade rows are not preferred over traded rows by default

### On-chain assumptions

- the vault is already deployed
- the caller is the vault `owner` or `operator`
- the vault uses the prototype interface defined in [contracts.md](contracts.md)
- the mock stablecoin is treated as approximately `1 token = 1 USD`

### Economic assumptions

- `base_nav_assets` is a demo starting NAV configured in YAML
- `reported_nav_assets = base_nav_assets + summary_pnl_assets`
- this is a simplified accounting mirror, not a complete live accounting engine
- the off-chain leaderboard result is trusted by the operator in this prototype

## Files

The main implementation is:

- [pipeline.py](../src/funding_arb/integration/pipeline.py)
- [default.yaml](../configs/integration/default.yaml)
- [sync_vault.py](../scripts/integration/sync_vault.py)

## CLI Usage

Unified CLI:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main sync-vault --config configs/integration/default.yaml
```

Wrapper script:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts/integration/sync_vault.py --config configs/integration/default.yaml
```

## Demo Walkthrough

### 1. Generate upstream artifacts

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main generate-signals --source baseline --config configs/signals/default.yaml
& 'd:\MG\anaconda3\python.exe' -m src.main backtest --config configs/backtests/default.yaml
```

### 2. Deploy the vault locally

```powershell
cd contracts
forge script script/DeployLocal.s.sol:DeployLocal --rpc-url http://127.0.0.1:8545 --broadcast
```

Update your environment so `VAULT_ADDRESS` points to the deployed vault and `PRIVATE_KEY` points to the owner/operator key.

### 3. Run the integration in dry-run mode first

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main sync-vault --config configs/integration/default.yaml
```

This writes:

- `selected_strategy_summary.json`
- `vault_update_plan.json`
- `contract_call_summary.json`
- `integration_report.md`

under:

- `data/artifacts/integration/binance/btcusdt/1h/mock_operator_default/`

### 4. Switch to broadcast mode

Edit [default.yaml](../configs/integration/default.yaml):

- set `contract.broadcast: true`
- keep only one of `update_nav: true` or `update_pnl: true`

Then rerun:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main sync-vault --config configs/integration/default.yaml
```

### 5. Inspect the result

The integration artifact directory will include the planned payload plus the execution summary.

In a local demo, this gives a simple narrative:

1. research and backtesting produced a current strategy view
2. the operator module converted that view into a compact on-chain update
3. the vault recorded the state change on-chain

## Educational Value

This module is intentionally simple because it helps explain the hybrid architecture clearly:

- ML and backtesting stay off-chain
- vault accounting stays on-chain
- a trusted operator bridges the two
- mark-to-market backtest risk is represented off-chain, while the vault only receives simplified state and NAV/PnL updates

That is exactly the point this course project is trying to demonstrate.
