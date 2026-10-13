# Reviewer Guide

This page is for a grader, instructor, or teammate who wants the fastest way to understand what this repository does without reverse-engineering the whole codebase.

## In One Sentence

This repository is a course-project prototype that studies delta-neutral funding-rate arbitrage off-chain, evaluates it with explicit backtesting, mirrors strategy state into a Solidity vault prototype, and packages the whole story into a demo-friendly frontend.

## What Is Already Working

The project has a coherent runnable path across all major layers:

1. market data ingestion and canonicalization
2. exploratory reporting and data-quality checks
3. feature engineering
4. post-cost label generation
5. baseline and deep-learning model-zoo modeling
6. unified signal generation
7. delta-neutral backtesting
8. robustness analysis
9. Solidity vault accounting prototype
10. mock off-chain to on-chain sync flow
11. frontend snapshot export and dashboard

## Best 10-Minute Review Path

1. Read [architecture.md](architecture.md) for the overall design.
2. Read [demo.md](demo.md) for the one-command demonstration path.
3. Inspect the latest workflow summary:
   - `data/artifacts/demo/workflow/full_demo_default/demo_workflow_summary.json`
4. Inspect the latest dashboard snapshot:
   - `frontend/public/demo/demo_snapshot.json`
5. When a model or strategy row shows zero trades or zero signals, inspect the corresponding manifest/report `status` and `reason` fields rather than relying on the numeric columns alone.
5. If you want code-level confirmation, scan:
   - `src/funding_arb/data/`
   - `src/funding_arb/features/`
   - `src/funding_arb/labels/`
   - `src/funding_arb/models/`
   - `src/funding_arb/backtest/`
   - `contracts/src/`

## Best Commands To Run

### Full demo

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml
```

### Python tests

```powershell
& 'd:\MG\anaconda3\python.exe' -m pytest tests/unit -q
```

### Solidity tests

```powershell
cd contracts
forge test -vv
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## What To Look At In The Results

### Quant side

- `reports/data_quality/`
  Shows whether the historical sample is usable and presentation-worthy.
- `data/artifacts/models/baselines/`
  Shows interpretable benchmark behavior.
- `data/artifacts/models/dl/`
  Shows the first sequence-model experiment, including whether a run was healthy or degenerate.
- `data/artifacts/backtests/`
  Shows full and primary-split trade logs, mark-to-market equity curves, realized-only audit columns, test-primary leaderboard, summary metrics, and no-trade statuses.
- `reports/robustness/`
  Shows whether the results are fragile under changed assumptions.

### On-chain side

- `contracts/src/DeltaNeutralVault.sol`
  Main vault prototype.
- `contracts/test/DeltaNeutralVault.t.sol`
  Contract behavior tests.
- `data/artifacts/integration/`
  Mock operator payloads bridging off-chain outputs to the vault.

### Presentation side

- `frontend/public/demo/demo_snapshot.json`
  Aggregated local state consumed by the dashboard.
- `frontend/public/demo/assets/`
  Demo charts used directly in the frontend.

## What Is Deliberately Simplified

This repository is not pretending to be a production trading system.

Important prototype simplifications:

- one main working market path: Binance `BTCUSDT`
- one primary research frequency: `1h`
- explicit but simplified execution and slippage model
- accounting-oriented vault instead of live multi-venue execution
- trusted operator sync instead of decentralized oracle infrastructure
- lightweight local dashboard instead of wallet-heavy app architecture

## What A Reviewer Should Notice About Zero Rows

The repository now distinguishes two very different situations:

- a genuine flat or inactive strategy that still had a valid model-selection path
- a degenerate experiment where validation/test never produced tradable signals

That distinction appears in manifests and reports through fields such as:

- `degenerate_experiment`
- `status`
- `reason`
- `diagnostic_reason`
- `signal_count_by_split`

So if you see `0 trades` or `NaN` Sharpe/drawdown, treat those status fields as the authoritative explanation.

## How To Interpret "Completed"

For this course-project prototype, "completed" means:

- the pipeline is coherent end to end
- major assumptions are explicit
- outputs are reproducible enough for local demos
- artifacts are inspectable by a reviewer
- each layer is testable and documented

It does not mean:

- production safety
- exchange-grade execution
- trustless oracle infrastructure
- multi-asset, multi-venue deployment readiness

## If You Only Read Three Things

1. [architecture.md](architecture.md)
2. [demo.md](demo.md)
3. [contracts.md](contracts.md)
