# End-to-End Demo Workflow

This document defines the clearest runnable path for presenting the full repository story:

1. prepare market data
2. engineer features and labels
3. fit baseline and optional deep-learning models
4. compare the Phase 1/2 deep-learning model zoo when artifacts are available
5. convert outputs into standardized signals
6. backtest the strategy
7. prepare a mock off-chain to on-chain vault update
8. export frontend-ready demo artifacts
9. open the local dashboard

The goal is a presentation-friendly prototype flow, not a production job scheduler.

## Recommended Demo Entry Point

The main entry point is the unified CLI:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml
```

This workflow reuses the existing module-level configs and runs the full chain in order.

A compatible wrapper script is also available:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts/demo/run_demo_workflow.py --config configs/demo/workflow.yaml
```

It writes a workflow summary under:

- `data/artifacts/demo/workflow/full_demo_default/`

and, if the final snapshot stage succeeds, it refreshes:

- `data/artifacts/demo/demo_snapshot.json`
- `frontend/public/demo/demo_snapshot.json`

## What `run-demo` Executes

By default, `configs/demo/workflow.yaml` runs these stages:

1. `fetch-data`
2. `report-data-quality`
3. `build-features`
4. `build-labels`
5. `train-baseline`
6. `train-dl` as an optional stage
7. `compare-dl` as an optional Phase 2 stage
8. `generate-signals --source baseline`
9. `generate-signals --source dl` as an optional stage
10. `backtest`
11. `sync-vault`
12. `scripts/demo/export_demo_snapshot.py`

Design notes:

- Baseline training is treated as required because the backtest and demo currently depend on it.
- Deep learning and model-zoo comparison are treated as optional so the demo can still complete if `torch` is unavailable or if you want a faster presentation-only run.
- The default `dl` signal source points to the current Phase 2 comparison winner, while the LSTM remains the stable reference single-model config.
- `sync-vault` reuses the existing integration config and stays in dry-run mode until you explicitly switch it to broadcast mode.
- If a stage fails but the expected downstream artifact already exists locally, the workflow records a warning and reuses the existing artifact so the demo can still proceed.
- Default deep-learning configs now fail fast on degenerate validation threshold-selection paths. In the demo workflow that is acceptable because the DL stages are optional and the workflow summary will record the warning or artifact reuse explicitly.

## Exact Commands To Run In Order

### A. Core end-to-end demo

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml
```

### B. Open the frontend dashboard

```powershell
cd frontend
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

### C. Optional local-chain extension

If you want the vault stage to submit real local/testnet transactions instead of dry-run artifacts:

1. Start a local chain such as `anvil`
2. Deploy the vault:

```powershell
cd contracts
forge script script/DeployLocal.s.sol:DeployLocal --rpc-url http://127.0.0.1:8545 --broadcast
```

3. Update environment variables:
   - `VAULT_ADDRESS`
   - `PRIVATE_KEY`
4. Edit `configs/integration/default.yaml` and set:
   - `contract.broadcast: true`
   - keep exactly one of `update_nav` or `update_pnl` enabled
5. Rerun the full workflow or rerun just the integration stage:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main sync-vault --config configs/integration/default.yaml
```

## Dependencies

### Required for the core Python demo

- `d:\MG\anaconda3\python.exe`
- Python dependencies already represented in the repository and environment:
  - `pandas`
  - `numpy`
  - `PyYAML`
  - `scikit-learn`
  - `matplotlib`
  - `pyarrow`
  - `web3`
  - `torch` only if you keep the deep-learning stage enabled

### Required for the frontend

- Node.js
- `npm install` inside `frontend/`

### Required for the Solidity extension

- Foundry
- a local/testnet RPC endpoint

## Fallback Paths

### If you want the fastest presentation-ready path

Disable the optional deep-learning stages in `configs/demo/workflow.yaml`:

- `stages.train_deep_learning.enabled: false`
- `stages.compare_deep_learning.enabled: false`
- `stages.generate_deep_learning_signals.enabled: false`

Then rerun:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml
```

The snapshot exporter now degrades gracefully and will still package a valid frontend demo.

### If deep-learning training fails because the experiment degenerates

This is now treated as a modeling diagnostic, not as a silent success case.

Recommended interpretation:

- if the optional DL stage fails and the workflow reuses an existing artifact, read the workflow summary and the DL manifest `status` / `reason` fields
- if there is no existing artifact, keep the DL stages disabled for a pure baseline demo path
- only enable `allow_degenerate_fallback` intentionally when you want a diagnostic artifact that clearly documents the degenerate run

### If you want to skip the on-chain part for a pure research demo

Disable:

- `stages.sync_vault.enabled: false`

The frontend will still load, and the vault section will show placeholder execution status instead of a fresh operator payload.

### If you only need to refresh the dashboard after results already exist

You can skip the full workflow and only refresh the exported snapshot:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts/demo/export_demo_snapshot.py --config configs/demo/default.yaml
cd frontend
npm run dev
```

## Main Output Locations

- Workflow summary:
  - `data/artifacts/demo/workflow/full_demo_default/demo_workflow_summary.json`
  - `data/artifacts/demo/workflow/full_demo_default/demo_workflow_report.md`
  - stage rows record whether an optional failure was reused from existing artifacts
- Frontend snapshot:
  - `frontend/public/demo/demo_snapshot.json`
- Frontend image assets:
  - `frontend/public/demo/assets/`
- Backtest artifacts:
  - `data/artifacts/backtests/binance/btcusdt/1h/baseline_signals_default/`
  - primary leaderboard metrics are test-split and mark-to-market by default
  - `primary_trade_log` is the test-split trade log used for primary plots and summaries
  - combined metrics are kept separately as secondary diagnostics
  - leaderboard rows now include `status` and `diagnostic_reason` for no-trade strategies
- Vault integration artifacts:
  - `data/artifacts/integration/binance/btcusdt/1h/mock_operator_default/`

## Separate Synthetic Showcase Mode

For presentation scenarios where you need illustrative but fully isolated
results, use:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main build-demo-showcase --config configs/demo/showcase.yaml
```

This command writes only to the demo-only paths:

- `data/demo_artifacts/showcase/binance/btcusdt/1h/`
- `reports/demo_showcase/binance/btcusdt/1h/`
- `frontend/public/demo_showcase/`

The frontend can then load that bundle via:

```text
http://127.0.0.1:5173/?mode=demo_showcase
```

Every generated artifact in that branch is labeled `DEMO ONLY`, and the
default real pipeline plus real report directories remain unchanged.

## Why This Workflow Is Presentation-Friendly

- It uses one primary entry point.
- It is deterministic enough for repeated classroom demos.
- It clearly separates required and optional stages.
- It produces inspectable reports, tables, figures, and frontend-ready artifacts.
- It supports both a pure local research demo and a local-chain extension without pretending to be production infrastructure.
