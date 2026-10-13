# Deep Learning-Based Delta-Neutral Statistical Arbitrage on Perpetual Funding Rates

This repository is a course-project prototype for a hybrid quantitative-research and DeFi-system workflow.

The project asks one concrete question:

> Can we identify post-cost, delta-neutral funding-rate arbitrage opportunities off-chain, evaluate them rigorously, and then mirror strategy state into a simple on-chain vault prototype?

The answer here is intentionally prototype-scoped rather than production-scoped.
The repository is designed to tell a clean end-to-end story:

- historical perpetual and spot data ingestion
- feature engineering and post-cost label construction
- rule-based, simple ML, and sequence-model benchmarks
- explicit delta-neutral backtesting with costs and funding
- a Solidity vault for accounting and state updates
- a lightweight frontend for demo and presentation use

## Why This Project Matters

Perpetual funding-rate arbitrage is a strong course-project topic because it combines:

- a real market microstructure mechanism: funding keeps perpetual prices anchored to spot
- a tractable delta-neutral framing: short perp plus long spot or index proxy
- a practical ML question: not "will price go up," but "is the dislocation still worth trading after costs?"
- a natural hybrid architecture: modeling happens off-chain, while accounting and state can be represented on-chain

That makes the project financially meaningful, technically layered, and still small enough to explain clearly in a final presentation.

## What Is Implemented

The repository already contains working prototypes for all major layers:

| Layer | Status | Main output |
| --- | --- | --- |
| Historical data pipeline | Implemented | canonical hourly market dataset |
| Data-quality / EDA reporting | Implemented | tables, charts, markdown report |
| Feature engineering | Implemented | interpretable feature table |
| Label generation | Implemented | post-cost supervised dataset |
| Rule-based and simple ML baselines | Implemented | artifacts, metrics, predictions |
| Deep-learning model zoo | Implemented | LSTM/GRU/TCN/Transformer artifacts and comparison reports |
| Unified signal layer | Implemented | standardized `signals.parquet` |
| Delta-neutral backtest | Implemented | trade log, mark-to-market equity, test-primary leaderboard |
| Robustness analysis | Implemented | sensitivity tables and figures |
| Solidity vault prototype | Implemented | contract, tests, deploy/update scripts |
| Off-chain to on-chain mock integration | Implemented | operator/oracle-style sync flow |
| Final report generator | Implemented | markdown + HTML report with copied figures |
| Showcase website | Implemented | static-buildable dashboard plus final-report entry point |
| End-to-end demo workflow | Implemented | one-command orchestration path |

## Architecture At A Glance

The system is organized around one data flow:

`market data -> canonical dataset -> features -> labels -> model outputs -> signals -> backtest -> vault sync payload -> frontend/demo`

Main code areas:

- `src/funding_arb/`
  Core Python package for data, features, labels, models, signals, backtesting, reporting, integration, and demo orchestration.
- `contracts/`
  Foundry-based Solidity workspace for the vault prototype and mocks.
- `frontend/`
  Lightweight Vite + TypeScript dashboard for presentation.

## Repository Map

```text
.
+-- Proposal.md
+-- README.md
+-- AGENTS.md
+-- configs/
+-- data/
+-- docs/
+-- scripts/
+-- src/funding_arb/
+-- tests/
+-- contracts/
`-- frontend/
```

Key directories:

- `configs/`
  Centralized YAML assumptions for data, features, labels, models, backtests, reports, integration, and demo workflow.
- `data/`
  Raw, interim, processed, and artifact outputs.
- `docs/`
  Design notes, module docs, reviewer guidance, and demo instructions.
- `scripts/`
  Thin command-line wrappers around the Python modules.
- `tests/`
  Fast Python unit tests plus Solidity tests under `contracts/test/`.
- `contracts/`
  Vault contract, Foundry tests, and deploy/update scripts.
- `frontend/`
  Demo dashboard that reads exported local JSON plus chart assets.

## Setup

### Python

All Python commands in this repository should use:

```text
d:\MG\anaconda3\python.exe
```

If you are already using that environment, the core dependencies are available.

If you need to recreate the environment elsewhere:

```powershell
& 'd:\MG\anaconda3\python.exe' -m pip install -e .[dev]
```

If you want to ensure deep-learning extras in a fresh environment:

```powershell
& 'd:\MG\anaconda3\python.exe' -m pip install -e .[dev,ml]
```

### Solidity

The Solidity prototype uses Foundry under `contracts/`.

```powershell
cd contracts
forge build
forge test -vv
```

### Frontend

The active frontend workspace is `frontend/`.
The active Solidity workspace is `contracts/`.

```powershell
cd frontend
npm install
npm run dev
```

## Fastest Demo Path

The cleanest end-to-end presentation command is:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml
```

That workflow:

1. fetches or reuses market data
2. regenerates data-quality outputs
3. rebuilds features and labels
4. trains baseline models
5. optionally trains the reference LSTM
6. optionally compares the Phase 1/2 deep-learning model zoo
7. regenerates standardized signals
8. reruns the baseline backtest
9. prepares the mock vault update payload
10. exports frontend-ready demo artifacts

Workflow outputs:

- `data/artifacts/demo/workflow/full_demo_default/demo_workflow_summary.json`
- `data/artifacts/demo/workflow/full_demo_default/demo_workflow_report.md`
- `data/artifacts/demo/demo_snapshot.json`
- `frontend/public/demo/demo_snapshot.json`

Then open the dashboard:

```powershell
cd frontend
npm install
npm run dev
```

Default local URL:

```text
http://127.0.0.1:5173
```

## Synthetic Demo Mode

The repository now includes a completely separate synthetic showcase path for
presentation-only artifacts.

Use it when you need a polished demo bundle with:

- `DEMO ONLY` labels on generated JSON, CSV, parquet, markdown, HTML, and chart assets
- isolated output directories that do not overwrite real experiment artifacts
- a coherent strict-vs-exploratory story for reports, dashboards, and screenshots

Rebuild the full synthetic bundle with one command:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main build-demo-showcase --config configs/demo/showcase.yaml
```

Synthetic showcase output directories:

- `data/demo_artifacts/showcase/binance/btcusdt/1h/`
- `reports/demo_showcase/binance/btcusdt/1h/`
- `frontend/public/demo_showcase/`

Frontend launch:

```powershell
cd frontend
npm install
npm run dev
```

Then open the isolated bundle at:

```text
http://127.0.0.1:5173/?mode=demo_showcase
```

The synthetic final report is copied to:

- `frontend/public/demo_showcase/report/`

while the default real-report path under `frontend/public/report/` stays unchanged.

## Main Commands

### Core pipeline

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main fetch-data
& 'd:\MG\anaconda3\python.exe' -m src.main report-data-quality
& 'd:\MG\anaconda3\python.exe' -m src.main build-features
& 'd:\MG\anaconda3\python.exe' -m src.main build-labels
```

### Modeling and signals

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main train-baseline
& 'd:\MG\anaconda3\python.exe' -m src.main evaluate-baseline
& 'd:\MG\anaconda3\python.exe' -m src.main train-dl
& 'd:\MG\anaconda3\python.exe' -m src.main compare-dl --config configs/experiments/dl/regression_all.yaml
& 'd:\MG\anaconda3\python.exe' -m src.main generate-signals --source baseline
& 'd:\MG\anaconda3\python.exe' -m src.main generate-signals --source dl
```

`train-dl` keeps LSTM as the stable reference model. The default `dl` signal source points to the current Phase 2 comparison winner from the regression bundle, so run `compare-dl` before refreshing deep-learning signals if you want the latest model-zoo result reflected downstream.

## Degenerate Experiment Guardrails

The repository now treats degenerate model-selection paths as first-class diagnostics instead of silent success cases.

Default behavior:

- label manifests record split-level `tradeable_rate`, `profitable_rate`, and degenerate reasons
- baseline and deep-learning threshold search fail fast when validation cannot support threshold selection
- checkpoint selection that only works through a fallback metric is no longer silently accepted by default
- signal manifests record `status`, `reason`, `degenerate_experiment`, `signal_count_by_split`, `selected_threshold`, and `threshold_search_summary`
- backtest leaderboards keep `trades = 0` when nothing executes, but also add `status` and `diagnostic_reason`
- no-trade Sharpe and drawdown fields are written as `NaN` rather than misleading `0`

This matters because a table full of `0 trades / 0 pnl / 0 sharpe` can otherwise hide the fact that validation/test produced no tradable signals at all.

If you explicitly want a diagnostic artifact even for a degenerate run, set the relevant config flags to `true`:

- baseline: `threshold_search.allow_degenerate_fallback`
- deep learning: `threshold_search.allow_degenerate_fallback`
- deep learning checkpoint selection: `training.allow_degenerate_fallback`

Those fallback modes are opt-in. They are meant for inspection and report generation, not for pretending the experiment was healthy.

### Evaluation and demo

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main backtest
& 'd:\MG\anaconda3\python.exe' -m src.main robustness-report
& 'd:\MG\anaconda3\python.exe' -m src.main generate-final-report --config configs/reports/final_report.yaml
& 'd:\MG\anaconda3\python.exe' -m src.main sync-vault
& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml
```

## Suggested Reading Order

If you are grading or reviewing the repository, this is the most efficient order:

1. [Proposal.md](Proposal.md)
2. [docs/review_guide.md](docs/review_guide.md)
3. [docs/architecture.md](docs/architecture.md)
4. [docs/demo.md](docs/demo.md)
5. Layer-specific docs as needed:
   - [docs/data_schema.md](docs/data_schema.md)
   - [docs/features.md](docs/features.md)
   - [docs/labels.md](docs/labels.md)
   - [docs/baselines.md](docs/baselines.md)
   - [docs/models.md](docs/models.md)
   - [docs/signals.md](docs/signals.md)
   - [docs/backtest.md](docs/backtest.md)
   - [docs/contracts.md](docs/contracts.md)
   - [docs/integration.md](docs/integration.md)
   - [docs/final_checklist.md](docs/final_checklist.md)

The docs index with a more detailed reading path is in [docs/README.md](docs/README.md).

## Testing

Python unit tests:

```powershell
& 'd:\MG\anaconda3\python.exe' -m pytest tests/unit -q
```

Foundry contract tests:

```powershell
cd contracts
forge test -vv
```

Frontend production build:

```powershell
cd frontend
npm run build
```

The final report and showcase entry point are copied into `frontend/public/report/`, so the built static site can expose both the dashboard and the report together.

When validating modeling changes, it is worth checking the report and manifest status fields in addition to whether a command exited successfully. The intended outcome is:

- healthy experiment: `status = ok`
- warning-only continuation: explicit warning plus `degenerate_experiment = true`
- invalid model-selection path: command fails with a diagnostic error instead of silently writing normal-looking zero results

## Important Artifact Locations

- Canonical market data:
  - `data/processed/binance/btcusdt/1h/`
- Features:
  - `data/processed/features/binance/btcusdt/1h/`
- Supervised datasets:
  - `data/processed/supervised/binance/btcusdt/1h/`
- Baseline artifacts:
  - `data/artifacts/models/baselines/binance/btcusdt/1h/btcusdt_24h_default/`
- Deep-learning artifacts:
  - `data/artifacts/models/dl/binance/btcusdt/1h/lstm_regression_24h_default/`
- Signals:
  - `data/artifacts/signals/binance/btcusdt/1h/`
- Backtest outputs:
  - `data/artifacts/backtests/binance/btcusdt/1h/baseline_signals_default/`
  - includes `primary_trade_log`, mark-to-market equity, test-primary leaderboard, combined metrics as secondary diagnostics, and strategy-level `status` / `diagnostic_reason`
- Robustness report:
  - `reports/robustness/binance/btcusdt/1h/`
- Integration artifacts:
  - `data/artifacts/integration/binance/btcusdt/1h/mock_operator_default/`
- Frontend snapshot:
  - `frontend/public/demo/`
- Final report artifacts:
  - `reports/final/binance/btcusdt/1h/`
  - `frontend/public/report/`

## Current Limitations

This repository is intentionally prototype-level in several ways:

- data currently centers on one clear working source path: Binance `BTCUSDT`
- backtesting includes mark-to-market equity, costs, funding, and leverage checks, but is not a full exchange microstructure simulator
- the vault is an accounting and state-management prototype, not a live execution protocol
- the off-chain to on-chain bridge is a trusted operator flow, not a decentralized oracle network
- the frontend is a presentation dashboard, not a production dApp
- some generated artifacts are intentionally committed so a grader can inspect results without rerunning every stage

## Complete vs Prototype-Level

Complete enough for course-project demonstration:

- end-to-end research pipeline
- baseline and first DL benchmark
- cost-aware signal evaluation
- vault contract prototype with tests
- mock integration flow
- presentation-ready frontend and demo workflow
- formal final report and static-showcase packaging

Still prototype-level:

- multi-exchange data support
- richer position sizing and portfolio construction
- intrabar execution, liquidation, margin, and order-book simulation
- live oracle security model
- real wallet-connected vault UX

## Submission Notes

- Primary submission/demo path:
  - run `& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml`
  - then open the dashboard from `frontend/`
- Primary grading docs:
  - [docs/review_guide.md](docs/review_guide.md)
  - [docs/demo.md](docs/demo.md)
  - [docs/final_checklist.md](docs/final_checklist.md)
- If live Binance fetch fails because of temporary network conditions, the demo workflow can reuse the most recent local artifacts and still produce a valid demo snapshot.
- The repository is intentionally prototype-scoped and explicitly documents its simplifications instead of presenting itself as production-ready.

## More Documentation

- Instructor/reviewer guide: [docs/review_guide.md](docs/review_guide.md)
- Final submission checklist: [docs/final_checklist.md](docs/final_checklist.md)
- Documentation index: [docs/README.md](docs/README.md)
- Frontend notes: [frontend/README.md](frontend/README.md)
- Contract notes: [contracts/README.md](contracts/README.md)
