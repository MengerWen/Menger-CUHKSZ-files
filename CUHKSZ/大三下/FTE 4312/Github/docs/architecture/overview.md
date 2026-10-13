# Architecture Overview

## Goal

Build a course-project prototype that demonstrates a complete path from historical perpetual funding-rate data to quant signals, backtest results, and a Solidity vault demo.

## Main Layers

### 1. Quantitative research pipeline

The Python package under `src/funding_arb/` will own:

- market-data ingestion
- cleaning and canonical alignment
- feature engineering
- label generation
- baseline models
- deep-learning experiments
- cost-aware backtesting
- evaluation metrics

### 2. Solidity vault prototype

The `contracts/` workspace will hold a single vault-oriented prototype for:

- deposits and withdrawals
- share accounting
- owner-controlled NAV/PnL reporting
- pause controls
- event logs for demo transparency

### 3. Frontend demo

The `frontend/` app will visualize:

- backtest summaries
- current demo strategy state
- vault balances and shares
- mock deposit/withdraw interactions

## Data Flow

1. Pull raw perpetual, spot, and reference/index data into `data/raw/`.
2. Normalize and clean into `data/interim/`.
3. Build features and labels into `data/processed/`.
4. Train baseline or deep-learning models and save artifacts under `data/artifacts/`.
5. Run cost-aware backtests and export summary outputs.
6. Surface those artifacts in the frontend and, where useful, mirror state in the vault prototype.

## Scope Notes

- No live trading engine is planned in this prototype.
- No backend API is required initially.
- Off-chain modeling drives the research workflow; on-chain logic focuses on accounting and demo state.

