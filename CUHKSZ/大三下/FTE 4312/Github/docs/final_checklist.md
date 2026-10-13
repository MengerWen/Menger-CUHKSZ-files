# Final Submission Checklist

This checklist is intended for the final repository review before course submission.

## Repository Story

- [x] The repository explains the problem, prototype scope, and end-to-end system motivation.
- [x] The README points to the main demo path and the main technical docs.
- [x] The docs directory has a coherent reading order for graders and teammates.

Primary reading path:

1. [../README.md](../README.md)
2. [review_guide.md](review_guide.md)
3. [architecture.md](architecture.md)
4. [demo.md](demo.md)

## Quantitative Pipeline

- [x] Historical market data pipeline exists and writes canonical hourly outputs.
- [x] Feature engineering is implemented and configurable.
- [x] Label generation is implemented and cost-aware.
- [x] Rule-based and simple ML baselines are implemented.
- [x] A deep-learning sequence-model zoo and comparison workflow are implemented.
- [x] Standardized signals are generated for backtesting.
- [x] Degenerate validation/test regimes are surfaced explicitly through diagnostics rather than silent zero-filled reports.

Main Python entrypoint:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml
```

## Backtesting

- [x] The backtest consumes standardized signals.
- [x] Entry/exit logic is explicit and documented.
- [x] Fees, slippage, and funding are modeled explicitly.
- [x] Full and primary-split trade logs, summary metrics, and figures are saved.
- [x] Backtest assumptions are documented in [backtest.md](backtest.md).
- [x] No-trade strategies are labeled with explicit status and reason fields instead of only `0` risk metrics.

## Solidity Vault Prototype

- [x] Deposit, withdraw, and share accounting are implemented.
- [x] NAV / PnL update flow is implemented.
- [x] Strategy state update flow is implemented.
- [x] Pause / unpause and owner/operator permissions are implemented.
- [x] Foundry tests exist and pass locally.
- [x] Deployment/update scripts exist for local/testnet demos.

Contract validation command:

```powershell
cd contracts
forge test -vv
```

## Frontend / Demo

- [x] The frontend is presentation-ready and lightweight.
- [x] The dashboard shows project overview, charts, metrics, vault state, and activity log.
- [x] A local simulation flow for deposit / withdraw / strategy update / NAV update is included.
- [x] Demo snapshot export exists and feeds the frontend directly.
- [x] Production build completes successfully.

Frontend validation command:

```powershell
cd frontend
npm run build
```

## Disclosure Of Limitations

- [x] Prototype-level simplifications are documented in the README.
- [x] The vault is clearly described as an accounting/state prototype, not a real execution engine.
- [x] The off-chain to on-chain bridge is clearly described as a trusted operator flow, not a production oracle network.
- [x] Demo fallback behavior is documented when network or optional artifacts are unavailable.
- [x] Model-selection fallback behavior is opt-in and documented when degenerate experiments are intentionally kept for diagnostics.

## Validation Status

Validated during the final audit:

- [x] `& 'd:\MG\anaconda3\python.exe' -m pytest tests/unit -q`
- [x] `cd contracts && forge test -vv`
- [x] `cd frontend && npm run build`

Final audit note:

- the end-to-end demo workflow is runnable and submission-friendly
- if live Binance fetch fails because of temporary network conditions, the workflow can reuse existing local artifacts and still produce a valid demo snapshot
- if validation/test degenerates into no tradable signals, manifests and reports now say so explicitly instead of silently presenting ordinary zero rows

## Remaining Prototype-Level Items

These are known limitations, not submission blockers:

- multi-exchange normalization is not implemented yet
- index-price ingestion is still limited
- intratrade mark-to-market equity is now the default backtest risk view, but intrabar execution and margin/liquidation modeling remain out of scope
- the vault is not production-audited
- the frontend is a demo dashboard, not a live wallet-connected application
