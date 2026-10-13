# Frontend Demo

## Goal

Provide a lightweight dashboard for presenting the project during demos and the final course presentation.

## Implemented Views

- project summary and architecture snapshot
- key research and backtest charts
- strategy metrics and model comparison cards
- vault status and operator-sync summary
- activity log
- local mock interaction panel for:
  - deposit
  - withdraw
  - strategy activation
  - NAV/PnL update
  - share/accounting status reset

## Stack Choice

The frontend uses `Vite + vanilla TypeScript` to keep setup and maintenance light.

## Data Flow

1. Python pipelines generate research, model, backtest, and integration artifacts.
2. `scripts/demo/export_demo_snapshot.py` packages those results into:
   - `data/artifacts/demo/demo_snapshot.json`
   - `frontend/public/demo/demo_snapshot.json`
   - `frontend/public/demo/assets/*`
3. `scripts/reports/generate_final_report.py` packages the final report into:
   - `reports/final/binance/btcusdt/1h/*`
   - `frontend/public/report/*`
4. The dashboard fetches `demo/demo_snapshot.json` and renders the full story without a backend.

## Startup

```powershell
& 'd:\MG\anaconda3\python.exe' scripts/demo/export_demo_snapshot.py --config configs/demo/default.yaml
& 'd:\MG\anaconda3\python.exe' scripts/reports/generate_final_report.py --config configs/reports/final_report.yaml
cd frontend
npm install
npm run dev
```

## Static deployment

```powershell
cd frontend
npm install
npm run build
```

The repository now includes a GitHub Pages workflow at `.github/workflows/deploy-showcase.yml`.
The frontend uses relative asset paths and build-time base URLs, so the same site can be served locally or from a GitHub Pages subpath.

## Caveats

- The dashboard is optimized for demo clarity, not wallet-grade production UX.
- The local vault console is an educational simulator and intentionally mirrors mock accounting assumptions.
- The frontend does not execute trades or act as a live oracle network.
