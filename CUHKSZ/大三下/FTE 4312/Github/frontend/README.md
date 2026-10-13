# Frontend Demo

This directory contains the presentation-friendly dashboard for the course-project prototype.

The frontend is intentionally lightweight:

- `Vite + TypeScript`
- no backend required
- reads one exported local snapshot plus copied chart assets
- includes a generated final report under `public/report/`
- builds into a static site that can be published with GitHub Pages

## What the dashboard shows

- project overview and architecture story
- key data-quality and backtest charts
- baseline and deep-learning model-zoo comparison cards
- strategy metrics table with test-primary, mark-to-market backtest fields
- vault status and mock operator update summary
- local deposit / withdraw / strategy update / NAV update simulation
- example activity log

## Recommended startup flow

1. Run the end-to-end demo workflow once:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml
```

2. If you only need to refresh the snapshot without rerunning the whole pipeline, export the local demo snapshot:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts/demo/export_demo_snapshot.py --config configs/demo/default.yaml
```

3. Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

4. Open the local Vite URL, usually:

```text
http://127.0.0.1:5173
```

## Synthetic Showcase Bundle

An isolated frontend-ready synthetic bundle is also available for presentation use.

Rebuild it with:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main build-demo-showcase --config configs/demo/showcase.yaml
```

Then open:

```text
http://127.0.0.1:5173/?mode=demo_showcase
```

This reads from `frontend/public/demo_showcase/` and keeps the default
`frontend/public/demo/` plus `frontend/public/report/` outputs untouched.

For the presentation-first standalone HTML page, open:

```text
http://127.0.0.1:5173/pre/index.html
```

To enter the slide-style PPT mode directly, open:

```text
http://127.0.0.1:5173/pre/index.html?ppt=1
```

This page reads the same isolated `demo_showcase` bundle but arranges it as a
one-page presentation flow for the pre/final demo. It is aligned to the course
presentation rubric and is intended to cover project idea, technical design,
implementation, evaluation/results, and system demo in one page.

For a fully portable copy that does not need Vite, `frontend/public/demo_showcase/`,
or any local server, use:

```text
frontend/public/pre/standalone.html
```

This single HTML file embeds the same presentation CSS, JavaScript, JSON
snapshots, and chart PNGs used by `pre/index.html`. It can be copied to another
computer and opened directly in a browser. Rebuild it after changing the
presentation page or demo artifacts with:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts/demo/build_standalone_presentation.py
```

## Production / static build

To build the showcase as a static site:

```bash
cd frontend
npm install
npm run build
```

The built output lands in `frontend/dist/`.
Because the dashboard now uses relative asset paths plus `import.meta.env.BASE_URL`, the same build works locally and on GitHub Pages-style subpaths.

The repository also includes `.github/workflows/deploy-showcase.yml` for GitHub Pages deployment.

## Re-export after new results

If you rerun upstream pipeline stages and want the dashboard to reflect new artifacts, rerun:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts/demo/export_demo_snapshot.py --config configs/demo/default.yaml
& 'd:\MG\anaconda3\python.exe' scripts/reports/generate_final_report.py --config configs/reports/final_report.yaml
```

## Caveats

- The demo console is local and simulated; it is not a wallet-connected production dApp.
- The operator/oracle flow is a prototype bridge from local artifacts to mock vault updates.
- Backtest cards use mark-to-market drawdown/Sharpe by default, while realized-only metrics remain in the exported artifacts for auditability.
- Broadcasted on-chain updates still depend on your local/testnet vault deployment and environment variables.
