# Final Report And Showcase Design

## Goal

Close the two remaining course-deliverable gaps:

1. a complete final report with technical detail and charts
2. a showcase webpage that can be deployed statically instead of existing only as a local demo

## Chosen Approach

### Option selected

Use the existing artifact pipeline as the single source of truth, then layer two delivery outputs on top:

- a generated final report sourced from `data/artifacts/demo/demo_snapshot.json` plus robustness summaries
- a static-buildable Vite showcase that links directly to the generated report

### Why this approach

- It avoids writing a disconnected “report document” by hand.
- It keeps the final report consistent with the latest demo snapshot and charts.
- It preserves the course-project prototype scope instead of introducing a backend or CMS.
- It makes the public-facing site and the formal report share the same evidence base.

## Deliverables

### Final report

- Generator module under `src/funding_arb/reporting/final_report.py`
- CLI command: `generate-final-report`
- Config: `configs/reports/final_report.yaml`
- Outputs:
  - `reports/final/binance/btcusdt/1h/final_report.md`
  - `reports/final/binance/btcusdt/1h/final_report.html`
  - copied assets and summary JSON
  - mirrored copies under `frontend/public/report/`

### Showcase website

- Continue using the existing lightweight frontend
- Upgrade it from “local dashboard” to “static showcase” by:
  - adding report/download entry points
  - using relative asset paths
  - honoring `import.meta.env.BASE_URL`
  - adding a GitHub Pages deployment workflow

## Key Constraints

- Keep the implementation prototype-scoped and demo-friendly.
- Do not introduce a backend service just to host the showcase.
- Be explicit about the main empirical conclusion:
  predictive structure exists, but positive post-cost out-of-sample alpha is not currently demonstrated.

## Validation Plan

- Unit tests for final report generation
- Existing CLI parser tests updated for the new command
- Run the report generator against real project artifacts
- Build the frontend with `npm run build`
