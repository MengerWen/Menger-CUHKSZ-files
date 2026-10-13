# Data Quality Reporting Module

## Purpose

Create presentation-friendly exploratory summaries for the canonical hourly market dataset produced by `fetch-data`.

## Inputs

- canonical dataset: `data/processed/<provider>/<symbol>/<frequency>/hourly_market_data.parquet`
- optional manifest: `data/processed/<provider>/<symbol>/<frequency>/manifest.json`
- report config: `configs/reports/data_quality.yaml`

## Outputs

The reporting command writes artifacts under:

- `reports/data_quality/<provider>/<symbol>/<frequency>/tables/`
- `reports/data_quality/<provider>/<symbol>/<frequency>/figures/`
- `reports/data_quality/<provider>/<symbol>/<frequency>/report.md`
- `reports/data_quality/<provider>/<symbol>/<frequency>/summary.json`

## Generated Summaries

- missingness summary by column
- time coverage and gap summary
- distribution summary for price, funding, spread, return, and volume variables
- basic correlation matrix for key derived variables

## Generated Figures

- funding-rate time-series plot
- perp-versus-spot spread plot
- rolling return-volatility plot
- correlation heatmap

## Command

```bash
& 'd:\MG\anaconda3\python.exe' -m src.main report-data-quality --config configs/reports/data_quality.yaml
```

## Module Ownership

- `src/funding_arb/reporting/data_quality.py`: summary computation, plotting, markdown generation
- `configs/reports/data_quality.yaml`: report scope and styling defaults
- `scripts/reports/report_data_quality.py`: wrapper entrypoint for users who prefer script-based execution
- `reports/data_quality/`: generated artifacts only, not hand-edited source files

## Practical Notes

- The report assumes the canonical dataset already exists, so `fetch-data` should run first.
- The current plots are optimized for course-project presentation use rather than publication-grade statistical diagnostics.
- Future extensions can add seasonality views, regime slices, exchange comparison panels, or notebook exports without changing the command structure.