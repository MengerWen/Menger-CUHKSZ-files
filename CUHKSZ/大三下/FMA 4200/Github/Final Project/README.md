# FMA4200 Final Project

This project analyzes six Kenneth French value-weighted portfolios using the course-provided `Data.csv` file. The reproducible setup in this folder now covers environment verification, data cleaning, univariate return modeling, exogenous predictive modeling with cached Fama-French monthly factors, and the full trading-strategies stage including VAR analysis, cointegration-based stat-arb, and rolling mean-variance backtests.

## Canonical Report Sources

- `report/sections/03_individual_returns_modeling.md` and `report/sections/appendix_individual_returns_modeling.md` are the canonical Section 3 narrative sources.
- Those Section 3 files are written by the shared writers in `src/fma4200_project/univariate_modeling.py`.
- `src/fma4200_project/predictive_modeling.py` extends the same Section 3 source instead of maintaining a separate parallel draft.
- `src/fma4200_project/final_report_builder.py` now reads the canonical Section 3 and conclusions source files when building `report/final_report.md`, which reduces draft-versus-final drift.
- `report/guidance_lecture_mapping.md` records how the assignment guidance maps to lecture content, code, outputs, and final-report sections.

## Canonical Structure

- `src/fma4200_project/`: reusable project code for environment checks, data cleaning, and modeling
- `scripts/`: runnable scripts
- `data/processed/`: cleaned datasets and the data dictionary
- `output/figures/`: baseline figures
- `output/tables/`: sanity-check tables and descriptive statistics
- `output/models/`: reserved for later model outputs
- `report/sections/`: report section drafts
- `logs/`: environment and pipeline logs

## Return Units

The raw Kenneth French monthly value-weighted section reports returns in **percent** units. For example, `1.0866` means `1.0866%`, not `108.66%`.

- The canonical cleaned dataset is `data/processed/monthly_portfolio_returns_clean.csv`.
- Its six portfolio columns end with `_pct` and remain in percent units.
- The companion decimal-form dataset is `data/processed/monthly_portfolio_returns_decimal.csv`.
- Each `_dec` column is created by dividing the corresponding `_pct` column by `100.0`.

## Main Files

- Cleaned data: `data/processed/monthly_portfolio_returns_clean.csv`
- Decimal companion data: `data/processed/monthly_portfolio_returns_decimal.csv`
- Data dictionary: `data/processed/data_dictionary.csv`
- Environment summary: `environment_used.md`
- Pipeline log: `logs/pipeline_run.log`
- Data-processing report draft: `report/sections/02_data_source_processing.md`
- Individual-modeling report draft: `report/sections/03_individual_returns_modeling.md`
- Modeling appendix: `report/sections/appendix_individual_returns_modeling.md`
- Trading-strategies report draft: `report/sections/04_trading_strategies.md`
- Conclusions section draft: `report/sections/05_conclusions.md`
- Final integrated report source: `report/final_report.md`
- Final integrated report notebook: `report/final_report.ipynb`
- Final integrated report HTML: `report/final_report.html`
- Final integrated report PDF: `report/final_report.pdf`
- Final appendices: `report/final_appendices.md`
- Export notes: `report/export_notes.md`
- Guidance and lecture mapping: `report/guidance_lecture_mapping.md`
- Clean bibliography: `report/references.md`
- Reference verification notes: `report/references_verification.md`
- Rubric checklist: `report/rubric_checklist.md`
- Audit report: `report/audit_report.md`
- Submission runbook: `SUBMISSION_RUNBOOK.md`
- Zip-ready bundle: `submission_ready/`
- Zip-ready archive: `submission_ready.zip`
- Cached Fama-French factors: `data/processed/fama_french_3f_monthly.csv`
- Predictor panel: `data/processed/predictor_dataset_monthly.csv`
- Predictor source summary: `data/processed/predictor_source_summary.csv`
- Wealth index panel: `data/processed/portfolio_wealth_indices.csv`

## Reproducible Commands

Run the environment check:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\check_environment.py'
```

Run only the cleaning and sanity-check step:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\clean_data.py'
```

Run only the individual portfolio modeling step:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\run_individual_modeling.py'
```

Run only the predictive-modeling extension:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\run_predictive_modeling.py'
```

Run only the trading-strategies stage:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\run_trading_strategies.py'
```

Build only the final integrated report package:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\build_final_report.py'
```

Run the strict audit and prepare the clean submission bundle:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\audit_and_prepare_submission.py'
```

Run the full current pipeline:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\run_pipeline.py'
```

## Current Outputs

The current pipeline saves:

- cleaned percent and decimal datasets,
- a data dictionary,
- date coverage, duplicate, missingness, and descriptive-statistics tables,
- baseline portfolio figures,
- per-portfolio univariate diagnostics and model summaries,
- MLE-based Normal, Student-t, and NIG distribution-fit comparisons plus `arch`-based Gaussian, Student-t, Skewed Student-t, and GED GARCH comparisons,
- combined ARIMA and GARCH comparison tables, and
- a cached Fama-French monthly factor file,
- a monthly predictor dataset with lagged factor and internal signals,
- per-portfolio predictive forecast plots, parameter tables, model summaries, and interpretations,
- combined predictive comparison and forecast tables, and
- VAR lag-selection, stability, IRF, and FEVD outputs,
- cointegration order-testing and Johansen/VECM outputs on wealth representations,
- statistical-arbitrage signal and backtest files,
- efficient-frontier tables and figures,
- rolling strategy weights and performance comparisons for equal-weight and mean-variance portfolios, and
- report drafts for Sections 2 through 5,
- a final integrated Markdown report,
- a final report notebook, HTML export, and PDF,
- a rubric checklist and strict audit report, and
- a clean `submission_ready/` folder plus `submission_ready.zip` for Blackboard submission.

The canonical current output folder is `output/`.
