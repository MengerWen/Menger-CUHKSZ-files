# Guidance and Lecture Mapping

## Purpose

This note is an audit aid for keeping the implementation, saved outputs, and final report aligned with `Guidance.md` and the course lecture materials. It is not a replacement for the report itself. Its main role is to make the source-of-truth chain explicit so that future reruns do not update a section draft while leaving `report/final_report.md` behind.

## Source-of-Truth Rule

- `report/sections/03_individual_returns_modeling.md` and `report/sections/appendix_individual_returns_modeling.md` are generated from the shared Section 3 writers in `src/fma4200_project/univariate_modeling.py`.
- `src/fma4200_project/predictive_modeling.py` no longer maintains a separate Section 3 writer; it calls the shared Section 3 writer so the predictive stage extends the same document instead of overwriting it with a parallel draft.
- `src/fma4200_project/final_report_builder.py` now reads the generated Section 3 and conclusion source files when building `report/final_report.md`, so the final report reflects the latest section source instead of a duplicated hardcoded narrative.

## Guidance-to-Implementation Map

| Guidance requirement | Lecture knowledge point | Main code file(s) | Main output file(s) | Final report section |
| --- | --- | --- | --- | --- |
| Introduction: motivation, background, literature, contributions, key findings | Course framing plus empirical-finance literature context | `report/sections/01_introduction.md`, `src/fma4200_project/final_report_builder.py` | `report/final_report.md`, `report/references.md` | `Introduction` |
| Data source, cleaning, variable definitions, sample description | Data handling and return conventions from course setup | `src/fma4200_project/data_pipeline.py`, `scripts/clean_data.py` | `data/processed/monthly_portfolio_returns_clean.csv`, `data/processed/data_dictionary.csv`, `output/tables/descriptive_statistics_pct.csv`, `output/figures/monthly_returns_overview.png` | `Data Source and Processing` |
| Section 3: distributional properties of each portfolio | `1_Univariate Data Analysis.md`, `2_Time Series Data Analysis.md` | `src/fma4200_project/univariate_modeling.py`, `scripts/run_individual_modeling.py` | `output/tables/individual_returns/portfolio_statistical_test_summary.csv`, `output/figures/individual_returns/`, `output/models/individual_returns/` | `Modeling the Individual Portfolio Returns` |
| Section 3: AR/MA/ARMA/ARIMA/GARCH diagnostics | `2_Time Series Data Analysis.md` | `src/fma4200_project/univariate_modeling.py` | `output/tables/individual_returns/portfolio_model_comparison_summary.csv`, `output/tables/individual_returns/portfolio_garch_summary.csv` | `Modeling the Individual Portfolio Returns` |
| Section 3: predictive modeling with exogenous variables | Predictive regressions and conditional mean design consistent with Lecture 2 time-series workflow | `src/fma4200_project/predictive_modeling.py`, `scripts/run_predictive_modeling.py` | `data/processed/predictor_dataset_monthly.csv`, `output/tables/predictive_individual_returns/predictive_model_summary.csv`, `output/tables/predictive_individual_returns/predictive_forecast_metrics.csv` | `Modeling the Individual Portfolio Returns` |
| Section 4: joint multivariate modeling and cointegration | `3_Multivariate Time Series Data Analysis.md` | `src/fma4200_project/trading_strategies.py`, `scripts/run_trading_strategies.py` | `output/tables/trading_strategies/var_lag_selection.csv`, `output/tables/trading_strategies/cointegration_summary.csv`, `output/figures/trading_strategies/` | `Trading Strategies` |
| Section 4: statistical arbitrage backtest | `3_Multivariate Time Series Data Analysis.md` statistical-arbitrage and cointegration material | `src/fma4200_project/trading_strategies.py` | `output/tables/trading_strategies/stat_arb_backtest.csv`, `output/tables/trading_strategies/stat_arb_signals.csv`, `output/models/trading_strategies/strategy_rules.md` | `Trading Strategies` |
| Section 4: mean-variance frontier and improved allocation rules | `4_Portfolio Optimization.md` | `src/fma4200_project/trading_strategies.py` | `output/tables/trading_strategies/strategy_metrics.csv`, `output/tables/trading_strategies/efficient_frontier_points.csv`, `output/figures/trading_strategies/efficient_frontier.png` | `Trading Strategies` |
| Conclusions, references, formatting, reproducibility | Report integration and submission requirements in `Guidance.md` | `src/fma4200_project/final_report_builder.py`, `scripts/build_final_report.py`, `scripts/audit_and_prepare_submission.py` | `report/final_report.md`, `report/final_report.pdf`, `report/final_appendices.md`, `SUBMISSION_RUNBOOK.md` | `Conclusions`, `References`, `Appendices` |

## Section 3: Exact Lecture-Method Mapping

### Lecture 1 to Section 3

- `1_Univariate Data Analysis.md` motivates the single-series distribution workflow used in Section 3: mean, median, standard deviation, skewness, kurtosis, and key quantiles.
- The lecture's fitted-distribution logic is used directly in `src/fma4200_project/univariate_modeling.py` through MLE-based comparisons of `Normal`, `Student-t`, and `NIG`, together with log-likelihood, AIC, BIC, and KS goodness-of-fit checks.
- The lecture's graphical diagnostics are reflected in `output/figures/individual_returns/<portfolio>/histogram_density.png`, `qq_plot.png`, and `recommended_distribution_diagnostic.png`.
- The lecture's normality-testing emphasis is reflected in the saved Jarque-Bera and Shapiro-Wilk outputs and in `output/tables/individual_returns/<portfolio>/distribution_fit_comparison.csv`.

### Lecture 2 to Section 3

- `2_Time Series Data Analysis.md` provides the stationarity and serial-dependence toolkit used for ADF, KPSS, ACF/PACF, and Ljung-Box diagnostics.
- The lecture's AR, MA, ARMA, and ARIMA identification logic is implemented in `src/fma4200_project/univariate_modeling.py` through a small interpretable candidate grid, explicit residual diagnostics, and tie-breaking based on fit and parsimony.
- The lecture's GARCH estimation and innovation-distribution discussion is mapped to the canonical `arch` implementation comparing Gaussian, Student-t, Skewed Student-t, and GED GARCH(1,1) models.
- The lecture's residual-diagnostics logic is reflected in `output/figures/individual_returns/<portfolio>/garch_diagnostics.png` and `output/tables/individual_returns/<portfolio>/garch_candidate_models.csv`.
- The predictive extension stays within the same Lecture 2 time-series framework by comparing ARIMA benchmarks with ARIMAX and predictive-regression variants under both in-sample and expanding out-of-sample evaluation.

## Section 4 and Section 5: Lecture Mapping

### Section 4 to Lecture 3

- `3_Multivariate Time Series Data Analysis.md` maps directly to the VAR stage: lag selection, stability checks, residual whiteness checks, and reduced-form impulse-response / FEVD interpretation.
- The same lecture provides the conceptual warning that raw returns are stationary, so cointegration should be tested on a defensible nonstationary representation such as cumulative log wealth, not on raw returns themselves.
- Johansen-style multivariate cointegration and spread-based statistical arbitrage in `src/fma4200_project/trading_strategies.py` are the course-aligned implementations of Lecture 3's cointegration and stat-arb material.
- The saved evidence is concentrated in `output/tables/trading_strategies/var_*.csv`, `cointegration_*.csv`, `stat_arb_*.csv`, and `output/models/trading_strategies/strategy_rules.md`.

### Section 5 to Lecture 4

- `4_Portfolio Optimization.md` maps directly to the efficient-frontier and rolling mean-variance backtest stage.
- The plug-in allocator implements the lecture's sample mean-variance logic, while the improved allocator follows the lecture's practical-improvement message by adding shrinkage, no-short / bounded weights, and turnover control.
- Equal weight is kept as the lecture-consistent benchmark that highlights estimation-error fragility in unconstrained optimization.
- The corresponding evidence is in `output/tables/trading_strategies/strategy_metrics.csv`, `strategy_weights.csv`, `efficient_frontier_points.csv`, and the efficient-frontier / cumulative-wealth figures.

## Most Important Alignment Conclusion

The key alignment result is that the project now follows a clean one-directional chain:

1. Shared pipeline writers generate the canonical section drafts.
2. `final_report_builder.py` consumes those canonical section sources for the most actively regenerated parts of the report, especially Section 3 and the conclusions.
3. The saved outputs cited in the report are produced by the same modeling scripts that write the section text.

That chain is what prevents the earlier drift problem in which Section 3 diagnostics changed in the pipeline but `report/final_report.md` still carried stale volatility and distribution language.
