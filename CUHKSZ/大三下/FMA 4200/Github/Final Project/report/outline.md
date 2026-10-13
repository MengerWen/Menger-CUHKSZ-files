# Report Outline

This outline follows the assessment structure in `Guidance.md` and maps each major section to the artifacts already created in the project folder.

## 1. Introduction

File: `report/sections/01_introduction.md`

Planned subsections:
- Motivation
- Research background
- Literature review
  - Monthly return dynamics and predictability
  - ARIMA and GARCH-type volatility modeling
  - Predictive modeling with exogenous variables
  - Cointegration and statistical arbitrage
  - Mean-variance optimization and improvements
- Main contributions of this project
- Preview of current and expected findings

Status:
- Drafted in this step.

## 2. Data Source and Processing

File: `report/sections/02_data_source_processing.md`

Planned subsections:
- Kenneth French source and portfolio construction
- Raw file structure
- Cleaning and preprocessing pipeline
- Variable definitions
- Sample period and coverage
- Descriptive statistics and initial visual evidence

Core supporting outputs:
- `data/processed/monthly_portfolio_returns_clean.csv`
- `data/processed/monthly_portfolio_returns_decimal.csv`
- `data/processed/data_dictionary.csv`
- `output/tables/date_coverage_check.csv`
- `output/tables/portfolio_summary_snapshot.csv`
- `output/tables/portfolio_correlation_matrix.csv`
- `output/figures/monthly_returns_overview.png`
- `output/figures/cumulative_growth_of_1.png`
- `output/figures/portfolio_correlation_heatmap.png`

Status:
- Drafted in this step.

## 3. Modeling the Individual Portfolio Returns

Planned subsections:
- Distributional properties of each monthly return series
- Baseline AR, MA, ARMA, and ARIMA specifications
- Volatility modeling with rolling variance benchmarks and GARCH-type models
- Diagnostic plots and residual checks
- Predictive models with exogenous variables

Planned outputs:
- Distribution and QQ plots
- Model comparison tables
- Residual diagnostics
- Forecast evaluation tables

Status:
- Not yet drafted.

## 4. Trading Strategies

Planned subsections:
- Multivariate time-series setup
- Cointegration tests and spread construction
- Statistical arbitrage backtests
- Mean-variance frontier and benchmark comparison
- Improvements to plug-in mean-variance strategies

Planned outputs:
- Cointegration test tables
- Spread charts and backtests
- Efficient frontier figure
- Portfolio weight and performance tables

Status:
- Not yet drafted.

## 5. Conclusions and References

Planned subsections:
- Main findings
- Contributions and limitations
- Directions for extension
- References

Supporting files:
- `report/references.md`

Status:
- References file drafted in this step.

## Current Figure and Table Plan

Suggested numbering for the eventual report:
- Table 1. Variable definitions and units
- Table 2. Sample coverage and data checks
- Table 3. Portfolio summary statistics
- Table 4. Correlation matrix
- Figure 1. Monthly return series by portfolio
- Figure 2. Growth of $1 by portfolio
- Figure 3. Correlation heatmap

## Current Writing Status

- `01_introduction.md`: drafted
- `02_data_source_processing.md`: drafted
- `references.md`: drafted with verified source details
- Remaining analytical and conclusion sections: pending after model estimation and backtesting
