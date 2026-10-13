# Appendices

## Appendix A. Portfolio-by-Portfolio Modeling Notes

This appendix records concise portfolio-by-portfolio notes from the univariate benchmark stage and, when available, the exogenous predictive-modeling stage.

## Small LoBM

- Distribution: mean = 0.974% per month, median = 1.109%, volatility = 7.432% per month, skewness = 0.566, kurtosis = 9.924, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and recommended marginal fit = Student-t with KS p-value = 0.5621.
- Stationarity and dependence: ADF p-value = 0.0000, KPSS p-value = 0.1000, raw Ljung-Box p-value at lag 12 = 0.0000, and raw ARCH-LM p-value = 0.0000.
- Selected ARIMA model: (0, 0, 2) with AIC = 8147.23, BIC = 8167.57, and residual Ljung-Box p-value at lag 12 = 0.3142.
- Selected volatility model: Student-t GARCH(1,1) with innovation distribution Student-t, alpha = 0.1415, beta = 0.8370, persistence = 0.9785, innovation KS p-value = 0.7917, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.9581, and standardized-residual ARCH-LM p-value = 0.9575.
- Predictive benchmark: RMSE = 6.6525, MAE = 4.8813, directional accuracy = 0.608.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with RMSE = 6.7337, MAE = 4.9446, directional accuracy = 0.533, and RMSE change vs benchmark = -1.22%.
- Most informative predictive terms: market_vol_12m_lag1_pct (positive, p=0.067); value_spread_lag1_pct (positive, p=0.082); ff_hml_lag1_pct (negative, p=0.083)
- Saved outputs: `output/figures/individual_returns/small_lobm/`, `output/tables/individual_returns/small_lobm/`, and `output/models/individual_returns/small_lobm/`.
- Distribution-fit comparison: `output/tables/individual_returns/small_lobm/distribution_fit_comparison.csv`; volatility-model comparison: `output/tables/individual_returns/small_lobm/garch_candidate_models.csv`.

## ME1 BM2

- Distribution: mean = 1.236% per month, median = 1.519%, volatility = 6.942% per month, skewness = 1.105, kurtosis = 16.406, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and recommended marginal fit = Student-t with KS p-value = 0.5919.
- Stationarity and dependence: ADF p-value = 0.0000, KPSS p-value = 0.1000, raw Ljung-Box p-value at lag 12 = 0.0000, and raw ARCH-LM p-value = 0.0000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7970.82, BIC = 8001.32, and residual Ljung-Box p-value at lag 12 = 0.0053.
- Selected volatility model: Skewed Student-t GARCH(1,1) with innovation distribution Skewed Student-t, alpha = 0.1275, beta = 0.8465, persistence = 0.9740, innovation KS p-value = 0.6206, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.8881, and standardized-residual ARCH-LM p-value = 0.8958.
- Predictive benchmark: RMSE = 6.1671, MAE = 4.6652, directional accuracy = 0.575.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with RMSE = 6.2379, MAE = 4.6976, directional accuracy = 0.583, and RMSE change vs benchmark = -1.15%.
- Most informative predictive terms: value_spread_lag1_pct (positive, p=0.056); ff_hml_lag1_pct (negative, p=0.057)
- Saved outputs: `output/figures/individual_returns/me1_bm2/`, `output/tables/individual_returns/me1_bm2/`, and `output/models/individual_returns/me1_bm2/`.
- Distribution-fit comparison: `output/tables/individual_returns/me1_bm2/distribution_fit_comparison.csv`; volatility-model comparison: `output/tables/individual_returns/me1_bm2/garch_candidate_models.csv`.

## Small HiBM

- Distribution: mean = 1.420% per month, median = 1.584%, volatility = 8.080% per month, skewness = 1.960, kurtosis = 23.855, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and recommended marginal fit = Student-t with KS p-value = 0.6922.
- Stationarity and dependence: ADF p-value = 0.0000, KPSS p-value = 0.1000, raw Ljung-Box p-value at lag 12 = 0.0000, and raw ARCH-LM p-value = 0.0000.
- Selected ARIMA model: (2, 0, 2) with AIC = 8327.37, BIC = 8357.87, and residual Ljung-Box p-value at lag 12 = 0.0022.
- Selected volatility model: Student-t GARCH(1,1) with innovation distribution Student-t, alpha = 0.1281, beta = 0.8502, persistence = 0.9783, innovation KS p-value = 0.6205, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.9682, and standardized-residual ARCH-LM p-value = 0.9680.
- Predictive benchmark: RMSE = 6.8517, MAE = 5.1519, directional accuracy = 0.542.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with RMSE = 6.9063, MAE = 5.2668, directional accuracy = 0.550, and RMSE change vs benchmark = -0.80%.
- Most informative predictive terms: value_spread_lag1_pct (positive, p=0.098); ff_hml_lag1_pct (negative, p=0.099)
- Saved outputs: `output/figures/individual_returns/small_hibm/`, `output/tables/individual_returns/small_hibm/`, and `output/models/individual_returns/small_hibm/`.
- Distribution-fit comparison: `output/tables/individual_returns/small_hibm/distribution_fit_comparison.csv`; volatility-model comparison: `output/tables/individual_returns/small_hibm/garch_candidate_models.csv`.

## Big LoBM

- Distribution: mean = 0.958% per month, median = 1.295%, volatility = 5.266% per month, skewness = -0.128, kurtosis = 8.203, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and recommended marginal fit = NIG with KS p-value = 0.8402.
- Stationarity and dependence: ADF p-value = 0.0000, KPSS p-value = 0.1000, raw Ljung-Box p-value at lag 12 = 0.0196, and raw ARCH-LM p-value = 0.0000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7336.09, BIC = 7366.59, and residual Ljung-Box p-value at lag 12 = 0.9774.
- Selected volatility model: Skewed Student-t GARCH(1,1) with innovation distribution Skewed Student-t, alpha = 0.1270, beta = 0.8423, persistence = 0.9694, innovation KS p-value = 0.7767, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.6356, and standardized-residual ARCH-LM p-value = 0.6408.
- Predictive benchmark: RMSE = 4.7003, MAE = 3.6290, directional accuracy = 0.625.
- Preferred predictive model: ARIMAX with lagged Fama-French factors with RMSE = 4.7330, MAE = 3.6268, directional accuracy = 0.650, and RMSE change vs benchmark = -0.70%.
- Most informative predictive terms: ma.L2 (positive, p=0.000); ar.L2 (negative, p=0.000); ma.L1 (positive, p=0.000)
- Saved outputs: `output/figures/individual_returns/big_lobm/`, `output/tables/individual_returns/big_lobm/`, and `output/models/individual_returns/big_lobm/`.
- Distribution-fit comparison: `output/tables/individual_returns/big_lobm/distribution_fit_comparison.csv`; volatility-model comparison: `output/tables/individual_returns/big_lobm/garch_candidate_models.csv`.

## ME2 BM2

- Distribution: mean = 0.962% per month, median = 1.248%, volatility = 5.602% per month, skewness = 1.160, kurtosis = 20.201, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and recommended marginal fit = Student-t with KS p-value = 0.5545.
- Stationarity and dependence: ADF p-value = 0.0000, KPSS p-value = 0.1000, raw Ljung-Box p-value at lag 12 = 0.0000, and raw ARCH-LM p-value = 0.0000.
- Selected ARIMA model: (2, 0, 2) with AIC = 7457.81, BIC = 7488.31, and residual Ljung-Box p-value at lag 12 = 0.0567.
- Selected volatility model: Student-t GARCH(1,1) with innovation distribution Student-t, alpha = 0.1320, beta = 0.8297, persistence = 0.9617, innovation KS p-value = 0.2626, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.3782, and standardized-residual ARCH-LM p-value = 0.4276.
- Predictive benchmark: RMSE = 4.8070, MAE = 3.4897, directional accuracy = 0.583.
- Preferred predictive model: ARIMAX with lagged Fama-French factors with RMSE = 4.6677, MAE = 3.3890, directional accuracy = 0.592, and RMSE change vs benchmark = 2.90%.
- Most informative predictive terms: ar.L2 (negative, p=0.000); ar.L1 (positive, p=0.000); ff_hml_lag1_pct (positive, p=0.000)
- Saved outputs: `output/figures/individual_returns/me2_bm2/`, `output/tables/individual_returns/me2_bm2/`, and `output/models/individual_returns/me2_bm2/`.
- Distribution-fit comparison: `output/tables/individual_returns/me2_bm2/distribution_fit_comparison.csv`; volatility-model comparison: `output/tables/individual_returns/me2_bm2/garch_candidate_models.csv`.

## Big HiBM

- Distribution: mean = 1.213% per month, median = 1.438%, volatility = 7.081% per month, skewness = 1.443, kurtosis = 20.449, Jarque-Bera p-value = 0.0000, Shapiro-Wilk p-value = 0.0000, and recommended marginal fit = Student-t with KS p-value = 0.5120.
- Stationarity and dependence: ADF p-value = 0.0000, KPSS p-value = 0.1000, raw Ljung-Box p-value at lag 12 = 0.0000, and raw ARCH-LM p-value = 0.0000.
- Selected ARIMA model: (2, 0, 2) with AIC = 8027.05, BIC = 8057.55, and residual Ljung-Box p-value at lag 12 = 0.0673.
- Selected volatility model: Student-t GARCH(1,1) with innovation distribution Student-t, alpha = 0.1331, beta = 0.8289, persistence = 0.9620, innovation KS p-value = 0.7416, standardized-squared-residual Ljung-Box p-value at lag 12 = 0.6114, and standardized-residual ARCH-LM p-value = 0.6704.
- Predictive benchmark: RMSE = 6.4570, MAE = 4.8316, directional accuracy = 0.558.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with RMSE = 6.1340, MAE = 4.5624, directional accuracy = 0.625, and RMSE change vs benchmark = 5.00%.
- Most informative predictive terms: No non-constant terms are significant at the 10% level.
- Saved outputs: `output/figures/individual_returns/big_hibm/`, `output/tables/individual_returns/big_hibm/`, and `output/models/individual_returns/big_hibm/`.
- Distribution-fit comparison: `output/tables/individual_returns/big_hibm/distribution_fit_comparison.csv`; volatility-model comparison: `output/tables/individual_returns/big_hibm/garch_candidate_models.csv`.

## Appendix B. Supplementary Diagnostic Material

The main body cites the highest-signal tables and figures only. To keep the report body compact, the following diagnostics remain in the saved output folders rather than being reproduced inline:

- `output/figures/individual_returns/<portfolio>/`: time-series plots, histograms with Gaussian/Student-t/NIG overlays, QQ plots, recommended-distribution diagnostics, ACF/PACF, residual diagnostics, and volatility-clustering figures for each portfolio.
- `output/tables/individual_returns/<portfolio>/`: detailed descriptive statistics, Jarque-Bera and Shapiro-Wilk output, fitted Normal/Student-t/NIG comparisons, ADF and KPSS tests, Ljung-Box diagnostics, ARCH-LM tests, and candidate-model comparison tables.
- `output/figures/predictive_individual_returns/<portfolio>/`: forecast-comparison figures for the exogenous predictive models.
- `output/tables/predictive_individual_returns/<portfolio>/`: fitted predictive-model summaries and forecast diagnostics.
- `output/figures/trading_strategies/var_irf_grid.png`: reduced-form impulse-response overview for the VAR.
- `output/figures/trading_strategies/var_fevd.png`: forecast-error variance decomposition summary for the VAR.
- `output/figures/trading_strategies/stat_arb_signal.png`: rolling spread z-score and trading signals for the cointegration strategy.

## Appendix C. Reproducibility Note

All tables, figures, cleaned datasets, and report outputs in this project are generated inside `Final Project/` with the fixed interpreter required by the assignment. The canonical end-to-end rerun command remains:

```powershell
& 'd:\MG\anaconda3\python.exe' 'scripts\run_pipeline.py'
```
