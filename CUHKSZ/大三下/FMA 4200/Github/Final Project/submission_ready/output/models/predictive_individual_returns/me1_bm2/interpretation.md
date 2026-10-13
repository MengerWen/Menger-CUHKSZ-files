# ME1 BM2 Predictive Modeling

- Predictor source: cached authoritative Fama-French monthly factors.
- Benchmark model: Benchmark ARIMA with out-of-sample RMSE = 6.1671, MAE = 4.6652, and directional accuracy = 0.575.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with out-of-sample RMSE = 6.2379, MAE = 4.6976, and directional accuracy = 0.583.
- Full-sample fit: AIC = 7917.17, BIC = 7967.93, Ljung-Box p-value at lag 12 = 0.000, significant-share = 0.000.
- Most informative terms: value_spread_lag1_pct (positive, p=0.056); ff_hml_lag1_pct (negative, p=0.057)
- Interpretation: The preferred predictive model remains economically interpretable, but the benchmark still wins on RMSE by 1.15%.
