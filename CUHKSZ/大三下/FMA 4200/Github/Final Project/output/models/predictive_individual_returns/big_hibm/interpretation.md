# Big HiBM Predictive Modeling

- Predictor source: cached authoritative Fama-French monthly factors.
- Benchmark model: Benchmark ARIMA with out-of-sample RMSE = 6.4570, MAE = 4.8316, and directional accuracy = 0.558.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with out-of-sample RMSE = 6.1340, MAE = 4.5624, and directional accuracy = 0.625.
- Full-sample fit: AIC = 7985.20, BIC = 8035.96, Ljung-Box p-value at lag 12 = 0.000, significant-share = 0.000.
- Most informative terms: No non-constant terms are significant at the 10% level.
- Interpretation: The preferred predictive model improves out-of-sample RMSE by 5.00% and MAE by 5.57% relative to the univariate benchmark.
