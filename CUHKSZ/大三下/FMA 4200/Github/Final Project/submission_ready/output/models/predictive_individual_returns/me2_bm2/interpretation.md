# ME2 BM2 Predictive Modeling

- Predictor source: cached authoritative Fama-French monthly factors.
- Benchmark model: Benchmark ARIMA with out-of-sample RMSE = 4.8070, MAE = 3.4897, and directional accuracy = 0.583.
- Preferred predictive model: ARIMAX with lagged Fama-French factors with out-of-sample RMSE = 4.6677, MAE = 3.3890, and directional accuracy = 0.592.
- Full-sample fit: AIC = 7416.46, BIC = 7467.19, Ljung-Box p-value at lag 12 = 0.000, significant-share = 0.500.
- Most informative terms: ar.L2 (negative, p=0.000); ar.L1 (positive, p=0.000); ff_hml_lag1_pct (positive, p=0.000)
- Interpretation: The preferred predictive model improves out-of-sample RMSE by 2.90% and MAE by 2.89% relative to the univariate benchmark.
