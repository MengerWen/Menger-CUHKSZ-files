# Big LoBM Predictive Modeling

- Predictor source: cached authoritative Fama-French monthly factors.
- Benchmark model: Benchmark ARIMA with out-of-sample RMSE = 4.7003, MAE = 3.6290, and directional accuracy = 0.625.
- Preferred predictive model: ARIMAX with lagged Fama-French factors with out-of-sample RMSE = 4.7330, MAE = 3.6268, and directional accuracy = 0.650.
- Full-sample fit: AIC = 7272.10, BIC = 7322.83, Ljung-Box p-value at lag 12 = 0.989, significant-share = 0.500.
- Most informative terms: ma.L2 (positive, p=0.000); ar.L2 (negative, p=0.000); ma.L1 (positive, p=0.000)
- Interpretation: The preferred predictive model remains economically interpretable, but the benchmark still wins on RMSE by 0.70%.
