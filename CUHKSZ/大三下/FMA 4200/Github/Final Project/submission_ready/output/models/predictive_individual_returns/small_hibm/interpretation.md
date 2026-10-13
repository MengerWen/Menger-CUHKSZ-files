# Small HiBM Predictive Modeling

- Predictor source: cached authoritative Fama-French monthly factors.
- Benchmark model: Benchmark ARIMA with out-of-sample RMSE = 6.8517, MAE = 5.1519, and directional accuracy = 0.542.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with out-of-sample RMSE = 6.9063, MAE = 5.2668, and directional accuracy = 0.550.
- Full-sample fit: AIC = 8265.32, BIC = 8316.08, Ljung-Box p-value at lag 12 = 0.000, significant-share = 0.000.
- Most informative terms: value_spread_lag1_pct (positive, p=0.098); ff_hml_lag1_pct (negative, p=0.099)
- Interpretation: The preferred predictive model remains economically interpretable, but the benchmark still wins on RMSE by 0.80%.
