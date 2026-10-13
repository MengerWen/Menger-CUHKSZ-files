# Small LoBM Predictive Modeling

- Predictor source: cached authoritative Fama-French monthly factors.
- Benchmark model: Benchmark ARIMA with out-of-sample RMSE = 6.6525, MAE = 4.8813, and directional accuracy = 0.608.
- Preferred predictive model: Predictive regression with lagged factors and internal signals with out-of-sample RMSE = 6.7337, MAE = 4.9446, and directional accuracy = 0.533.
- Full-sample fit: AIC = 8084.84, BIC = 8135.60, Ljung-Box p-value at lag 12 = 0.113, significant-share = 0.000.
- Most informative terms: market_vol_12m_lag1_pct (positive, p=0.067); value_spread_lag1_pct (positive, p=0.082); ff_hml_lag1_pct (negative, p=0.083)
- Interpretation: The preferred predictive model remains economically interpretable, but the benchmark still wins on RMSE by 1.22%.
