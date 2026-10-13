# ME1 BM2

- Distribution: mean = 1.236% per month, volatility = 6.942% per month, skewness = 1.105, kurtosis = 16.406.
- Normality: Jarque-Bera and Shapiro-Wilk both reject normality. The recommended marginal model is Student-t, which improves AIC by 359.5 points relative to the Gaussian fit.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(2, 0, 2) chosen because lowest BIC within 2 AIC points because no candidate met the preferred residual-and-significance filter.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.005.
- Volatility model: Skewed Student-t GARCH(1,1) estimated on ARIMA residuals with alpha = 0.128, beta = 0.847, and persistence = 0.974. Assumed innovation distribution parameters: eta=6.8339, lambda=-0.1374.
- Volatility clustering: ARIMA-residual ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.8958. ARCH effects are materially weaker after the arch-based volatility filter.
- Innovation fit: standardized-residual KS p-value under the selected Skewed Student-t assumption = 0.6206.
