# Small HiBM

- Distribution: mean = 1.420% per month, volatility = 8.080% per month, skewness = 1.960, kurtosis = 23.855.
- Normality: Jarque-Bera and Shapiro-Wilk both reject normality. The recommended marginal model is Student-t, which improves AIC by 494.9 points relative to the Gaussian fit.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(2, 0, 2) chosen because lowest BIC within 2 AIC points because no candidate met the preferred residual-and-significance filter.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.002.
- Volatility model: Student-t GARCH(1,1) estimated on ARIMA residuals with alpha = 0.128, beta = 0.850, and persistence = 0.978. Assumed innovation distribution parameters: nu=5.6206.
- Volatility clustering: ARIMA-residual ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.9680. ARCH effects are materially weaker after the arch-based volatility filter.
- Innovation fit: standardized-residual KS p-value under the selected Student-t assumption = 0.6205.
