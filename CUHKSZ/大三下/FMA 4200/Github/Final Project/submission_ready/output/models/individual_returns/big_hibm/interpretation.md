# Big HiBM

- Distribution: mean = 1.213% per month, volatility = 7.081% per month, skewness = 1.443, kurtosis = 20.449.
- Normality: Jarque-Bera and Shapiro-Wilk both reject normality. The recommended marginal model is Student-t, which improves AIC by 432.0 points relative to the Gaussian fit.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(2, 0, 2) chosen because lowest BIC within 2 AIC points among models with acceptable residual autocorrelation and mostly significant dynamic terms.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.067.
- Volatility model: Student-t GARCH(1,1) estimated on ARIMA residuals with alpha = 0.133, beta = 0.829, and persistence = 0.962. Assumed innovation distribution parameters: nu=7.4638.
- Volatility clustering: ARIMA-residual ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.6704. ARCH effects are materially weaker after the arch-based volatility filter.
- Innovation fit: standardized-residual KS p-value under the selected Student-t assumption = 0.7416.
