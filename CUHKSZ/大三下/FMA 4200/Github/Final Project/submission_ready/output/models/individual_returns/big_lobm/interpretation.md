# Big LoBM

- Distribution: mean = 0.958% per month, volatility = 5.266% per month, skewness = -0.128, kurtosis = 8.203.
- Normality: Jarque-Bera and Shapiro-Wilk both reject normality. The recommended marginal model is NIG, which improves AIC by 171.5 points relative to the Gaussian fit.
- Stationarity: Both ADF and KPSS support treating the return series as stationary in levels.
- Selected mean model: ARIMA(2, 0, 2) chosen because lowest BIC within 2 AIC points among models with acceptable residual autocorrelation and mostly significant dynamic terms.
- Residual autocorrelation: Ljung-Box p-value at lag 12 = 0.977.
- Volatility model: Skewed Student-t GARCH(1,1) estimated on ARIMA residuals with alpha = 0.127, beta = 0.842, and persistence = 0.969. Assumed innovation distribution parameters: eta=9.1616, lambda=-0.1549.
- Volatility clustering: ARIMA-residual ARCH-LM p-value = 0.0000; standardized-residual ARCH-LM p-value = 0.6408. ARCH effects are materially weaker after the arch-based volatility filter.
- Innovation fit: standardized-residual KS p-value under the selected Skewed Student-t assumption = 0.7767.
