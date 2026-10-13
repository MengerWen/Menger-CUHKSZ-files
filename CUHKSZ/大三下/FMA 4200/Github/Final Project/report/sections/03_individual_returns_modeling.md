# Modeling the Individual Portfolio Returns

## Distributional Properties and Stationarity

Each portfolio was analyzed with the lecture-style univariate toolkit from Lecture Slides 1 and 2: time-series plots, histogram-density plots with fitted Gaussian, Student-t, and NIG densities, normal-versus-recommended QQ diagnostics, recommended-fit CDF diagnostics, full descriptive statistics (mean, median, standard deviation, quantiles, skewness, and kurtosis), Jarque-Bera and Shapiro-Wilk tests, MLE-based distribution fitting, KS goodness-of-fit tests, ADF and KPSS stationarity checks, ACF/PACF, Ljung-Box tests, and ARCH-LM diagnostics.

**Table 3. Distribution, normality, stationarity, and ARCH diagnostics.**

| Portfolio | Skewness | Excess kurtosis | Jarque-Bera p | Shapiro-Wilk p | Recommended fit | Best-fit KS p | AIC gain vs Gaussian | ADF p | ARCH-LM p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small LoBM | 0.57 | 6.92 | <0.001 | <0.001 | Student-t | 0.562 | 206.6 | <0.001 | <0.001 |
| ME1 BM2 | 1.10 | 13.41 | <0.001 | <0.001 | Student-t | 0.592 | 359.5 | <0.001 | <0.001 |
| Small HiBM | 1.96 | 20.85 | <0.001 | <0.001 | Student-t | 0.692 | 494.9 | <0.001 | <0.001 |
| Big LoBM | -0.13 | 5.20 | <0.001 | <0.001 | NIG | 0.840 | 171.5 | <0.001 | <0.001 |
| ME2 BM2 | 1.16 | 17.20 | <0.001 | <0.001 | Student-t | 0.555 | 403.2 | <0.001 | <0.001 |
| Big HiBM | 1.44 | 17.45 | <0.001 | <0.001 | Student-t | 0.512 | 432.0 | <0.001 | <0.001 |

Table 3 shows that the six monthly return series are stationary in levels but decisively non-Gaussian. Jarque-Bera and Shapiro-Wilk p-values are effectively zero across the panel, while the preferred marginal-fit counts are **NIG: 1, Student-t: 5**. The most extreme departure from the Gaussian benchmark appears in **Small HiBM**, where the recommended **Student-t** fit improves AIC by **494.9** points relative to the normal fit.

The stationarity evidence supports modeling monthly returns directly rather than differencing them again: ADF rejects a unit root throughout the panel, while KPSS does not overturn the practical use of level ARIMA models. At the same time, the raw series exhibit clear volatility clustering. ARCH-LM tests reject homoskedasticity for all six portfolios, with the strongest raw ARCH signal in **ME2 BM2**.

Detailed artifacts for this subsection are saved under:

- `output/figures/individual_returns/<portfolio>/`
- `output/tables/individual_returns/<portfolio>/`
- `output/models/individual_returns/<portfolio>/`

## ARIMA Benchmarks and `arch`-Based Volatility Models

AR, MA, and ARMA/ARIMA candidates were estimated with `statsmodels` using the lecture-slide logic of ACF/PACF identification, low-order interpretable grids, and explicit residual checks. Candidate mean models were compared with AIC, BIC, residual Ljung-Box tests, and the significance share of dynamic parameters. The volatility stage then used the canonical `arch` package on the selected ARIMA residuals, comparing Gaussian, Student-t, Skewed Student-t, and GED GARCH(1,1) specifications.

Preferred volatility selection is intentionally multi-criterion rather than single-metric. It combines AIC/BIC, Ljung-Box tests on standardized residuals and squared standardized residuals, standardized-residual ARCH-LM tests, innovation KS checks under the assumed distribution, and the significance share of the core volatility parameters.

**Table 4. Selected mean and volatility models.**

| Portfolio | Selected ARIMA | Selected volatility | Innovation dist. | Residual Ljung-Box p (12) | GARCH persistence | Std. sq. resid. LB p (12) | Std. resid. ARCH-LM p | Innovation KS p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small LoBM | (0, 0, 2) | Student-t GARCH(1,1) | Student-t | 0.314 | 0.979 | 0.958 | 0.958 | 0.792 |
| ME1 BM2 | (2, 0, 2) | Skewed Student-t GARCH(1,1) | Skewed Student-t | 0.005 | 0.974 | 0.888 | 0.896 | 0.621 |
| Small HiBM | (2, 0, 2) | Student-t GARCH(1,1) | Student-t | 0.002 | 0.978 | 0.968 | 0.968 | 0.621 |
| Big LoBM | (2, 0, 2) | Skewed Student-t GARCH(1,1) | Skewed Student-t | 0.977 | 0.969 | 0.636 | 0.641 | 0.777 |
| ME2 BM2 | (2, 0, 2) | Student-t GARCH(1,1) | Student-t | 0.057 | 0.962 | 0.378 | 0.428 | 0.263 |
| Big HiBM | (2, 0, 2) | Student-t GARCH(1,1) | Student-t | 0.067 | 0.962 | 0.611 | 0.670 | 0.742 |

Mean dynamics remain modest. The selected ARIMA models are low-order benchmarks, and the clearest remaining residual autocorrelation at lag 12 appears in **ME1 BM2, Small HiBM**. By contrast, the conditional-variance evidence is stronger and more systematic. The selected volatility-model counts are **Skewed Student-t: 2, Student-t: 4**, and the most persistent selected process belongs to **Small LoBM** with alpha + beta = **0.979**.

The portfolio with the highest average monthly return remains **Small HiBM**, at **1.420%** per month. Across the panel, the evidence supports a standard finance interpretation: conditional mean dynamics are weak, but conditional risk is persistent and is better captured by heavy-tailed innovation assumptions than by the Gaussian benchmark alone.

## Predictive Models with Exogenous Variables

The predictive extension adds lagged exogenous information to the univariate benchmarks. In the current run, the project uses **cached authoritative Fama-French monthly factors**, together with internally constructed signals such as lagged size and value spreads, a 12-month rolling market-volatility proxy, a 12-month momentum proxy, and a drawdown proxy. Three classes are compared portfolio by portfolio: the selected ARIMA benchmark, an ARIMAX specification with lagged exogenous predictors, and a predictive regression with lagged returns plus the broader predictor set.

Predictive evaluation follows the course requirement to compare both in-sample fit and out-of-sample performance. Full-sample model quality is summarized with AIC, BIC, residual Ljung-Box diagnostics, and parameter significance. Out-of-sample performance is evaluated with a 120-month expanding one-step-ahead forecast exercise using RMSE, MAE, and directional accuracy.

**Table 5. Preferred predictive model versus the univariate benchmark.**

| Portfolio | Preferred model | Benchmark RMSE | Predictive RMSE | RMSE gain vs benchmark (%) | Directional accuracy |
| --- | --- | --- | --- | --- | --- |
| Small LoBM | Predictive regression | 6.652 | 6.734 | -1.22 | 0.533 |
| ME1 BM2 | Predictive regression | 6.167 | 6.238 | -1.15 | 0.583 |
| Small HiBM | Predictive regression | 6.852 | 6.906 | -0.80 | 0.550 |
| Big LoBM | ARIMAX | 4.700 | 4.733 | -0.70 | 0.650 |
| ME2 BM2 | ARIMAX | 4.807 | 4.668 | 2.90 | 0.592 |
| Big HiBM | Predictive regression | 6.457 | 6.134 | 5.00 | 0.625 |

The most common preferred predictive family is **Predictive regression with lagged factors and internal signals**. The out-of-sample gains are modest rather than dramatic, which is consistent with the literature on monthly return predictability. Even so, **2 of 6** portfolios improve on the benchmark RMSE once the best predictive extension is used. The strongest improvement occurs for **Big HiBM**, where the preferred **Predictive regression** changes RMSE by **5.00%** relative to the benchmark.

These results still point to the same economic conclusion: exogenous signals can help selectively, but they do not overturn the broader Section 3 lesson that conditional variance and tail behavior are more reliable features of the monthly portfolio returns than large and stable conditional-mean predictability.

Predictive-modeling artifacts are saved under:

- `data/processed/predictor_dataset_monthly.csv`
- `data/processed/predictor_source_summary.csv`
- `output/figures/predictive_individual_returns/<portfolio>/`
- `output/tables/predictive_individual_returns/<portfolio>/`
- `output/models/predictive_individual_returns/<portfolio>/`
- `output/tables/predictive_individual_returns/predictive_model_summary.csv`
- `output/tables/predictive_individual_returns/predictive_forecast_metrics.csv`
- `output/tables/predictive_individual_returns/predictive_forecasts.csv`

Overall, Section 3 delivers three main conclusions. First, the monthly portfolio returns are heavy tailed enough that Gaussian-only diagnostics are too narrow, so the analysis now uses multiple normality checks, MLE-based Normal/Student-t/NIG comparisons, and KS goodness-of-fit tests. Second, low-order ARIMA models are adequate mean benchmarks, but they reveal only modest conditional-mean structure. Third, volatility clustering is strong enough to justify `arch`-based GARCH modeling, and non-Gaussian innovation assumptions often dominate the Gaussian benchmark even at the monthly frequency.

Portfolio-by-portfolio notes and the denser output inventory are moved to `report/sections/appendix_individual_returns_modeling.md`.
