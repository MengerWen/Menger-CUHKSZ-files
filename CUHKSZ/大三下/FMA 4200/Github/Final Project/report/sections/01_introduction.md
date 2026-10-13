# Introduction

## Motivation

Understanding how equity portfolio returns behave over long samples is central to both empirical asset pricing and practical portfolio construction. The six size and book-to-market portfolios in this project are a compact but economically rich laboratory because they embed two of the most studied cross-sectional dimensions in finance: firm size and value-versus-growth exposure. A century-long monthly sample provides enough variation to study not only differences in average returns, but also the dynamic behavior of volatility, the scope for predictive signals, the possibility of long-run comovement, and the portfolio consequences of estimation error.

This motivation fits the course brief closely. The assignment does not ask only for data description; it asks for a full bridge from descriptive evidence to univariate time-series modeling, multivariate dependence analysis, statistical arbitrage, and mean-variance portfolio choice. That bridge matters because investors face two linked questions at once: how returns evolve over time, and how those dynamics should influence trading and allocation decisions.

## Research Background

The background for this project combines classic asset-pricing ideas with econometric time-series tools. On the asset-pricing side, size and book-to-market characteristics are central because they have long been associated with systematic differences in expected returns and common risk exposures. Fama and French (1993) show that size and book-to-market are tied to important common factors in stock returns, which makes these portfolios natural objects for both modeling and strategy design. On the portfolio-allocation side, Markowitz (1952) established the mean-variance framework that still underlies efficient-frontier analysis, but later work shows that estimation error can substantially distort portfolio weights and out-of-sample performance.

On the econometrics side, the literature emphasizes that financial return series often show weak and unstable predictability in the conditional mean, but much stronger structure in second moments. Hamilton (1994) provides the general time-series framework used for ARIMA-class models, while Engle (1982) and Bollerslev (1986) formalize volatility dynamics through ARCH and GARCH models. Once multiple portfolios are studied jointly, cointegration methods and relative-value trading ideas become relevant because common economic forces can induce strong comovement and occasional mean reversion in spreads. At the same time, the portfolio-allocation literature warns that naive plug-in optimization can be fragile unless covariance estimation and portfolio constraints are handled carefully.

## Literature Review

### Monthly Return Dynamics and Predictability

The starting point for monthly return modeling is modesty about mean predictability. Hamilton (1994) gives the general logic of ARIMA-style modeling: past values and shocks can capture serial dependence when such dependence exists, but the model should earn its complexity through diagnostics and forecasting performance rather than assumption. In empirical finance, predictable variation in expected returns is often studied through slowly moving state variables instead of large autoregressive coefficients in raw returns alone. Fama and French (1988) show that dividend yields can forecast stock returns, especially at longer horizons, and Campbell and Shiller (1988) connect the dividend-price ratio to expectations of future dividends and discount factors through a present-value framework. Together, these papers imply that monthly returns may contain limited short-run mean structure in isolation, but richer predictive content can emerge when economically motivated predictors are brought into the model.

For the current project, that literature suggests a disciplined sequence. First, each of the six return series should be tested for simple serial dependence using low-order AR, MA, or ARIMA specifications. Second, any attempt to forecast the conditional mean should compare pure univariate models with models that add exogenous information, because the literature suggests that financial predictability is more plausibly linked to valuation or state variables than to large standalone autoregressive effects.

### ARIMA and GARCH-Type Volatility Modeling

Financial return volatility is much more persistent than the conditional mean, and that stylized fact motivates the project's volatility section. Engle (1982) introduced ARCH models to capture time-varying conditional variance, showing that squared residuals can carry predictable structure even when raw returns look close to serially uncorrelated. Bollerslev (1986) generalized this framework to GARCH, allowing volatility persistence to be modeled more parsimoniously and making the approach especially useful for financial applications with clustering and mean reversion in volatility.

This literature is directly relevant to monthly portfolio returns. Even at the monthly frequency, episodes such as the Great Depression, the 1970s inflationary period, the dot-com crash, the global financial crisis, and the COVID shock can induce large swings in volatility. A useful implication for the project is that ARIMA and GARCH-type models should be treated as complementary rather than competing tools: ARIMA-style terms target the conditional mean, while ARCH/GARCH components target conditional second moments. Model evaluation should therefore focus on residual diagnostics, volatility persistence, and whether the more complex specifications materially improve fit or forecasting over simpler benchmarks.

### Predictive Modeling with Exogenous Variables

Predictive modeling with exogenous variables is important because return predictability is often state dependent. Fama and French (1988) and Campbell and Shiller (1988) both show that valuation ratios can contain information about future returns, while the broader predictive-return literature treats macro-financial variables as proxies for time-varying discount rates and risk premia. In practical terms, this means the project should go beyond pure univariate dynamics when building return forecasts.

Because the current dataset contains only the six portfolio returns, the first exogenous-variable layer can still be built internally from the panel of portfolios. For example, lagged cross-portfolio spreads, lagged market-style aggregates, or valuation-style spread proxies derived from the six returns can be evaluated as candidate predictors. If later extensions add external market factors or macro variables, the project can connect even more closely to the predictive-regression literature. The main lesson from the literature is not that any single predictor will dominate, but that economically motivated state variables usually provide a more credible forecasting design than purely mechanical lag selection.

### Cointegration and Statistical Arbitrage

Once the six portfolios are analyzed jointly, long-run dependence becomes central. Engle and Granger (1987) provide the foundational framework for cointegration and error-correction modeling. Their key insight is that nonstationary series can move together around a stable long-run relation even when the individual series themselves require differencing. In finance, that idea supports spread trading because deviations from equilibrium may partially mean revert.

Gatev, Goetzmann, and Rouwenhorst (2006) provide one of the best-known empirical studies of relative-value trading through pairs trading. Their results show that carefully constructed spreads can generate economically meaningful excess returns, although implementation details and transaction costs matter. For this project, the six size and book-to-market portfolios are not literal matched-stock pairs, but they are close substitutes in the sense that they are exposed to common macro and equity-market forces while differing along size and value dimensions. That makes cointegration tests, spread definitions, and simple statistical-arbitrage rules a natural extension of the descriptive evidence.

### Mean-Variance Optimization and Portfolio Improvements

The portfolio-choice part of the project is anchored by Markowitz (1952), which defines the efficient frontier in terms of expected return and variance. That framework is still the right baseline for an academic finance project because it gives a transparent benchmark against which alternative strategies can be judged. However, later research shows that mean-variance optimization is highly sensitive to estimation error. Jagannathan and Ma (2003) show that seemingly ad hoc portfolio constraints can improve performance by implicitly shrinking problematic covariance estimates. Ledoit and Wolf (2004) develop a shrinkage-based covariance estimator that is better conditioned than the sample covariance matrix in high-dimensional settings. DeMiguel, Garlappi, and Uppal (2009) provide an especially important cautionary result: out of sample, naive diversification can outperform many optimized strategies because estimation error overwhelms the theoretical gains from optimization.

This literature gives the project a clear strategy for the trading section. The efficient frontier and plug-in mean-variance optimizer should be implemented as required by the assignment, but the empirical discussion should treat them as baselines rather than unquestioned winners. Equal weighting, covariance shrinkage, and simple portfolio constraints are all relevant comparison points precisely because the literature shows that robust implementation often matters more than formal optimality in sample.

## Main Contributions of This Project

This project contributes in five practical ways within the course setting:

1. It converts the raw Kenneth French file into a reproducible cleaned dataset with explicit percent-versus-decimal return conventions and saved data-quality checks.
2. It studies the six portfolios both individually and jointly, which links univariate return modeling to multivariate trading and allocation decisions.
3. It connects descriptive evidence to an integrated methodological pipeline: ARIMA-style mean models, GARCH-type volatility models, predictive regressions with exogenous information, cointegration tests, and portfolio optimization.
4. It evaluates not only a textbook mean-variance strategy but also improved variants motivated by the estimation-error literature.
5. It documents each stage in saved outputs under `output/`, making the project reproducible and easier to extend.

## Preview of Key Findings and Expected Findings

At the current stage, the project already has a few verified descriptive findings from the cleaned sample covering **July 1926 through January 2026**. The six portfolios are highly positively correlated, with pairwise correlations ranging from about **0.78 to 0.96**. The **small, high book-to-market** portfolio has the highest average monthly return in the sample at about **1.42%**, but it also has the highest monthly volatility at about **8.08%**. These facts already suggest that the main differences across portfolios are not simple sign reversals, but differences in exposure, average payoffs, and risk intensity.

The expected findings from the next analytical stages are more provisional. Based on the literature, it is reasonable to expect weak or unstable conditional-mean dynamics in simple ARIMA models, stronger persistence in conditional volatility, and modest but possibly useful gains from predictive models that include economically sensible lagged signals. In the multivariate setting, some long-run relations among the six portfolios may support spread-based mean-reversion tests, although profitability will depend on specification and implementation details. For allocation, the literature strongly suggests that naive sample-based mean-variance weights may be unstable, while shrinkage and constraints are likely to produce more credible out-of-sample behavior. These expectations should be treated as hypotheses to evaluate, not as conclusions to assume.
