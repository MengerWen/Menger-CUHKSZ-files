# Machine Learning Data Analysis
## Empirical Asset Pricing Models

**Gongqiu Zhang**
SSE, CUHK-Shenzhen

---

## 1 **Introduction**

To operationalize mean-variance analysis, we need estimates of expected returns, variances, and covariances.

- Expected returns are especially hard to estimate.
- It will be useful to have a theory of what expected returns should be.

In this part, we will study the use of multiple-factor models to explain the cross-section of expected returns.

---

## 2 **The CAPM and Applications**

### CAPM Theory and Foundations

#### Mean-Variance in Equilibrium $\Rightarrow$ CAPM

Equilibrium is an economic concept that characterizes a situation where no investor wants to do anything different.

- In equilibrium, all assets must be fully purchased by investors.
- If a stock's price is too high, no one will want to buy it.
- The price must adjust until investors are willing to hold all available shares.

The CAPM gives an answer based on all investors using Markowitz MV analysis, leading to a linear relationship between expected returns and risk.

#### CAPM Assumptions

A number of assumptions are necessary to formally derive the CAPM:

1. No transaction costs or taxes.
2. All assets are tradable and infinitely divisible.
3. No individual can effect security prices (perfect competition).
4. Investors care only about expected returns and variances.
5. Unlimited short sales, borrowing, and lending.
6. Homogeneous expectations.

#### MV Analysis and The Tangency Portfolio

Assumptions 4, 5 and 6 imply that everyone solves the same passive MV portfolio problem: The tangency portfolio must be the same for everybody.

![[Pasted image 20260416191545.png]]
*Figure: Minimum Variance Frontier and The Tangency Portfolio*

According to Markowitz, everyone's optimal investment portfolio is therefore comprised of the risk-free asset and the same tangency portfolio.

#### The Tangency and Market Portfolio

The tangency portfolio equals the market portfolio in equilibrium.
- Investors on average hold the market portfolio.
- The market portfolio contains all risky assets in proportion to their market values.
- This includes stocks, bonds, real estate, and human capital.
- All these assets must be tradeable for the theory to work.

#### The Capital Market Line (CML)

The CML is the line that passes through the risk-free asset and the market portfolio.

![[Pasted image 20260416191607.png]]
*Figure: The Capital Market Line (CML)*

Every investor faces the same CAL in equilibrium.

The CML gives the return on efficient portfolios $r_e$:
$$ E(r_e) = r_f + \frac{E(r_m) - r_f}{\sigma_m} \sigma_e. $$

- Investors should hold market portfolio and risk-free asset
- Market portfolio includes all risky securities
- Popular index funds/ETFs:
    - SPY (S&P 500)
    - QQQ (Nasdaq 100)
    - IWM (Russell 2000)

#### The Share of Ownership by ETFs

![[Pasted image 20260416192022.png]]
*Figure: Share of U.S. Stock Ownership by Exchange Traded Funds (ETFs) (1990 to 2016)*

#### The Security Market Line (SML) and CAPM

The CAPM aims to predict expected returns for all assets and portfolios.

- The market portfolio should be optimal - no benefit from deviating.
- Each asset's risk must be balanced by its return.
- All assets should have the same risk-return tradeoff at the margin.
- This leads to the Security Market Line (SML) and CAPM.
- Returns relate to market covariance, not total risk.

---

### A Formal Derivation of the CAPM

Suppose that you are currently holding the market portfolio, but decide to borrow a small additional fraction $\delta_{GM}$ at the risk-free rate to invest in GM

$$ r_c = r_m - \delta_{GM} \cdot r_f + \delta_{GM} \cdot r_{GM} $$

The expected return and variance on the new portfolio will be:

$$ E(r_c) = E(r_m) + \delta_{GM} \cdot (E(r_{GM}) - r_f) $$
$$ \sigma_c^2 = \sigma_m^2 + \delta_{GM}^2 \sigma_{GM}^2 + 2 \cdot \delta_{GM} \cdot \text{cov}(r_{GM}, r_m) $$

For small values of $\delta_{GM}$, the changes in the expected return and variance are:

$$ \Delta E(r_c) = \delta_{GM} \cdot (E(r_{GM}) - r_f) $$
$$ \Delta \sigma_c^2 = 2 \cdot \delta_{GM} \cdot \text{cov}(r_{GM}, r_m) $$

*Note*: we ignore the $\delta_{GM}^2$ term in the variance equation

Now suppose that you invest $\delta_{GM} > 0$ more in GM, and invest just enough less $\delta_{IBM} < 0$ in IBM so that your portfolio variance $\sigma_c^2$ stays the same.
From before, the change in the variance will be:

$$ \Delta \sigma_c^2 = 2 \cdot \delta_{GM} \cdot \text{cov}(r_{GM}, r_m) + 2 \cdot \delta_{IBM} \cdot \text{cov}(r_{IBM}, r_m) $$

Setting this equal to zero (i.e., $\Delta \sigma_c^2 = 0$) and solving:

$$ \delta_{IBM} = -\delta_{GM} \cdot \frac{\text{cov}(r_{GM}, r_m)}{\text{cov}(r_{IBM}, r_m)} $$

From before, the change in the expected return for this $\Delta \sigma_c^2 = 0$ portfolio will be:

$$
\begin{aligned}
\Delta E(r_c) &= \delta_{GM} \cdot (E(r_{GM}) - r_f) + \delta_{IBM} \cdot (E(r_{IBM}) - r_f) \\
&= \delta_{GM} \cdot \left[ (E(r_{GM}) - r_f) - (E(r_{IBM}) - r_f) \cdot \frac{\text{cov}(r_{GM}, r_m)}{\text{cov}(r_{IBM}, r_m)} \right].
\end{aligned}
$$

Since the market portfolio has the highest Sharpe Ratio, we cannot improve on it.
- For any portfolio with the same variance as the market portfolio, its expected return cannot exceed the market portfolio's return.
- Therefore, when $\Delta \sigma_c^2 = 0$, we must have $\Delta E(r_c) = 0$:

$$ \frac{E(r_{GM}) - r_f}{\text{cov}(r_{GM}, r_m)} = \frac{E(r_{IBM}) - r_f}{\text{cov}(r_{IBM}, r_m)} = \lambda $$

- This ratio $\lambda$ is constant across all assets and portfolios.

For the market portfolio:

$$ \frac{E(r_{GM}) - r_f}{\text{cov}(r_{GM}, r_m)} = \frac{E(r_m) - r_f}{\sigma_m^2} = \lambda $$

Therefore:

$$ E(r_{GM}) - r_f = (E(r_m) - r_f) \cdot \beta_{GM} $$

where $\beta_{GM} = \frac{\text{cov}(r_{GM}, r_m)}{\sigma_m^2}$

This is the Security Market Line (SML), which shows how expected returns relate to beta.

The tangent portfolio $P$ has the highest possible Sharpe ratio - no combination with other assets can achieve a better risk-return tradeoff.

This leads to the key CAPM equation, where excess returns are proportional to beta:

> [!info]+ CAPM Equation
> $$ E(r_i) - r_f = (E(r_m) - r_f)\beta_i $$

This equation shows that an asset's excess return equals its systematic risk ($\beta_i$) multiplied by the market risk premium ($E(r_m) - r_f$).

#### The Security Market Line (SML)

![[Pasted image 20260416192148.png]]
*Figure: The Security Market Line (SML) Plot*

---

### An Example

| Asset | $E(r)$ | $\sigma$ |
|---|---|---|
| A | 8% | 20% |
| B | 10% | 23% |
| C | 11% | 25% |
| D | 6% | 27% |

| Assets | A | B | C | D |
|---|---|---|---|---|
| A | 1.0 | 0.0 | 0.3 | -0.2 |
| B | 0.0 | 1.0 | 0.2 | -0.2 |
| C | 0.3 | 0.2 | 1.0 | -0.2 |
| D | -0.2 | -0.2 | -0.2 | 1.0 |

Further assume that $r_f = 3.5\%$. The resulting MVE portfolio has weights of:

$$ \boldsymbol{w}_{MVE} = \begin{bmatrix} 0.2515 \\ 0.3053 \\ 0.2270 \\ 0.2161 \end{bmatrix} $$

If these were the only assets, the CAPM would imply that this is the market portfolio.

The $\beta$ for each of the four assets w.r.t. this market/tangent portfolio:

$$ \beta_i = \frac{\text{cov}(r_i, r_m)}{\sigma_m^2} $$

- Use the equation for the covariance of a portfolio:
	
	$$
	\begin{aligned}
	\text{cov}(r_A, r_m) &= \text{cov}(r_A, w_A r_A + w_B r_B + w_C r_C + w_D r_D) \\
	&= w_A \text{cov}(r_A, r_A) + w_B \text{cov}(r_A, r_B) \\
	&\quad + w_C \text{cov}(r_A, r_C) + w_D \text{cov}(r_A, r_D)
	\end{aligned}
	$$

- Use the equation for the variance of a portfolio:
	$$
	\begin{aligned}
	\sigma_m^2 &= \sum_{i=A}^D \sum_{j=A}^D w_i w_j \text{cov}(r_i, r_j) \\
	&= w_A^2 \sigma_A^2 + w_B^2 \sigma_B^2 + w_C^2 \sigma_C^2 + w_D^2 \sigma_D^2 + \underbrace{2 w_A w_B \text{cov}(r_A, r_B) + \dots}_{\text{6 covariance terms}}
	\end{aligned}
	$$

![[Pasted image 20260416192247.png]]
*Figure: Expected Returns vs Beta (The expected returns for all assets lie on the SML)*

![[Pasted image 20260416192303.png]]
*Figure: Security Market Line vs Capital Market Line Plot*

Every security lies on the SML, but only the market portfolio and the risk-free asset lie on the CML.

SML plots rewards versus systematic risk, while CML plots rewards versus total risk (systematic+idiosyncratic).

![[Pasted image 20260416192323.png]]
*Figure: Minimum Variance Frontier and CML Plot*

- Any portfolio combining asset A and the market portfolio lies inside the efficient frontier.
- The slope at the market portfolio equals the slope of the CML.
- This applies to all assets.

#### Some Comments on the CAPM

The CAPM predicts expected returns in market equilibrium.

- If CAPM fails, better returns than the market portfolio are possible.
- The model's empirical validity remains debated.
- CAPM provides a useful benchmark for expected returns.
- Later extensions build on its core concepts.

---

### The Single-Index Model and Risk Breakdown

#### The Single-Index Model

The CAPM is a special case of the Single-Index Model (SIM).

The single-index model reads:

$$ \tilde{r}_{i,t} = \alpha_i + \beta_i \tilde{r}_{m,t} + \varepsilon_{i,t}, \quad t = 1, 2, \dots, T, $$

where $\varepsilon_{i,t}$ is the idiosyncratic shock, which is uncorrelated with the market return.

If CAPM is true, then $\alpha_i = 0$ for all $i$.

#### Systematic and Idiosyncratic Risk

Asset return variance has two components:

$$ \sigma_i^2 = \beta_i^2 \sigma_m^2 + \sigma_{\varepsilon_i}^2, $$

where $\beta_i^2 \sigma_m^2$ is systematic (market) risk and $\sigma_{\varepsilon_i}^2$ is idiosyncratic risk.

In CAPM, only systematic risk matters - idiosyncratic risk can be diversified away.

The $R^2$ from regressing returns on market returns measures how well the market explains asset returns:

$$ R^2 = 1 - \frac{\sigma_{\varepsilon_i}^2}{\sigma_i^2}. $$

#### Mutual Funds' $\alpha$

![[Pasted image 20260416192941.png]]
*Figure: Mutual Funds' alpha Frequency Distribution Plot*

There is even survivorship bias here.

---

### Estimating $\beta$

We can estimate $\beta$ by regressing excess returns on market excess returns:

$$ r_{i,t} - r_{f,t} = \alpha_i + \beta_i(r_{m,t} - r_{f,t}) + \varepsilon_{i,t}, $$

where $r_{i,t}$ is portfolio return, $r_{m,t}$ is market return, and $r_{f,t}$ is risk-free rate (e.g., 1-month T-bill rate depending on the frequency of the data).

We typically use 5 years of historical data to estimate betas.
- Longer periods like 10 or 25 years are not used because betas change over time.

Betas can change due to:
- Changes in firm leverage;
- Changes in operations;
- Mergers and expansions;
- Market composition changes.

Below is the rolling $\beta$ estimates of ATT:
![[Pasted image 20260416193010.png]]
*Figure: Rolling beta estimates of ATT (Jan-66 to Dec-95)*

To allow for time-varying betas, we need more sophisticated models and statistical methods.

Cross-sectional distribution of firm betas, July 1927 to December 2012 (Cederburg and O'Doherty (2016, Journal of Finance)).
![[Pasted image 20260416193032.png]]
*Figure: Firm-level Beta (Median and 90% Interval) from 1930 to 2010*

In estimating betas we typically use monthly data.
- To get better estimates, we could use higher frequency weekly, daily, or even intraday data.
- Many more observations, and in turn more accurate estimates.

But, the use of higher frequency data (especially intraday data) also presents a number of complications:
- Non-synchronous prices.
- Bid-ask bounce effects.

More suitable for actively traded stocks.

The traditional ways of estimating $\beta$ s rely on historical returns.

How would you estimate the $\beta$ for a new company without any historical data?

Standard industry practice is to use "comparables"
- Find a similar company, for which you have historical data and use the estimated beta for that company.

What if a "comparable" company can not be found?
- Construct a model-predicted beta based on company characteristics.

Potential characteristics: Industry, Firm Size, Financial Leverage, Operating Leverage, Growth/Value, Percentage of revenue from exports...

1. Estimate $\beta_i$ for a cross-section of companies using historical data.
2. Regress the estimated $\hat{\beta}_i$ s on various characteristics that supposedly determine the betas:
	
	$$ \hat{\beta}_i = a_0 + \gamma \cdot \text{INDUSTRY}_i + a_1 \cdot \text{LEV}_i + a_2 \cdot \text{SIZE}_i + \dots + u_i $$
	
	where:
	- $\text{INDUSTRY}_i$: industry dummy
	- $\text{LEV}_i$: financial leverage
	- $\text{SIZE}_i$: market value
	- $\dots$

3. The beta for the new company XYZ in the tech industry may then be predicted as:
	
	$$ \beta_{xyz} = \hat{a}_0 + \hat{\gamma}_{tech} + \hat{a}_1 \cdot \text{LEV}_{xyz} + \hat{a}_2 \cdot \text{SIZE}_{xyz} + \dots $$

---

### CAPM and Portfolio Management

To create an optimal portfolio we need to estimate the efficient frontier and the best capital allocation line (CAL).

- To do so we need estimates for the expected returns, variances, and covariances for all of the assets.
- One approach is to use the sample means, variances and covariances based on past historical returns.
- We have already seen that there are problems with this approach.

1. Run the CAPM regression:
	
	$$ r_{i,t} - r_{f,t} = \alpha_i + \beta_i(r_{m,t} - r_{f,t}) + \varepsilon_{i,t}, $$
	
	to get the estimated $\hat{\alpha}_i$ and $\hat{\beta}_i$.

2. The CAPM estimate of the expected return is:
	
	$$ \hat{E}[r_i] = r_f + \hat{\beta}_i(E[r_m] - r_f), $$
	
	where $E[r_m]$ is the expected return on the market portfolio.

3. The CAPM estimate for the variance and covariance are:
	
	$$ \hat{\sigma}_i^2 = \hat{\beta}_i^2 \hat{\sigma}_m^2 + \hat{\sigma}_{\varepsilon_i}^2, $$
	$$ \hat{\sigma}_{ij} = \hat{\beta}_i \hat{\beta}_j \hat{\sigma}_m^2, \quad i \neq j. $$

These procedures involves $2N + 1$ estimates, where $N$ is the number of assets.

To incorporate views on the expected returns for active portfolio management, we can further apply the Black-Litterman model.

---

## 3 **Empirical Evidence on the CAPM**

### Testing the CAPM

#### Statistical Tests of the CAPM

Two key approaches to testing the CAPM:

1. Test whether $\alpha_i = 0$ in the time-series regression:
	
	$$ r_{i,t} - r_{f,t} = \alpha_i + \beta_i(r_{m,t} - r_{f,t}) + \varepsilon_{i,t}, \quad t = 1, 2, \dots, T $$

2. Given the $\beta_i$ s, test whether $\gamma_0 = 0$ and $\gamma_1 \approx E(r_m) - r_f$ in the cross-sectional regression:
	
	$$ r_i - r_f = \gamma_0 + \gamma_1 \hat{\beta}_i + u_i, \quad i = 1, 2, \dots, n. $$
	
	- This second test was first implemented empirically by Fama and MacBeth (1973).
	- Eugene F. Fama, 2013 Nobel Prize in Economics.
	- The same idea may be used in testing for other priced risk factors.

#### Fama-MacBeth Regression

- Data: Monthly returns of 100 largest U.S. stocks (1963-2008)
- Step 1: Estimate $\beta_i$ for each stock using time-series regression
- Step 2: Run cross-sectional regression of excess returns $(\bar{r}_i - \bar{r}_f)$ on $\hat{\beta}_i$
- Results:
	
	| | Estimate | Std Error | t-stat |
	|---|---|---|---|
	| $\gamma_0$ | 6.01% | 1.8% | 3.5 |
	| $\gamma_1$ | 0.17% | 1.7% | 0.1 |
	| $R^2$ | 2% | | |

- The low $R^2$, significant $\gamma_0$, and insignificant $\gamma_1$ suggest CAPM performs poorly for this sample.

![[Pasted image 20260416193157.png]]
*Figure: Fama-MacBeth Regression Plot (Beta vs Monthly Return)*

Too flat than the CAPM line.

#### Comments on the Fama-MacBeth Regression

Two main challenges in testing CAPM with the Fama-MacBeth approach:
- Expected returns are hard to estimate from realized returns.
- Beta estimates contain measurement errors that can bias results.

Using diversified portfolios instead of individual stocks helps address these issues:
- Sort firms into portfolios based on their beta.
- Portfolio betas are more stable and precise than individual stock betas.

Black, Jensen and Scholes (1972) pioneered this approach, which is now standard in both research and practice.

#### CAPM: BJS Test (1st pass)

![[Pasted image 20260416213754.png]]
*Figure: Table 2 Summary of Statistics for Time Series Tests, Entire Period (January, 1931-December, 1965)*

- High (low) betas are associated with high (low) returns.
- But, high (low) betas are also associated with low (high) alphas.

#### CAPM: BJS Test (2nd pass)

![[Pasted image 20260416213810.png]]
*Figure: Average excess monthly returns (%) vs Systematic risk*

Much better fit than for the individual stocks we looked at before.

---

### Market Anomalies

#### The Size Effect

Research shows that firm size (market capitalization) helps explain stock returns.

Here are size deciles for NYSE stocks in 2007:

| Dec | MKCap(m$) | NYSE | AMEX | NASDAQ | Total |
| --- | --------- | ---- | ---- | ------ | ----- |
| 10  | 511,391   | 172  | 5    | 80     | 257   |
| 9   | 10,486    | 172  | 3    | 81     | 256   |
| 8   | 4,428     | 172  | 5    | 136    | 313   |
| 7   | 2,237     | 172  | 5    | 166    | 343   |
| 6   | 1,387     | 172  | 5    | 217    | 394   |
| 5   | 889       | 172  | 11   | 254    | 437   |
| 4   | 534       | 172  | 15   | 251    | 438   |
| 3   | 353       | 172  | 32   | 400    | 604   |
| 2   | 198       | 172  | 73   | 551    | 796   |
| 1   | 95        | 172  | 412  | 1,399  | 1,983 |

Average historical excess portfolio returns 1926-2007:
![[Pasted image 20260416213837.png]]
*Figure: Excess return, % (historical average) by MKCAP Decile*

Small cap stocks have historically outperformed large cap stocks. Does this violate the CAPM?

Small firms tend to have larger betas. To resolve this, Fama and French (1992) did a double sort:
- First sort on size.
- Then sort on market beta.

| | All | Low- $\beta$ | $\beta$ -2 | $\beta$ -3 | $\beta$ -4 | $\beta$ -5 | $\beta$ -6 | $\beta$ -7 | $\beta$ -8 | $\beta$ -9 | High- $\beta$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| All | | 1.22 | 1.30 | 1.32 | 1.35 | 1.36 | 1.34 | 1.29 | 1.34 | 1.14 | 1.10 |
| Small-ME | 1.78 | 1.74 | 1.76 | 2.08 | 1.91 | 1.92 | 1.72 | 1.77 | 1.91 | 1.56 | 1.46 |
| ME-2 | 1.44 | 1.41 | 1.35 | 1.33 | 1.61 | 1.72 | 1.59 | 1.40 | 1.62 | 1.24 | 1.11 |
| ME-3 | 1.36 | 1.21 | 1.40 | 1.22 | 1.47 | 1.34 | 1.51 | 1.33 | 1.57 | 1.33 | 1.21 |
| ME-4 | 1.28 | 1.26 | 1.29 | 1.19 | 1.27 | 1.51 | 1.30 | 1.19 | 1.56 | 1.18 | 1.00 |
| ME-5 | 1.24 | 1.22 | 1.30 | 1.28 | 1.33 | 1.21 | 1.37 | 1.41 | 1.31 | 0.92 | 1.06 |
| ME-6 | 1.23 | 1.21 | 1.32 | 1.37 | 1.09 | 1.34 | 1.10 | 1.40 | 1.21 | 1.22 | 1.08 |
| ME-7 | 1.17 | 1.08 | 1.23 | 1.37 | 1.27 | 1.19 | 1.34 | 1.10 | 1.11 | 0.87 | 1.17 |
| ME-8 | 1.15 | 1.06 | 1.18 | 1.26 | 1.25 | 1.26 | 1.17 | 1.16 | 1.05 | 1.08 | 1.04 |
| ME-9 | 1.13 | 0.99 | 1.18 | 1.00 | 1.24 | 1.26 | 1.31 | 1.15 | 1.11 | 1.09 | 1.05 |
| Large-ME | 0.95 | 0.99 | 1.01 | 1.12 | 1.01 | 0.89 | 0.95 | 0.95 | 1.00 | 0.90 | 0.68 |

The relation between average returns and market $\beta$ s within each size decile is generally negative. This effect is especially strong for the smallest firms.

Critics of the size effect argue:
- Alternative size measures don't show the same effect.
- The effect has weakened since 2000.
- The effect may just reflect a liquidity premium, since small stocks are harder to trade.

#### The Value Effect

Research shows that value stocks historically outperform growth stocks, e.g., Graham and Dodd (1934) "Security Analysis".

- Value stocks: Low market price relative to book value.
    - May be undervalued, offering good investment opportunities.
- Growth stocks: High market price relative to book value.
    - Also called glamor stocks.
    - May be overvalued, leading to poor returns.

This value investing approach is used by successful investors like Warren Buffett.

Sort all stocks into 10 portfolios based on the ratio of book-value-of-equity to the market-value-of-equity, re-balancing every year, 1926-2007:

| Sort | Lo | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Hi | Hi-Lo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $E(R_i) - r_f$ | 6.77 | 8.01 | 7.85 | 7.98 | 8.93 | 9.37 | 9.66 | 11.41 | 12.12 | 13.17 | 6.40 |
| $(t)$ | (2.22) | (2.13) | (2.07) | (2.34) | (2.18) | (2.39) | (2.59) | (2.70) | (2.95) | (3.62) | (2.57) |
| $\sigma$ | 19.98 | 19.16 | 18.59 | 21.06 | 19.60 | 21.46 | 23.29 | 24.22 | 26.48 | 32.49 | 23.10 |
| $\alpha$ | -1.01 | 0.42 | 0.53 | -0.23 | 1.38 | 1.13 | 0.92 | 2.46 | 2.43 | 1.97 | 2.98 |
| $(t)$ | (0.74) | (0.60) | (0.59) | (0.73) | (0.78) | (0.87) | (1.04) | (1.14) | (1.34) | (1.84) | (2.30) |
| $\beta_{MKT}$ | 1.01 | 0.98 | 0.95 | 1.06 | 0.98 | 1.06 | 1.13 | 1.16 | 1.25 | 1.45 | 0.44 |
| $(t)$ | (0.02) | (0.02) | (0.02) | (0.04) | (0.03) | (0.04) | (0.06) | (0.07) | (0.05) | (0.10) | (0.11) |
| $R^2(\%)$ | 89.80 | 92.74 | 91.86 | 89.94 | 88.02 | 87.23 | 83.44 | 80.80 | 79.38 | 70.35 | 12.89 |

Even more of a puzzle over the 1962-2007 sample period:

| Sort | Lo | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Hi | Hi-Lo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $E(R_i) - r_f$ | 3.95 | 5.59 | 6.07 | 6.29 | 6.28 | 7.28 | 8.33 | 8.67 | 9.64 | 11.11 | 7.16 |
| $(t)$ | (2.67) | (2.42) | (2.39) | (2.37) | (2.22) | (2.21) | (2.21) | (2.19) | (2.37) | (2.73) | (2.28) |
| $\sigma$ | 18.01 | 16.35 | 16.14 | 16.00 | 15.01 | 14.90 | 14.92 | 14.78 | 16.01 | 18.46 | 15.39 |
| $\alpha$ | -2.07 | -0.04 | 0.54 | 0.95 | 1.38 | 2.38 | 3.63 | 4.05 | 4.66 | 5.71 | 7.78 |
| $(t)$ | (1.05) | (0.73) | (0.76) | (0.95) | (1.01) | (0.92) | (1.10) | (1.12) | (1.23) | (1.71) | (2.44) |
| $\beta_{MKT}$ | 1.10 | 1.03 | 1.01 | 0.97 | 0.89 | 0.90 | 0.86 | 0.84 | 0.91 | 0.99 | -0.11 |
| $(t)$ | (0.02) | (0.02) | (0.02) | (0.03) | (0.03) | (0.03) | (0.03) | (0.03) | (0.04) | (0.05) | (0.06) |
| $R^2(\%)$ | 86.02 | 91.34 | 90.27 | 85.63 | 81.92 | 83.31 | 76.58 | 75.20 | 74.27 | 65.82 | 1.07 |

Even a negative, relation between returns and betas.

#### The Size and Value Effects

Two key explanations:
- Value stocks have higher distress risk, leading to higher risk premiums.
- Small stocks have higher trading costs, requiring higher returns.

Does this fit with CAPM?

How can we extend CAPM for size and value?

#### Other Patterns in Returns

Harvey, Liu and Zhu (2016, RFS) document many return patterns, including:
- Momentum.
- Media coverage.
- Idiosyncratic volatility.
- ...

*Note*: As trading strategies become widely known, their profitability tends to diminish.

---

### Factor Models and Critiques

#### The Fama-French 3-Factor Model

The Fama-French 3-factor model extends CAPM by adding two factors: size and value.

The model is:

$$ r_i - r_f = \beta_{i,M}(r_M - r_f) + \beta_{i,SMB} r_{SMB} + \beta_{i,HML} r_{HML} + \varepsilon_i $$

Three risk factors:
- Market $(r_M - r_f)$
- Size ($\text{SMB}$ = small minus big)
- Value ($\text{HML}$ = high minus low book-to-market)

Fama and French create 6 portfolios by sorting stocks on size and value:

| Market Cap | Low Book-to-market | Medium Book-to-market | High Book-to-market |
|---|---|---|---|
| **Small** | Portfolio 1: Small growth | Portfolio 2: Small core | Portfolio 3: Small value |
| **Large** | Portfolio 4: Large growth | Portfolio 5: Large core | Portfolio 6: Large value |

- Small minus Big (SMB): $\text{SMB} = \frac{1}{2}(\text{SG} + \text{SV}) - \frac{1}{2}(\text{LG} + \text{LV})$
- High minus Low (HML): $\text{HML} = \frac{1}{2}(\text{SV} + \text{LV}) - \frac{1}{2}(\text{SG} + \text{LG})$.

These are zero-cost portfolios. Alternative definitions exist.

#### The Roll Critique

Roll (1977) argues that the CAPM cannot be properly tested because:
- The true market portfolio is unobservable.
- Common market proxies exclude important assets like real estate and human capital.
- Tests only show if the chosen proxy is mean-variance efficient.

Due to these limitations, Roll suggests using the Arbitrage Pricing Theory (APT) instead.

---

## 4 **Multifactor Asset Pricing Models**

### Fundamentals of Arbitrage Pricing Theory

#### Motivations

The CAPM's empirical failures can be explained by its unrealistic assumptions:

- The model relies on strong assumptions that don't reflect reality.
- It assumes perfectly rational investors with identical views.
- The market portfolio is impossible to measure accurately.

#### The Arbitrage Pricing Theory (APT)

The basic assumptions for the APT are:
- Finite expected returns and variances for all securities.
- Ability to form well-diversified portfolios.
- No market frictions (taxes, transaction costs, etc.).

These assumptions are weaker than those for the CAPM.

The APT prices assets relative to each other:
- Price restrictions are based on no-arbitrage.
- Similar results hold if we exclude near-arbitrage or extremely good deals.

#### No Arbitrage

No-Arbitrage in financial markets means you can't have a security with zero cost and a positive return.

- You can't create a portfolio with this feature either.
- Securities or portfolios with the same returns must have the same price.
- No "free lunch".

In a well-functioning market, arbitrage opportunities shouldn't last:
- If they do, not for long...
- This no-arbitrage concept is key in pricing options and derivatives.

Unlike CAPM's equilibrium ideas, this no-arbitrage rule needs just one smart investor.

#### Example of Arbitrage

Suppose that there are only two possible and equally likely states of nature for inflation and real interest rates: high or low.

Suppose that the four securities A, B, C and D are all currently selling for 100, and that the known payoffs in each of the four possible states are:

| State/Stock | High Real Int. Rates (High Infl.) | High Real Int. Rates (Low Infl.) | Low Real Int. Rates (High Infl.) | Low Real Int. Rates (Low Infl.) |
|---|---|---|---|---|
| A | -20 | 20 | 40 | 60 |
| B | 0 | 70 | 30 | -20 |
| C | 90 | -20 | -10 | 70 |
| D | 15 | 23 | 15 | 36 |

Expected returns, standard deviations, and correlations:

| Stock | Price | Return(%) | SD (%) | A | B | C | D |
|---|---|---|---|---|---|---|---|
| A | 100 | 25.00 | 29.58 | 1.00 | -0.15 | -0.29 | 0.68 |
| B | 100 | 20.00 | 33.91 | -0.15 | 1.00 | -0.87 | -0.38 |
| C | 100 | 32.50 | 48.15 | -0.29 | -0.87 | 1.00 | 0.22 |
| D | 100 | 22.25 | 8.58 | 0.68 | -0.38 | 0.22 | 1.00 |

Consider the return/payoff of an equally weighted portfolio of A, B and C, and compare this with the return/payoff of D:

| State/Port. | High Real Int. Rates (High Infl.) | High Real Int. Rates (Low Infl.) | Low Real Int. Rates (High Infl.) | Low Real Int. Rates (Low Infl.) |
|---|---|---|---|---|
| Portfolio | 23.33 | 23.33 | 20.00 | 36.67 |
| D | 15 | 23 | 15 | 36 |

- The return/payoff on the portfolio made up of A, B and C is higher than D in all states of nature
- This is an arbitrage opportunity
- What would happen to the price of D in a well functioning market?

This numerical example is obviously too simplistic:
- In reality there is a continuum of possible states of nature and multiple sources of risks

---

### Multi-Factor Models Framework

#### A Multi-Factor Model

A $K$ -factor model is defined as:

$$ r_i = b_{i,0} + b_{i,1}f_1 + b_{i,2}f_2 + \dots + b_{i,K} f_K + \varepsilon_i $$

- $f_j$ (common factors): These are the $K$ factors affecting assets, such as economic growth or interest rates.
- $b_{i,j}$ (factor loading): Indicates how asset $i$ 's return changes with factor $j$.
- $\varepsilon_i$ (idiosyncratic risk): Represents the unique risk of asset $i$, assumed uncorrelated with others, i.e. $\text{cov}(\varepsilon_i, \varepsilon_h) = 0$ for $i \neq h$.

Factor models provide a convenient framework for realistically describing how security returns move with economy-wide risks, and in turn with one another.
- A factor model is a multivariate statistical/mathematical model for returns (return generating process).
- The sources of co-movement are called factors (systematic risks).
- The sensitivities of the assets to the different factors are called factor loadings (factor betas or factor sensitivities).
- In the absence of arbitrage, we can price assets relative to one another based on their comovements with the factors.
- This is the basic idea behind the Arbitrage Pricing Theory (APT).

The $K$ -factor model can be expressed using factor surprises $\tilde{f}_j$:

$$ r_i = a_i + b_{i,1}\tilde{f}_1 + b_{i,2}\tilde{f}_2 + \dots + b_{i,K} \tilde{f}_K + \varepsilon_i $$

Interpretations:
- $E(\tilde{f}_j) = 0$: Factors are defined as deviations from expectations.
- $a_i = E(r_i)$: The intercept represents the expected return.
- $\text{cov}(\tilde{f}_j, \tilde{f}_k) = 0$ for $j \neq k$: Assumed for simplicity, achievable through statistical methods.

#### A Multi-Factor Model: Example

Suppose that two factors have been identified for the U.S. economy:
- The growth rate of industrial production (IP)
- The inflation rate (Inf)

Given information:
- Expected IP growth: 4%
- Expected inflation: 6%
- Stock betas: $b_{IP} = 1.0$ and $b_{Inf} = 0.4$
- Expected stock return: 14%

Actual outcomes:
- Actual IP growth: 5% (1% surprise)
- Actual inflation: 7% (1% surprise)

Revised return estimate:
$$ r = 14\% + 1.0 \times 1\% + 0.4 \times 1\% = 14\% + 1\% + 0.4\% = 15.4\%. $$

#### Multi-Factor Models: Implications on Covariance

The factor model has important implications about asset return variances and covariances.

Consider a two-factor model:
$$
\begin{aligned}
\text{var}(r_i) &= \text{var}(b_{i,1}f_1 + b_{i,2}f_2 + \varepsilon_i) \\
&= b_{i,1}^2 \text{var}(f_1) + b_{i,2}^2 \text{var}(f_2) + 2b_{i,1}b_{i,2}\text{cov}(f_1, f_2) + \sigma_{\varepsilon,i}^2
\end{aligned}
$$
If the factors are uncorrelated:

$$ \text{var}(r_i) = b_{i,1}^2 \text{var}(f_1) + b_{i,2}^2 \text{var}(f_2) + \sigma_{\varepsilon,i}^2 $$

- $b_{i,1}^2 \text{var}(f_1) + b_{i,2}^2 \text{var}(f_2)$ represents the systematic variance
- $\sigma_{\varepsilon,i}^2$ is the idiosyncratic variance

The general formula with $K$ factors:

$$ \text{var}(r_i) = \sum_{j=1}^K \sum_{k=1}^K b_{i,j} \cdot b_{i,k}\sigma_{j,k} + \sigma_{\varepsilon,i}^2 $$

where:
- $\sigma_{j,k}$ for $j \neq k$ denotes the covariance between the $j$'th and $k$'th factors
- $\sigma_{j,j} \equiv \sigma_j^2$ denotes the variance of the $j$'th factor
- The systematic variance is given by $\sum_{j=1}^K \sum_{k=1}^K b_{i,j} \cdot b_{i,k}\sigma_{j,k}$
- The idiosyncratic variance is $\sigma_{\varepsilon,i}^2$

If the factors are uncorrelated the formula simplifies to:

$$ \text{var}(r_i) = \sum_{k=1}^K b_{i,k}^2 \sigma_k^2 + \sigma_{\varepsilon,i}^2 $$

Now consider the covariance between stocks $i$ and $j$ for the two-factor model:
$$
\begin{aligned}
\text{cov}(r_i, r_j) &= \text{cov}(b_{i,1}f_1 + b_{i,2}f_2 + \varepsilon_i, \; b_{j,1}f_1 + b_{j,2}f_2 + \varepsilon_j) \\
&= b_{i,1}b_{j,1}\text{var}(f_1) + b_{i,2}b_{j,2}\text{var}(f_2) + (b_{i,1}b_{j,2} + b_{j,1}b_{i,2})\text{cov}(f_1, f_2)
\end{aligned}
$$
If the factors are uncorrelated:

$$ \text{cov}(r_i, r_j) = b_{i,1}b_{j,1}\text{var}(f_1) + b_{i,2}b_{j,2}\text{var}(f_2) $$

#### Diversified Portfolios

APT suggests only systematic risk is rewarded, as idiosyncratic risk can be diversified away.

A diversified portfolio has no idiosyncratic risk. In a $K$-factor model, its return is:

$$ r_p = E(r_p) + b_{p,1}\tilde{f}_1 + \dots + b_{p,K} \tilde{f}_K $$

The return depends on the factor model and factor surprises $\tilde{f}_j$.

We assume investors can create such portfolios.

---

### APT Pricing and Derivation

#### The Arbitrage Pricing Theory

The APT pricing equation (which we will develop both intuitively and more formally) states that:

> [!info]+ APT Pricing Equation
> $$ E(r_i) = \lambda_0 + \lambda_1 b_{i,1} + \dots + \lambda_K b_{i,K} $$

- The $\lambda_j$s are called factor risk premia
- They represent the extra return for an extra unit of the $j$'th risk
- There is one $\lambda_j$ for each of the systematic risk factors, plus one additional $\lambda_0$
- If there is a risk-free asset, then $\lambda_0 = r_f$
- If this equation is not satisfied for all well diversified portfolios, there is an arbitrage opportunity

#### Example of Arbitrage

Suppose that Apple (AAPL) is currently (time 0) selling for $100 per share.

Suppose also that Apple is going to pay a liquidating dividend in exactly one year from now (time 1), and this is the only future payment that Apple will ever make.

The dividend that Apple will pay is uncertain, and depends on how well the economy is doing:
- If the economy is in an expansion the dividend will be $140
- If the economy is in a recession the dividend will only be $100
- Assume that the two states are equally likely, so that the expected cash flow from Apple equals $E(CF_1^{AAPL}) = \$120$

Now let's consider a second stock, Starbucks (SBUX), which like Apple, is going to pay a liquidating dividend exactly one year from now:

| | AAPL | SBUX |
|---|---|---|
| Boom Payoff (Pr=0.5) | $140 | $160 |
| Bust Payoff (Pr=0.5) | $100 | $80 |
| E(CF$_1$) | $120 | $120 |
| Time 0 Price | $100 | ? |

- Since Apple and Starbucks have the same expected cash-flows, one might naturally think that the price for Starbucks would also be $100
- But, that is not necessarily the case

Comparing the cash-flows of Starbucks and Apple:
- Both have the same expected value ($120)
- Starbucks has more risk - higher in boom ($160) but lower in recession ($80)
- During recessions, cash is more valuable as other investments also perform poorly
- During booms, extra cash is less valuable as other investments are already doing well
- Given this risk profile, investors would prefer Apple at the same price
- Therefore, Starbucks must be priced below $100 to attract investors. We may assume that the price of Starbucks is $P_{SBUX} = \$90$.

Quantifying the business cycle factor $f_{BC}$:
- Based on NBER business cycle indicator: $f_{BC} = 1$ during expansion, $f_{BC} = 0$ during recession
- We need the unexpected component of this factor
- With 50/50 chance of expansion/recession, expected indicator value is $E(f_{BC}) = 0.5$
- Surprise component: $+0.5$ in expansions ($1 - 0.5$), $-0.5$ in recessions ($0 - 0.5$)

Let's calculate the factor loadings $b_{AAPL,BC}$ and $b_{SBUX,BC}$

- Using the regression model:
	$$ r_{AAPL,t} = a_{AAPL} + b_{AAPL,BC} \cdot \tilde{f}_{BC,t} + e_{AAPL,t} $$
- With boom/bust equations:
	$$ 0.40 = a_{AAPL} + b_{AAPL,BC} \cdot 0.5 \quad (\text{boom}) $$
	$$ 0.00 = a_{AAPL} + b_{AAPL,BC} \cdot (-0.5) \quad (\text{bust}) $$
- Solving yields $a_{AAPL} = 0.20$ and $b_{AAPL,BC} = 0.4$ for Apple
- For Starbucks, we get $a_{SBUX} = 0.33$ and $b_{SBUX,BC} = 0.89$
- These loadings show each stock's exposure to business cycle risk
- Expected returns are $E(r_{AAPL}) = 0.20$ and $E(r_{SBUX}) = 0.33$

Finally, let's calculate the factor risk premia $\lambda_0$ and $\lambda_{BC}$:
- To determine how investors price the risks we need the APT pricing equation:
	$$ E(r_i) = \lambda_0 + \lambda_{BC} \cdot b_{i,BC} $$
- $\lambda_{BC}$ represents the price of business cycle risk
    - How much more investors discount the cash flows as a result of having one extra unit of business cycle factor risk
- $\lambda_0$ is the required return for a security with no risk
    - The time value of money
- Solving these two equations based on our previous estimates for $b_{AAPL,BC}$ and $b_{SBUX,BC}$, yields $\lambda_0 = 0.0909$ and $\lambda_{BC} = 0.2727$

Arbitrage opportunities arise when risk pricing is inconsistent across assets
- This happens when the APT equation is violated:
	$$ E(r_i) = \lambda_0 + \lambda_{BC} \cdot b_{i,BC} $$
- Example: Add a risk-free asset with 5% return
- We found $\lambda_0 = 0.0909$ from Apple and Starbucks
- This means we can create a synthetic risk-free portfolio with 9.09% return
- Arbitrage: Borrow at 5%, invest in the 9.09% synthetic portfolio

To create a risk-free portfolio with Apple and Starbucks, we solve:
$$ b_{p,BC} = w_{AAPL} \cdot b_{AAPL,BC} + (1 - w_{AAPL}) \cdot b_{SBUX,BC} = 0 $$

- This portfolio is risk-free because it has zero business cycle exposure
- Solution:
	$$ w_{AAPL} = 1.8182 $$
	$$ w_{SBUX} = -0.8182 $$
- Arbitrage strategy: Buy $1.8182 of Apple, short $0.8182 of Starbucks, borrow $1
- This requires no initial investment but guarantees positive returns

To verify that this works, let's look at the return on the stock portion of the portfolio in the two possible states:
$$ w_{AAPL} \cdot (140/100) + (1 - w_{AAPL}) \cdot (160/90) = 1.0909 \quad (\text{boom}) $$
$$ w_{AAPL} \cdot (100/100) + (1 - w_{AAPL}) \cdot (80/90) = 1.0909 \quad (\text{bust}) $$

The payoff from the zero-investment arbitrage portfolio is $1.0909 - 1.05 = 0.0409$ in both states.
- This portfolio is risk-free (same return in both states)
- For a 1 million investment, we get a risk-free payoff of $40,900
- We have created a "money pump".

- Investors require different returns for securities based on their risk profiles
- Inconsistent risk pricing creates arbitrage opportunities
- APT assumes investors exploit these opportunities, pushing prices back to equilibrium
- This assumes arbitrageurs have unlimited capital and patience
- In reality, there may be limits to arbitrage
- APT provides a framework for identifying mispriced securities

#### Formally Deriving the APT

Construct an arbitrage portfolio:
- Zero cost:
	$$ w_1 + w_2 + \dots + w_N = 0 $$
- No systematic risk:
	$$
	\begin{aligned}
	w_1 b_{1,1} + w_2 b_{2,1} + \dots + w_N b_{N,1} &= 0 \\
	w_1 b_{1,2} + w_2 b_{2,2} + \dots + w_N b_{N,2} &= 0 \\
	&\vdots \\
	w_1 b_{1,K} + w_2 b_{2,K} + \dots + w_N b_{N,K} &= 0
	\end{aligned}
	$$

- Note, if $N > K$ we can always construct such a portfolio

> [!info]+ No-Arbitrage Condition
> - Arbitrage portfolios should have zero expected returns.
> - Otherwise, arbitrage opportunities exist.
> 
> Using linear algebra:
> - If $\mathbf{w}^T \boldsymbol{\alpha} = 0$ for all $\mathbf{w}$ where $\mathbf{w}^T \mathbf{1} = 0$ (zero cost) and $\mathbf{w}^T \mathbf{B} = \mathbf{0}$ (no systematic risk).
> - Then $\boldsymbol{\alpha}$ is a combination of $\mathbf{1}$ and columns of $\mathbf{B}$.

>[!info]+ APT Pricing Equation
> $$ \alpha_i = E(r_i) = \lambda_0 + \lambda_1 b_{i,1} + \lambda_2 b_{i,2} + \dots + \lambda_K b_{i,K} $$

**Interpretation:**
- $\lambda_0$: risk-free rate.
- $\lambda_j$: risk-premium for factor $j$.
- $b_{i,j}$: asset $i$'s sensitivity to factor $j$.

#### Summary of APT

The APT can replace the CAPM for:
- Calculating expected returns and cost of capital
- Performance evaluation
- Risk management

Unlike the CAPM, APT does not specify the systematic risks driving returns.

APT uses a statistical factor model to describe systematic risks and return co-movements:
- Its effectiveness depends on identifying the correct factors

How do you determine these factors?

---

### Approaches to Factor Identification

#### The Macroeconomic Approach

**Macroeconomic Approach**
- Factors $f_j$ are treated as primitives.
- Factors may include variables like inflation and GDP growth.
- These variables capture systematic risks in the economy.
- The Chen, Roll and Ross (CRR) model is an example of this approach.

Common macro-economic risk factors:
- GDP growth.
- Inflation.
- Interest Rates.
- Sentiment.
- Business Cycle Indicators.
- ...

#### The Chen, Roll and Ross (CRR) Model

Chen, Roll, and Ross (1986) identify these five factors:
1. Unexpected growth in industrial production (IP).
2. Changes in expected inflation, measured by $r_{TBill}$ change (EI).
3. Unexpected inflation (UI).
4. Unexpected changes in bond return spread, measured by $r_{Baa} - r_{AAA}$ (Default Spread).
5. Unexpected changes in the term structure slope, measured by $r_{TBond} - r_{TBill}$ (Term Spread).
6. They also consider the return on equal-weighted (EWNY) and value-weighted (VWNY) NYSE market portfolios for pricing.

#### The Macroeconomic Approach (continued)

After selecting the factors and calculating the surprises, $\tilde{f}_k = f_k - E(f_k)$, estimate the factor loadings, $b_{i,k}$.

For each security $i$, perform the regression:

$$ r_{i,t} = a_i + b_{i1}\tilde{f}_{1,t} + b_{i2}\tilde{f}_{2,t} + \dots + b_{iK}\tilde{f}_{K,t} + e_{i,t} $$

- $\hat{b}_{i,k}$ shows how security $i$ typically moves with factor $k$.
- $e_{i,t}$ is the idiosyncratic risk for security $i$.
- APT assumes $e_{i,t}$ and $e_{j,t}$ for different assets are mostly uncorrelated and can be diversified.

Estimate the market prices of risk, $\lambda_1, \dots, \lambda_K$, using a cross-sectional regression:

$$ \bar{r}_i = \lambda_0 + \lambda_1\hat{b}_{i1} + \lambda_2\hat{b}_{i2} + \dots + \lambda_K\hat{b}_{iK} + u_i $$

where $\bar{r}_i$ is the average return on asset $i$.

- This is like the Fama-MacBeth method used for CAPM testing.
- $\hat{\lambda}_k$ estimates the expected return difference for a one-unit change in factor $k$ exposure.
- If APT is perfect, all $u_i$ should be zero.

#### The Fundamental Approach

The "fundamental" approach identifies firm characteristics that may indicate different sensitivities to systematic risks.

- Differences in characteristics should relate to differences in expected returns.
- Firms with similar characteristics should move together.
- Portfolios of stocks sorted by these characteristics can be used as proxies for the factors.

Tests of the CAPM showed that small and value firms had higher returns than predicted.

The 3-factor Fama-French model is based on these findings.

The Fama-French 3-factor model is widely used as an alternative to the CAPM:

$$ r_i = r_f + \beta_{i,M} \cdot r_M + \beta_{i,SMB} \cdot r_{SMB} + \beta_{i,HML} \cdot r_{HML} + \epsilon_i $$

- A stock's risk is captured by three betas: market, size, and value.
- SMB (small minus big) and HML (high minus low book-to-market) are factor portfolios.

The Fama-French-Carhart 4-factor model includes a momentum factor.

#### Statistical Approaches

Principal Component Analysis (PCA) is a method to find common factors in historical returns:

$$ r_{i,t} = b_{i,0} + b_{i,1}f_{1,t} + b_{i,2}f_{2,t} + \dots + b_{i,K}f_{K,t} + e_{i,t} $$

- PCA describes return variations using the fewest factors $K$.
- It identifies key data information, treating $e_{i,t}$ as noise.
- For U.S. stocks, PCA usually finds 4-8 key factors.
- The first factor often aligns with the market portfolio.
- For interest rates, PCA typically identifies 3 factors:
    - "Level," "slope," and "curvature."

$$ \min_{\mathbf{B},\mathbf{F}} \sum_{i=1}^N \sum_{t=1}^T \left( r_{i,t} - \sum_{k=0}^K b_{i,k} f_{k,t} \right)^2. $$

---

### Applications of Factor Models

#### Factor Tilting

Factor Tilting uses forecasts to adjust risk factor exposure.
- Similar to "market timing" in CAPM.
- APT allows specific factor views, unlike CAPM's market view.

With good forecasts, you can earn more by adjusting factor loadings.
- Increase loading if a factor is expected to outperform.
- Decrease loading if a factor is expected to underperform.

The return process is:

$$ r_{p,t} = E[r_{p,t}] + b_{p,1}\tilde{f}_{1,t} + \dots + b_{p,n}\tilde{f}_{n,t} + e_{p,t} $$

where $\tilde{f}$ are factor surprises.

- The market assumes $E[\tilde{f}_{k,t}] = 0$.
- If you believe $\tilde{f}_{1,t} \neq 0$, you have an edge.
- Use this edge to adjust $b_{p,1}$ in your portfolio.

Suppose that you have estimated the following 2-factor model for the three securities A, B and C:
$$
\begin{aligned}
r_A &= 0.12 + 1 \cdot \tilde{f}_1 + 1 \cdot \tilde{f}_2 + e_A \\
r_B &= 0.12 + 1 \cdot \tilde{f}_1 + 2 \cdot \tilde{f}_2 + e_B \\
r_C &= 0.12 + 3 \cdot \tilde{f}_1 + 2 \cdot \tilde{f}_2 + e_C
\end{aligned}
$$
where factor 1 is a foreign income factor, and factor 2 is an interest rate factor.

- What are the expected returns for each of the three securities?
- How would you find the factor risk premia?

You believe that:
- Europe and Japan will recover economically, boosting U.S. exports.
- Analysts' U.S. interest rate expectations are accurate.

How can you use this forecast with the factor model?
- Construct a portfolio with high factor 1 risk and low factor 2 risk.
- Which of the three stocks would you invest in more heavily?

Assume we want a loading of 10 for factor 1 and 0 for factor 2:
$$
\begin{aligned}
w_A + w_B + w_C &= 1, \\
w_A + w_B + 3w_C &= 10, \\
w_A + 2w_B + 2w_C &= 0.
\end{aligned}
$$
The first equation ensures the portfolio weights sum to one.

Solving these equations gives:

$$ w_A = 2, \quad w_B = -5.5, \quad w_C = 4.5. $$

If the foreign income factor "surprise" is 2%, the expected return on this portfolio is 0.32.

- This is much higher than the 0.12 without tilting.
- The portfolio has a loading of 10 on factor 1 and 0 on factor 2.
- If you had better information on U.S. interest rates, you could create a portfolio loading only on factor 2.

A portfolio loading on just one factor is a **factor-mimicking portfolio**.

#### Factor Mimicking Portfolios

Factor-mimicking portfolios are widely used in portfolio tilting and hedging strategies.

Common examples include:
- Oil futures for oil price factors.
- TIPS (Treasury Inflation-Protected Securities) for inflation factors.
- REITs (Real Estate Investment Trusts) for real-estate factors.
- Commodity futures for commodity price factors.

Many ETFs also serve as factor-mimicking portfolios.

---

## 5 **Machine Learning and Dynamic Factor Models**

### Instrumented Principal Component Analysis (IPCA)

#### Introduction to IPCA

Instrumented Principal Component Analysis (IPCA) combines:
- Factor analysis (like PCA).
- Conditioning information from firm characteristics.

Key innovation: IPCA allows factor loadings to vary with observable firm characteristics.

- Traditional factor models assume constant loadings.
- IPCA recognizes that firm exposures to risk factors change over time.
- These changes are related to observable firm characteristics

#### IPCA Framework

The IPCA model can be expressed as:

$$ r_{i,t+1} = \alpha_{i,t} + \beta_{i,t}^\top f_{t+1} + \varepsilon_{i,t+1}, $$

Where:
- $r_{i,t+1}$ is the excess return of asset $i$.
- $\alpha_{i,t} = z_{i,t}' \Gamma_\alpha + \nu_{\alpha,i,t}$ is the intercept.
- $\beta_{i,t} = z_{i,t}' \Gamma_\beta + \nu_{\beta,i,t}$ are time-varying factor loadings.
- $z_{i,t}$ are observable firm characteristics.
- $\Gamma_\alpha$ and $\Gamma_\beta$ are matrices of coefficients mapping characteristics to intercept and loadings, respectively.
- $\nu_{\alpha,i,t}$ and $\nu_{\beta,i,t}$ are error terms.
- $f_{t+1}$ are latent factors.

#### Advantages of IPCA

IPCA offers several advantages over traditional methods:
- Accounts for changing risk exposures.
- Uses firm characteristics to adjust factor loadings.
- Simplifies data while keeping economic relevance.
- Performs better out-of-sample than traditional models.
- Supports conditional asset pricing tests.

IPCA combines:
- Statistical methods (PCA).
- Fundamental methods (Fama-French).

#### Estimation of IPCA

The estimation minimizes the sum of squared residuals:

$$ \min_{\Gamma_\alpha, \Gamma_\beta, f} \sum_{i,t} \left( r_{i,t+1} - z'_{i,t}(\Gamma_\alpha + \Gamma_\beta f_{t+1}) \right)^2. $$

The estimation is done iteratively.

#### $R^2$ Measures in IPCA

Several $R^2$ measures help evaluate IPCA performance:

- **Total** $R^2$: Overall explanatory power of the model
	
	$$ R^2_{total} = 1 - \frac{\sum_{i,t} (r_{i,t+1} - z'_{i,t}(\hat{\Gamma}_\alpha + \hat{\Gamma}_\beta \hat{f}_{t+1}))^2}{\sum_{i,t} r_{i,t+1}^2} $$

- **Predictive** $R^2$: Out-of-sample forecasting power
	
	$$ R^2_{pred} = 1 - \frac{\sum_{i,t} (r_{i,t+1} - z'_{i,t}(\hat{\Gamma}_\alpha + \hat{\Gamma}_\beta \hat{\lambda}))^2}{\sum_{i,t} r_{i,t+1}^2}. $$

Higher predictive $R^2$ indicates better model performance.

---

### Deep Learning in Asset Pricing

#### Autoencoder Framework for Asset Pricing

The factor loadings:
$$
\begin{aligned}
z_{i,t}^{(0)} &= z_{i,t-1}, \\
z_{i,t}^{(l)} &= g \left( b^{(l-1)} + W^{(l-1)} z_{i,t}^{(l-1)} \right), \quad l = 1, \dots, L_\beta, \\
\beta_{i,t} &= b^{(L_\beta)} + W^{(L_\beta)} z_{i,t}^{(L_\beta)},
\end{aligned}
$$
where $z_{i,t}^{(0)} = z_{i,t-1}$ is the initial input, $z_{i,t}^{(l)}$ is the output of the $l$-th layer, and $\beta_{i,t}$ is the final output.

The factor values:
$$
\begin{aligned}
r_t^{(0)} &= r_t, \\
r_t^{(l)} &= \tilde{g} \left( \tilde{b}^{(l-1)} + \tilde{W}^{(l-1)} r_t^{(l-1)} \right), \quad l = 1, \dots, L_f, \\
f_t &= \tilde{b}^{(L_f)} + \tilde{W}^{(L_f)} r_t^{(L_f)}.
\end{aligned}
$$