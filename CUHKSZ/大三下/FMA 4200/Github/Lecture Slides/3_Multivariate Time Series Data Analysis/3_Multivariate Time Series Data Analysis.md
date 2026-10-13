# Multivariate Time Series Data Analysis

**Gongqiu Zhang** 
SSE, CUHK-Shenzhen

---

## 0 **Multivariate Financial Time Series Data**

In practice, we often need to analyze multiple financial time series simultaneously, as they may be related to each other.

Some examples:
- The SPX and SPX futures price.
- Prices of the same stock in different markets.
- The prices of different cryptocurrencies.
- ...

The purposes of multivariate data analysis include:
- Identifying causal relationships and forecasting.
- Portfolio construction and risk management.
- ...

---

## 1 **Multivariate Time Series Models**

### Multivariate Time Series Models

> [!info]+ Definition (Stationarity)
> Let $\boldsymbol{X}_t = (X_{1,t}, X_{2,t}, \cdots, X_{d,t})^\prime$ be a $d$ -dimensional vector process.
> We say $\boldsymbol{X}_t$ is stationary if:
> - The mean vector $\boldsymbol{\mu} = \mathbb{E}[\boldsymbol{X}_t]$ is independent of $t$.
> - The autocovariance matrix $\boldsymbol{\Gamma}(k) = \mathbb{E}[(\boldsymbol{X}_{t+k} - \boldsymbol{\mu})(\boldsymbol{X}_t - \boldsymbol{\mu})^\top]$ is independent of $t$ for all $k$.

The autocorrelation matrix is defined as:
$$
\boldsymbol{R}(k) = (\text{diag}(\boldsymbol{\Gamma}(0)))^{-1/2} \boldsymbol{\Gamma}(k) (\text{diag}(\boldsymbol{\Gamma}(0)))^{-1/2}.
$$

Writing separately for each element, we have:
$$
R_{ij}(k) = \frac{\gamma_{ij}(k)}{\sqrt{\gamma_{ii}(0)\gamma_{jj}(0)}}.
$$

It is not difficult to see that
$$
\boldsymbol{\Gamma}(-k) = \boldsymbol{\Gamma}(k)^\top, \quad \boldsymbol{R}(-k) = \boldsymbol{R}(k)^\top.
$$

### Vector White Noise Processes

> [!info]+ Definition (Vector White Noise Process)
> We say $\boldsymbol{\varepsilon}_t \sim \text{WN}(\boldsymbol{0}, \boldsymbol{\Sigma}_\varepsilon)$ is a vector white noise process if:
> - $\mathbb{E}[\boldsymbol{\varepsilon}_t] = \boldsymbol{0}$.
> - $\text{cov}(\boldsymbol{\varepsilon}_t) = \boldsymbol{\Sigma}_\varepsilon$.
> - $\boldsymbol{\varepsilon}_t$ is serially uncorrelated, i.e., $\text{cov}(\boldsymbol{\varepsilon}_t, \boldsymbol{\varepsilon}_s) = \boldsymbol{0}$ for $t \neq s$.

Vector white noise processes are stationary.

### Vector Moving Average Processes

> [!info]+ Vector Moving Average Processes
> A vector moving average process of order $q$: $\boldsymbol{X}_t \sim \text{MA}(q)$ is defined as:
> $$
> \boldsymbol{X}_t = \boldsymbol{\mu} + \sum_{i=1}^q \boldsymbol{B}_i \boldsymbol{\varepsilon}_{t-i} + \boldsymbol{\varepsilon}_t,
> $$
> where $\boldsymbol{\varepsilon}_t \sim \text{WN}(\boldsymbol{0}, \boldsymbol{\Sigma}_\varepsilon)$.
> 
> We have that
> $$
> \begin{aligned}
> \mathbb{E}[\boldsymbol{X}_t] &= \boldsymbol{\mu}, \\
> \boldsymbol{\Gamma}(k) &= \boldsymbol{B}_k \boldsymbol{\Sigma}_\varepsilon + \sum_{i=k+1}^q \boldsymbol{B}_i \boldsymbol{\Sigma}_\varepsilon \boldsymbol{B}_{i-k}^\top, \quad 0 \leq k \leq q,
> \end{aligned}
> $$
> where $\boldsymbol{B}_0 = \boldsymbol{I}_d$ and $\boldsymbol{\Gamma}(k) = \boldsymbol{0}$ for $k > q$.

### Sample Autocovariance Matrix

> [!info]+ Formula (Sample Autocovariance and Autocorrelation Matrix)
> The sample autocovariance matrix is defined as:
> $$
> \widehat{\boldsymbol{\Gamma}}(k) = \frac{1}{T} \sum_{t=1}^{t-k} (\boldsymbol{X}_{t+k} - \widehat{\boldsymbol{\mu}})(\boldsymbol{X}_t - \widehat{\boldsymbol{\mu}})^\top,
> $$
> where $\widehat{\boldsymbol{\mu}} = \frac{1}{T} \sum_{t=1}^T \boldsymbol{X}_t$.
> 
> The sample autocorrelation matrix is defined as:
> $$
> \widehat{\boldsymbol{R}}(k) = (\text{diag}(\widehat{\boldsymbol{\Gamma}}(0)))^{-1/2} \widehat{\boldsymbol{\Gamma}}(k) (\text{diag}(\widehat{\boldsymbol{\Gamma}}(0)))^{-1/2}.
> $$

### Exploratory Data Analysis: FTSE Indices

#### A Stylistic Example

We consider the daily simple returns of FTSE, Mid Cap (250), and Small Cap (350) from 2023-01-01 to 2024-12-31.

![[Pasted image 20260412181006.png]]
*Figure: FTSE 100 Price, FTSE Mid Cap 250 Price, and FTSE Small Cap Price from 2023-01-01 to 2024-12-31*

#### The ACF of Returns

![[Pasted image 20260412181029.png]]
*Figure: The ACF of Returns (FTSE Return ACF, FTSE Mid Cap Return ACF, FTSE Small Cap Return ACF)*

#### The ACF of Squared Returns

![[Pasted image 20260412181044.png]]
*Figure: The ACF of Squared Returns (FTSE Squared Return ACF, FTSE Mid Cap Squared Return ACF, FTSE Small Cap Squared Return ACF)*

#### Sample Autocorrelation Matrix of FTSE, Mid Cap, and Small Cap Returns

![[Pasted image 20260412181100.png]]
*Figure: Sample Autocorrelation Matrix of FTSE, Mid Cap, and Small Cap Returns*

#### Sample Autocorrelation Matrix of FTSE, Mid Cap, and Small Cap Squared Returns

![[Pasted image 20260412181120.png]]
*Figure: Sample Autocorrelation Matrix of FTSE, Mid Cap, and Small Cap Squared Returns*

---

### The Vector Autoregressive (VAR) Model

>[!info]+ The Vector Autoregressive (VAR) Model
> The $d$ -dimensional vector autoregressive (VAR) model of order $p$ is given by:
> $$
> \boldsymbol{X}_t = \boldsymbol{c} + \sum_{i=1}^p \boldsymbol{A}_i \boldsymbol{X}_{t-i} + \boldsymbol{\varepsilon}_t,
> $$
> where $\boldsymbol{\varepsilon}_t \sim \text{WN}(\boldsymbol{0}, \boldsymbol{\Sigma}_\varepsilon)$, $\boldsymbol{c}$ is a $d$ -dimensional vector, and $\boldsymbol{A}_i$ is a $d \times d$ matrix.
> 
> An example of the VAR(1) model is:
> $$
> \begin{aligned}
> \boldsymbol{X}_t &= \boldsymbol{A}_1 \boldsymbol{X}_{t-1} + \boldsymbol{\varepsilon}_t \\
> &= \boldsymbol{\varepsilon}_t + \boldsymbol{A}_1(\boldsymbol{\varepsilon}_{t-1} + \boldsymbol{A}_1 \boldsymbol{X}_{t-2}) \\
> &= \boldsymbol{\varepsilon}_t + \sum_{i=1}^\infty \boldsymbol{A}_1^i \boldsymbol{\varepsilon}_{t-i},
> \end{aligned}
> $$
> which converges as long as the eigenvalues of $\boldsymbol{A}_1$ are less than 1 in absolute value.

#### The Vector Autoregressive (VAR) Model: Parameter Estimation and Model Selection

For a fixed $p$, the VAR( $p$ ) model can be estimated by
- Least square estimation.
- Maximum likelihood estimation.

The optimal $p$ can be selected by minimizing the AIC, BIC or HQIC:
$$
\begin{aligned}
\text{AIC}(p) &= \log(|\widehat{\boldsymbol{\Sigma}}_\varepsilon(p)|) + \frac{2d^2p}{T}, \\
\text{BIC}(p) &= \log(|\widehat{\boldsymbol{\Sigma}}_\varepsilon(p)|) + \frac{d^2p \log(T)}{T}, \\
\text{HQIC}(p) &= \log(|\widehat{\boldsymbol{\Sigma}}_\varepsilon(p)|) + \frac{2d^2p \log(\log(T))}{T}.
\end{aligned}
$$

### Empirical Application: Fitting VAR Models to FTSE Data

#### Fitting the FTSE, Mid Cap, and Small Cap Returns

We fit the VAR models with up to 3 lags to the FTSE, Mid Cap, and Small Cap returns.

| Lag | AIC | BIC | HQIC |
| :--- | :--- | :--- | :--- |
| 1 | −32.15 | −32.05 | −32.11 |
| 2 | −32.16 | −31.99 | −32.09 |
| 3 | −32.15 | −31.90 | −32.05 |

The optimal lag is 1 according to BIC and HQIC, while 2 according to AIC.

##### Regression Results

|       | coefficient | std. error | $t$ -stat | $p$ -value |
| :---- | :---------- | :--------- | :----- | :------ |
| const | 0.000173    | 0.000295   | 0.586  | 0.558   |
| L1.y1 | 0.009911    | 0.068759   | 0.144  | 0.885   |
| L1.y2 | −0.015984   | 0.069037   | −0.232 | 0.817   |
| L1.y3 | 0.025658    | 0.105680   | 0.243  | 0.808   |

|       | coefficient | std. error | $t$ -stat | $p$ -value |
| :---- | :---------- | :--------- | :------- | :------ |
| const | 0.000162    | 0.000383   | 0.422    | 0.673   |
| L1.y1 | 0.062812    | 0.089214   | 0.704    | 0.481   |
| L1.y2 | −0.118409   | 0.089574   | −1.322   | 0.186   |
| L1.y3 | 0.179964    | 0.137119   | 1.312    | 0.189   |

| | coefficient | std. error | $t$ -stat | $p$ -value |
| :--- | :--- | :--- | :--- | :--- |
| const | 0.000157 | 0.000244 | 0.647 | 0.518 |
| L1.y1 | 0.059941 | 0.056717 | 1.057 | 0.291 |
| L1.y2 | 0.071465 | 0.056946 | 1.255 | 0.209 |
| L1.y3 | −0.012529 | 0.087172 | −0.144 | 0.886 |

##### Autocorrelation of Residuals

![[Pasted image 20260412181247.png]]
*Figure: ACF Matrix of VAR Model Residuals*

#### Fitting the FTSE, Mid Cap, and Small Cap Squared Returns

We fit the VAR models with up to 3 lags to the FTSE, Mid Cap, and Small Cap squared returns.

| Lag | AIC | BIC | HQIC |
| :--- | :--- | :--- | :--- |
| 1 | −56.76 | −56.66 | −56.72 |
| 2 | −56.78 | −56.60 | −56.71 |
| 3 | −56.77 | −56.52 | −56.67 |

The optimal lag is 1 according to BIC and HQIC, while 2 according to AIC.

##### Regression Results (Squared Returns)

| | coefficient | std. error | $t$ -stat | $p$ -value |
| :--- | :--- | :--- | :--- | :--- |
| const | 0.000037 | 0.000005 | 7.547 | 0.000 |
| L1.y1 | 0.016570 | 0.055430 | 0.299 | 0.765 |
| L1.y2 | 0.002043 | 0.044024 | 0.046 | 0.963 |
| L1.y3 | 0.196716 | 0.113897 | 1.727 | 0.084 |

| | coefficient | std. error | $t$ -stat | $p$ -value |
| :--- | :--- | :--- | :--- | :--- |
| const | 0.000068 | 0.000008 | 8.557 | 0.000 |
| L1.y1 | −0.038335 | 0.090032 | −0.426 | 0.670 |
| L1.y2 | −0.049049 | 0.071507 | −0.686 | 0.493 |
| L1.y3 | 0.377232 | 0.184999 | 2.039 | 0.041 |

| | coefficient | std. error | $t$ -stat | $p$ -value |
| :--- | :--- | :--- | :--- | :--- |
| const | 0.000025 | 0.000003 | 7.976 | 0.000 |
| L1.y1 | −0.056434 | 0.035790 | −1.577 | 0.115 |
| L1.y2 | 0.024296 | 0.028426 | 0.855 | 0.393 |
| L1.y3 | 0.198985 | 0.073541 | 2.706 | 0.007 |

##### Autocorrelation of Residuals

![[Pasted image 20260412181314.png]]
*Figure: ACF Matrix of VAR Model Residuals (Squared Returns)*

---

## 2 **Granger Causality**

### Granger Causality

> [!info]+ Definition (Granger Causality)
> The time series $Z_t$ is said to Granger cause time series $Y_t$ if $Z_t$ helps predict $Y_t$ better than the past values of $Y_t$ alone:
> $$
> \mathcal{L}(Y_t | Y_{t-1}, Y_{t-2}, \dots) \neq \mathcal{L}(Y_t | Y_{t-1}, Y_{t-2}, \dots, Z_{t-1}, Z_{t-2}, \dots).
> $$
> Similary, Granger causality in mean is defined as:
> $$
> \mathbb{E}[Y_t | Y_{t-1}, Y_{t-2}, \dots] \neq \mathbb{E}[Y_t | Y_{t-1}, Y_{t-2}, \dots, Z_{t-1}, Z_{t-2}, \dots].
> $$

#### Granger Causality Tests

> [!info]+ Granger Causality Tests
> The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another.
> 
> The null hypothesis is that the coefficients on the lags of the dependent variable are zero, meaning that the lags of the dependent variable are not useful in forecasting the dependent variable.
> 
> The test statistic is given by:
> $$
> F = \frac{(\text{RSS}_r - \text{RSS})/p}{\text{RSS}/(2T - 4p - 2)},
> $$
> where $\text{RSS}_r$ is the residual sum of squares of the restricted model, $\text{RSS}$ is the residual sum of squares of the unrestricted model, $p$ is the number of lags, and $T$ is the sample size.
> 
> The critical value is given by the F-distribution with $(p, 2T - 4p - 2)$ degrees of freedom.

### Empirical Application: Granger Causality on FTSE Data

#### Granger Causality Tests on Returns

| Causality Test | Test Statistic | Critical Value | $p$ -value | df |
| :--- | :--- | :--- | :--- | :--- |
| Mid Cap $\to$ FTSE | 0.05361 | 3.848 | 0.817 | (1, 1500) |
| Small Cap $\to$ FTSE | 0.05894 | 3.848 | 0.808 | (1, 1500) |
| FTSE $\to$ Mid Cap | 0.4957 | 3.848 | 0.482 | (1, 1500) |
| Small Cap $\to$ Mid Cap | 1.723 | 3.848 | 0.190 | (1, 1500) |
| FTSE $\to$ Small Cap | 1.117 | 3.848 | 0.291 | (1, 1500) |
| Mid Cap $\to$ Small Cap | 1.575 | 3.848 | 0.210 | (1, 1500) |
| Joint on FTSE | 0.03363 | 3.002 | 0.967 | (2, 1500) |
| Joint on Mid Cap | 1.488 | 3.002 | 0.226 | (2, 1500) |
| Joint on Small Cap | 1.999 | 3.002 | 0.136 | (2, 1500) |

Table: Granger Causality Test Results. * indicates significance at 5% level.

#### Granger Causality Tests on Squared Returns

| Causality Test | Test Statistic | Critical Value | $p$ -value | df |
| :--- | :--- | :--- | :--- | :--- |
| Mid Cap $\to$ FTSE | 0.002153 | 3.848 | 0.963 | (1, 1500) |
| Small Cap $\to$ FTSE | 2.983 | 3.848 | 0.084 | (1, 1500) |
| FTSE $\to$ Mid Cap | 0.1813 | 3.848 | 0.670 | (1, 1500) |
| Small Cap $\to$ Mid Cap | 4.158 | 3.848 | 0.042* | (1, 1500) |
| FTSE $\to$ Small Cap | 2.486 | 3.848 | 0.115 | (1, 1500) |
| Mid Cap $\to$ Small Cap | 0.7306 | 3.848 | 0.393 | (1, 1500) |
| Joint on FTSE | 2.850 | 3.002 | 0.058 | (2, 1500) |
| Joint on Mid Cap | 2.112 | 3.002 | 0.121 | (2, 1500) |
| Joint on Small Cap | 1.436 | 3.002 | 0.238 | (2, 1500) |

Table: Granger Causality Test Results. * indicates significance at 5% level.

---

## 3 **Cointegration and Statistical Arbitrage**

### Statistical Arbitrage

Statistical arbitrage is a class of trading strategies that exploit the mean reversion or the stationarity of some asset or portfolio.

Stationary versus mean reverting:
- **Stationary** : The mean and covariances of the series are constant over time.
- **Mean reverting** : The series tends to revert to a long-term mean.

These two concepts are inherently connected, but different.
- A stationary process is not necessarily mean reverting.
- A mean reverting process is not necessarily stationary.

### Unit-Root Stationarity

Unit-root stationarity is a specific type of stationarity modeled with an autoregressive (AR) model without unit roots.

A time series with a unit root is non-stationary and tends to diverge over time, e.g., random walk model for log-prices:
$$
y_t = \mu + y_{t-1} + \epsilon_t,
$$
where $\mu$ is the drift and $\epsilon_t$ the residual.

Example of AR(1) model without unit root (stationary):
$$
y_t = \mu + \rho y_{t-1} + \epsilon_t,
$$
where $|\rho| < 1$. When $\rho < 0$, the series is mean reverting.

### Cointegration

Cointegration means that two (or more) assets, while not stationary individually, are stationary with respect to each other.

Cointegration occurs when series contain stochastic trends (nonstationary) but move closely together, making their difference stable (stationary).

Intuitive example: Drunken man and dog wandering the streets: both paths are nonstationary, but the distance between them is mean-reverting and stationary.

#### Cointegration of Two Series

> [!info]+ Model (Cointegration of Two Series)
> A common model for two cointegrated time series:
> $$
> \begin{aligned}
> y_{1,t} &= \gamma x_t + w_{1,t}, \\
> y_{2,t} &= x_t + w_{2,t},
> \end{aligned}
> $$
> where
> $$
> x_t = x_{t-1} + w_t,
> $$
> $w_{1,t}, w_{2,t}, w_t$ are i.i.d. residual terms, with variances $\sigma_1^2, \sigma_2^2, \sigma^2$, and $\gamma$ is the coefficient determining the cointegration relationship.

Each time series $y_1$ and $y_2$ is a random walk plus noise, hence nonstationary.

The spread is the linear combination without the trend:
$$
z_t = y_{1,t} - \gamma y_{2,t} = w_{1,t} - \gamma w_{2,t},
$$
which is stationary.

#### Correlation versus Cointegration

Correlated time series are not necessarily cointegrated:
$$
\begin{aligned}
y_{1,t} &= x_t + w_t, \\
y_{2,t} &= w_t,
\end{aligned}
$$
where $x_t$ is a common stochastic trend and $w_t$ are i.i.d. residual terms.

Cointegrated time series are not necessarily highly correlated:
$$
\begin{aligned}
y_{1,t} &= x_t + w_{1,t}, \\
y_{2,t} &= x_t + w_{2,t},
\end{aligned}
$$
where $w_t$ and $w_{1,t}, w_{2,t}$ are independent i.i.d. residual terms.

Correlation concerns short-term fluctuations, while cointegration concerns long-term relationships.

---

### Pairs Trading

Historical context:
- Developed in the mid-1980s by Nunzio Tartaglia's team at Morgan Stanley.
- Achieved significant success but the team disbanded in 1989.

Trading strategies classification:
- **Momentum-based strategies** (or directional trading): Capture market trends. Treat fluctuations as undesired noise (risk).
- **Pairs trading** (or statistical arbitrage): Market neutral. Trade stationary fluctuations of relative mispricings between two securities.

#### A Simple Implementation of Pairs Trading

Based on comparing the spread of two time series $y_{1,t}$ and $y_{2,t}$ to a threshold $s_0$.

Spread defined as (in terms of log-prices or prices):
$$
z_t = y_{1,t} - \gamma y_{2,t} - \mu,
$$
where $\mu$ is the mean of the spread.

Trading strategy:
- Buy signal: Buy if the spread is low: $z_t < -s_0$.
- Short-sell signal: Short-sell if the spread is high: $z_t > s_0$.
- Unwinding the position: Unwind the position when the spread reverts back to the mean (after $k$ periods).
- This ensures a difference of at least $|z_{t+k} - z_t| \geq s_0$.

#### Pre-Screening for Cointegrated Pairs

Pre-screening for pairs trading: is a simple and cost-effective process to discard many pairs and select potential pairs for further analysis.

> [!info]+ Normalized Price Distance (NPD)
> Normalized Price Distance (NPD): is a common heuristic proxy for cointegration:
> $$
> \text{NPD} \triangleq \sum_{t=1}^T \left( \frac{p_{1,t}}{p_{1,0}} - \frac{p_{2,t}}{p_{2,0}} \right)^2,
> $$
> where $p_{1,t}$ and $p_{2,t}$ are the prices of the two assets at time $t$, and $p_{1,0}$ and $p_{2,0}$ are the prices of the two assets at time 0.

The NPD is a measure of the distance between the two price series, normalized by the initial prices.

### Cointegration Tests

Cointegration tests are used to determine if a pair of time series are cointegrated.

> [!info]+ Engle–Granger test
> Engle–Granger test: Simple and direct method for testing cointegration (Engle and Granger 1987). Two-step process:
> - Obtain $\gamma$ via least squares regression.
> - Test the residual for stationarity, e.g., Dickey-Fuller test.
> 
> Regression model:
> $$
> y_{1,t} = \gamma y_{2,t} + \mu + r_t,
> $$
> Residual $r_t$ is checked for unit-root stationarity or mean-reversion.

#### Cointegration Tests on FTSE Mid Cap, and Small Cap Prices

We apply the Engle-Granger test to the FTSE Mid Cap, and Small Cap prices from 2023-01-01 to 2024-12-31.

The test results are:

| | |
| :--- | :--- |
| Test Statistic | $-3.8709$ |
| $p$ -value | $0.0109$ |
| **Critical Values:** | |
| 1% | $-3.9183$ |
| 5% | $-3.3483$ |
| 10% | $-3.0529$ |

### Persistency of Cointegration

Persistence of cointegration:
- Discovering a cointegrated pair and passing tests does not guarantee persistent profitability.
- Challenges in practice: Cointegrated pairs found in historical data may lose cointegration in subsequent out-of-sample periods.
- Factors affecting persistence: management decisions, competition, company-specific news, etc.

Time-varying cointegration: Use of Kalman filtering to model time-varying cointegration.

---

### Trading with Cointegrated Pairs

> [!info]+ Thresholded Strategy
> Define the z-score:
> $$
> \hat{z}_t = \frac{z_t - \mathbb{E}[z_t]}{\sqrt{\text{Var}(z_t)}},
> $$
> where $z_t$ is the spread, and $\hat{z}_t$ is the estimated z-score.
> 
> Thresholded strategy: All-in or all-out sizing based on thresholds. Compare estimated z-score to a thresholds $s_0$ and $s_1$:
> $$
> s_t = \begin{cases}
> -1 & \text{if } \hat{z}_t < -s_0, \\
> 0 & \text{if } \hat{z}_t \in[-s_1, s_1], \\
> 1 & \text{if } \hat{z}_t > s_0.
> \end{cases}
> $$

In other regions, hold the position from the previous period.

#### Optimal Threshold

After $N_{\text{trades}}$ successful trades, the total (uncompounded) profit is:
$$
N_{\text{trades}} \times \sigma \times s_0,
$$
where $\sigma$ is the standard deviation of the z-score.

The number of successful trades decreases with $s_0$:
$$
\max_{s_0} \mathbb{E}[N_{\text{trades}} \times \sigma \times s_0].
$$

##### A Parametric Approach

Assuming that $\hat{z}_t \sim \mathcal{N}(0, 1)$, the number of successful trades is:
$$
\mathbb{E}[N_{\text{trades}}] = T(1 - \Phi(s_0)),
$$
where $\Phi$ is the cumulative distribution function of the standard normal distribution.

The optimization problem is:
$$
\max_{s_0} (1 - \Phi(s_0)) \times s_0,
$$
which can be optimized by bisection search.

##### A Parametric Approach: Results

![[Pasted image 20260412181637.png]]
*Figure: Expected Profit Rate/Standard Deviation vs. Threshold (Standard Deviations)*

##### A Non-Parametric Approach

Given $T$ observations of the estimated z-score, $\hat{z}_t$ for $t = 1, \dots, T$, and $J$ discretized threshold values, $s_{01}, \dots, s_{0J}$:
$$
\bar{f}_j = \frac{1}{T} \sum_{t=1}^T \mathbf{1}_{\{\hat{z}_t > s_{0j}\}}.
$$

Optimal threshold:
$$
s_0^\ast = \text{argmax}_{s_{0j} \in \{s_{01}, s_{02}, \dots, s_{0J}\}} s_{0j} \times \bar{f}_j.
$$

##### A Non-Parametric Approach: Results

![[Pasted image 20260412181654.png]]
*Figure: Empirical Expected Profit/Standard Deviation vs. Threshold (Standard Deviations)*

---

### Multivariate Cointegration

A univariate time series $X_t$ is said to have $k$ unit roots if its $k$ -th difference is stationary: $X_t \sim I(k)$.

Examples:
- $X_t \sim I(1)$ is said to be integrated of order 1.
- $X_t \sim I(0)$ is stationary.

>[!info]+ Definition (Multivariate Cointegration)
> A multivariate time series $\boldsymbol{X}_t$ is said to be cointegrated of order $(k, h)$ ($k \geq h \geq 1$) if:
> - All components of $\boldsymbol{X}_t$ are integrated of order $k$, i.e., $\boldsymbol{X}_t \sim I(k)$.
> - There exists a nonzero vector $\boldsymbol{\beta}$ such that $\boldsymbol{\beta}^\top \boldsymbol{X}_t \sim I(k - h)$.

We denote this as $\boldsymbol{X}_t \sim CI(k, h)$. The most frequent case is $k = h = 1$, i.e., $\boldsymbol{X}_t \sim CI(1, 1)$.

The vector $\boldsymbol{\beta}$ is called the cointegrating vector.

#### Johansen Test

> [!info]+ Johansen Test
> There are at most $d$ cointegrating vectors for a $d$ -dimensional time series.
> $$
> \boldsymbol{V} = (\boldsymbol{\beta}_1, \boldsymbol{\beta}_2, \dots, \boldsymbol{\beta}_r), \quad r \leq d,
> $$
> is a matrix of cointegrating vectors.
> 
> The Johansen test is a procedure for testing the number of cointegrating vectors.
> $$
> \begin{aligned}
> \text{H}_0 &: r = 0 \quad \text{vs} \quad \text{H}_1 : r \geq 1, \\
> \text{H}_0 &: r \leq 1 \quad \text{vs} \quad \text{H}_1 : r \geq 2, \\
> &\dots \\
> \text{H}_0 &: r \leq d - 1 \quad \text{vs} \quad \text{H}_1 : r = d.
> \end{aligned}
> $$

#### To Detect Cointegration among FTSE, Mid Cap, and Small Cap Prices

The test statistics for the data from 2023-01-01 to 2023-12-31 are:

| Null Hypothesis | Trace Statistic | 5% Critical Value |
| :--- | :--- | :--- |
| $r = 0$ | 39.5686 | 35.0116 |
| $r \leq 1$ | 15.3957 | 18.3985 |
| $r \leq 2$ | 2.7247 | 3.8415 |

The cointegration vectors are:
$$
\begin{pmatrix}
0.00192515 & -0.00880598 & -0.0034959 \\
-0.00525423 & 0.00176697 & -0.00171257 \\
0.01467691 & -0.00372121 & 0.01380514
\end{pmatrix}.
$$

#### Statistical Arbitrage

Define the spread as:
$$
S_t = \boldsymbol{\beta}^\top \boldsymbol{X}_t.
$$

The statistical arbitrage strategy is to buy the spread when it is low and sell it when it is high:
$$
\begin{aligned}
\text{Position} &= 1 \quad \text{if} \quad S_t < -\text{threshold}_1, \\
\text{Position} &= 0 \quad \text{if} \quad -\text{threshold}_2 \leq S_t \leq \text{threshold}_2, \\
\text{Position} &= -1 \quad \text{if} \quad S_t > \text{threshold}_1.
\end{aligned}
$$

We use the first half of the sample to estimate the cointegration vector and the second half of the sample to backtest the strategy.

##### Statistical Arbitrage: Results

![[Pasted image 20260412181719.png]]
*Figure: Spread Z-Score, Trading Positions, and Cumulative Returns (Test Period)*