# Time Series Data Analysis

**Gongqiu Zhang**
**SSE, CUHK-Shenzhen**

---

## 1 **Stylized Facts of Financial Returns**

### The Data

We will use the S&P 500 index data ( 2001-01-08 to 2024-12-31 ) from Yahoo Finance to illustrate the stylized facts of financial returns.

![[Pasted image 20260123150232.png]]
*Figure: S&P 500 Adjusted Closing Price & Daily Log Returns*

### Stationarity

The prices of an asset recorded over time are often **not stationary**, due to, for example,
* the steady expansion of an economy,
* the increase of productivity due to technological progress,
* economic recessions and financial crises.

However, the returns are often **stationary**, fluctuating around a constant level.

### Distributional Properties
![[Pasted image 20260123150322.png]]
*Figure: Distribution of Daily Returns, Weekly Returns, Monthly Returns, Quarterly Returns*

![[Pasted image 20260123150345.png]]
*Figure: Q-Q Plot of Daily, Weekly, Monthly, Quarterly Returns vs Normal Distribution*

From the QQ plot, we can see that the S&P 500 returns are **heavy-tailed** and **left-skewed**.

As the sampling frequency increases, the distribution of returns becomes more and more **heavy-tailed**, while the **left-skewness** is quite persistent for various sampling frequencies.

### Volatility Clustering

![[Pasted image 20260123150541.png]]
*Figure: S&P 500 Daily, Weekly, and Monthly Log Returns over time*

As the sampling frequency increases, the volatility clustering becomes more and more pronounced.
The high volatility periods are followed by high volatility periods, e.g., the financial crisis in 2008.

### Long-Range Dependence

The autocorrelation function (ACF) is defined as $\rho(k) = \gamma(k) / \gamma(0)$, where $\gamma(k)$ is the autocovariance function:

$$\gamma(k) = \text{Cov}(r_t, r_{t-k}) \approx \frac{1}{T-k} \sum_{t=k+1}^T (r_t - \bar{r})(r_{t-k} - \bar{r}).$$

For log returns, $\rho(k) \approx 0$ for $k \geq 1$, while for the squared returns, $\rho(k)$ decays slowly, indicating **long-range dependence**.

![[Pasted image 20260123150616.png]]
*Figure: ACF of Daily Returns vs ACF of Squared Daily Returns*

### Leverage Effect: VIX and S&P 500 Returns

When the SPX index drops, the firms are more leveraged, and the risk of the firms increases, leading to increasing volatility ( proxied by VIX ).

![[Pasted image 20260123150657.png]]
*Figure: S&P 500 and VIX time series*

![[Pasted image 20260123150721.png]]
*Figure: S&P 500 Returns vs VIX Changes (Correlation: -0.71)*

---

## 2 **Efficient Market Hypothesis and Statistical Models for Financial Returns**

### Efficient Market Hypothesis

The efficient market hypothesis (EMH) states that the prices of an asset reflect all available information, and it is impossible to consistently earn abnormal returns by exploiting any information.

There are three forms of the EMH:
* **Weak form**: the current price of an asset reflects all past publicly available information,
* **Semi-strong form**: the current price of an asset reflects all publicly available information,
* **Strong form**: the current price of an asset reflects all information, both public and private.

### EMH and The General Dynamics of Asset Returns

Under the EMH, the asset return process can be expressed as:

$$r_t = \mu_t + \varepsilon_t,$$

where:
* $\mu_t$ is the rational expectation ( or the least square prediction ) of $r_t$ based on all publicly available information at time $t-1$; according to the stylized fact, we may set $\mu_t = \mu$ as a constant;
* $\varepsilon_t$ is the unpredictable return termed **innovation** due to new information arriving between $t-1$ and $t$, specified as a **white noise** process:
    * $\varepsilon_t$ is serially uncorrelated, i.e., $\text{Cov}(\varepsilon_t, \varepsilon_{t-k}) = 0$ for $k \geq 1$,
    * $\text{Var}(\varepsilon_t) = \sigma^2$.

### Two Stronger Assumptions on the Innovations

> [!info] + IID innovations
> $$\varepsilon_t \sim \text{i.i.d.}(0, \sigma^2).$$

> [!info] + Martingale difference (MD) innovations
> $$\mathbb{E}[\varepsilon_t | r_{t-1}, r_{t-2}, \cdots] = \mathbb{E}[\varepsilon_t | \varepsilon_{t-1}, \varepsilon_{t-2}, \cdots] = 0,$$
> where $\varepsilon_{t-1}, \varepsilon_{t-2}, \cdots$ are the innovations at time $t-1, t-2, \cdots$.

A frequently used example of martingale difference innovations:
$$\varepsilon_t = \sigma_t \eta_t,$$
where $\eta_t \sim \text{i.i.d.}(0, 1)$.

#### The Relationship Between the Three Types of Innovations
$$\text{IID} \implies \text{Martingale Difference} \implies \text{White Noise}.$$

The i.i.d. innovations are the strongest assumption but it contradicts the stylized fact that the squared returns are long-range dependent. The martingale difference innovations are a compromise and the most commonly used assumption:
* The returns are **not predictable**;
* but volatility **may be predictable**.

### Tests for White Noise: the Ljung-Box Test

The Ljung-Box $Q(m)$ statistic is defined as:

$$Q(m) = T(T+2) \sum_{k=1}^m \frac{\hat{\rho}(k)^2}{T-k},$$

where $\hat{\rho}(k)$ is the sample autocorrelation function (ACF) of the returns and $m$ is a prespecified positive integer.
We reject the null hypothesis of white noise at the significance level $\alpha$ if $Q(m) > \chi^2_{1-\alpha}(m)$, or the p-value $\mathbb{P}(Q > Q(m)) < \alpha$ with $Q \sim \chi^2(m)$.

#### Ljung-Box Test for S&P 500 Returns (2023-01-01 to 2024-12-31)

| | $m=6$ | $m=12$ | $m=24$ |
| :--- | :--- | :--- | :--- |
| **Returns** | | | |
| Q-stat | 4.5393 | 7.6178 | 24.4364 |
| p-value | 0.6041 | 0.8142 | 0.4369 |
| **Absolute Returns** | | | |
| Q-stat | 16.6896 | 28.7464 | 35.9555 |
| p-value | 0.0105 | 0.0043 | 0.0554 |
| **Squared Returns** | | | |
| Q-stat | 19.7053 | 32.8507 | 36.6592 |
| p-value | 0.0031 | 0.0010 | 0.0473 |

Using too long data may lead to the rejection of the null hypothesis of white noise.

### Augmented Dickey-Fuller (ADF) Test

The ADF test is a popular test for the null hypothesis of a **unit root**, i.e., the returns are non-stationary.
There are three versions:
* **no constant/trend**: $\ln S_t = \alpha \ln S_{t-1} + \sum_{i=1}^p \beta_i \Delta \ln S_{t-i} + \varepsilon_t$;
* **constant only**: $\ln S_t = \alpha \ln S_{t-1} + \beta + \sum_{i=1}^p \beta_i \Delta \ln S_{t-i} + \varepsilon_t$;
* **constant and trend**: $\ln S_t = \alpha \ln S_{t-1} + \beta + \gamma t + \sum_{i=1}^p \beta_i \Delta \ln S_{t-i} + \varepsilon_t$.

Null hypothesis: $\alpha = 1$; alternative hypothesis: $\alpha < 1$. Test statistic: $$\text{DF} = \frac{\hat{\alpha}-1}{\text{SE}(\hat{\alpha})}$$

#### ADF Test Results for S&P 500 Returns

##### Augmented Dickey-Fuller Test Results for Log SPX Price

|                    | No Const/Trend | Const Only | Const& Trend |
| :----------------- | :------------: | :--------: | :----------: |
| Test Statistic     |     1.7952     |   0.6539   |   –2.7549    |
| p-value            |     0.9832     |   0.9889   |    0.2140    |
| 1% Critical Value  |    –2.5661     |  –3.4314   |   –3.9603    |
| 5% Critical Value  |    –1.9410     |  –2.8620   |   –3.4112    |
| 10% Critical Value |    –1.6168     |  –2.5670   |   –3.1275    |

##### Augmented Dickey-Fuller Test Results for SPX Returns

|                    | No Const/Trend | Const Only | Const& Trend |
| :----------------- | :------------: | :--------: | :----------: |
| Test Statistic     |    –21.3870    |  –21.5970  |   –21.5787   |
| p-value            |     0.0000     |   0.0000   |    0.0000    |
| 1% Critical Value  |    –2.5702     |  –3.4435   |   –3.9770    |
| 5% Critical Value  |    –1.9415     |  –2.8673   |   –3.4193    |
| 10% Critical Value |    –1.6163     |  –2.5699   |   –3.1322    |

---

## 3 **Time Series Models for Financial Returns**

### Stationary Time Series

A **stationary**$^1$ time series is a time series $\{X_t\}$ that has a constant mean and variance, and the covariance between the returns at different times is constant, i.e.,

$$\mathbb{E}[X_t], \quad \text{Var}(X_t), \quad \text{Cov}(X_t, X_{t-k}),$$

are independent of $t$.

The ACF of a stationary time series is given by:

$$\rho(k) = \frac{\gamma_k}{\gamma_0},$$

where $\gamma_k = \text{Cov}(X_t, X_{t-k})$.

$^1$**Remark**: **Strong stationary** means the joint distribution of $(X_t, X_{t-1}, \cdots, X_{t-k})$ is the same as that of $(X_{t+h}, X_{t+h-1}, \cdots, X_{t+h-k})$ for any $h \geq 0$ and $t$.

### Moving Average (MA) Models

> [!info] + Moving Average (MA) Models
> The MA($q$) model is defined as:
> $$X_t = \mu + \varepsilon_t + \sum_{i=1}^q a_i \varepsilon_{t-i},$$
> where $\varepsilon_t \sim \text{WN}(0, \sigma^2)$ is the unobservable innovations at time $t$, and $a_1, a_2, \cdots, a_q$ are the parameters.

The MA($q$) model is always stationary.

$$
\begin{aligned}
\mathbb{E}[X_t] &= \mu, \\
\text{Var}(X_t) &= \sigma^2(1 + a_1^2 + a_2^2 + \cdots + a_q^2), \\
\text{Cov}(X_t, X_{t-|k|}) &= \sigma^2 \left( a_{|k|} + \sum_{i=1}^{q-|k|} a_{|k|+i} a_i \right), \quad |k| \leq q, \\
\text{Cov}(X_t, X_{t-k}) &= 0, \quad |k| > q.
\end{aligned}
$$

#### The ACF of the MA($q$) Model

The ACF of the MA($q$) model is given by:

$$
\rho(k) = \begin{cases} \frac{a_{|k|} + \sum_{i=1}^{q-|k|} a_{|k|+i} a_i}{1 + a_1^2 + a_2^2 + \cdots + a_q^2}, & |k| \leq q, \\ 0, & |k| > q. \end{cases}
$$

The ACF of the MA($q$) model is independent of $\sigma^2, \mu,$ and cuts off after $q$ lags.

#### To Determine the Order of the MA Model

It is known that the ACF of the MA($q$) model cuts off after $q$ lags. Further assuming that $\varepsilon_t \sim \text{i.i.d.}(0, \sigma^2)$ and $\mathbb{E}[\varepsilon_t^4] < \infty$, we have that

$$\sqrt{T}\hat{\rho}(k) \xrightarrow{d} \mathcal{N}\left(0, 1 + 2 \sum_{i=1}^q \rho(i)^2\right), \quad k > q,$$

as the sample size $T$ goes to infinity, where $\hat{\rho}(k)$ is the sample ACF of the returns.

We can use the above equation to determine the order of the MA model by testing the null hypothesis that $\rho(k) = 0$ and set $q$ as the largest integer such that the null hypothesis is rejected.

As an example, we generate 1000 observations from the following models.
* MA(1) model with $a_1 = 0.7$,
* MA(2) model with $a_1 = 0.7$ and $a_2 = -0.4$,
* MA(4) model with $a_1 = 0.7, a_2 = -0.4, a_3 = 0.6,$ and $a_4 = 0.8$.

We plot the sample ACF and the 95% confidence intervals.

![[Pasted image 20260123153152.png]]
*Figure: Sample ACF of MA(1) Process, Sample ACF of MA(2) Process, Sample ACF of MA(4) Process*

#### Identification and Invertibility of MA Models

Consider two MA(1) models:

$$
\begin{aligned}
X_t &= \varepsilon_t + a \varepsilon_{t-1}, \quad \varepsilon_t \sim \text{i.i.d.}(0, \sigma^2), \\
Y_t &= e_t + a^{-1} e_{t-1}, \quad e_t \sim \text{i.i.d.}(0, a^2 \sigma^2).
\end{aligned}
$$

The two models are observationally equivalent, i.e., they have the same ACVF and hence the same ACF. Therefore, we cannot identify the two models from the ACF. To solve this issue, we can impose that $|a| < 1$.

This is equivalent to the invertibility condition of the MA model. For MA($q$) models, the invertibility condition is that the roots of the characteristic polynomial $1 + a_1 z + a_2 z^2 + \cdots + a_q z^q = 0$ are all outside the unit circle.

### Autoregressive (AR) Models

> [!info] + Autoregressive (AR) Models
> The AR($p$) model is defined as:
> $$X_t = c + \sum_{i=1}^p b_i X_{t-i} + \varepsilon_t,$$
> where $\varepsilon_t \sim \text{WN}(0, \sigma^2)$ is the unobservable innovations at time $t$, and $b_1, b_2, \cdots, b_p$ are the parameters.

The AR($p$) model is not always stationary. For example, the AR(1) model $X_t = b X_{t-1} + \varepsilon_t$ is stationary if $|b| < 1$, and non-stationary if $|b| \geq 1$.

#### Stationary AR($p$) Models

The AR($p$) model is stationary if the $p$ roots of the characteristic polynomial $1 - b_1 z - b_2 z^2 - \cdots - b_p z^p = 0$ are all outside the unit circle, i.e., the modulus of the roots are greater than 1.

The ACF of the stationary AR($p$) model decays exponentially, i.e., $\rho(k) = \mathcal{O}(a^k)$ for some $a < 1$ as $k \to \infty$.

Suppose the AR($p$) model is stationary, the mean is given by:

$$\mathbb{E}[X_t] = \frac{c}{1 - b_1 - b_2 - \cdots - b_p}.$$

The ACVF satisfies the so-called **Yule-Walker equations**:

$$\gamma(k) = b_1 \gamma(k-1) + b_2 \gamma(k-2) + \cdots + b_p \gamma(k-p), \quad k \geq 1.$$

We can first solve the above equations for $\gamma(k)$ for $k \leq p$, and then use the Yule-Walker equations to calculate the ACVF for $k > p$.

For example, consider the AR(1) model $X_t = b X_{t-1} + \varepsilon_t$. The ACVF and ACF are given by:

$$
\gamma(k) = \begin{cases} \sigma^2 / (1 - b^2), & k = 0, \\ \sigma^2 b^{|k|} / (1 - b^2), & |k| > 0. \end{cases}
$$

$$
\rho(k) = \begin{cases} 1, & k = 0, \\ b^{|k|}, & |k| > 0. \end{cases}
$$

### Autoregressive (AR) Models are MA($\infty$) Models

Consider the AR(1) model $X_t = b X_{t-1} + \varepsilon_t$. We can write the model as:

$$
\begin{aligned}
X_t &= \varepsilon_t + b \varepsilon_{t-1} + b^2 \varepsilon_{t-2} + \cdots \\
&= \sum_{i=0}^\infty b^i \varepsilon_{t-i},
\end{aligned}
$$

where the convergence is in mean square when $|b| < 1$.

The above derivation can be reformulated using the lag operator $B$, i.e., $B X_t = X_{t-1}$. Then we have:

$$X_t = (1 - bB)^{-1} \varepsilon_t = \sum_{i=0}^\infty b^i B^i \varepsilon_t = \sum_{i=0}^\infty b^i \varepsilon_{t-i},$$

where the convergence is in mean square when $|b| < 1$.

Autoregressive models are always invertible.

#### AR Models are Mean Reverting

Let

$$\mu = \mathbb{E}[X_t] = \frac{c}{1 - \sum_{i=1}^p b_i}.$$

Then,

$$X_t - \mu = \sum_{i=1}^p b_i (X_{t-i} - \mu) + \varepsilon_t.$$

#### To Determine the Order of the AR Model

The partial autocorrelation function (PACF) is defined as the correlation between $X_{k+1}$ and $X_1$ after removing the linear dependence on the intermediate variables $X_2, X_3, \cdots, X_k$.

Suppose we run a regression:

$$\min_{\beta_0, \beta_1, \beta_2, \cdots, \beta_k} \mathbb{E} \left[ \left( X_{k+1} - \beta_0 - \sum_{i=1}^k \beta_i X_{k-i+1} \right)^2 \right].$$

The PACF is given by $\pi(k) = \beta_k^*$, which cuts off after $p$ lags, i.e., $\pi(k) = 0$ for $k > p$.

The sample PACF can be estimated by a regression based on the observed series:

$$\min_{\beta_0, \beta_1, \beta_2, \cdots, \beta_k} \sum_{t=k+1}^T \left( X_t - \beta_0 - \sum_{i=1}^k \beta_i X_{t-i} \right)^2.$$

Suppose the true model is an stationary AR($p$) model with $\varepsilon_t \sim \text{i.i.d.}(0, \sigma^2)$ and $\mathbb{E}[\varepsilon_t^4] < \infty$,

$$\sqrt{T} \hat{\pi}(k) \xrightarrow{d} \mathcal{N}(0, 1), \quad k > p.$$

As an example, we generate 1000 observations from the following models.
* AR(1) model with $b_1 = 0.7$,
* AR(2) model with $b_1 = 0.7$ and $b_2 = -0.4$,
* AR(4) model with $b_1 = 0.7, b_2 = -0.4, b_3 = 0.6,$ and $b_4 = -0.8$.

![[Pasted image 20260123153302.png]]
*Figure: Sample PACF of AR(1) Process, Sample PACF of AR(2) Process, Sample PACF of AR(4) Process*

### ARMA Models

> [!info] + ARMA Models
> The ARMA($p, q$) model is defined as:
> $$X_t = c + \sum_{i=1}^p b_i X_{t-i} + \varepsilon_t + \sum_{i=1}^q a_i \varepsilon_{t-i},$$
> where $\varepsilon_t \sim \text{WN}(0, \sigma^2)$ is the unobservable innovations at time $t$, and $b_1, b_2, \cdots, b_p$ and $a_1, a_2, \cdots, a_q$ are the parameters.

The **stationary condition** of the ARMA($p, q$) model is that the $p$ roots of the characteristic polynomial $b(z) = 1 - b_1 z - b_2 z^2 - \cdots - b_p z^p$ are all outside the unit circle.

The **invertible condition** of the ARMA($p, q$) model is that the $q$ roots of the characteristic polynomial $a(z) = 1 + a_1 z + a_2 z^2 + \cdots + a_q z^q$ are all outside the unit circle.

#### Stationary ARMA Models

For a stationary ARMA($p, q$) model, the mean, variance, and autocovariance function are given by:

$$\mathbb{E}[X_t] = \mu = \frac{c}{1 - b_1 - b_2 - \cdots - b_p}.$$

The ARMA model can be written as:

$$b(B) X_t = c + a(B) \varepsilon_t.$$

If it is stationary, we can write the ARMA model as:

$$X_t = \mu + a(B)^{-1} (1 + b(B)) \varepsilon_t = \mu + \varepsilon_t + \sum_{i=1}^\infty \alpha_i \varepsilon_{t-i},$$

with $\sum_{i=1}^\infty |\alpha_i| < \infty$, which is an MA($\infty$) process (belongs to the class of **causal processes**).

#### Invertibility of the ARMA Model

For an invertible ARMA($p, q$) model, we can write the ARMA model as:

$$
\begin{aligned}
\varepsilon_t &= a^{-1}(B) (b(B) X_t - c) \\
&= X_t - \mu^* - \sum_{i=1}^\infty \beta_i X_{t-i},
\end{aligned}
$$

for some constants $\mu^*$ and $\beta_i$. This is an AR($\infty$) process.

For example, consider the ARMA(1, 1) model $X_t = c + b X_{t-1} + \varepsilon_t + a \varepsilon_{t-1}$. We can write the model as:

$$
\begin{aligned}
\varepsilon_t &= (1 + aB)^{-1} (X_t - b X_{t-1} - c) \\
&= (1 - aB + a^2 B^2 - \cdots) (X_t - b X_{t-1} - c) \\
&= \sum_{i=0}^\infty (-a)^i B^i (X_t - b X_{t-1} - c).
\end{aligned}
$$

#### The Autocorrelation Functions of the ARMA Model

**Autocorrelation**:
* The autocorrelation function of an ARMA($p, q$) process exhibits exponential decay towards zero: it does not cut off but gradually dies out as $|k|$ increases, possibly damped oscillations.
* The autocorrelation function of an ARMA($p, q$) process displays the shape of that of an AR($p$) process for $|k| > \max(p, q + 1)$.

**Partial Autocorrelation**:
* The partial autocorrelation function of an ARMA($p, q$) process will gradually die out (the same property as a MA($q$) model).

It is more difficult to determine the order of the ARMA model than the AR and MA models.

#### Gaussian MLE of ARMA Models

Though the innovations are not necessarily Gaussian, we can still use the **Gaussian maximum likelihood estimation (MLE)** to estimate the parameters of the ARMA model which can lead to consistent estimates of the parameters.

In this case, $\boldsymbol{X} = (X_1, X_2, \cdots, X_T)$ are jointly normal with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ depending on the model parameters $(c, \boldsymbol{a}, \boldsymbol{b}, \sigma^2)$. Then joint density function of $\boldsymbol{X}$ is given by:

$$f(\boldsymbol{X}) = \frac{1}{(2\pi)^{T/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{X} - \boldsymbol{\mu}) \right).$$

The log-likelihood function is given by:

$$\ell(\boldsymbol{\theta}) = \log L(\boldsymbol{\theta}) = -\frac{T}{2} \log(2\pi) - \frac{1}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2} (\boldsymbol{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{X} - \boldsymbol{\mu}).$$

If the true distribution is not Gaussian, the resulting estimators are **quasi-MLE**.

#### Statistical Inference of ARMA Models

Once we have the MLE $\hat{\boldsymbol{\theta}}$ of the parameters, we can conduct the statistical inference of the parameters.

For example, we can conduct the Wald test to test the significance of the parameters. The Wald test is given by:

$$W = (\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}_0)^T \widehat{\text{Var}}(\hat{\boldsymbol{\theta}})^{-1} (\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}_0) \xrightarrow{a \sim H_0} \chi^2(p),$$

where $p$ is the number of parameters, $\widehat{\text{Var}}(\hat{\boldsymbol{\theta}}) = -\partial^2 \ell(\hat{\boldsymbol{\theta}}) / \partial \boldsymbol{\theta} \partial \boldsymbol{\theta}^T$ is the covariance matrix of $\hat{\boldsymbol{\theta}}$ (also called the **Fisher information matrix**).

Suppose we want to calculate the standard error of $f(\hat{\boldsymbol{\theta}})$ for some smooth function $f$, we can use the Delta Method:

$$\text{Var}(f(\hat{\boldsymbol{\theta}})) \approx \partial f(\hat{\boldsymbol{\theta}}) / \partial \boldsymbol{\theta}^T \widehat{\text{Var}}(\hat{\boldsymbol{\theta}}) \partial f(\hat{\boldsymbol{\theta}}) / \partial \boldsymbol{\theta}.$$

Asymptotically, we can assume that the distribution of $\hat{\boldsymbol{\theta}}$ is Gaussian.

#### Fitting SPX Returns to an ARMA(1, 1) Model

We fit the SPX returns to an ARMA(1, 1) model and report the summary of the model.

| Parameter | Coef | Std Err | Z | p value |
| :--- | :--- | :--- | :--- | :--- |
| const | 0.0002 | 0.000 | 1.690 | 0.091 |
| ar.L1 | $-0.0129$ | 0.048 | $-0.266$ | 0.790 |
| ma.L1 | $-0.0922$ | 0.049 | $-1.869$ | 0.062 |
| sigma2 | 0.0001 | $1.12 \times 10^{-6}$ | 130.141 | 0.000 |

For the unconditional mean, we use the Delta method to calculate the standard error and the 95% confidence interval:

| Metric | Value |
| :--- | :--- |
| Estimated Mean | $2.46 \times 10^{-4}$ |
| Standard Error | $1.44 \times 10^{-4}$ |
| 95% CI | $[-3.60, 52.9] \times 10^{-5}$ |

#### To Determine the Order of the ARMA Model

For ARMA models, there is no simple way to determine the order of the model.

We can use the **AIC (Akaike Information Criterion)** or **BIC (Bayesian Information Criterion)** to determine the order of the model.

The AIC and BIC are defined as:

$$
\begin{aligned}
\text{AIC} &= -2 \log \hat{L} + 2(p + q + 1), \\
\text{BIC} &= -2 \log \hat{L} + \log(T)(p + q + 1),
\end{aligned}
$$

where $\hat{L}$ is the maximized log-likelihood function and $p$ and $q$ are the orders of the AR and MA parts, respectively.

#### Fitting SPX Returns to ARMA Models

We fit the SPX returns to ARMA models with different orders ($p$ and $q$) and report the AIC and BIC of the models.

| $p$ | $q$ | AIC | BIC |
| :--- | :--- | :--- | :--- |
| 0 | 0 | $-36103.89$ | $-36090.47$ |
| 0 | 1 | $-36168.02$ | $-36147.91$ |
| 0 | 2 | $-36166.03$ | $-36139.21$ |
| 0 | 3 | $-36164.12$ | $-36130.60$ |
| 0 | 4 | $-36169.50$ | $-36129.27$ |
| 1 | 0 | $-36167.44$ | $-36147.32$ |
| 1 | 1 | $-36166.04$ | $-36139.22$ |
| 1 | 2 | $-36164.03$ | $-36130.50$ |
| 1 | 3 | $-36163.01$ | $-36122.78$ |
| 1 | 4 | $-36166.37$ | $-36119.44$ |
| 2 | 0 | $-36166.15$ | $-36139.34$ |
| 2 | 1 | $-36163.94$ | $-36130.41$ |

#### The Best AIC and BIC Models

**Best AIC Model Summary**:

|        | coef    | std err  | z       | $P > \ | z \|$ |
| :----- | :------ | :------- | :------ | :------------ |
| const  | 0.0002  | 0.000    | 1.652   | 0.099         |
| ma.L1  | -0.1038 | 0.007    | -15.492 | 0.000         |
| ma.L2  | -0.0010 | 0.005    | -0.192  | 0.848         |
| ma.L3  | 0.0123  | 0.007    | 1.885   | 0.059         |
| ma.L4  | -0.0314 | 0.006    | -5.202  | 0.000         |
| sigma2 | 0.0001  | 1.14e-06 | 127.352 | 0.000         |

**Best BIC Model Summary**:

|        | coef    | std err  | z       | $P > \ | z \|$ |
| :----- | :------ | :------- | :------ | :------------ |
| const  | 0.0002  | 0.000    | 1.724   | 0.085         |
| ma.L1  | -0.1051 | 0.006    | -16.204 | 0.000         |
| sigma2 | 0.0001  | 1.11e-06 | 130.871 | 0.000         |

#### Model Diagnostics: Residual Analysis

After fitting a model, we can calculate the residuals:

$$\hat{\varepsilon}_t = X_t - \sum_{i=1}^p \hat{b}_i X_{t-i} - \sum_{i=1}^q \hat{a}_i \hat{\varepsilon}_{t-i} - \hat{c}, \quad t = p + 1, p + 2, \cdots, T,$$

where the initial values $\hat{\varepsilon}_{p+1-i}, i = 1, 2, \cdots, q$ can be set to zero.

Most computer packages revert to AR($\infty$) to calculate the residuals.

Residual plots are useful for checking the model adequacy.
* Plot residuals against time or fitted values.
* Plot the autocorrelation function of the residuals.
* Plot the QQ plot of the residuals.

Also, we can use the Ljung-Box test to test the significance of the autocorrelations of the residuals, with the degrees of freedom $m - p - q$, where $m$ is the lag length.

![[Pasted image 20260123153919.png]]
*Figure: Residual Plots: Best AIC Model (Residuals Over Time, Residuals vs Fitted Values)*

![[Pasted image 20260123153936.png]]
*Figure: Residual ACF Plots: Best AIC Model (ACF of Residuals, ACF of Squared Residuals)*

##### Ljung-Box Test: Best AIC Model

| Lag                         | Statistic | p-value |
| :-------------------------- | :-------- | :------ |
| $\hat{\varepsilon}_t$       |           |         |
| 10                          | 25.46     | 0.0001  |
| 15                          | 54.39     | 0.0000  |
| 20                          | 83.70     | 0.0000  |
| $\ |\hat{\varepsilon}_t\|$   |           |         |
| 10                          | 6638.29   | 0.0000  |
| 15                          | 8873.29   | 0.0000  |
| 20                          | 10656.59  | 0.0000  |
| $\ |\hat{\varepsilon}_t\|^2$ |           |         |
| 10                          | 6010.40   | 0.0000  |
| 15                          | 7573.39   | 0.0000  |
| 20                          | 8653.20   | 0.0000  |

##### Residual QQ Plots: Best AIC Model

**Jarque-Bera Test Results**:
* Statistic: 26788.88
* p-value: 0.0000

![[Pasted image 20260123154147.png]]
*Figure: Q-Q Plot of Residuals (Best AIC Model), Histogram of Residuals (Best AIC Model)*

#### Forecasting with ARMA Models

In general, we are interested in the point prediction of the future values of the series, given that the model parameters have been estimated.

$$X_{T+h} = c + \sum_{i=1}^p b_i X_{T+h-i} + \varepsilon_{T+h} + \sum_{i=1}^q a_i \varepsilon_{T+h-i}.$$

Taking expectation on both sides based on the information up to time $T$, we get:

$$\mathbb{E}_T [X_{T+h}] = c + \sum_{i=1}^p b_i \mathbb{E}_T [X_{T+h-i}] + \sum_{i=1}^q a_i \mathbb{E}_T [\varepsilon_{T+h-i}],$$

where

$$\mathbb{E}_T [\varepsilon_{T+h-i}] = \begin{cases} 0, & i < h, \\ \varepsilon_{T+h-i}, & i \geq h. \end{cases}$$

To calculate the prediction error, we reply the MA($\infty$) representation of the ARMA model:

$$X_t = \mu + \varepsilon_t + \sum_{i=1}^\infty \psi_i \varepsilon_{t-i}.$$

This implies that

$$X_{T+h} = \mu + \varepsilon_{T+h} + \sum_{i=1}^\infty \psi_i \varepsilon_{T+h-i}.$$

Then,

$$\text{Var}_T [X_{T+h}] = \sigma^2 \left( 1 + \sum_{i=1}^{h-1} \psi_i^2 \right).$$

### ARIMA Models

Sometimes, the series is not stationary, but the first difference of the series is stationary. In this case, we can take the first difference of the series and then fit an ARMA model to the differenced series.

The ARIMA($p, d, q$) model is defined as:

$$\nabla^d X_t = c + \sum_{i=1}^p b_i \nabla^d X_{t-i} + \varepsilon_t + \sum_{i=1}^q a_i \varepsilon_{t-i},$$

where $\nabla^d$ is the $d$ -th order difference operator, i.e., $\nabla^d X_t = (\nabla^{d-1} X_t) - (\nabla^{d-1} X_{t-1})$ with $\nabla^0 X_t = X_t$.

The $d$ -th order difference removes the trend of the series in polynomial form $t^d$.

#### Seasonal ARIMA Models

Some series are seasonal, i.e., the series has a periodic pattern. In this case, we can take the seasonal difference of the series and then fit an ARIMA model to the seasonal differenced series.

For example, the monthly sales of iPhones are seasonal, and the seasonal difference is taken to remove the seasonal pattern:

$$Y_t = X_t - X_{t-4} = (1 - B^4) X_t.$$

#### The Earnings per Share (EPS) of Johnson & Johnson

![[Pasted image 20260123154343.png]]
*Figure: Johnson & Johnson Quarterly Earnings (Time series plot)*

We fit a seasonal AR(1) model to the EPS series.

![[Pasted image 20260123154410.png]]
*Figure: The Residuals of the Seasonal AR(1) Model (Residuals vs Time, Residuals vs Fitted Values)*

![[Pasted image 20260123154542.png]]
*Figure: The ACF of the Residuals of the Seasonal AR(1) Model (ACF of Residuals, ACF of Absolute Residuals, ACF of Squared Residuals)*

##### The Ljung-Box Test of the Residuals of the Seasonal AR(1) Model

| Ljung-Box Test Results | Lag 10 | Lag 15 | Lag 20 |
| :--- | :--- | :--- | :--- |
| **Residuals** | | | |
| Statistic | 24.0607 | 29.6714 | 41.4760 |
| p-value | 0.0022 | 0.0053 | 0.0013 |
| **Absolute residuals** | | | |
| Statistic | 54.9417 | 64.7628 | 82.3944 |
| p-value | 0.0000 | 0.0000 | 0.0000 |
| **Squared residuals** | | | |
| Statistic | 44.9642 | 47.1785 | 53.2538 |
| p-value | 0.0000 | 0.0000 | 0.0000 |

##### The QQ Plot of the Residuals of the Seasonal AR(1) Model

**Jarque-Bera Test Results**:
* Statistic: 13.9346
* p-value: 0.0009

![[Pasted image 20260123154602.png]]
*Figure: Q-Q Plot of Standardized Residuals*

##### Prediction of the Seasonal AR(1) Model

![[Pasted image 20260123154700.png]]
*Figure: Johnson & Johnson Quarterly Earnings: Actual vs Predicted*

---

## 4 **Volatility Modeling via GARCH Models**

### Volatility Estimation

**Volatility** means the degree of fluctuation of asset prices over time. It is important in many aspects:
* **Option pricing**: volatility is a key parameter in the Black-Scholes option pricing model.
* **Volatility** is an important risk measure for risk management and evaluating risk-adjusted returns.

However, volatility is not directly observable. We need to estimate it from the observed price series.
* **Implied volatility**: the volatility implied by the option price, e.g., VIX.
* **Realized volatility**: estimated from the observed price series, computed as the sum of squared intraday returns for a particular day.
* **Conditional volatility**: estimated from dynamic models such as the ARCH and GARCH type models.

### ARCH Models

The ARCH model has been introduced by Engle (1982) $^2$:

$\text{ARCH} = \text{AutoRegressive Conditional Heteroskedasticity}.$

Example: an ARCH(1) process.

$$
\begin{cases}
X_t = Z_t \sigma_t, \\
\sigma_t^2 = \alpha_0 + \alpha_1 X_{t-1}^2,
\end{cases}
$$

where $Z_t$ is a sequence of i.i.d. random variables with mean 0 and variance 1, $\alpha_0 \geq 0$ and $0 \leq \alpha_1 < 1$.
* Heteroscedasticity refers to a time-varying conditional variance.
* The ARCH model assumes that the conditional variance is a linear function of the past squared returns.

$^2$**Remark**: Robert F. Engle Nobel Prize 2003; Engle, R.F. (1982), AutoRegressive Conditional Heteroskedasticity with Estimates of the Variance of U.K. Inflation, Econometrica, 50, 987-1008.

#### Conditional Variance

The conditional variance:

$$\sigma_t^2 = \text{Var}_{t-1}[X_t].$$

$\sigma_t^2$ is known based on the information up to time $t-1$.

An equivalent formulation:

$$
\begin{cases}
X_t = Z_t \sqrt{h_t}, \\
h_t = \alpha_0 + \alpha_1 X_{t-1}^2,
\end{cases}
$$

where $h_t$ is the conditional variance.

#### The Trajectory of an ARCH(1) Process

Set $\alpha_0 = 0.1$ and $\alpha_1 = 0.8$ and $Z_t \sim \mathcal{N}(0, 1)$.

![[Pasted image 20260123174808.png]]
*Figure: ARCH(1) Returns, ARCH(1) Conditional Volatility*

#### The Properties of the ARCH Model: AR Representation

We have the following AR representation:

$$X_t^2 = \alpha_0 + \alpha_1 X_{t-1}^2 + v_t,$$

where $v_t = X_t^2 - \sigma_t^2$ satisfying that

$$\mathbb{E}_{t-1}[v_t] = 0.$$

The AR representation shows that the conditional variance of $X_t^2$ is an AR(1) process. Then

$$\mathbb{E}[X_t^2] = \frac{\alpha_0}{1 - \alpha_1}, \quad \text{Cov}[X_t^2, X_{t-k}^2] = \alpha_1^k \text{Var}[X_t^2] \neq 0,$$

which is known as the **ARCH effect** (corresponding to volatility clustering).

The ARCH(1) process is stationary if $\alpha_0 > 0$ and $0 \leq \alpha_1 < 1$.

#### The Properties of the ARCH Model: ACF of $X_t$

$X_t$ is a martingale difference sequence with respect to the information up to $t-1$, i.e.,

$$\mathbb{E}_{t-1}[X_t] = 0.$$

The point prediction of $X_t$ is the same as the ARMA prediction.

The ACF of $X_t$ is given by:

$$\rho_k(X_t) = 0, \quad k \geq 1.$$

This is consistent with the empirical fact that that the returns are not serially correlated.

#### The Properties of the ARCH Model: Kurtosis of $X_t$

The kurtosis of $X_t$ is given by:

$$\text{Kurt}(X_t) = \frac{\mathbb{E}[X_t^4]}{\mathbb{E}[X_t^2]^2} = \frac{\mathbb{E}[Z_t^4]\mathbb{E}[\sigma_t^4]}{\mathbb{E}[Z_t^2]^2\mathbb{E}[\sigma_t^2]^2} \geq \text{Kurt}(Z_t).$$

#### The Properties of the ARCH Model: Fourth Moments

Assume $Z_t \sim \mathcal{N}(0, 1)$.

The conditional fourth moment/kurtosis of $X_t$ is given by:

$$\mathbb{E}[X_t^4 | \mathcal{F}_{t-1}] = 3 \sigma_t^4 \implies \text{Kurt}(X_t | \mathcal{F}_{t-1}) = 3.$$

The unconditional fourth moment of $X_t$ is given by ($\alpha_1^2 < 1/3$):

$$\mathbb{E}[X_t^4] = \frac{3 \alpha_0^2 (1 + \alpha_1)}{(1 - 3 \alpha_1^2)(1 - \alpha_1)} \implies \text{Kurt}(X_t) = \frac{3(1 - \alpha_1^2)}{(1 - 3 \alpha_1^2)} > 3.$$

The unconditional distribution of $X_t$ is **leptokurtic** even if its conditional distribution is normal.

#### ARCH($p$) Process

It is straightforward to extend the ARCH(1) model to the ARCH($p$) model:

$$
\begin{cases}
X_t = Z_t \sigma_t, \\
\sigma_t^2 = \alpha_0 + \sum_{i=1}^p \alpha_i X_{t-i}^2,
\end{cases}
$$

where $\alpha_0 > 0, \alpha_i \geq 0$ for $i = 1, 2, \cdots, p$ and $\sum_{i=1}^p \alpha_i < 1$.

The ARCH($p$) process is stationary if $\alpha_0 > 0$ and $\sum_{i=1}^p \alpha_i < 1$.

### ARMA-ARCH Models

The return series can be modeled as an ARMA process with ARCH errors.

The returns can be decomposed as:

$$r_t = \mu_t + \varepsilon_t,$$

where $\varepsilon_t$ follows an ARCH($p$) process:

$$\varepsilon_t = \sigma_t Z_t,$$

where $Z_t$ is a sequence of i.i.d. random variables with mean 0 and variance 1.

$$
\begin{aligned}
\mu_t &= \mathbb{E}_{t-1}[r_t] = \mu_t(r_{:t-1}; \theta), \\
\sigma_t^2 &= \text{Var}_{t-1}[r_t] = \sigma_t^2(r_{:t-1}; \theta).
\end{aligned}
$$

With an ARMA-ARCH specification, we can specify the conditional mean and variance of the returns.

$$
\begin{aligned}
\mu_t &= \phi_0 + \sum_{i=1}^{p_1} \phi_i r_{t-i} + \sum_{j=1}^{q_1} \theta_j \varepsilon_{t-j}, \\
\sigma_t^2 &= \alpha_0 + \sum_{i=1}^{p_2} \alpha_i \varepsilon_{t-i}^2.
\end{aligned}
$$

#### Estimation of ARMA-ARCH Models

In most of the software, the parameters of the ARMA-ARCH models are estimated by maximizing the log-likelihood function assuming that $Z_t \sim \mathcal{N}(0, 1)$.

After the estimation, we can solve for the conditional variance $\sigma_t^2$ and the residuals $\varepsilon_t$ from the model.

The residuals $\varepsilon_t$ can be used to test the model specification by the **standardized residuals** $\hat{Z}_t = \varepsilon_t / \hat{\sigma}_t$.
* The Ljung-Box Q-statistics of $\hat{Z}_t$ can be used to check the adequacy of the mean equation.
* The Ljung-Box Q-statistics of $\hat{Z}_t^2$ can be used to check the adequacy of the volatility equation.
* The skewness, kurtosis, and QQ-plot of $\hat{Z}_t$ can be used to check the validity of the distribution assumption on $Z_t$.

#### Fitting the AR(0)-ARCH(1) Model to SPX Returns

We fit the AR(0)-ARCH(1) model with constant mean to the SPX returns. In the estimation, the log returns have been scaled by 100 to ensure numerical stability.
* Log-Likelihood: $-9237.52$.
* AIC: $18481.0$.
* BIC: $18501.2$.

##### Mean Model

|       | coef   | std err                | $t$   | $P > \ |t\|$            |
| :---- | :----- | :--------------------- | :---- | :--------------------- |
| Const | 0.0562 | $1.629 \times 10^{-2}$ | 3.453 | $5.554 \times 10^{-4}$ |

##### Volatility Model

|          | coef   | std err                | $t$    | $P > \ |t\|$             |
| :------- | :----- | :--------------------- | :----- | :---------------------- |
| omega    | 0.9220 | $5.041 \times 10^{-2}$ | 18.289 | $1.010 \times 10^{-74}$ |
| alpha[1] | 0.4017 | $5.532 \times 10^{-2}$ | 7.262  | $3.803 \times 10^{-13}$ |

#### The AR(0)-ARCH(1) Model Diagnostics

![[Pasted image 20260123174956.png]]
*Figure: Standardized Residuals, Q-Q Plot, ACF of Standardized Residuals, ACF of Squared Standardized Residuals*

##### Ljung-Box Test Results

| | Lag 10 | Lag 15 | Lag 20 |
| :--- | :--- | :--- | :--- |
| **Residuals** | | | |
| Statistic | 19.09 | 34.08 | 46.59 |
| p-value | 0.0079 | 0.0007 | 0.0001 |
| **Absolute Residuals** | | | |
| Statistic | 1963.42 | 2674.30 | 3327.08 |
| p-value | 0.0000 | 0.0000 | 0.0000 |
| **Squared Residuals** | | | |
| Statistic | 1640.73 | 2133.21 | 2472.98 |
| p-value | 0.0000 | 0.0000 | 0.0000 |

##### Jarque-Bera Test for Normality
* Statistic: 15248.44
* p-value: 0.0000

##### Descriptive Statistics of Standardized Residuals
* Mean: $-0.0287$
* Std Dev: 0.9996
* Skewness: $-0.6004$
* Kurtosis: 7.6960

##### Conditional Volatility of SPX Returns Estimated by the AR(0)-ARCH(1) Model

![[Pasted image 20260123175052.png]]
*Figure: S&P 500 Returns, Conditional Volatility (annualized in percent)*

#### Fitting the AR(0)-ARCH(10) Model to SPX Returns

We fit the AR(0)-ARCH(10) model with constant mean to the SPX returns. In the estimation, the log returns have been scaled by 100 to ensure numerical stability.
* Log-Likelihood: $-8207.74$.
* AIC: $16439.5$.
* BIC: $16519.9$.

##### Mean Model

|       | coef   | std err                | $t$   | $P > \ |t\|$             |
| :---- | :----- | :--------------------- | :---- | :---------------------- |
| Const | 0.0630 | $1.022 \times 10^{-2}$ | 6.161 | $7.249 \times 10^{-10}$ |

##### Volatility Model

|           | coef     | std err                | $t$      | $P > \ |t\|$             |
| :-------- | :------- | :--------------------- | :------- | :---------------------- |
| omega     | 0.2090   | $2.251 \times 10^{-2}$ | 9.283    | $1.650 \times 10^{-20}$ |
| alpha[1]  | 0.0749   | $2.134 \times 10^{-2}$ | 3.510    | $4.489 \times 10^{-4}$  |
| alpha[2]  | 0.1381   | $2.026 \times 10^{-2}$ | 6.817    | $9.305 \times 10^{-12}$ |
| alpha[3]  | 0.1145   | $1.936 \times 10^{-2}$ | 5.912    | $3.385 \times 10^{-9}$  |
| alpha[4]  | 0.1254   | $2.561 \times 10^{-2}$ | 4.896    | $9.770 \times 10^{-7}$  |
| alpha[5]  | 0.0786   | $1.692 \times 10^{-2}$ | 4.646    | $3.382 \times 10^{-6}$  |
| $\cdots$  | $\cdots$ | $\cdots$               | $\cdots$ | $\cdots$                |
| alpha[10] | 0.0493   | $1.651 \times 10^{-2}$ | 2.985    | $2.837 \times 10^{-3}$  |

#### The AR(0)-ARCH(10) Model Diagnostics

![[Pasted image 20260123175245.png]]
*Figure: Standardized Residuals, Q-Q Plot, ACF of Standardized Residuals, ACF of Squared Standardized Residuals*

##### Ljung-Box Test Results

| | Lag 15 | Lag 20 | Lag 30 |
| :--- | :--- | :--- | :--- |
| **Residuals** | | | |
| Statistic | 27.36 | 31.32 | 47.24 |
| p-value | 0.0000 | 0.0001 | 0.0002 |
| **Absolute Residuals** | | | |
| Statistic | 16.17 | 25.85 | 43.89 |
| p-value | 0.0010 | 0.0011 | 0.0006 |
| **Squared Residuals** | | | |
| Statistic | 9.78 | 11.29 | 19.16 |
| p-value | 0.0206 | 0.1858 | 0.3822 |

##### Jarque-Bera Test for Normality
* Statistic: 948.15
* p-value: 0.0000

##### Descriptive Statistics of Standardized Residuals
* Mean: $-0.0447$
* Std Dev: 0.9990
* Skewness: $-0.5158$
* Kurtosis: 1.6457

##### Conditional Volatility of SPX Returns Estimated by the AR(0)-ARCH(10) Model

![[Pasted image 20260123175304.png]]
*Figure: S&P 500 Returns, Conditional Volatility (annualized in percent)*

### GARCH Models

Due to the large persistence in volatility, ARCH models often require a large $p$ to fit the data. A more parsimonious specification is provided by GARCH models $^3$.

$$\text{GARCH} = \text{Generalized AutoRegressive Conditional Heteroskedasticity}.$$

The asset return $r_t$ can be decomposed as:

$$
\begin{cases}
r_t = \mu_t + \varepsilon_t, \\
\varepsilon_t = \sigma_t Z_t,
\end{cases}
$$

where $Z_t$ is a sequence of i.i.d. random variables with mean zero and variance 1, and,

$$
\begin{aligned}
\mu_t &= \mathbb{E}[r_t | \mathcal{F}_{t-1}], \\
\sigma_t^2 &= \text{Var}[r_t | \mathcal{F}_{t-1}].
\end{aligned}
$$

The GARCH($p, q$) process is given by:

$$\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2,$$

where $\omega > 0, \alpha_i \geq 0, \beta_j \geq 0,$ and $\sum_{i=1}^p \alpha_i + \sum_{j=1}^q \beta_j < 1$.

The parameters $\alpha_i$ are called **ARCH parameters** and $\beta_j$ **GARCH parameters**.

From a practical point, GARCH(1, 1) specifications are generally sufficient to capture the dynamics of the conditional variance and higher-order lags are not required.

The GARCH(1, 1) process is given by:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2.$$

The overall impact of a shock can be decomposed into a **contemporaneous effect**, which depends on $\alpha$, and a **persistence effect**, which depends on $\beta$.

One often observes that:
* The sum of the estimates of $\alpha$ and $\beta$ are generally close (but below 1).
* The estimate of $\beta$ is generally greater than the one of $\alpha$.
* The estimate of $\beta$ is generally larger than 0.90 for daily returns and the estimate of $\alpha$ is below 0.1.

$^3$**Remark**: Bollerslev, T. (1986), Generalized Autoregressive Conditional Heteroskedasticity. Journal of Econometrics, 31, 307-327.

#### Properties of GARCH Models

The main properties of a GARCH process are similar to those of an ARCH process.
* The process $\varepsilon_t^2$ has an ARMA representation.
* The process $\varepsilon_t$ is a martingale difference.
* The process $\varepsilon_t$ is a stationary process under some conditions on the parameters $\alpha$ and $\beta$.
* The process $\varepsilon_t$ is (unconditionally) homoscedastic.
* The process $\varepsilon_t$ is conditionally heteroscedastic.
* The (marginal) distributions of $r_t$ and $Z_t$ are leptokurtic.
* If $Z_t$ has a normal distribution, the conditional distributions of $r_t$ is normal.

#### ARMA Representation of GARCH Models

It is not difficult to show that:

$$\varepsilon_t^2 = \omega + \sum_{i=1}^{\max(p, q)} (\alpha_i + \beta_i) \varepsilon_{t-i}^2 + v_t - \sum_{j=1}^q \beta_j v_{t-j},$$

where $v_t = \varepsilon_t^2 - \sigma_t^2$ satisfies that $\mathbb{E}_{t-1}[v_t] = 0$.

#### Unconditional Moments of GARCH Models

The unconditional mean and variance of $\varepsilon_t$ is given by:

$$
\begin{aligned}
\mathbb{E}[\varepsilon_t] &= 0, \\
\text{Var}[\varepsilon_t] &= \frac{\omega}{1 - \sum_{i=1}^p \alpha_i - \sum_{j=1}^q \beta_j}.
\end{aligned}
$$

Therefore, the variance equation can be written as:

$$\sigma_t^2 = \text{Var}[\varepsilon_t] + \sum_{i=1}^p \alpha_i (\varepsilon_{t-i}^2 - \text{Var}[\varepsilon_t]) + \sum_{j=1}^q \beta_j (\sigma_{t-j}^2 - \text{Var}[\varepsilon_t]).$$

#### Kurtosis of GARCH Models

Assuming that $Z_t$ is a standard normal random variable, Bollerslev (1986) shows that the kurtosis coefficient of a GARCH(1,1) process is equal to:

$$\text{Kurt}(\varepsilon_t) = \frac{3(1 - (\alpha + \beta)^2)}{1 - (\alpha + \beta)^2 - 2 \alpha^2},$$

when $(\alpha + \beta)^2 + 2 \alpha^2 < 1$.

The kurtosis coefficient is infinite if $(\alpha + \beta)^2 + 2 \alpha^2 \geq 1$ which is the case for many practical applications.

#### Estimation of GARCH Models

The set of parameters $\theta$ of an ARMA-GARCH model is estimated by **Maximum Likelihood (ML)** or **Quasi Maximum Likelihood (QML)**. When the model is estimated by ML, the most often used distributions for $Z_t$ are:
1. The normal distribution, $Z_t \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$.
2. The Student t-distribution, $Z_t \overset{\text{i.i.d.}}{\sim} t(\nu)$, which is symmetric and leptokurtic (if $\nu$ is "small").
3. The skewed Student t-distribution, $Z_t \overset{\text{i.i.d.}}{\sim} \text{Skewed-}t(\delta, \nu)$, which is asymmetric (if $\delta \neq 1$) and leptokurtic (if $\nu$ is "small").
4. The Generalized Error Distribution (GED), $Z_t \overset{\text{i.i.d.}}{\sim} \text{GED}(\nu)$, which is symmetric and leptokurtic (if $\nu < 2$).

Suppose the density of $Z_t$ is known as $f(\cdot)$. Then, the log-likelihood function of the GARCH model is given by:

$$\ell(\theta) = \sum_{t=\nu+1}^T \log \left( \frac{1}{\sigma_t} f \left( \frac{\varepsilon_t}{\sigma_t} \right) \right) = \sum_{t=\nu+1}^T \left( -\log \sigma_t + \log f \left( \frac{\varepsilon_t}{\sigma_t} \right) \right),$$

where $\frac{\varepsilon_t}{\sigma_t}$ is the standardized innovation. When $Z_t \sim \mathcal{N}(0, 1)$, the log-likelihood function is given by:

$$
\begin{aligned}
\ell(\theta) &= \sum_{t=\nu+1}^T \left( -\log \sigma_t + \log f \left( \frac{\varepsilon_t}{\sigma_t} \right) \right) \\
&= \sum_{t=\nu+1}^T \left( -\log \sigma_t - \frac{\varepsilon_t^2}{2 \sigma_t^2} + \log \frac{1}{\sqrt{2\pi}} \right).
\end{aligned}
$$

**Why to consider non-normal distributions for $Z_t$?**

Why consider non-Gaussian distributions for the innovation $Z_t$?
1. The use of a leptokurtic distribution for $Z_t$ allows to increase the kurtosis of $r_t$:
	$$\text{kurtosis of a GARCH process} = \text{kurtosis from model dynamics} + \text{kurtosis of } Z_t$$
	To reproduce the high kurtosis observed in financial returns, the kurtosis generated by the model dynamics alone is not sufficient. This motivates the use of leptokurtic distributions for $Z_t$ like Student-$t$ or GED.
2. The use of a skewed distribution for $Z_t$ allows to reproduce the skewness observed in financial returns:
	$$\text{skewed distribution for } Z_t \implies \text{skewed distribution for } r_t$$

#### Fitting GARCH Models to SPX Returns

| | Gaussian | Student-t | Skewed Student-t | GED |
| :--- | :--- | :--- | :--- | :--- |
| Log-Likelihood | $-8202.17$ | $-8068.88$ | $-8047.69$ | $-8061.94$ |
| AIC | 16412.3 | 16147.8 | 16107.4 | 16133.9 |
| BIC | 16439.2 | 16181.3 | 16147.6 | 16167.4 |
| **Mean Model** | | | | |
| Const | 0.0629 | 0.0796 | 0.0602 | 0.0751 |
| t-stat | 6.088 | 8.605 | 6.066 | 9.392 |
| p-value | $1.144 \times 10^{-9}$ | $7.636 \times 10^{-18}$ | $1.308 \times 10^{-9}$ | $5.869 \times 10^{-21}$ |
| **Volatility Model** | | | | |
| $\omega$ | 0.0240 | 0.0148 | 0.0143 | 0.0183 |
| t-stat | 4.953 | 4.699 | 4.774 | 5.047 |
| p-value | $7.317 \times 10^{-7}$ | $2.620 \times 10^{-6}$ | $1.806 \times 10^{-6}$ | $4.479 \times 10^{-7}$ |
| $\alpha_1$ | 0.1197 | 0.1225 | 0.1191 | 0.1215 |
| t-stat | 9.794 | 10.992 | 11.307 | 10.819 |
| p-value | $1.193 \times 10^{-22}$ | $4.169 \times 10^{-28}$ | $1.216 \times 10^{-29}$ | $2.795 \times 10^{-27}$ |
| $\beta_1$ | 0.8616 | 0.8731 | 0.8742 | 0.8677 |
| t-stat | 66.455 | 81.728 | 83.809 | 76.779 |
| p-value | 0.000 | 0.000 | 0.000 | 0.000 |

![[Pasted image 20260123175517.png]]
*Figure: Fitting GARCH Models to SPX Returns (continued) (GARCH(1,1) Models with Different Distributions (AR(0)) plots: Gaussian, Student-t, Skewed Student-t, GED)*

#### The Gaussian AR(0)-GARCH(1,1) Model Diagnostics

![[Pasted image 20260123175543.png]]
*Figure: Standardized Residuals, Q-Q Plot, ACF of Standardized Residuals, ACF of Squared Standardized Residuals*

##### The Ljung-Box Test for the Gaussian AR(0)-GARCH(1,1) Model Residuals

| | Lag 10 | Lag 20 | Lag 30 |
| :--- | :--- | :--- | :--- |
| **Residuals** | | | |
| Statistic | 18.78 | 31.12 | 46.91 |
| p-value | 0.0046 | 0.0130 | 0.0072 |
| **Absolute Residuals** | | | |
| Statistic | 34.93 | 40.79 | 52.95 |
| p-value | 0.0000 | 0.0006 | 0.0014 |
| **Squared Residuals** | | | |
| Statistic | 14.96 | 18.74 | 29.46 |
| p-value | 0.0206 | 0.2824 | 0.2906 |

##### The Jarque-Bera Test for the Gaussian AR(0)-GARCH(1,1) Model Residuals

**Jarque-Bera Test for Normality**
* Statistic 999.30
* P-value 0.0000

**Descriptive Statistics of Standardized Residuals**
* Mean: $-0.0446$
* Std Dev: 0.9989
* Skewness: $-0.5327$
* Kurtosis: 1.6854

**Fitting GARCH Models to SPX Returns (continued)**

![[Pasted image 20260123175640.png]]
*Figure: SPX Returns, Conditional Volatility (Gaussian AR(0)-GARCH(1,1))*

---

## 5 **Applications and Extensions of GARCH Models**

### Volatility Forecasting

Volatility forecasting is an important application of GARCH models. We note that:

$$\varepsilon_{t+h}^2 = \omega + \sum_{i=1}^{\max(p, q)} \alpha_i \varepsilon_{t+h-i}^2 + v_{t+h} + \sum_{j=1}^q \beta_j v_{t+h-j},$$

where $v_t = \varepsilon_t^2 - \sigma_t^2$. Then,

$$\sigma_{t+h}^2 = \mathbb{E}_t [\varepsilon_{t+h}^2] = \omega + \sum_{i=1}^{\max(p, q)} \alpha_i \mathbb{E}_t [\varepsilon_{t+h-i}^2] + \sum_{j=1}^q \beta_j \mathbb{E}_t [v_{t+h-j}],$$

where

$$\mathbb{E}_t [\varepsilon_{t+h-i}^2] = \begin{cases} \sigma_{t+h-i}^2, & \text{if } i < h, \\ \varepsilon_{t+h-i}^2, & \text{if } i \geq h. \end{cases}, \quad \mathbb{E}_t [v_{t+h-j}] = \begin{cases} 0, & \text{if } j < h, \\ v_{t+h-j}, & \text{if } j \geq h. \end{cases}$$

### Forecasting Value-at-Risk (VaR) with GARCH Models

Suppose you are holding SPX, the VaR can be computed as:

$$\text{VaR}_\alpha = \sigma_{t+1} \times F_Z^{-1}(\alpha),$$

where $F_Z^{-1}(\alpha)$ can be estimated either by the inverse of the cumulative distribution function of the assumed distribution for $Z_t$ or by empirical quantiles.

![[Pasted image 20260123175718.png]]
*Figure: SPX Returns and Value at Risk Estimates (Returns, 5% Normal VaR, 1% Normal VaR, 5% Historical VaR, 1% Historical VaR)*

### GARCH Extensions

Some relevant extensions of the GARCH model have been proposed in order to accommodate particular features of financial series (asymmetry, leverage effect, etc.).

Among others, GARCH models have been refined by introducing:
1. **Asymmetric responses** to negative and positive innovations to handle the observed asymmetry in the reaction of conditional volatility to the arrivals of news.
2. **Persistence**.
3. **Long-memory** (the dependency for a large number of lags).

The following GARCH models are often encountered in the empirical financial literature as well as in the industry:
1. **Asymmetric GARCH models**: Exponential GARCH model (EGARCH), Threshold GARCH model (TGARCH), GJR model;
2. **Integrated GARCH model** (IGARCH).
3. **Long-memory GARCH model** (LMGARCH) or Fractionally integrated GARCH model (FIGARCH).

**GARCH Extensions: IGARCH**

> [!info] + Integrated GARCH (IGARCH)
> Integrated GARCH (IGARCH) $^4$ models are a special case of GARCH models where the sum of the ARCH and GARCH parameters is equal to 1.

Example: IGARCH(1, 1):

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + (1 - \alpha) \sigma_{t-1}^2,$$

The IGARCH model has several important properties:
1. The impact of past squared shocks $\varepsilon_{t-k}^2 - \sigma_{t-k}^2$ in the ARMA representation is persistent.
2. The unconditional variance of $\varepsilon_t$ is not defined under an IGARCH(1, 1) model.
3. A special case of the IGARCH(1, 1) is the **RiskMetrics** volatility model defined as: $\sigma_t^2 = \lambda \varepsilon_{t-1}^2 + (1 - \lambda) \sigma_{t-1}^2$, which is commonly used to compute Value-at-Risk.

$^4$**Remark**: Nelson, D. B. (1990), Stationarity and persistence in the GARCH(1, 1) model, Econometric Theory, 6, 318-334.

#### GARCH Extensions: GARCH-M

The return of a security may depend on its volatility. To model such a phenomenon, one may consider the GARCH-M$^5$ model, where M stands for GARCH in mean.

$$
\begin{aligned}
r_t &= c + \delta \sigma_t^2 + \varepsilon_t \quad & (\text{Conditional variance effect}) \\
r_t &= c + \delta \sigma_t + \varepsilon_t \quad & (\text{Conditional volatility effect}) \\
r_t &= c + \delta \ln \sigma_t + \varepsilon_t \quad & (\text{Log-linear specification})
\end{aligned}
$$

The parameter $\delta$ is called the **risk premium parameter**. A positive $\delta$ indicates that the return is positively related to its past volatility.

$^5$**Remark**: Engle, R., Lilien, D., and R. Robins (1987). Estimating Time Varying Risk Premia in the Term Structure: The ARCH-M Model. Econometrica, 55(2), 391-407.

#### GARCH Extensions: Asymmetric GARCH Models

Asymmetric GARCH models:
* The GARCH model assumes that positive and negative shocks have the same effects on volatility because it depends on the square of the previous shocks.
* In practice, the return of a financial asset responds differently to positive and negative shocks.
* The GARCH model does not allow to capture the leverage effect.

As asset prices decline, companies become more leveraged (debt to equity ratios increase) and riskier, and hence their stock prices become more volatile.

On the other hand, when stock prices become more volatile, investors demand high returns and hence stock prices go down.

The asymmetric GARCH models are designed to capture the nonlinearities of the conditional variance dynamics, including the leverage effect.

Many asymmetric GARCH models have been proposed: GJR-GARCH, TGARCH, EGARCH, APARCH, VSGARCH, QGARCH, LSTGARCH, ANSTGARCH, etc.

One of the most often used asymmetric models is the GJR-GARCH$^6$ model.

$^6$**Remark**: Glosten, L., Jagannathan, R., and D. Runkle, D. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. The Journal of Finance, 48(5), 1779-1801.

The GJR-GARCH model is given by:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \gamma \varepsilon_{t-1}^2 \cdot I(\varepsilon_{t-1} < 0) + \beta \sigma_{t-1}^2.$$

The impact of past shocks on the conditional variance is:

$$\frac{\partial \sigma_t^2}{\partial \varepsilon_{t-1}^2} = \begin{cases} \alpha + \gamma, & \text{if } \varepsilon_{t-1} < 0, \\ \alpha, & \text{otherwise}. \end{cases}$$

A leverage effect implies that $\gamma > 0$, i.e., the increase in volatility caused by a negative shock is larger than the appreciation due a positive shock of the same magnitude.

#### GARCH Extensions: TGARCH

The TGARCH$^7$ model is given by:

$$\sigma_t = \omega + \alpha_+ \varepsilon_{t-1} I(\varepsilon_{t-1} \geq 0) + \alpha_- \varepsilon_{t-1} I(\varepsilon_{t-1} < 0) + \beta \sigma_{t-1},$$

where $I(\cdot)$ is the indicator function.

The TGARCH allows to capture an asymmetry between positive and negative shocks, as the leverage effect implies that $|\alpha_-| > |\alpha_+|$, i.e. the increase in volatility caused by a negative return is larger than the appreciation due a positive return of the same magnitude.

$^7$**Remark**: Zakoian J.M. (1994), Threshold Heteroskedastic Models. Journal of Economic Dynamic and Control, 18, 931-955, 1994.

#### GARCH Extensions: EGARCH

The EGARCH, where "E" stands for Exponential, is an asymmetric GARCH model.

The EGARCH is designed to capture both (1) the asymmetric effects between positive and negative shocks on the returns and (2) the effects of big shocks.

The EGARCH$^8$ model is given by:

$$\ln \sigma_t^2 = \omega + \alpha Z_{t-1} + \gamma (|Z_{t-1}| - \mathbb{E}[|Z_{t-1}|]) + \beta \ln \sigma_{t-1}^2,$$

where $Z_{t-1} = \frac{\varepsilon_{t-1}}{\sigma_{t-1}}$.

The EGARCH model does not require any restriction on the parameters because, since the equation is on log variance instead of variance itself, the positivity of the variance is automatically satisfied $\forall (\omega, \alpha, \gamma, \beta) \in \mathbb{R}^4$:

$$\sigma_t^2 = \exp(\omega + \alpha Z_{t-1} + \gamma (|Z_{t-1}| - \mathbb{E}[|Z_{t-1}|]) + \beta \ln \sigma_{t-1}^2) > 0$$

The EGARCH model captures the asymmetric effects between positive and negative shocks on the returns, since:

$$\frac{\partial \ln \sigma_t^2}{\partial |Z_{t-1}|} = \begin{cases} \gamma - \alpha, & \text{if } Z_{t-1} < 0, \\ \gamma + \alpha, & \text{otherwise}. \end{cases}$$

The leverage effect, i.e., the fact that negative shocks at time $t-1$ have a stronger impact on the variance at time $t$ than positive shocks, implies that $\alpha < 0$.

The term $(|Z_{t-1}| - \mathbb{E}[|Z_{t-1}|])$ measures the magnitude of the (positive or negative) shocks.

If the parameter $\gamma$ is positive, then the **big** shocks (compared to their expected value) have a stronger impact on the variance than the **small** shocks.

The mean $\mathbb{E}[|Z_{t-1}|]$ is a constant that depends on the distribution of $Z_t$. For the Gaussian distribution, we have:

$$\mathbb{E}[|Z_t|] = \sqrt{\frac{2}{\pi}}$$

$^8$**Remark**: Nelson, D. B. (1991), Conditional heteroskedasticity in asset returns: A new approach, Econometrica, 59, 347-370.

### Fitting SPX Returns with the GJR-GARCH Model

| Parameter | Coefficient | Std Error | t-stat | p-value |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Model** | | | | |
| Const | 0.0250 | 0.0106 | 2.362 | 0.0182 |
| **Volatility Model** | | | | |
| $\omega$ | 0.0210 | 0.0084 | 2.508 | 0.0122 |
| $\alpha_1$ | 0.0000 | 0.0190 | 0.000 | 1.0000 |
| $\gamma_1$ | 0.1647 | 0.0342 | 4.811 | $1.503 \times 10^{-6}$ |
| $\beta_1$ | 0.8961 | 0.0362 | 24.725 | $5.722 \times 10^{-135}$ |

### Fitting SPX Returns with the EGARCH Model

| Parameter | Coefficient | Std Error | t-stat | p-value |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Model** | | | | |
| Const | 0.0253 | $9.972 \times 10^{-3}$ | 2.534 | $1.127 \times 10^{-2}$ |
| **Volatility Model** | | | | |
| $\omega$ | $1.382 \times 10^{-3}$ | $3.120 \times 10^{-3}$ | 0.443 | 0.658 |
| $\alpha_1$ | 0.1555 | $1.904 \times 10^{-2}$ | 8.170 | $3.084 \times 10^{-16}$ |
| $\gamma_1$ | $-0.1370$ | $1.621 \times 10^{-2}$ | $-8.450$ | $2.917 \times 10^{-17}$ |
| $\beta_1$ | 0.9724 | $4.577 \times 10^{-3}$ | 212.425 | 0.000 |

### Performance of GARCH Models

| Model | AIC | BIC | Log-Likelihood |
| :--- | :--- | :--- | :--- |
| EGARCH | 16187.68 | 16221.20 | $-8088.84$ |
| GJR-GARCH | 16207.67 | 16241.20 | $-8098.84$ |
| GARCH | 16412.35 | 16439.17 | $-8202.17$ |

### Conditional Volatility of SPX Returns from the GJR-GARCH and EGARCH Models

![[Pasted image 20260123180445.png]]
*Figure: GJR-GARCH and EGARCH Conditional Volatility (Annualized Volatility (%) plots comparison)*

### GARCH Models for Options Pricing

GARCH models have been extensively used to perform option pricing due to its ease of implementation and the flexibility of the model to capture the volatility dynamics of the underlying asset.

Compared to one-period models, GARCH models are able to time-varying dynamics of the volatility and higher moments, which have potential to fit the term structure of implied volatility.

The model construction follows the following steps:
1. Specify the GARCH model under the **physical probability measure**.
2. Specify the Radon-Nikodym derivative of the **risk-neutral probability measure** with respect to the physical probability measure.
3. Derive the GARCH model under the risk-neutral probability measure.

Compared to continuous time models, the fitting results are easier to interpret.

#### The Physical Probability Measure Dynamics

Following the standard GARCH model specified previously, we have:

$$
\begin{cases}
r_t = \ln(S_t / S_{t-1}) = r + \lambda \sqrt{h_t} - \frac{1}{2} h_t + \sqrt{h_t} Z_t, \\
h_t = \omega + \alpha h_{t-1} Z_{t-1}^2 + \beta h_{t-1},
\end{cases}
$$

where $Z_t \sim \text{i.i.d.} \mathcal{N}(0, 1)$ under $\mathbb{P}$.

This fits into the GARCH-M framework, under which we have that

$$r_t = c + \sqrt{h_t} Z_t,$$

where $c = r + \lambda \sqrt{h_t} - \frac{1}{2} h_t$.

#### The Risk-Neutral Probability Measure Dynamics

There remains the question of how to specify the Radon-Nikodym derivative of the risk-neutral probability measure with respect to the physical probability measure.

We can follow the locally risk-neutral valuation relationship (LRNVR) approach, which is based on the following relationship:

$$\mathbb{E}_{t-1}^{\mathbb{Q}}[e^{r_t}] = e^r, \quad \text{Var}_{t-1}^{\mathbb{Q}}(r_t) = \text{Var}_{t-1}^{\mathbb{P}}(r_t) = h_t.$$

Under the LRNVR, the asset return dynamics is given by:

$$
\begin{cases}
r_t = r - \frac{1}{2} h_t + \sqrt{h_t} Z_t^*, \\
h_t = \omega + \alpha h_{t-1} (Z_{t-1}^* - \lambda)^2 + \beta h_{t-1},
\end{cases}
$$

where $Z_t^* = Z_t + \lambda \sim \text{i.i.d.} \mathcal{N}(0, 1)$ under $\mathbb{Q}$.

#### To Consider Conditional Leverage Effect

We note that the change-of-measure result still holds if we vary the dynamics of $h_t$.

In general, its dynamics can be written as:

$$h_t = \omega + \alpha h_{t-1} f(Z_{t-1}) + \beta h_{t-1},$$

where $f(Z_{t-1})$ is a function of $Z_{t-1}$, which may induce asymmetric effect of innovation on the volatility and hence conditional leverage effect.

For example, we can consider the following functions:
* **Leverage**: $f(Z_{t-1}) = (Z_{t-1} - \theta)^2$.
* **News**: $f(Z_{t-1}) = (|Z_{t-1} - \theta| - \kappa(Z_{t-1} - \theta))^2$.
* **Power**: $f(Z_{t-1}) = (Z_{t-1} - \theta)^\gamma$.

#### To Fit The Parameters

We can fit the parameters of the GARCH model under the physical probability measure using the historical data by maximum likelihood estimation.

The log-likelihood function is given by:

$$\ell(\theta) = -(T-1) \ln(2\pi) - \frac{1}{2} \sum_{t=2}^T \ln(h_t) - \frac{1}{2} \sum_{t=1}^T \frac{(r_t - r - \lambda \sqrt{h_t} + \frac{1}{2} h_t)^2}{h_t}.$$

The estimated parameters can then be used to price options. In most cases, analytical pricing formula is not available, so we can use Monte Carlo simulation to obtain approximate option prices.

In the LRNVR framework, the option price is dependent on the risk-premium $\lambda$, which can be estimated from the option data.

### The Heston-Nandi Model

The Heston-Nandi model$^9$ is an affine GARCH model that allows for tractable pricing of options.

$$
\begin{cases}
r_t = r + \lambda \sqrt{h_t} - \frac{1}{2} h_t + \sqrt{h_t} Z_t, \\
h_t = \omega + \alpha (Z_{t-1} - \theta \sqrt{h_{t-1}})^2 + \beta h_{t-1},
\end{cases}
$$

where $Z_t \sim \text{i.i.d.} \mathcal{N}(0, 1)$ under $\mathbb{P}$.

Under this model and the LRNVR, the log price admits closed-form characteristic function.

$^9$**Remark**: Heston, S., Nandi, S., 2000. A closed-form GARCH option valuation model. Review of Financial Studies 13, 585-625.