# Univariate Data Analysis

**Gongqiu Zhang**
SSE, CUHK-Shenzhen

---

## Part 1: Introduction to Univariate Data Analysis

### Univariate Data Analysis
Sometimes, we are **not interested** in the relationship between variables, but only in the **distribution** of a single variable or even some of its **distributional characteristics**.

**Examples:**
- The distribution of the daily returns of a **stock/index**.
- The distribution of the daily returns of a **trading strategy**.

In such cases, we often regard the daily returns as **independent and identically distributed (i.i.d.)** samples from the same distribution, or say the daily returns are **realizations of a random variable**.

The goal of univariate data analysis is to understand the distribution of the random variable.

### Simple and Log Returns
**Log return/continuously compounded return:**
$$r_t = \log(S_t/S_{t-1})$$
where $S_t$ is the price (adjusted for dividends) at day $t$.

**Simple return:**
$$R_t = S_t/S_{t-1} - 1 \approx r_t, \text{ for small } r_t$$

**We use log return more often because:**
- Log return is **additive** in the sense that the log return over a period is the sum of the log returns over the subperiods.
- Simple returns are distributed in $[-1, \infty)$, while log returns are distributed in $(-\infty, \infty)$.

### k-Period Returns
**One-period gross return:**
$$G_t := \frac{S_t}{S_{t-1}}$$

**k-period simple return:**
$$R_t(k) := \frac{S_t}{S_{t-k}} - 1 = \prod_{i=0}^{k-1}(1 + R_{t-i}) - 1$$

**k-period log return:**
$$r_t(k) := \ln\left(\frac{S_t}{S_{t-k}}\right) = \sum_{i=0}^{k-1} r_{t-i}$$

### The Daily Returns of SPX
The daily returns of S&P 500 index from 2000-12-18 to 2024-12-05. Data source: Yahoo Finance.

![[Pasted image 20260113151019.png]]

*Figure: The adjusted closing price and the log daily returns of S&P 500 index*

### Distributional Properties of SPX Returns
Leptokurtic and heavy-tailed.

![[Pasted image 20260113151448.png]]

*Figure: Distribution of S&P 500 Daily Log Returns*

### Some Key Statistics of SPX Returns

| Statistic | Value |
| :--- | :--- |
| Mean | 0.0003 |
| Median | 0.0007 |
| Standard Deviation | 0.0122 |
| Skewness | −0.4077 |
| Kurtosis | 13.9822 |
| Minimum | −0.1277 |
| Maximum | 0.1096 |

*Table: Key Statistics of S&P 500 Daily Log Returns.*

### Skewness and Kurtosis
The statistics above reveal important properties of the return distribution.

**Skewness** measures the asymmetry of the distribution:
- Normal distribution has skewness of 0.
- Negative skewness (−0.4077) indicates the distribution is left-skewed.
- More extreme negative returns than positive returns.

**Kurtosis** measures the heaviness of tails:
- Normal distribution has kurtosis of 3.
- Much higher kurtosis (13.9822) indicates heavy tails.
- Heavy tails mean extreme returns occur more frequently than predicted by normal distribution.

---

## Part 2: Random Variables and Distributions

### Random Variables
A **random variable** is a mapping from the sample space to the real numbers.
$$X : \Omega \to \mathbb{R}$$
- $\Omega$ is the **sample space** containing all possible outcomes of an experiment.
- $X(\omega)$ is the value of the random variable $X$ at the outcome $\omega$.

**Example:**
- $\Omega = \{HH, HT, TH, TT\}$: the sample space of the coin toss experiment.
- $X(\omega)$ = number of heads corresponding to $\omega \in \{HH, HT, TH, TT\}$.

### Discrete Random Variables
**Probability mass function (pmf):**
$$p_X(x) = \mathbb{P}(X = x) = \mathbb{P}(\{\omega \in \Omega : X(\omega) = x\})$$

**Cumulative distribution function (cdf):**
$$F_X(x) = \mathbb{P}(X \le x) = \sum_{x_i \le x} p_X(x_i)$$

**Expectation of a discrete random variable:**
$$\mathbb{E}[X] = \sum_{x} x p_X(x)$$

### Continuous Random Variables
**Cumulative distribution function (cdf):**
$$F_X(x) = \int_{-\infty}^{x} f_X(t)dt$$

**Probability density function (pdf):**
$$f_X(x) = \frac{d}{dx} F_X(x)$$

**Expectation** of a continuous random variable:
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x)dx$$

### Moments of a Random Variable
The **$k$-th moment** of a random variable $X$ is defined as:
$$\mathbb{E}[X^k] = \int_{-\infty}^{\infty} x^k f_X(x)dx$$

The **$k$-th central moment** of a random variable $X$ is defined as:
$$\mathbb{E}[(X - \mathbb{E}[X])^k] = \int_{-\infty}^{\infty} (x - \mathbb{E}[X])^k f_X(x)dx$$

The **moment generating function (mgf)** of a random variable $X$ is defined as:
$$M_X(t) = \mathbb{E}[e^{tX}]$$
We have that $\mathbb{E}[X^k] = M_X^{(k)}(0)$ (the $k$-th derivative of $M_X(t)$ at $t = 0$).

### Skewness and Kurtosis of a Random Variable
Skewness and kurtosis are the third and fourth central normalized moments of a random variable, respectively.

The **skewness** of a random variable $X$ is defined as:
$$\text{Skewness}(X) = \frac{\mathbb{E}[(X - \mathbb{E}[X])^3]}{(\mathbb{E}[(X - \mathbb{E}[X])^2])^{3/2}}$$

The **kurtosis** of a random variable $X$ is defined as:
$$\text{Kurtosis}(X) = \frac{\mathbb{E}[(X - \mathbb{E}[X])^4]}{(\mathbb{E}[(X - \mathbb{E}[X])^2])^2}$$

The **excess kurtosis** of a random variable $X$ is defined as:
$$\text{Excess Kurtosis}(X) = \text{Kurtosis}(X) - 3$$

### Summary Table of Distributions

#### Uniform Distribution
| Property | Formula/Value |
| :--- | :--- |
| Parameters | $a < b, a, b \in \mathbb{R}$ (lower and upper bounds) |
| PDF | $f_X(x) = \frac{1}{b-a}$ for $x \in [a, b]$, 0 otherwise |
| MGF | $\frac{e^{bt}-e^{at}}{t(b-a)}$ for $t \neq 0$ and $1$ for $t = 0$ |
| Mean | $\mathbb{E}[X] = \frac{a+b}{2}$ |
| Variance | $\text{Var}(X) = \frac{(b-a)^2}{12}$ |
| Skewness | 0 |
| Kurtosis | 9/5 |

#### Normal Distribution
| Property   | Formula/Value                                                                             |
| :--------- | :---------------------------------------------------------------------------------------- |
| Parameters | $\mu \in \mathbb{R}$ (mean), $\sigma^2 > 0$ (variance)                                    |
| PDF        | $f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(x-\mu)^2}{2\sigma^2} \right)$ |
| MGF        | $M_X(t) = \exp(\mu t + \frac{\sigma^2 t^2}{2})$                                           |
| Mean       | $\mathbb{E}[X] = \mu$                                                                     |
| Variance   | $\text{Var}(X) = \sigma^2$                                                                |
| Skewness   | 0                                                                                         |
| Kurtosis   | 3                                                                                         |

#### Student's $t$ Distribution

| Property | Formula/Value |
| :--- | :--- |
| Parameters | $\nu > 0$ (degrees of freedom), $\mu \in \mathbb{R}$ (location), $\sigma^2 > 0$ (scale) |
| PDF | $f_X(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\nu\pi}\sigma} \left( 1 + \frac{(x-\mu)^2}{\nu\sigma^2} \right)^{-\frac{\nu+1}{2}}$ |
| MGF | Does not exist for $t \neq 0$ |
| Mean | $\mathbb{E}[X] = \mu$ for $\nu > 1$, otherwise undefined |
| Variance | $\text{Var}(X) = \frac{\nu}{\nu-2}\sigma^2$ for $\nu > 2$, otherwise undefined |
| Skewness | 0 |
| Kurtosis | $3 + \frac{6}{\nu-4}$ for $\nu > 4$, otherwise undefined |

*Note: The Student's $t$ Distribution converges to the normal distribution as the degrees of freedom increases to infinity.*

#### Log-Normal Distribution
| Property | Formula/Value |
| :--- | :--- |
| Parameters | $\mu \in \mathbb{R}$ (location), $\sigma^2 > 0$ (scale) |
| PDF | $f_X(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp \left( -\frac{(\ln x-\mu)^2}{2\sigma^2} \right)$ |
| MGF | Does not exist |
| Mean | $\mathbb{E}[X] = e^{\mu + \frac{\sigma^2}{2}}$ |
| Variance | $\text{Var}(X) = e^{2\mu+\sigma^2}(e^{\sigma^2} - 1)$ |
| Skewness | $(e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$ |
| Kurtosis | $e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 3$ |

#### Exponential Distribution
| Property | Formula/Value |
| :--- | :--- |
| Parameters | $\lambda > 0$ (rate) |
| PDF | $f_X(x) = \lambda e^{-\lambda x}$ for $x \ge 0$, 0 otherwise |
| MGF | $\frac{\lambda}{\lambda-t}$ for $t < \lambda$, otherwise undefined |
| Mean | $\mathbb{E}[X] = 1/\lambda$ |
| Variance | $\text{Var}(X) = 1/\lambda^2$ |
| Skewness | 2 |
| Kurtosis | 9 |

*Note: Exponential distribution is often used to model the distribution of first arrival time.*

#### Poisson Distribution
| Property | Formula/Value |
| :--- | :--- |
| Parameters | $\lambda > 0$ (rate) |
| PMF | $f_X(x) = \frac{\lambda^x}{x!}e^{-\lambda}, x \in \{0, 1, 2, \dots\}$ |
| MGF | $e^{\lambda(e^t-1)}$ |
| Mean | $\mathbb{E}[X] = \lambda$ |
| Variance | $\text{Var}(X) = \lambda$ |
| Skewness | $1/\sqrt{\lambda}$ |
| Kurtosis | $3 + 1/\lambda$ |

*Note: Poisson distribution is the distribution of the number of times an event occurs in a fixed interval of time or space, given the average number of times the event occurs in that interval.*

#### Gamma Distribution
| Property | Formula/Value |
| :--- | :--- |
| Parameters | $\alpha > 0$ (shape), $\beta > 0$ (rate) |
| PDF | $f_X(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$ |
| MGF | $(\frac{\beta}{\beta-t})^\alpha$ for $t < \beta$, otherwise undefined |
| Mean | $\mathbb{E}[X] = \alpha/\beta$ |
| Variance | $\text{Var}(X) = \alpha/\beta^2$ |
| Skewness | $2/\sqrt{\alpha}$ |
| Kurtosis | $3 + 6/\alpha$ |

*Note: Gamma distribution is a continuous probability distribution that is often used to model the time until an event occurs in a process that follows an exponential distribution.*

#### Inverse Gaussian Distribution
| Property | Formula/Value |
| :--- | :--- |
| Parameters | $\mu \in \mathbb{R}, \lambda > 0$ |
| PDF | $f_X(x) = \sqrt{\frac{\lambda}{2\pi x^3}} \exp \left( -\frac{\lambda(x-\mu)^2}{2\mu^2 x} \right), x > 0$ |
| MGF | $\exp \left( \frac{\lambda}{\mu} (1 - \sqrt{1 - 2\mu^2 t/\lambda}) \right)$ |
| Mean | $\mathbb{E}[X] = \mu$ |
| Variance | $\text{Var}(X) = \mu^3/\lambda$ |
| Skewness | $3\sqrt{\mu/\lambda}$ |
| Kurtosis | $3 + 15\mu/\lambda$ |

#### Normal Inverse Gaussian (NIG) Distribution
| Property   | Formula/Value                                                                                                                                                  |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Parameters | $\mu \in \mathbb{R}$ (location), $\alpha > 0$ (tail heaviness), $\beta \in \mathbb{R}$ (asymmetry), $\delta > 0$ (scale), $\gamma = \sqrt{\alpha^2 - \beta^2}$ |
| PDF        | $f_X(x) = \frac{\alpha\delta}{\pi} \frac{K_1(\alpha\sqrt{\delta^2+(x-\mu)^2})}{\sqrt{\delta^2+(x-\mu)^2}} \exp(\delta\gamma + \beta(x - \mu))$                 |
| MGF        | $\exp(\mu t + \delta(\gamma - \sqrt{\alpha^2 - (\beta + t)^2}))$ for $\|\beta + t\| < \alpha$, otherwise undefined                                             |
| Mean       | $\mathbb{E}[X] = \mu + \delta\beta/\gamma$                                                                                                                     |
| Variance   | $\text{Var}(X) = \delta\alpha^2/\gamma^3$                                                                                                                      |
| Skewness   | $3\beta/(\alpha\sqrt{\delta\gamma})$                                                                                                                           |
| Kurtosis   | $3 + 3(1 + 4\beta^2/\alpha^2)/(\delta\gamma)$                                                                                                                  |

*Note: NIG is a generalization of the Normal distribution allowing for both location and scale parameters. $K_1(x)$ is the modified Bessel function of the second kind of order 1.*

#### Chi-Square Distribution
| Property   | Formula/Value                                            |
| :--------- | :------------------------------------------------------- |
| Parameters | $k > 0$ (degrees of freedom)                             |
| PDF        | $f_X(x) = \frac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2}$ |
| MGF        | $(1 - 2t)^{-k/2}$ for $t < 1/2$, otherwise undefined     |
| Mean       | $\mathbb{E}[X] = k$                                      |
| Variance   | $\text{Var}(X) = 2k$                                     |
| Skewness   | $\sqrt{8/k}$                                             |
| Kurtosis   | $3 + 12/k$                                               |

*Note: Chi-square is the distribution of the sum of the squares of $k$ independent standard normal random variables.*

#### F Distribution
| Property   | Formula/Value                                                                                          |
| :--------- | :----------------------------------------------------------------------------------------------------- |
| Parameters | $d_1 > 0, d_2 > 0$ (degrees of freedom)                                                                |
| PDF        | $f_X(x) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x B(d_1/2, d_2/2)}$    |
| MGF        | Does not exist                                                                                         |
| Mean       | $\mathbb{E}[X] = \frac{d_2}{d_2-2}$ for $d_2 > 2$, otherwise undefined                                 |
| Variance   | $\frac{2d_2^2(d_1+d_2-2)}{d_1(d_2-2)^2(d_2-4)}$ for $d_2 > 4$, otherwise undefined                     |
| Skewness   | $\frac{2(2d_1+d_2-2)\sqrt{8(d_2-4)}}{(d_2-6)\sqrt{d_1(d_1+d_2-2)}}$ for $d_2 > 6$, otherwise undefined |
| Kurtosis   | $3 + \frac{12(d_1(5d_2-22)+(d_2-4)^2)}{d_1(d_2-6)(d_2-8)}$ for $d_2 > 8$, otherwise undefined          |

*Note: F distribution is the distribution of the ratio of two chi-square random variables.*

#### Beta Distribution
| Property | Formula/Value |
| :--- | :--- |
| Parameters | $\alpha > 0$ (shape), $\beta > 0$ (shape) |
| PDF | $f_X(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$ |
| MGF | $\frac{1}{B(\alpha,\beta)} \int_0^1 e^{tx} x^{\alpha-1}(1-x)^{\beta-1} dx$ |
| Mean | $\mathbb{E}[X] = \frac{\alpha}{\alpha+\beta}$ |
| Variance | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| Skewness | $\frac{2(\beta-\alpha)\sqrt{\alpha+\beta+1}}{(\alpha+\beta+2)\sqrt{\alpha\beta}}$ |
| Kurtosis | $\frac{6((\alpha-\beta)^2(\alpha+\beta+1)-\alpha\beta(\alpha+\beta+2))}{\alpha\beta(\alpha+\beta+2)(\alpha+\beta+3)} + 3$ |

*Note: Beta distribution is often used to model the distribution of a random variable that takes values in a finite interval [0, 1].*

### Fitting SPX Returns by Normal Distribution
A normal distribution is fully characterized by its mean and variance, we take the mean and variance of SPX returns as the parameters of the normal distribution (method of moments):
$$\hat{\mu} = \mathbb{E}[X] = \frac{1}{T} \sum_{t=1}^{T} r_t$$
$$\hat{\sigma}^2 = \text{Var}(X) = \frac{1}{T - 1} \sum_{t=1}^{T} (r_t - \hat{\mu})^2$$
The denominator of the variance is $T - 1$ instead of $T$ because we use the **sample mean** to estimate the **population mean**.

### The Method of Moments
The **method of moments** is a simple and intuitive method for estimating the parameters of a distribution.

To match the first $k$ moments of the distribution, we set the first $k$ **sample moments** equal to the first $k$ **population moments**.

The **parameters** are then the solutions to the equations.

The method of moments is simple and intuitive, but it has several **Drawbacks:**
- The estimate parameters may be outside of the parameter space (infrequent for small samples).
- The estimates are not necessarily sufficient statistics; they sometimes fail to take into account all relevant information in the sample.

### The Maximum Likelihood Estimation (MLE)
The **maximum likelihood estimation (MLE)** is a method for estimating the parameters of a distribution by maximizing the likelihood function.

The log likelihood function is defined as:
$$\log L(\theta) = \sum_{i=1}^{n} \log p(x_i; \theta)$$
where $p(x_i; \theta)$ is the probability density function of the distribution, and $\theta$ is the parameter vector.

The MLE is the parameter vector that maximizes the log likelihood function.

*For a normal distribution, the MLE estimator is the same as the method of moments estimator.*

### Goodness of Fit of SPX Returns by Normal Distribution

![[Pasted image 20260115135254.png]]
*Figure: Q-Q Plot of Returns vs Normal Distribution and Empirical vs Fitted Normal Distribution*

### Q-Q Plot
The Q-Q plot is a graphical method for assessing the fit of a distribution to a sample.

It is a plot of the quantiles of the sample against the quantiles of the distribution.
- If the sample is drawn from the distribution, the points in the Q-Q plot will lie on the line $y = x$.
- From the Q-Q plot, we can see that the **SPX returns do not fit the normal distribution well.**

### Hypothesis Testing
The **hypothesis testing** is a statistical method for testing whether a hypothesis is true or false.
- The **null hypothesis** is the hypothesis that we want to test, and the **alternative hypothesis** is the hypothesis that we want to reject.
- The **test statistic** is a function of the sample data used to test the null hypothesis.
- The **p-value** is the probability of making a type I error, i.e., rejecting the null hypothesis when it is true. To calculate it, we need the distribution of the test statistic under the null.

### Example: Test Whether the Mean of SPX Returns is Greater Than Zero
- **Null Hypothesis:** Mean is zero.
- **Alternative Hypothesis:** Mean is greater than zero.
- **Test Statistic:** Scaled sample mean $Z = \sqrt{T}\hat{\mu}/\hat{\sigma}$, which approximately follows a standard normal distribution under the null hypothesis.
- **Results:** The p-value is 0.028 (probability of statistic > observed value 1.906).
- **Conclusion:** The threshold for rejecting at 5% significance is 1.645. Since 1.906 > 1.645, we reject the null.

### Example: Test Whether the Mean of SPX Returns is Equal to Zero
- **Null Hypothesis:** Mean is zero.
- **Alternative Hypothesis:** Mean is not zero.
- **Results:** The p-value is 0.057 (probability of $|statistic| \ge 1.906$).
- **Conclusion:** The threshold for rejecting at 5% significance is 1.96. Since 1.906 < 1.96, we fail to reject the null.

### The Shapiro-Wilk Test

The Shapiro-Wilk test is a test of normality.

The test statistic is given by:
$$W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n(x_i - \bar{x})^2}$$
where $x_{(i)}$ is the $i$-th order statistic of the sample, and $a_i$ is the $i$-th element of the vector $(a_1, a_2, \dots, a_n)$ given by:
$$a_i = \frac{m_i}{\sqrt{m_0 m_i}},$$
where $m_i$ is the $i$-th moment of the standard normal distribution, and $m_0$ is the sample mean.

Under the normal distribution, the test statistic $W$ follows the distribution:
$$W \sim \text{Beta}\left(\frac{n}{2}, \frac{n}{2}\right)$$
where $n$ is the sample size.

The $p$-value is the probability of making a type I error, i.e., rejecting the null hypothesis when it is true:
$$p = P(W \le w_{obs}|H_0) = \int_0^{w_{obs}} f_W(w)dw$$
where $w_{obs}$ is the observed value of the test statistic and $f_W(w)$ is the probability density function of the $\text{Beta}(\frac{n}{2}, \frac{n}{2})$ distribution.

### Testing Normality of SPX Returns

Apart from visual inspection, we can also use **hypothesis testing** to test whether the sample is drawn from the distribution.

The null hypothesis is that the sample is drawn from the distribution, and the alternative hypothesis is that the sample is not drawn from the distribution.

We can use the **Shapiro-Wilk test** to test whether the sample is drawn from the normal distribution.

The **$p$ -value** of the **Shapiro-Wilk test** is 0.0000, which is less than 0.05.
Therefore, **we reject the null hypothesis and conclude that the SPX returns do not follow the normal distribution.**

### Fitting SPX Returns by Student’s $t$ Distribution

The Student’s $t$ distribution is a generalization of the normal distribution that allows for heavy tails.

We fit the Student’s $t$ distribution to the SPX returns by **maximum likelihood estimation (MLE)**:
$$\max_{\nu,\mu,\sigma} \log L(\nu, \mu, \sigma) = \sum_{t=1}^T \log p(r_t; \nu, \mu, \sigma).$$

### Goodness of Fit of SPX Returns by Student’s $t$ Distribution

![[Pasted image 20260115141024.png]]
*Figure: Goodness of Fit of SPX Returns by Student’s $t$ Distribution.*

### The Kolmogorov-Smirnov Test

The **Kolmogorov-Smirnov test** is to test whether the sample is drawn from the distribution.

The test statistic is given by:
$$D_n = \sup_{x \in \mathbb{R}} |F_n(x) - F(x)|$$
where $F_n(x)$ is the **empirical distribution function** of the sample, and $F(x)$ is the distribution function of the hypothesized distribution.

Under the null hypothesis, the test statistic $\sqrt{n}D_n$ converges to the **Kolmogorov distribution**:
$$P(\sqrt{n}D_n > c) \approx 2 \sum_{k=1}^\infty (-1)^{k-1} e^{-2k^2c^2}.$$

### Test the Fit of SPX Returns by Student’s $t$ Distribution

We conduct the Kolmogorov-Smirnov test to test whether the SPX returns follow the Student’s $t$ distribution.

The $p$-value of the Kolmogorov-Smirnov test is 3.18e-02, which is less than 0.05. Therefore, **we reject the null hypothesis and conclude that the SPX returns do not follow the Student’s $t$ distribution.**

---

### Fitting SPX Returns by the Normal Inverse Gaussian Distribution

![[Pasted image 20260115141129.png]]
*Figure: Goodness of Fit of SPX Returns by the Normal Inverse Gaussian Distribution.*

### Kolmogorov-Smirnov Test for SPX Returns by the Normal Inverse Gaussian Distribution

Kolmogorov-Smirnov Test for Normal Inverse Gaussian Distribution:

| Measure | Value |
| :--- | :--- |
| Statistic | 0.0117 |
| p-value | 0.3792 |

Since the p-value (0.3792) is much larger than the significance level of 0.05, we fail to reject the null hypothesis.

This suggests that **the SPX returns can be well approximated by the Normal Inverse Gaussian distribution.**

---

## Part 3: Estimation of Risk Measures

### Risk Measures

**Risk measures** are used to quantify the risk of a portfolio.

Risk measures are useful for **risk management**, **performance evaluation**, **portfolio construction**, **regulatory compliance**, etc.

Examples of applications:
* Evaluate **risk-adjusted performance** of different trading strategies;
* Determine **margin requirements** for short positions or derivatives;
* Determine the **capital requirement** for a bank or insurance company;
* As criteria for trading and investment decisions;
* ...

Suppose we have a portfolio that has a return $X$ in a future time horizon $T$.

From the current perspective, the return $X$ is uncertain, and but we would like to quantify the risk of $X$, i.e, to establish **a mapping from $X$ to a real number**:
$$\rho : \mathcal{X} \to \mathbb{R}$$
where $\mathcal{X}$ is the space of random variables.

The function $\rho$ is called a **risk measure**.

### Value-at-Risk (VaR)

The value-at-risk (VaR) is the **quantile** of the portfolio return distribution:
$$\text{VaR}_\alpha(X) := -\inf\{x \in \mathbb{R} : P(X \le x) > \alpha\} = F_{-X}^{-1}(1 - \alpha).$$

The VaR is a popular risk measure, and it is also known as **quantile risk measure**.

It gives the **maximum loss** that the portfolio can suffer with a **confidence level** of $1 - \alpha$.

Typically, the confidence level is set to be 1% or 5%.

![[Pasted image 20260115141349.png]]
*Figure: Illustration of VaR.*

### Conditional Value-at-Risk (CVaR)

The Conditional Value-at-Risk (CVaR), also known as **Expected Shortfall (ES)**, is defined as:
$$\text{CVaR}_\alpha(X) := \mathbb{E}[-X| -X \ge \text{VaR}_\alpha(X)] = \frac{1}{\alpha} \int_0^\alpha \text{VaR}_\beta(X)d\beta.$$

CVaR measures the **expected loss** given that the loss exceeds VaR.

Properties of CVaR:
* CVaR is always **greater** than or equal to VaR.
* CVaR takes into account the **tail behavior** of the distribution.
* CVaR is a **coherent risk measure**.
* CVaR is more **sensitive to extreme events** than VaR.

![[Pasted image 20260115141448.png]]
*Figure: Illustration of CVaR.*

### Coherent Risk Measures

Coherent risk measures are a class of risk measures that satisfy the following properties:

* **Normalized:** $\rho(0) = 0$.
* **Monotonicity:** If $X \le Y$, then $\rho(X) \ge \rho(Y)$.
* **Subadditivity:** $\rho(X + Y) \le \rho(X) + \rho(Y)$.
* **Positive homogeneity:** $\rho(\lambda X) = \lambda\rho(X)$ for all $\lambda \ge 0$.
* **Translation invariance:** $\rho(X + A) = \rho(X) - a$ for all $A = a$ a.s.

The VaR is not a coherent risk measure as it does not satisfy the subadditivity property.

### Historical Simulation for Estimation of Risk Measures

We may use the historical simulation to estimate the risk measures.

The historical simulation is a non-parametric method that uses the historical data to estimate the risk measures.

The historical simulation is a simple method that does not require any distributional assumptions.

The procedure of historical simulation is as follows:
* Simulate the portfolio returns from the historical data.
* Sort the simulated returns in ascending order.
* Estimate the risk measures using the simulated returns.

### Historical Simulation for Estimation of VaR

Suppose we have simulated a sample of historical returns $\{X_1, \dots, X_T\}$.

We can estimate the VaR at confidence level $\alpha$ as:
$$\text{VaR}_\alpha(X) = -X_{(\lceil T\alpha \rceil)}$$
where:
* $X_{(i)}$ is the **$i$-th order statistic** of the sample $\{X_1, \dots, X_T\}$;
* $\lceil \cdot \rceil$ denotes the **ceiling function** that returns the smallest integer greater than or equal to the input.

### Historical Simulation for Estimation of CVaR

We can estimate the CVaR at confidence level $\alpha$ as:
$$\text{CVaR}_\alpha(X) = -\frac{1}{\lceil T\alpha \rceil} \sum_{i=1}^{\lceil T\alpha \rceil} X_{(i)},$$
which is the average of the largest $\lceil T\alpha \rceil$ losses.

### Calculate VaR and CVaR of Shorting SPX

As an example, we calculate the daily VaR and CVaR of shorting SPX.

For this position, the daily return is the negative of the SPX return:
$$X_t = 1 - e^{r_t^{\text{SPX}}}.$$

We regard the return sample $\{X_1, \dots, X_T\}$ as the historical simulation data, and use it to estimate the VaR and CVaR at confidence level $\alpha$.

#### Results

| Confidence Level | VaR | CVaR |
| :--- | :--- | :--- |
| 1.00% | 0.0343 | 0.0493 |
| 5.00% | 0.0184 | 0.0293 |
| 10.00% | 0.0124 | 0.0222 |

*Table: Risk Measures for Shorting SPX Position at Different Confidence Levels.*

---

### VaR and CVaR of Shorting SPX: Visualization

![[Pasted image 20260115141635.png]]
Figure: VaR and CVaR of Shorting SPX at 10% Confidence Level.

![[Pasted image 20260115141649.png]]
Figure: VaR and CVaR of Shorting SPX at 10% Confidence Level.

![[Pasted image 20260115141701.png]]
Figure: VaR and CVaR of Shorting SPX at 1% Confidence Level.

**The estimations tend to be unreliable when the confidence level is low as the sample size is small.**

### Estimation of Risk Measures by Parametric Models

| Confidence Level | Method       | VaR    | CVaR   |
| :--------------- | :----------- | :----- | :----- |
| 0.10%            | Historical   | 0.0692 | 0.0887 |
| 0.10%            | Normal       | 0.0386 | 0.0397 |
| 0.10%            | Student-$t$ | 0.0946 | 0.1076 |
| 0.10%            | NIG          | 0.0614 | 0.0656 |
| 0.50%            | Historical   | 0.0425 | 0.0597 |
| 0.50%            | Normal       | 0.0321 | 0.0352 |
| 0.50%            | Student-$t$ | 0.0493 | 0.0689 |
| 0.50%            | NIG          | 0.0413 | 0.0505 |
| 1.00%            | Historical   | 0.0339 | 0.0488 |
| 1.00%            | Normal       | 0.0290 | 0.0327 |
| 1.00%            | Student-$t$ | 0.0371 | 0.0551 |
| 1.00%            | NIG          | 0.0334 | 0.0434 |
| 5.00%            | Historical   | 0.0169 | 0.0277 |
| 5.00%            | Normal       | 0.0205 | 0.0255 |
| 5.00%            | Student-$t$ | 0.0183 | 0.0307 |
| 5.00%            | NIG          | 0.0178 | 0.0273 |

*Table: Comparison of Risk Measures across Different Methods.*

---

### Comparison of Fitted Distributions of SPX Returns

![[Pasted image 20260115141825.png]]
*Figure: Comparison of Fitted Distributions of SPX Returns.*

![[Pasted image 20260115141848.png]]
*Figure: Comparison of Fitted Distributions of SPX Returns.*

## Part 4: Extraction of Risk-Neutral Distributions from Option Data

### Forwards

A **forward contract** is a customized derivative that obligates the buyer to buy or sell an underlying asset at a specified price on a specified future date.

The specified price is called the **forward price**.

The payoff of a forward contract is given by:
$$\text{Payoff} = S_T - K,$$
where $K$ is the forward price.

### Options

**Options** are a type of **financial derivatives** that give the **buyer** the right, but not the obligation, to buy or sell an **underlying asset** at a specified price on or before a certain date.

The **underlying asset** can be a stock, an index, a bond, a commodity, or a currency.

The specified price is called the **strike price** or **exercise price**.

The specified date is called the **expiration date** or **maturity date**.

The buyer of an option pays a premium (**option price**) to the seller of the option.

#### European Call and Put Options
A **European call option** grants the buyer the right to buy the underlying asset at the strike price $K$ at the maturity date.
$$\text{Payoff} = \max(S_T - K, 0).$$

A **European put option** grants the buyer the right to sell the underlying asset at the strike price $K$ at the maturity date.
$$\text{Payoff} = \max(K - S_T, 0).$$

Other types of options include **American options**, which can be exercised at any time before the maturity date, and **exotic options**, which have payoffs that are more complex than the standard call and put options.

### Risk-Neutral Pricing of Options

Generally speaking, the option price is determined by the distribution of the underlying asset price at the maturity date $S_T$, and **market’s risk preferences**.

Then the option price can be expressed as:
$$C = \mathbb{E}_{\mathbb{P}}[M_T e^{-rT} f(S_T)],$$
where $M_T$ a non-negative random variable representing risk preferences, and $f(S_T)$ is the payoff function of the option.

#### The Case that $f(S_T) \equiv 1$
Suppose we have zero-coupon bond with maturity $T$ and face value 1.
$$C = \mathbb{E}_{\mathbb{P}}[M_T e^{-rT}] = e^{-rT} \Rightarrow \mathbb{E}_{\mathbb{P}}[M_T] = 1.$$

Then $M_T$ can serve as the Radon-Nikodym derivative that induces a new probability measure $\mathbb{Q}$ with respect to the real-world measure $\mathbb{P}$, i.e.,
$$\frac{d\mathbb{Q}}{d\mathbb{P}} = M_T,$$
i.e., for any event $A$, $\mathbb{Q}(A) = \mathbb{E}_{\mathbb{P}}[M_T \mathbb{1}_A]$.

Under this new probability measure $\mathbb{Q}$, the option pricing formula becomes:
$$C = \mathbb{E}_{\mathbb{P}}[M_T e^{-rT} f(S_T)] = \mathbb{E}_{\mathbb{Q}}[e^{-rT} f(S_T)].$$

#### The Case that $f(S_T) = S_T - K$ (a Forward Contract)
This is the case of holding a stock.
The contract value is given by:
$$C = \mathbb{E}_{\mathbb{P}}[e^{-rT} M_T(S_T - K)] = 0.$$

Therefore, the fair forward price is given by:
$$F_0 = \mathbb{E}_{\mathbb{P}}[M_T S_T].$$

### The Fair Forward Price

In the case of no dividend yield, by the no-arbitrage principle, the fair forward price is also given by:
$$F_0 = S_0 e^{rT},$$
by the no-arbitrage arguments as follows:
* If $F_0 > S_0 e^{rT}$, sell a forward contract and buy the underlying asset by borrowing $S_0$ at the risk-free rate.
* If $F_0 < S_0 e^{rT}$, sell the underlying asset and buy a forward contract, and invest $S_0$ at the risk-free rate.

Therefore, we have:
$$\mathbb{E}_{\mathbb{P}}[M_T S_T] = S_0 e^{rT}.$$

### The Black-Scholes Model

The **Black-Scholes model** specifies the underlying asset price $S_T$ as:
$$S_T = S_0 e^{(\mu - \frac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z},$$
where $Z$ is a standard normal random variable.

The pricing factor $M_T$ is given by:
$$M_T = e^{-\frac{1}{2}\theta^2 T - \theta \sqrt{T} Z},$$
for some $\theta \in \mathbb{R}$.

Noting that $\mathbb{E}_{\mathbb{P}}[M_T e^{-rT} S_T] = S_0$, we have
$$\theta = (\mu - r)/\sigma,$$
which is known as the **market price of risk**.

#### The Distribution of $S_T$ under the Risk-Neutral Measure
We note that $S_T$ is a function of $Z$, which is a standard normal random variable under the real-world measure $\mathbb{P}$.

To see its distribution under the risk-neutral measure $\mathbb{Q}$, we consider its moment generating function:
$$
\begin{aligned}
\mathbb{E}_{\mathbb{Q}}[e^{uZ}] &= \mathbb{E}_{\mathbb{P}}[e^{uZ} M_T] = \mathbb{E}_{\mathbb{P}}\left[e^{-\frac{1}{2}\theta^2 T + (u - \theta\sqrt{T})Z}\right] \\
&= e^{-\frac{1}{2}\theta^2 T + \frac{1}{2}(u - \theta\sqrt{T})^2} = e^{\frac{1}{2}u^2 - u\theta\sqrt{T}}.
\end{aligned}
$$

The right-hand side is the moment generating function of a normal random variable with mean $-\theta \sqrt{T}$ and variance 1. Then,
$$\tilde{Z} = Z + \theta \sqrt{T} \sim \mathcal{N}(0, 1), \text{ under } \mathbb{Q}.$$

And,
$$S_T = S_0 e^{(r - \frac{1}{2}\sigma^2)T + \sigma \sqrt{T} \tilde{Z}}.$$

### The Black-Scholes Formula

The Black-Scholes formula for **European call option** price is given by:
$$C = \mathbb{E}_{\mathbb{Q}}[e^{-rT} (S_T - K)^+] = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2),$$
where $\Phi$ is the cumulative distribution function of a standard normal random variable, and
$$d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}.$$

The Black-Scholes formula for **European put option** price is given by:
$$P = \mathbb{E}_{\mathbb{Q}}[e^{-rT} (K - S_T)^+] = K e^{-rT} \Phi(-d_2) - S_0 \Phi(-d_1).$$

### The Case of Dividend Yield

For simplicity, we consider constant proportional dividend yield $q$.

The pricing factor $M_T$ should satisfy that:
$$\mathbb{E}_{\mathbb{P}}[M_T e^{-rT} e^{qT} S_T] = S_0, \quad M_T = e^{-\frac{1}{2}\theta^2 T - \theta \sqrt{T} Z},$$
$$\theta = (\mu - (r - q))/\sigma.$$

In this case,
$$S_T = S_0 e^{(r - q - \frac{1}{2}\sigma^2)T + \sigma \sqrt{T} \tilde{Z}}, \quad \mathbb{E}_{\mathbb{Q}}[e^{-rT} S_T] = S_0 e^{-qT}.$$

#### The Black-Scholes Model with Dividend Yield
The Black-Scholes formula for **European call option** price with dividend yield is given by:
$$C = \mathbb{E}_{\mathbb{Q}}[e^{-rT} (S_T - K)^+] = S_0 e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2),$$
where $d_1 = \frac{\ln(S_0/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}$, and $d_2 = d_1 - \sigma \sqrt{T}$.

The Black-Scholes formula for **European put option** price with dividend yield is given by:
$$P = \mathbb{E}_{\mathbb{Q}}[e^{-rT} (K - S_T)^+] = K e^{-rT} \Phi(-d_2) - S_0 e^{-qT} \Phi(-d_1).$$

### The General Dividend Yield Case

In practice, dividends are not distributed continuously, but rather at discrete times. Moreover, it is not guaranteed that the dividend yield is constant.

In this case, we may recognize that the forward price process satisfies that
$$\mathbb{E}_{\mathbb{Q}}[F_T] = F_0, \quad F_T = S_T.$$

Therefore, we may specify that
$$F_T = F_0 e^{-\frac{1}{2}\sigma^2 T + \sigma \sqrt{T} \tilde{Z}}.$$

#### The Black-Scholes Formula by the Forward Price
And then the Black-Scholes formula for **European call option** price with dividend yield is given by:
$$C = \mathbb{E}_{\mathbb{Q}}[e^{-rT} (F_T - K)^+] = e^{-rT} (F_0 \Phi(d_1) - K \Phi(d_2)),$$
where $d_1 = \frac{\ln(F_0/K) + \frac{1}{2}\sigma^2 T}{\sigma \sqrt{T}}$, and $d_2 = d_1 - \sigma \sqrt{T}$.

The same argument applies to put options with formula:
$$P = \mathbb{E}_{\mathbb{Q}}[e^{-rT} (K - F_T)^+] = e^{-rT} (K \Phi(-d_2) - F_0 \Phi(-d_1)).$$

In the continuous dividend yield case, the forward price is given by:
$$F_0 = S_0 e^{(r-q)T}.$$

### The Put-Call Parity

The put-call parity is given by:
$$C - P = e^{-rT}(F_0 - K).$$

If it is violated, there exists an arbitrage opportunity as follows:
* If $C - P > e^{-rT}(F_0 - K)$, sell a call option and buy a put option, and enter a forward contract to buy the underlying asset at the forward price $F_0$.
* If $C - P < e^{-rT}(F_0 - K)$, buy a call option and sell a put option, and enter a forward contract to sell the underlying asset at the forward price $F_0$.

### Calibrating the Black-Scholes Model to Option Data: Implied Volatility

There is only one free parameter in the Black-Scholes model, which is the volatility $\sigma$.

The **implied volatility** is the volatility that makes the Black-Scholes formula fit the option price, i.e.,
$$C_{BS}(K, T; \sigma) = C_{MKT}(K, T),$$
where $C_{BS}$ is the Black-Scholes formula, and $C_{MKT}$ is the market price of the option.

This yields a **nonlinear equation** for $\sigma$, which can be solved by numerical methods:
$$\sigma = \sigma_{IV}(K, T).$$

### The Implied Volatility Surface

It is well-known that the **implied volatility surface** is not flat, but rather exhibits a smile or skew pattern.

* **Implied volatility curve:** the dependence of implied volatility on the strike price at a fixed maturity.
* **Term structure of implied volatility:** the dependence of implied volatility on the maturity date.

These dependence **violates the Black-Scholes model**, which assumes that the implied volatility is constant over time and strike prices.

However, implied volatility calculated by the Black-Scholes formula serves as a convenient measure of the highness of the option price.

### Existence of Implied Volatility

By the **no-arbitrage principle**, the market price of a call option must belong to a range:
$$(S_0 - K e^{-rT})^+ \le C_{MKT} \le S_0.$$

Moreover, we note that
$$\lim_{\sigma \to 0} C_{BS}(K, T; \sigma) = (S_0 - K e^{-rT})^+,$$
$$\lim_{\sigma \to \infty} C_{BS}(K, T; \sigma) = S_0.$$

Therefore, **the implied volatility must exist if the market price of the option belongs to the no-arbitrage range**. The same argument applies to put options.

### Numerical Methods for Implied Volatility

#### The bisection search method

1. Set the initial range $[\sigma_L, \sigma_H]$, where $\sigma_L = 0$ and $\sigma_H$ is a large number.
2. Calculate $\sigma_M = \frac{\sigma_L + \sigma_H}{2}$ and evaluate $C_{BS}(K, T; \sigma_M)$.
3. If $|C_{BS}(K, T; \sigma_M) - C_{MKT}| < \epsilon$, stop.
4. Otherwise:
	* If $C_{BS}(K, T; \sigma_M) > C_{MKT}$, set $\sigma_H = \sigma_M$
	* If $C_{BS}(K, T; \sigma_M) < C_{MKT}$, set $\sigma_L = \sigma_M$
5. Return to step 2.

The number of iterations to achieve the desired accuracy $\epsilon$ is given by:
$$\log_2 \left( \frac{\sigma_H - \sigma_L}{\epsilon} \right).$$

#### The Newton-Raphson Method
The **Newton-Raphson method** is a more efficient method for finding the implied volatility.

The iteration of the Newton-Raphson method is given by:
$$\sigma_{n+1} = \sigma_n - \frac{C_{BS}(K, T; \sigma_n) - C_{MKT}}{\partial_\sigma C_{BS}(K, T; \sigma_n)}.$$

The Newton-Raphson method converges faster than the bisection search method.

### SPX Options

The standard options written on the S&P 500 index are European style. The liquidity of SPX options is very high, and the data is publicly available, providing a good testbed for developing and testing **option pricing models**.

Apart from option pricing and trading, options embeds a lot of **forward looking** information about the market.

We are concerned about the following issues:
* How to estimate the implied volatility of SPX options?
* How to extract volatility, skewness and kurtosis from the option prices?
* How to construct a distribution model for $S_T$ under the risk-neutral measure to explain the implied volatility curve?

### The SPX Options Data

The December 13, 2024 trading data of SPX options expiring on January 17, 2025.

The current S&P 500 index level is 6051, the 3-month risk-free rate is 4.31% per annum, and the dividend yield is 1.213% per annum.

![[Pasted image 20260115144850.png]]
*Table: First and Last Few Rows of SPX Options Data*

### The Implied Volatility Curve

We calculate the implied volatility by OTM SPX call (converted to put options by the put-call parity) and put options (filtering out options with zero volume and open interest) by inverting the Black-Scholes formula assuming the dividend yield is 1.213% per annum. We see **inconsistency of the implied volatility implied from OTM put options and OTM call options**.

![[Pasted image 20260115174054.png]]
*Figure: Put Options: Implied Volatility vs Moneyness*

### Correction by the Implied Forward Price

We first calculate the implied forward price by the put-call parity using ATM call and put options, and then use the implied forward price to calculate the implied volatility.

![[Pasted image 20260115174149.png]]
*Figure: Put Options: Implied Volatility vs Moneyness - Corrected*

### The Volatility Index (VIX)

VIX is a volatility index that is widely used to measure the market’s expectation of volatility over the next 30 days. It is releaseed by the CBOE (Chicago Board Options Exchange) every day at 4pm EST from 2004.

The CBOE VIX index can be derived from the prices of SPX options using the following formula:
$$\text{VIX} = 100 \times \sqrt{\frac{2e^{rT}}{T} \left( \int_0^{F_0} \frac{1}{K^2} P(K)dK + \int_{F_0}^\infty \frac{1}{K^2} C(K)dK \right)},$$
where $F_0$ is the forward price, $P(K)$ and $C(K)$ are put and call option prices, and $r$ is the risk-free rate.

#### Calculating the VIX
To calculate the VIX, we need to **discretize the integral** using the mid-point rule, which leads to the following formula:
$$\text{VIX} = 100 \times \sqrt{\frac{2e^{rT}}{T} \sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i) - \frac{1}{T} \left( \frac{F_0}{K_0} - 1 \right)^2},$$
where $Q(K_i)$ is the midpoint of the bid-ask spread for each option with strike $K_i$, $\Delta K_i = (K_{i+1} - K_{i-1})/2$, and $K_0$ is the first strike below the forward price.

Using the SPX options data, we calculate the VIX to be **13.78** while the reported VIX is **13.81** (close value).

### Implied Skewness and Kurtosis

Bakshi (2003) propose methods to extract implied skewness and kurtosis of $\ln(S_T/S_0)$ from the option prices.

$$\text{SKEW}(T) = \frac{e^{rT}W(T) - 3\mu(T)e^{rT}V(T) + 2\mu(T)^3}{(e^{rT}V(T) - \mu(T)^2)^{3/2}},$$
$$\text{KURT}(T) = \frac{e^{rT}X(T) - 4\mu(T)e^{rT}W(T) + 6e^{rT}\mu(T)^2V(T) - 3\mu(T)^4}{(e^{rT}V(T) - \mu(T)^2)^2},$$
where
$$V(T) = \int_0^\infty \frac{2(1 - \ln(K/S_0))}{K^2} Q(K)dK,$$
$$X(T) = \int_0^\infty \frac{12(\ln(K/S_0))^2 - 4(\ln(K/S_0))^3}{K^2} Q(K)dK,$$
$$W(T) = \int_0^\infty \frac{6 \ln(K/S_0) - 3(\ln(K/S_0))^2}{K^2} Q(K)dK.$$

### Implied Mean and Volatility

Also we have that the implied mean and standard deviation of the log return is given by
$$\mu(T) = e^{rT} - 1 - \frac{e^{rT}V}{2} - \frac{e^{rT}W}{6} - \frac{e^{rT}X}{24},$$
$$\sigma(T) = \sqrt{V e^{rT} - \mu(T)^2}.$$

The results are as follows:

| BKM Risk-Neutral Moments (30 days). | Value |
| :--- | :--- |
| Mean | 0.0028 |
| Volatility | 0.0411 |
| Skewness | −5.7389 |
| Kurtosis | 72.0314 |

### The Implied NIG Model

Suppose $\ln(S_T/S_0)$ follows a normal inverse Gaussian distribution, i.e.,
$$\ln(S_T/S_0) \sim \text{NIG}(\alpha, \beta, \delta, \mu).$$

The parameters can be estimated by matching the mean, variance, skewness and kurtosis with those implied by BKM (2003).
$$\mu(T) = \mu + \delta\beta/\gamma,$$
$$\sigma^2(T) = \delta\alpha^2/\gamma^3,$$
$$\text{SKEW}(T) = 3\beta/(\alpha \sqrt{\delta\gamma}),$$
$$\text{KURT}(T) = 3(1 + 4\beta^2/\alpha^2)/(\delta\gamma) + 3.$$

#### The Estimated NIG Parameters
The estimated NIG parameters are as follows:

| Fitted NIG Parameters | Value |
| :--- | :--- |
| $\alpha$ | 15.0643 |
| $\beta$ | −9.9777 |
| $\delta$ | 0.0106 |
| $\mu$ | 0.0121 |

We check the moments of the NIG distribution and compare them with those implied by BKM (2003).

| Moments Comparison | BKM | NIG |
| :--- | :--- | :--- |
| Mean | 0.0028 | 0.0028 |
| Volatility | 0.0411 | 0.0409 |
| Skewness | −5.7389 | −5.7389 |
| Kurtosis | 72.0314 | 71.9375 |

### The Performance of the Implied NIG Model

We examine the performance of the implied NIG model by comparing the implied option prices and implied volatility with the observed.

The option prices implied by the NIG model are estimated by **Monte Carlo simulation**:
$$P_{NIG}(K, T) = \frac{1}{N} \sum_{i=1}^N e^{-rT} \max(0, S_0 e^{X_i} - K),$$
where $X_i$ is simulated from the fitted NIG distribution and $N = 100000$ is the number of simulations.

#### Errors for Different Moneyness Groups
Deep OTM ($K/S < 0.94$), OTM ($0.94 \le K/S < 0.97$), ATM ($0.97 \le K/S < 1.03$), ITM ($1.03 \le K/S < 1.06$), Deep ITM ($K/S \ge 1.06$).

|                | **NIG Model**  |                |  **BS Model**  |                |
| :------------- | :------------: | :------------: | :------------: | :------------: |
| **$K/S$**      | **Rel. Error** | **Abs. Error** | **Rel. Error** | **Abs. Error** |
| $< 0.94$       |     32.76%     |      1.46      |     93.82%     |      3.79      |
| $[0.94, 0.97)$ |     23.15%     |      3.86      |     22.45%     |      3.45      |
| $[0.97, 1.03)$ |     6.84%      |      4.32      |     35.16%     |     22.16      |
| $[1.03, 1.06)$ |     1.66%      |      4.36      |     9.56%      |     21.95      |
| $\ge 1.06$     |     0.99%      |      5.38      |     1.23%      |      6.41      |
| **Overall**    |   **15.14%**   |    **3.54**    |   **46.90%**   |   **14.27**    |
*Table: NIG Model vs BS Model Error Comparison*

### NIG Implied Volatility and Prices Visualization

![[Pasted image 20260115175333.png]]
*Figure: NIG Implied Volatility Curve*

![[Pasted image 20260115175347.png]]
*Figure: Market vs NIG and BS Prices*

### To Refine the NIG Model

We can refine the parameters of the NIG model by minizing the following objective function:
$$\min_{\alpha,\beta,\delta,\mu} \frac{1}{n} \sum_{i=1}^n (P_{NIG}(K_i, T) - P_{MKT}(K_i, T))^2,$$
where $P_{NIG}(K_i, T)$ is the price implied by the NIG model and $P_{MKT}(K_i, T)$ is the market price of the option with strike $K_i$ and maturity $T$.

Here the $P_{NIG}(K_i, T)$ is calculated by Monte Carlo simulation.

Additional **penalty terms** can be added to the objective function to ensure the validity of the NIG parameters ($\delta > 0$ and $|\alpha| > |\beta|$) and the no-arbitrage bounds of the option prices:
$$e^{-rT} (K - F_0)^+ \le P_{NIG}(K, T) \le e^{-rT} K.$$

### Simulating NIG Random Variables

To avoid repeated sampling for different parameters, we sample the NIG distribution as follows:
$$X = \mu + \delta(bx + \sqrt{x}W),$$
where $b = \beta\delta$, $W$ is a standard normal random variable and $x$ is sampled from the **inverse Gaussian distribution** as follows:
$$y = \mu_{ig} + \frac{1}{2}\mu_{ig}^2 Z^2 - \frac{1}{2}\mu_{ig} \sqrt{4\mu_{ig} Z^2 + \mu_{ig}^2 Z^4},$$
$$x = \begin{cases} y & \text{if } U \le \frac{\mu_{ig}}{\mu_{ig} + y}, \\ \frac{\mu_{ig}^2}{y} & \text{otherwise.} \end{cases}$$
where $a = \alpha\delta$, $\mu_{ig} = 1/\sqrt{a^2 - b^2}$, $U$ is sampled from the uniform distribution on $[0, 1]$ and $Z$ is sampled from the standard normal distribution.

### The Estimated NIG Parameters (Refined)

The parameters:

| Parameter | Value |
| :--- | :--- |
| $\alpha$ | 20.4877 |
| $\beta$ | −9.8971 |
| $\delta$ | 0.0188 |
| $\mu$ | 0.0142 |

The statistics of the fitted NIG distribution:

| Moment | Value |
| :--- | :--- |
| Mean | 0.0038 |
| Volatility | 0.0369 |
| Skewness | −2.4972 |
| Kurtosis | 20.2219 |

### In-Sample and Out-of-Sample Performance

We randomly split the data into in-sample (approximately 70%) and out-of-sample (approximately 30%). The parameters are estimated using the in-sample data.

|                | **In-Sample**  |                | **Out-of-Sample** |                |
| :------------- | :------------: | :------------: | :---------------: | :------------: |
| **$K/S$**      | **Rel. Error** | **Abs. Error** |  **Rel. Error**   | **Abs. Error** |
| $< 0.94$       |     54.59%     |      1.23      |      55.02%       |      1.37      |
| $[0.94, 0.97)$ |     1.93%      |      0.33      |       2.91%       |      0.67      |
| $[0.97, 1.03)$ |     0.96%      |      0.50      |       1.33%       |      0.50      |
| $[1.03, 1.06)$ |     0.97%      |      2.43      |       0.86%       |      2.03      |
| $\ge 1.06$     |     0.01%      |      0.10      |       0.16%       |      0.70      |
| **Overall**    |   **15.76%**   |    **0.74**    |    **16.36%**     |    **0.90**    |
*Table: In-Sample and Out-of-Sample Performance Errors*

### NIG Refined Model Visualization

![[Pasted image 20260115175526.png]]
*Figure: The NIG Implied Volatility Curve: In-Sample*

![[Pasted image 20260115175634.png]]
*Figure: The NIG Implied Volatility Curve: Out-of-Sample*

![[Pasted image 20260115175649.png]]
*Figure: The NIG Price vs Market Price: In-Sample*

![[Pasted image 20260115175704.png]]
*Figure: The NIG Price vs Market Price: Out-of-Sample*

### The DEJD Model

Here we consider a simple, intuitive and flexible model for the distribution of the log return, the **DEJD (double exponential jump diffusion)** model:
$$\ln(S_T/S_0) = \mu + \sigma \tilde{Z} + \begin{cases} \mathcal{E}_1/\eta_1 & \text{w.p. } p\lambda, \\ -\mathcal{E}_2/\eta_2 & \text{w.p. } (1 - p)\lambda, \end{cases}$$
where $\tilde{Z}$ is a standard normal random variable, $\mathcal{E}_1$ and $\mathcal{E}_2$ are two independent exponential random variables with mean 1.

* $\lambda$ is the probability of a jump.
* $p$ is the probability of a positive jump given a jump occurs.
* $1/\eta_1$ and $1/\eta_2$ are the mean of the positive and negative jumps, respectively.

#### The Estimated DEJD Parameters

| Parameter | Value |
| :--- | :--- |
| $\mu$ | 0.0048 |
| $\sigma$ | 0.0251 |
| $\lambda$ | 0.6195 |
| $p$ | 0.9742 |
| $\eta_1$ | 239.6988 |
| $\eta_2$ | 3.5663 |

### DEJD Model Performance

|                | **In-Sample**  |                | **Out-of-Sample** |                |
| :------------- | :------------: | :------------: | :---------------: | :------------: |
| **$K/S$**      | **Rel. Error** | **Abs. Error** |  **Rel. Error**   | **Abs. Error** |
| $< 0.94$       |    109.18%     |      3.62      |      149.83%      |      3.24      |
| $[0.94, 0.97)$ |     9.43%      |      1.65      |       4.39%       |      0.80      |
| $[0.97, 1.03)$ |     6.43%      |      2.96      |       5.17%       |      2.24      |
| $[1.03, 1.06)$ |     1.04%      |      2.13      |       0.76%       |      1.86      |
| $\ge 1.06$     |     0.12%      |      0.52      |       0.07%       |      0.45      |
| **Overall**    |   **35.59%**   |    **2.90**    |    **41.83%**     |    **2.17**    |
*Table: DEJD In-Sample and Out-of-Sample Performance Errors*

### DEJD Model Visualization

![[Pasted image 20260115175841.png]]
Figure: The DEJD Implied Volatility Curve: In-Sample

![[Pasted image 20260115175858.png]]
Figure: The DEJD Implied Volatility Curve: Out-of-Sample

![[Pasted image 20260115175911.png]]
Figure: The DEJD Price vs Market Price: In-Sample

![[Pasted image 20260115175922.png]]
Figure: The DEJD Price vs Market Price: Out-of-Sample