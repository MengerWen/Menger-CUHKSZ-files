# Portfolio Optimization

**Gongqiu Zhang** 
SSE, CUHK-Shenzhen

---

## 0 **Portfolio Allocation**

The general principle of portfolio allocation is to combine different assets to achieve a balance between risk and return.

- Minimize risk for a given level of return.
- Maximize return for a given level of risk.
- Maximize risk-adjusted return.

---

## 1 **The Mean-Variance Portfolio Optimization**

### Problem Formulation and Solution

#### A Motivating Example

Consider a one-period market with $n$ securities with identical expected returns and variances:
- $\mathbb{E}[R_i] = \mu$ for all $i = 1, 2, \dots, n$
- $\text{Var}(R_i) = \sigma^2$ for all $i = 1, 2, \dots, n$
- $\text{Cov}(R_i, R_j) = 0$ for all $i \neq j$

**Portfolio A** : 100% invested in security #1
$$
w_1 = 1, \quad w_i = 0 \text{ for } i = 2, \dots, n
$$
The expected return and variance of Portfolio A are:
$$
\mathbb{E}[R_A] = \mu, \quad \text{Var}(R_A) = \sigma^2
$$

**Portfolio B** : Equally-weighted portfolio
$$
w_i = \frac{1}{n} \text{ for } i = 1, \dots, n
$$
The expected return and variance of Portfolio B are:
$$
\mathbb{E}[R_B] = \mu, \quad \text{Var}(R_B) = \frac{\sigma^2}{n}.
$$

#### The Mean-Variance Portfolio Optimization

The mean-variance portfolio optimization problem is formulated as:
$$
\sigma^2 = \min_w \frac{1}{2} w^\top \Sigma w
$$
subject to
$$
w^\top \mu = z, \quad w^\top e = 1,
$$
where $z$ is the target return level.

#### Solving the Mean-Variance Portfolio Optimization

The Lagrangian for this optimization problem is:
$$
\mathcal{L}(w, \lambda_1, \lambda_2) = \frac{1}{2} w^\top \Sigma w - \lambda_1(w^\top \mu - z) - \lambda_2(w^\top e - 1)
$$

The first-order conditions are:
$$
\begin{aligned}
\Sigma w - \lambda_1 \mu - \lambda_2 e &= 0 \quad \text{(First-order condition)} \\
w^\top \mu &= z \quad \text{(Return constraint)} \\
w^\top e &= 1 \quad \text{(Budget constraint)}
\end{aligned}
$$

Solving these equations:
$$
w = \lambda_1 \Sigma^{-1} \mu + \lambda_2 \Sigma^{-1} e,
$$
where $\lambda_1$ and $\lambda_2$ are the Lagrange multipliers that can be determined from the constraints.

### Key Properties and Theorems

#### The Mean-Variance Frontier

The mean-variance frontier is the set of portfolios that minimize the variance for a given level of return.

In the $z - \sigma^2$ plane, the mean-variance frontier is a parabola.

#### The Global Minimum Variance Portfolio

The global minimum variance portfolio is the portfolio that has the minimum variance:
$$
\frac{1}{2}\sigma^2 = \min_w \frac{1}{2} w^\top \Sigma w, \text{ subject to } w^\top e = 1.
$$

The Lagrangian for this optimization problem is:
$$
\mathcal{L}(w, \lambda) = \frac{1}{2} w^\top \Sigma w - \lambda(w^\top e - 1)
$$

The first-order conditions are:
$$
\begin{aligned}
\Sigma w - \lambda e &= 0 \quad \text{(First-order condition)} \\
w^\top e &= 1 \quad \text{(Budget constraint)}
\end{aligned}
$$

Solving these equations:
$$
w = \lambda \Sigma^{-1} e, \text{ and } \lambda = \frac{1}{e^\top \Sigma^{-1} e}.
$$

#### A Synthetic Example

Consider a market with 10 stocks with given expected returns and covariance matrix.

![[Pasted image 20260412182226.png]]
*Figure: Mean-Variance Frontier*

#### The Two-Fund Theorem

> [!info]+ Theorem (The Two-Fund Theorem)
> Let $w_1$ and $w_2$ be (mean-variance) efficient portfolios corresponding to expected returns $r_1$ and $r_2$, respectively, with $r_1 \neq r_2$. Then all efficient portfolios can be obtained as linear combinations of $w_1$ and $w_2$.
> 
> **Proof:** Let $w$ be an efficient portfolio with expected return $r$. From the first-order conditions of the mean-variance optimization problem, we have:
> $$
> w = \lambda_1 \Sigma^{-1} e + \lambda_2 \Sigma^{-1} \mu
> $$
> for some constants $\lambda_1$ and $\lambda_2$. Since this holds for any efficient portfolio (including $w_1$ and $w_2$), all efficient portfolios must be linear combinations of $\Sigma^{-1} e$ and $\Sigma^{-1} \mu$, or equivalently, of any two different efficient portfolios.

#### A Simulation Study

Consider a market with 10 stocks such that $R \sim \mathcal{N}(\mu, \Sigma)$. Estimate the mean-variance frontier from the 24, 240, and 2400 periods of historical data, and plug in the estimated parameters to compute weights of efficient portfolios.

![[Pasted image 20260412182252.png]]
*Figure: Mean-Variance Frontiers (10 Simulations) - 24 periods*

![[Pasted image 20260412182309.png]]
*Figure: Mean-Variance Frontiers (10 Simulations) - 240 periods*

![[Pasted image 20260412182321.png]]
*Figure: Mean-Variance Frontiers (10 Simulations) - 2400 periods*

### Extensions to the Basic Model

#### The Mean-Variance Frontier with a Risk-Free Asset

Consider a market with a risk-free asset with expected return $r_f$ and variance $0$. The mean-variance optimization problem is:
$$
\frac{1}{2}\sigma^2 = \min_w \frac{1}{2} w^\top \Sigma w
$$
subject to
$$
w^\top \mu + (1 - w^\top e)r_f = z.
$$

The Lagrangian for this optimization problem is:
$$
\mathcal{L}(w, \lambda) = \frac{1}{2} w^\top \Sigma w - \lambda(w^\top \mu + (1 - w^\top e)r_f - z)
$$

The first-order conditions are:
$$
\Sigma w - \lambda(\mu - e r_f) = 0
$$

Solving these equations:
$$
w = \lambda \Sigma^{-1}(\mu - e r_f)
$$

The Lagrangian multiplier $\lambda$ is given by:
$$
\lambda = \frac{z - r_f}{(\mu - e r_f)^\top \Sigma^{-1} (\mu - e r_f)} = \frac{z - r_f}{\sigma_m^2},
$$
where $\sigma_m^2 = (\mu - e r_f)^\top \Sigma^{-1} (\mu - e r_f)$.

The minimum variance portfolio is given by:
$$
w^\top \Sigma w = \lambda^2 (\mu - e r_f)^\top \Sigma^{-1} (\mu - e r_f) = \frac{(z - r_f)^2}{\sigma_m^2}
$$

The mean-variance frontier should be a straight line in the $z - \sigma$ plane.

### Practical Limitations and Improvements

#### Weakness of Mean-Variance Analysis

The tendency to produce extreme portfolios combining extreme shorts with extreme longs. As a result, portfolio managers generally do not trust these extreme weights. This problem is typically caused by estimation errors in the mean return vector and covariance matrix.

In addition, it is worth emphasizing that in practice we may not have as many as $24$ relevant observations available. For example, if our data observations are weekly returns, then using $24$ of them to estimate the joint distribution of returns is hardly a good idea since we are generally more concerned with estimating conditional return distributions and so more weight should be given to more recent returns.

The problem becomes more severe when the number of assets $n$ is large and/or the returns are heavy-tailed. A more sophisticated estimation approach should therefore be used in practice.

The portfolio weights tend to be extremely sensitive to very small changes in the expected returns. Best and Grauer (1991) showed that
$$
\|w - \hat{w}\| \leq \|\mu - \hat{\mu}\| \frac{1}{\lambda_{\min}} \left( 1 + \frac{\lambda_{\max}}{\lambda_{\min}} \right)
$$
where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and smallest eigenvalues, respectively, of the covariance matrix, $\Sigma$.

This ratio $\frac{\lambda_{\max}}{\lambda_{\min}}$, when applied to the estimated covariance matrix, $\hat{\Sigma}$, typically becomes large as the number of assets increases and the number of sample observations is held fixed. As a result, we can expect large errors for large portfolios with relatively few observations.

#### Improving the Mean-Variance Analysis

As a result of these weaknesses, portfolio managers traditionally have had little confidence in mean-variance analysis and therefore applied it very rarely in practice.

Common approaches to overcome these problems that are now routinely used in general asset allocation include:
- Better estimation techniques:
    - Shrinkage estimators.
    - Robust estimators.
    - Bayesian techniques like the Black-Litterman framework (1990s).
- Portfolio constraints can help mitigate extreme allocations:
    - No short-sales constraints.
    - No-borrowing constraints.
    - Leverage constraints.
    - Sparsity constraints.

---

## 2 **The Black-Litterman Framework**

### Framework Overview

#### The Black-Litterman Framework

**Priori return dist.** $+$ **Views**
$\Rightarrow$ **Posterior return dist.**
$\Rightarrow$ **Mean-Variance optimization.**

### Key Components of the Framework

#### Priori Return Distribution

Ignoring the constraint, the mean-variance solution is:
$$
w = \frac{1}{\gamma} \Sigma^{-1} \mu.
$$

**Reverse optimization** : given that we observe the market proportion of each assets is $\omega$, then the market thinks that the **expected asset returns** are,
$$
\mu_0 = \gamma \Sigma \omega.
$$

Then the priori return distribution is assumed as:
$$
\mu \sim \mathcal{N}(\mu_0, \tau \Sigma).
$$
$\tau$ is a small scalar.

#### Views

People may have **prediction on the future returns** based on their expertise or information.

A **view (the $j$ -th view)** can be describe as:
$$
p_j^\top \mu = q_j + \epsilon_j.
$$
- Each $p_j$ is a $n$ -dim vector representing a portfolio.
- Each $q_j$ represents the predicted return of this portfolio.
- $\epsilon_j \sim \mathcal{N}(0, \sigma_j^2)$: $\sigma_j^2$ measures the confidence of the view.

All the views together can be described as:
$$
P\mu = q + \epsilon.
$$
Here: $K$ is the number of views.
$$
P =[p_1, p_2, \dots, p_K]^\top,
$$
$$
q =[q_1, q_2, \dots, q_K]^\top,
$$
$$
\epsilon \sim \mathcal{N}(0_K, \Omega),
$$
where $\Omega = \text{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_K^2)$.

**Different views are independent.**

#### Posterior Return Distribution

The posterior return distribution is also normal.

The posterior expected return is,
$$
\mu_p = \left( (\tau \Sigma)^{-1} + P^\top \Omega^{-1} P \right)^{-1} \left( (\tau \Sigma)^{-1} \mu_0 + P^\top \Omega^{-1} q \right)
$$

Then $\mu_p$ is **plugged into the mean-variance solution**.

---

## 3 **The Mean-CVaR Portfolio Optimization**

### Tail Risk Measures

#### Value-at-Risk (VaR)

Denote by $L$ the loss of a portfolio in the future with CDF $F_L(\cdot)$:
$$
F_L(x) = \mathbb{P}[L \leq x].
$$

The $\text{VaR}_\alpha(L)$ is defined as
$$
\text{VaR}_\alpha(L) := \sup\{x : F_L(x) \leq \alpha\} = F_L^{-1}(\alpha).
$$

It measures **tail risk**.

#### Conditional Value-at-Risk (CVaR)

The loss can be large beyond $\text{VaR}_\alpha(L)$.

$\text{CVaR}_\alpha(L)$ is defined as
$$
\begin{aligned}
\text{CVaR}_\alpha(L) &= \mathbb{E}[L | L > \text{VaR}_\alpha(L)] = \frac{\mathbb{E}\left[ L \mathbf{1}_{\{L > \text{VaR}_\alpha(L)\}} \right]}{\mathbb{P}[L > \text{VaR}_\alpha(L)]} \\
&= \frac{1}{1 - \alpha} \mathbb{E}\left[ L \mathbf{1}_{\{L > \text{VaR}_\alpha(L)\}} \right].
\end{aligned}
$$

The average loss beyond $\text{VaR}_\alpha(L)$.

It is also called **expected shortfall**.

### Defining the Mean-CVaR Problem

#### The Mean-CVaR Framework

$$
\min_x \text{CVaR}_\alpha\left( -R^\top x \right), \text{ s.t. } \mathbb{E}\left[ R^\top x \right] \geq z, \quad e^\top x = 1.
$$

### Numerical Solution via Linear Programming

#### Stochastic Optimization and Monte Carlo Simulation

**Stochastic optimization.**
$$
\begin{aligned}
\min_x \quad & \mathbb{E}[f(x, \xi)] \\
\text{s.t.} \quad & x \in \mathcal{X}.
\end{aligned}
$$

**Monte Carlo simulation** : take a sample $\xi_1, \dots, \xi_N$.
$$
\begin{aligned}
\min_x \quad & \frac{1}{N} \sum_{i=1}^N f(x, \xi_i) \\
\text{s.t.} \quad & x \in \mathcal{X}
\end{aligned}
$$

#### A Key Result

An **alternative formulation of CVaR** :
$$
\text{CVaR}_\alpha(L) = \min_{\xi \in \mathbb{R}} \left[ \xi + \frac{1}{1 - \alpha} \mathbb{E}[\max\{L - \xi, 0\}] \right],
$$
The minimum is attained at $\xi = \text{VaR}_\alpha(L)$.

Then the **Mean-CVaR** is reduced to:
$$
\min_{x, \xi} \left[ \xi + \frac{1}{1 - \alpha} \mathbb{E}[\max\{-R^\top x - \xi, 0\}] \right], \text{ s.t. } \mathbb{E}\left[ R^\top x \right] \geq z, \quad e^\top x = 1.
$$

#### Approximate Objective and Constraints

Take $N$ samples of returns.

The realized return for $R_i$ in sample $j$ is $r_{ij}$.
$$
\min_{\xi, x} \xi + \frac{1}{(1 - \alpha)N} \sum_{j=1}^N \max\{-r_j^\top x - \xi, 0\},
$$
subject to,
$$
\begin{aligned}
\bar{R}^\top x &\geq z, \\
e^\top x &= 1.
\end{aligned}
$$
Here, $\bar{R} = \frac{1}{N} \sum_{j=1}^N r_j$.

#### Mean-CVaR Optimization as Linear Programming

Let $u_j = \max\{-r_j^\top x - \xi, 0\}$ as new decision variables.
$$
\min_{\xi, x, u} \xi + \frac{1}{(1 - \alpha)N} \sum_{j=1}^N u_j,
$$
Additional constraints:
$$
\begin{aligned}
u_j &\geq -\sum_{i=1}^n x_i r_{ij} - \xi \\
u_j &\geq 0, \quad j = 1, \dots, N.
\end{aligned}
$$

#### A Simulation Study

Consider a market with 10 stocks such that $R \sim \mathcal{N}(\mu, \Sigma)$. Estimate the mean-CVaR frontier from the 1000 periods of historical data, and plug in the simulated returns to compute weights of efficient portfolios.

![[Pasted image 20260412182543.png]]
*Figure: Mean-CVaR Frontier*