# Problem Set 3
*MFE5340 - AI in FE: Quantitative Investment*

Spring 2026

**General Instructions:**

This assignment contains 2 problems. This assignment is due by **23:59:59 on April 5, 2026 (Sunday)**. Late submissions will incur a penalty unless prior arrangements have been made.

**Ethical and Responsible Usage of AI**

While you are encouraged to explore modern tools such as AI and machine learning platforms to enhance your learning, it is essential to use these resources responsibly. All solutions must be your original work. You may use AI to assist with understanding concepts or to guide you, but copying AI-generated answers directly without comprehension is not permitted. Always strive to understand the underlying methods and derivations behind your solutions.

**Additional Reminders:**

1. Independent Work: Each student must submit their own work. Direct copying from others is considered academic misconduct.
2. Formatting: Please submit your answers as a typed document (preferably in PDF format) or as clearly scanned handwritten work on BB.
3. References: If you use any external sources, including textbooks, articles, or online resources, make sure to cite them appropriately.

If you have any questions or need clarification on any of the questions, feel free to reach out before the due date.

**Good Luck!**

## Problem 1 True/False Questions (30 Points)

**Scoring Rules:** Each question is worth 3 points:

* To receive full credit (3 points) for each question, you need to provide the correct answer **and a correct explanation**.
* If the correct answer is selected **without an explanation**, or the explanation is incorrect, only 1 points will be awarded.
* If the answer is incorrect, no points (0 points) will be awarded, regardless of the explanation.

**NOTE:** We only expect concise answers. Focus on the core reasoning and avoid lengthy explanations. Below is an example of the expected answer format:

* Example Question: The capital asset pricing model (CAPM) implies that the market portfolio lies on the mean-variance frontier. (True / False)
* Answer: True. Under CAPM assumptions, the market portfolio is efficient and lies on the mean-variance frontier.

**Start of Questions:**

1. When predictors are highly multicollinear, the OLS estimator remains unbiased but its variance can be very large, leading to poor out-of-sample prediction performance. (True / False)

2. The regularization parameter $\lambda$ in Ridge and Lasso regression should be chosen to minimize in-sample training error. (True / False)

3. In PCA, the first principal component is the linear combination of variables that maximizes variance, subject to the loading vector having unit norm. (True / False)

4. Risk Premium PCA differs from standard PCA in that it only targets the cross-sectional variation in mean returns rather than total return variance. (True / False)

5. In the Chinese stock market, the TTM (Trailing Twelve Months) value of an income statement item, when the latest available report is not an annual report, equals the current latest report plus last year’s annual report minus the corresponding period from last year. (True / False)

6. Forward price adjustment maintains current prices unchanged while adjusting historical prices; backward adjustment keeps historical prices unchanged and adjusts current prices. (True / False)

7. A linear autoencoder and PCA solve the same optimization problem and therefore always produce identical latent representations. (True / False)

8. The conditional autoencoder of Gu, Kelly and Xiu (2021) extends the standard autoencoder by allowing factor loadings ($\beta$) to vary over time as a function of firm-specific characteristics, while keeping the latent factors ($f_t$) fixed and time-invariant. (True / False)

9. Empirical evidence suggests that polynomial terms (e.g., squared or cubed features) are the most important source of nonlinearity in predicting future stock returns, outperforming interaction terms between covariates. (True / False)

10. If the activation function in a neural network is linear, the network reduces to ordinary least squares (OLS) regression, regardless of the number of hidden layers. (True / False)

## Problem 2 Factor Zoo or Noise Zoo? (70 Points)

You just joined a quantitative research team as a junior analyst. Your manager hands you a dataset and says:

> *“We think profitability factors drive cross-sectional returns. Run some regressions and tell me which factors actually matter — and which ones are just noise.”*

You have monthly panel data covering a large universe of stocks. For each stock-month, the following features are available **at the end of the month** (i.e., no look-ahead bias):

| Variable | Description |
| :--- | :--- |
| `roic` | Return on Invested Capital |
| `roa` | Return on Assets |
| `roe` | Return on Equity |
| `bm` | Book-to-Market Ratio |
| `log_mcap` | Log of Market Capitalization |

The outcome variable `ret` is the **stock return in the following month**. Your task is to evaluate three estimation approaches — **OLS, Ridge, and Lasso** — and help your manager decide which factors to trust.

**Data Preparation (already completed for you).** The features provided have been preprocessed as follows: within each month, all features are first **winsorized** at the 1st and 99th percentiles (values are clipped, not dropped), and then **standardized** to zero mean and unit variance cross-sectionally. You may use the processed data directly without any further transformation.

### Part (a) OLS Baseline (10 points)

For each month $t$, run a cross-sectional OLS regression:

$$
r_{i,t+1} = \alpha_t + \beta_1 \mathtt{roic}_{i,t} + \beta_2 \mathtt{roa}_{i,t} + \beta_3 \mathtt{roe}_{i,t} + \beta_4 \mathtt{bm}_{i,t} + \beta_5 \mathtt{log\_mcap}_{i,t} + \varepsilon_{i,t}. \tag{1}
$$

(i) For each feature, compute the **time-series mean** of the monthly OLS coefficients and the corresponding **Newey–West $t$ -statistic** (lag length $L = \lfloor 4 \cdot (T/100)^{2/9} \rfloor$). Report these in a table. Which features appear statistically significant? *(5 points)*

(ii) Look carefully at the signs and magnitudes of the `roic` and `roa` coefficients. Do they make economic sense? What might explain the pattern you observe? *(5 points)*

> *Hint: Think about what happens when two features carry very similar information.*

### Part (b) Diagnosing Multicollinearity (10 points)

(i) Compute the **average cross-sectional correlation matrix** across all months. Report the full matrix and identify the most collinear **features or groups of features**. How does this relate to the coefficient signs in Part (a)(ii)? *(5 points)*

> *Hint: When computing the correlation matrix, ensure that each monthly cross-section uses only stocks for which **all** features are non-missing — i.e., apply the same `dropna()` filter used in your monthly regressions. This ensures the reported correlations reflect the actual estimation sample.*

(ii) Explain intuitively why multicollinearity causes OLS to produce **unstable and potentially misleading** coefficient estimates. A clear economic or geometric argument is sufficient; no derivations are required. *(5 points)*

### Part (c) Ridge Regression (15 points)

For each month $t$, solve the Ridge regression problem:

$$
\hat{\beta}^{\text{ridge}} = \arg\min_{\beta} \left[ \sum_i \left( r_{i,t+1} - \alpha - \beta^\top x_{i,t} \right)^2 + \lambda_{\text{ridge}} ||\beta||_2^2 \right]. \tag{2}
$$

Use $\lambda_{\text{ridge}} = 30.0$.

(i) Report the time-series mean coefficients and Newey–West $t$ -statistics for Ridge. *(10 points)*

(ii) Does Ridge resolve the sign problem for `roic` and `roa`? Why or why not? *(5 points)*

### Part (d) Lasso Regression (25 points)

For each month $t$, solve the Lasso regression problem:

$$
\hat{\beta}^{\text{lasso}} = \arg\min_{\beta} \left[ \sum_i \left( r_{i,t+1} - \alpha - \beta^\top x_{i,t} \right)^2 + \lambda_{\text{lasso}} ||\beta||_1 \right]. \tag{3}
$$

Use $\lambda_{\text{lasso}} = 0.003$. A coefficient is considered “selected” if its absolute value exceeds $10^{-6}$.

(i) For each feature, report the time-series mean coefficient (over **all** months), the Newey–West $t$ -statistic, and the **selection rate** (fraction of months with a non-zero coefficient). *(10 points)*

(ii) Re-compute the mean coefficient and Newey–West $t$ -statistic using **only the months in which each feature was selected** by Lasso. Report these alongside the full-sample statistics from (i) and discuss what the difference reveals about how Lasso handles low-information versus high-information features. *(5 points)*

(iii) Part (d)(i) reports a selection rate for each feature. Part (b) identified a group of highly collinear profitability features. Your manager notices that some features in this group have much lower selection rates than others, and concludes that they must be weaker predictors. Do you agree? Write a two-to-three sentence response suitable for a research memo. *(10 points)*

### Part (e) Synthesis (10 points)

(i) Fill in the summary table below using your results from Parts (a)–(d):

| Feature | OLS $t$ -stat | Ridge $t$ -stat | Lasso $t$ -stat | Lasso Selection Rate |
| :--- | :--- | :--- | :--- | :--- |
| `roic` | | | | |
| `roa` | | | | |
| `roe` | | | | |
| `bm` | | | | |
| `log_mcap` | | | | |

*(5 points)*

(ii) Based on all three methods, which features would you recommend including in the final factor model? Justify your answer. For each feature you recommend, state whether its estimated coefficient sign is consistent with the direction documented in the academic literature (e.g., the value premium for `bm`, the size effect for `log_mcap`, and profitability for the retained profitability measure), and briefly note any discrepancy. *(5 points)*

**Implementation Notes:**

* Use `sklearn.linear_model.Ridge` and `sklearn.linear_model.Lasso` with `fit_intercept=True`.
* For Newey–West standard errors, use `statsmodels.stats.sandwich_covariance` or equivalent.
* Hyperparameters: $\lambda_{\text{ridge}} = 30.0$, $\lambda_{\text{lasso}} = 0.003$.
* Please include all code in your submission, with comments for clarity and reproducibility.