# Problem Set 1
**MFE5340 - AI in FE: Quantitative Investment**

Spring 2026

**General Instructions:**

This assignment contains 4 problems. This assignment is due by **23:59:59 on February 1, 2025 (Sunday)**. Late submissions will incur a penalty unless prior arrangements have been made.

**Ethical and Responsible Usage of AI**

While you are encouraged to explore modern tools such as AI and machine learning platforms to enhance your learning, it is essential to use these resources responsibly. All solutions must be your original work. You may use AI to assist with understanding concepts or to guide you, but copying AI-generated answers directly without comprehension is not permitted. Always strive to understand the underlying methods and derivations behind your solutions.

**Additional Reminders:**

1. **Independent Work:** Each student must submit their own work. Direct copying from others is considered academic misconduct.
2. **Formatting:** Please submit your answers as a typed document (preferably in PDF format) or as clearly scanned handwritten work on BB.
3. **References:** If you use any external sources, including textbooks, articles, or online resources, make sure to cite them appropriately.

If you have any questions or need clarification on any of the questions, feel free to reach out before the due date.

**Good luck!**

***

## Problem 1 True/False Questions (30 Points)

**Scoring Rules:** Each question is worth 3 points:

* To receive full credit (3 points) for each question, you need to provide the correct answer **and a correct explanation**.
* If the correct answer is selected **without an explanation**, or the explanation is incorrect, only 1 points will be awarded.
* If the answer is incorrect, no points (0 points) will be awarded, regardless of the explanation.

**NOTE:** We only expect concise answers. Focus on the core reasoning and avoid lengthy explanations. Below is an example of the expected answer format:

* **Example Question:** The capital asset pricing model (CAPM) implies that the market portfolio lies on the mean-variance frontier. (True / False)
* **Answer:** True. Under CAPM assumptions, the market portfolio is efficient and lies on the mean-variance frontier.

**Start of Questions:**

1. In mean-variance analysis, the mean-variance frontier is the set of all portfolios with the lowest possible variance for each given level of expected return. (True / False)
2. The mean-variance efficient frontier is always a straight line, regardless of whether a risk-free asset is included. (True / False)
3. The Capital Market Line (CML) applies to all portfolios. (True / False)
4. According to the two-fund separation theorem, all investors, regardless of their risk preferences, will hold the same two portfolios in the same proportions. (True / False)
5. If a risk-free asset exists, the mean-variance optimization requires the constraint that risky asset weights sum to one, i.e., $w^\top \mathbf{1} = 1$. (True / False)
6. The Hansen–Jagannathan bound implies that an asset achieves the maximum Sharpe ratio when it is perfectly positively correlated with the SDF. (True / False)
7. In portfolio sorting, the sorting variable is assumed to be correlated with (not necessarily equal to) the true factor exposure. (True / False)
8. Independent double sorting can create portfolios with very few stocks when the two sorting variables are highly correlated, which can make factor returns more sensitive to outliers. (True / False)
9. The Newey-West estimator is necessary only when residuals are autocorrelated; if residuals are heteroscedastic but serially independent, Newey-West is invalid and White’s estimator must be used instead. (True / False)
10. In a time-series regression used to test an anomaly (intercept $\alpha$), the OLS $t$-statistic for $\hat{\alpha}$ is always reliable as long as the number of months $T$ is large. (True / False)

***

## Problem 2 (10 Points)

Consider a portfolio consisting of both risky assets and a risk-free asset with constant return $R_f$. Let $\boldsymbol{w}$ represent the vector of weights for the *risky* assets. The weight on the risk-free asset is $1 - \boldsymbol{w}^\top \mathbf{1}$. Besides, let $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ be the expected return vector and covariance matrix of the risky assets.

Then, the expected return of the portfolio is given by $\boldsymbol{w}^\top \boldsymbol{\mu} + (1 - \boldsymbol{w}^\top \mathbf{1}) R_f$; the variance of the portfolio is given by $\boldsymbol{w}^\top \boldsymbol{\Sigma} \boldsymbol{w}$. For notational convenience, we write expected excess return as $\boldsymbol{\mu}^e = \boldsymbol{\mu} - R_f \mathbf{1}$.

The goal is to find the optimal weights $\boldsymbol{w}$, denoted by $\boldsymbol{w}^\star$, that minimize the variance of the portfolio subject to a target expected return $\mu^\star$. In other word, you will need to solve the following optimization problem:

$$\min_{\boldsymbol{w}} \boldsymbol{w}^\top \boldsymbol{\Sigma} \boldsymbol{w} \quad \text{subject to} \quad \boldsymbol{w}^\top \boldsymbol{\mu}^e = \mu^\star - R_f.$$

Show that the optimal weights are:

$$\boxed{\boldsymbol{w}^\star = \left( \frac{\mu^\star - R_f}{(\boldsymbol{\mu}^e)^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}^e} \right) \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}^e.}$$

***

## Problem 3 (30 Points)

#### Task Background

In practical factor investing, literature replication is often required. Assume you are a quantitative researcher tasked by your manager to replicate the findings of Piotroski, J. D. and E. C. So’s paper, “Identifying expectation errors in value/glamour strategies: A fundamental analysis approach” (*Review of Financial Studies*, vol. 25, 2012, pp. 2841-2875). The paper is also provided along with this assignment.

#### Background on Expectation Errors

According to the mispricing explanation, value stocks outperform growth stocks because market participants underestimate the former and overestimate the latter. Here, overestimation and underestimation refer to prices relative to intrinsic value. Prices reflect investors’ market expectations, while intrinsic value reflects a stock’s fundamental expectations. The discrepancy between the two is defined by Piotroski and So as expectation errors. They argue that the outperformance of value stocks is due to the correction of these errors, using the F-Score to measure intrinsic value and the book-to-market ratio (BM) for market expectations. They employ a double portfolio sort to illustrate these expectation errors (refer to Figure on page 2847 of the original paper). Your task is as follows:

1. **F-Score and BM Independent Double Sort**
   At each cross-section, use the provided F-score data (`fscore.csv`) to categorize firms into three groups: 0 to 3 (Low), 4 to 6 (Middle), and 7 to 9 (High). For the book-to-market ratio (`bm.csv`), use the 30th and 70th percentiles for grouping.
2. **Construct Expectation Error Portfolio**
   Read the paper carefully and construct the expectation error portfolio. Note that this portfoilo is a long-short hedge portfolio (make sure you use the correct groups for long and short legs!).
3. **Statistical Test**
   Calculate the monthly average excess return of the expectation error portfolio and use it to perform a $t$-test. For the $t$-test, you may use the OLS formula for standard error **without** Newey-West adjustments. For simplicity, use **equal weighting** for the stocks within each group. Report the $t$-statistic and the corresponding $p$-value.

When submitting your assignment, please include all relevant code used for your analysis. Ensure that your code is well-commented for clarity and reproducibility.

***

## Problem 4 (30 Points)

In the lecture, we discussed how the profitability factor (constructed using ROE(TTM)) does not earn significant excess returns in the Chinese stock market due to its close relationship with market capitalization, which causes it to be influenced by the size factor. However, we have observed through double portfolio sort that controlling for market capitalization to some extent enables the profitability factor to generate significant excess returns.

In this problem, we take a different approach using the Fama-MacBeth regression. You are provided with the necessary data, including ROE(TTM), market capitalization, and stock returns (all in separate pkl files). Your task is as follows:

1. At the end of each month $t$, perform a cross-sectional regression of $t + 1$ stock returns on the two firm characteristics: ROE(TTM) and market capitalization (include an intercept term in the regression).
	**NOTE:** To mitigate the impact of the distribution of the original variables, for each period, transform both variables as follows:
	* Rank the variables within each cross-section.
	* Map the ranks to a uniform distribution between $[-1, 1]$, where the smallest rank corresponds to $-1$ and the largest rank corresponds to $1$.
2. Test whether the time-series mean of the regression coefficient for ROE(TTM) is significantly different from zero. When calculating the standard error, there is no need to perform Newey-West adjustment.

**IMPORTANT:** Please address the issue of missing data within each cross-section, rather than across the entire panel. Additionally, note that the two variables at time $t$ (as well as the return data at $t + 1$) may have different stock coverage. Therefore, ensure that the stocks are properly aligned across these three datasets at each cross-section before performing any transformations on the two variables.

**Deliverables:**

* Provide the time-series average of the regression coefficient for ROE(TTM) and its associated $t$-statistic.
* Submit your code and a brief explanation of your methodology, including how you transformed the variables and conducted the significance test.