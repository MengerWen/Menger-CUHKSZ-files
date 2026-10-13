# Problem Set 2
*MFE5340 - AI in FE: Quantitative Investment*

Spring 2026

**General Instructions:**

This assignment contains 2 problems. This assignment is due by **23:59:59 on March 8, 2026 (Sunday)**. Late submissions will incur a penalty unless prior arrangements have been made.

**Ethical and Responsible Usage of AI**

While you are encouraged to explore modern tools such as AI and machine learning platforms to enhance your learning, it is essential to use these resources responsibly. All solutions must be your original work. You may use AI to assist with understanding concepts or to guide you, but copying AI-generated answers directly without comprehension is not permitted. Always strive to understand the underlying methods and derivations behind your solutions.

**Additional Reminders:**

1. Independent Work: Each student must submit their own work. Direct copying from others is considered academic misconduct.
2. Formatting: Please submit your answers as a typed document (preferably in PDF format) or as clearly scanned handwritten work on BB.
3. References: If you use any external sources, including textbooks, articles, or online resources, make sure to cite them appropriately.

If you have any questions or need clarification on any of the questions, feel free to reach out before the due date.

**Good Luck!**

***

# Problem 1 True/False Questions (30 Points)

**Scoring Rules:** Each question is worth 3 points:

- To receive full credit (3 points) for each question, you need to provide the correct answer **and a correct explanation**.
- If the correct answer is selected **without an explanation**, or the explanation is incorrect, only 1 points will be awarded.
- If the answer is incorrect, no points (0 points) will be awarded, regardless of the explanation.

**NOTE:** We only expect concise answers. Focus on the core reasoning and avoid lengthy explanations. Below is an example of the expected answer format:

- Example Question: The capital asset pricing model (CAPM) implies that the market portfolio lies on the mean-variance frontier. (True / False)
- Answer: True. Under CAPM assumptions, the market portfolio is efficient and lies on the mean-variance frontier.

**Start of Questions:**

1.  Econometrics focuses on predictive accuracy, while machine learning prioritizes explaining causal relationships between variables. (True / False)
2.  The goal of the Stochastic Discount Factor (SDF) framework discussed in the lecture is to minimize in-sample pricing errors of published anomalies. (True / False)
3.  In machine learning, regularization is used to reduce model complexity and prevent overfitting by adding a penalty term to the loss function. (True / False)
4.  In multiple hypothesis testing, the probability of making at least one false discovery increases as the number of hypotheses tested increases. (True / False)
5.  The $p$ -value represents the probability that the null hypothesis is true given the observed data. (True / False)
6.  The Bayes Factor measures the marginal likelihood ratio of the observed data under the null hypothesis compared to the alternative hypothesis. (True / False)
7.  Controlling the False Discovery Rate (FDR) is stricter than controlling the Family-Wise Error Rate (FWER). (True / False)
8.  Reducing the Type I error rate (false positive rate) in multiple hypothesis testing will always lead to a reduction in the Type II error rate (false negative rate). (True / False)
9.  Block bootstrap is primarily used to retain cross-sectional dependencies. (True / False)
10. In the double-bootstrap procedure of Harvey and Liu (2020), the second-stage bootstrap uses the original sample dataset as the population for resampling. (True / False)

***

# Problem 2 Double Bootstrap (70 Points)

In class, we covered the double bootstrap methodology as proposed in Harvey and Liu (2020). The goal of this problem is to replicate the double bootstrap process, using a dataset of 100+ factor return series provided (`factor_returns.csv`). You may reference the original paper (also provided) for additional guidance on the implementation.

**Tasks:**

- Implement the double bootstrap method, with 100 iterations for each stage of bootstrap (that is, $I = J = 100$ and a total of 10,000 bootstraps).
- Calculate Type 1 error (false discovery rate), Type 2 error (miss rate), and the oratio (ratio of false discoveries to misses) for each scenario.
- Plot Type 1, Type 2, and oratio for $p_0 = 5\%, 10\%$ and $20\%$ as a function of $t$ -statistic cutoff ranging from 2.0 to 3.0 with a step size of 0.1.
- For each $p_0$ value, calculate the $t$ -statistic cutoff that controls the Type 1 error rate to be no higher than 5%.

**Deliverables:**

- **(45 Points)** Three plots (each for 20 points) for three $p_0$ values considered. Each plot should show how type1, type2, and oratio vary with $t$ -statistic cutoff. *(Hint: Your results should look similar to those on lecture slides.*
- **(25 Points)** The $t$ -statistic cutoff values that control the type 1 error rate under 5% for all three $p_0$ s, and a discussion of the results.

**IMPORTANT:** Make sure to use **block bootstrap**. The recommended block window is 4, but you are free to choose other values. Also, make sure to bootstrap factors together to retain cross-sectional correlations. Be mindful of computational efficiency; each round of bootstrap is limited to 100 iterations to save time.

Please submit all relevant code used for your analysis. Ensure that your code is well-commented for clarity and reproducibility.