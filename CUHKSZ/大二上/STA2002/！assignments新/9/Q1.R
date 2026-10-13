# --------------------------------------------------------------
# Poisson Parameter Testing: Likelihood Ratio, Wald, and Score Tests
# --------------------------------------------------------------

# ---------------------------
# Part iv: Given Dataset and Hypothesis
# ---------------------------

# Given dataset
data <- c(4, 1, 1, 2, 1, 3, 1, 3, 4, 3, 4, 0, 2, 2, 2, 1, 0, 1, 2, 3)

# Hypothesized lambda under H0
lambda0 <- 1.0

# Sample size
n <- length(data)

# Sum of observations
sum_X <- sum(data)

# Sample mean (MLE of lambda)
lambda_hat <- mean(data)

# Display basic statistics
cat("----- Part iv: Given Data and Basic Statistics -----\n")
cat("Data:", data, "\n")
cat("Sample Size (n):", n, "\n")
cat("Sum of Observations (ΣX_i):", sum_X, "\n")
cat("Sample Mean (λ̂):", lambda_hat, "\n\n")

# ---------------------------
# Part i: Likelihood Ratio Test
# ---------------------------

# Function to compute log-likelihood for Poisson
log_likelihood <- function(lambda, data) {
  sum(dpois(data, lambda, log = TRUE))
}

# Log-likelihood under H0
LL0 <- log_likelihood(lambda0, data)

# Log-likelihood under MLE (lambda_hat)
LL_hat <- log_likelihood(lambda_hat, data)

# Likelihood Ratio Test Statistic: Λ = -2 * (LL0 - LL_hat)
Lambda_stat <- -2 * (LL0 - LL_hat)

# P-value using chi-square distribution with 1 df
p_value_LRT <- pchisq(Lambda_stat, df = 1, lower.tail = FALSE)

# Display Likelihood Ratio Test results
cat("----- Part i: Likelihood Ratio Test -----\n")
cat("Log-Likelihood under H0 (LL0):", LL0, "\n")
cat("Log-Likelihood under MLE (LL̂):", LL_hat, "\n")
cat("Test Statistic (Λ):", Lambda_stat, "\n")
cat("P-value:", p_value_LRT, "\n")

# Decision at 5% significance level
if(p_value_LRT < 0.05){
  cat("Result: Reject H0 at 5% significance level.\n\n")
} else {
  cat("Result: Do not reject H0 at 5% significance level.\n\n")
}

# ---------------------------
# Part ii: Wald Test
# ---------------------------

# Variance of lambda_hat under H0
var_lambda_hat <- lambda0 / n

# Wald Test Statistic: W = (λ̂ - λ0)^2 / Var(λ̂)
Wald_stat <- (lambda_hat - lambda0)^2 / var_lambda_hat

# P-value using chi-square distribution with 1 df
p_value_Wald <- pchisq(Wald_stat, df = 1, lower.tail = FALSE)

# Display Wald Test results
cat("----- Part ii: Wald Test -----\n")
cat("Variance of λ̂ under H0 (Var(λ̂)):", var_lambda_hat, "\n")
cat("Test Statistic (W):", Wald_stat, "\n")
cat("P-value:", p_value_Wald, "\n")

# Decision at 5% significance level
if(p_value_Wald < 0.05){
  cat("Result: Reject H0 at 5% significance level.\n\n")
} else {
  cat("Result: Do not reject H0 at 5% significance level.\n\n")
}

# ---------------------------
# Part iii: Score Test
# ---------------------------

# Score Function U(λ0) = Σ(X_i - λ0) / λ0
score_U <- (sum_X - n * lambda0) / lambda0

# Fisher Information I(λ0) = n / λ0
fisher_info <- n / lambda0

# Score Test Statistic: (U(λ0))^2 / I(λ0)
Score_stat <- (score_U)^2 / fisher_info

# Alternatively, Score_stat can be computed as:
# Score_stat <- (sum_X - n * lambda0)^2 / (n * lambda0)

# P-value using chi-square distribution with 1 df
p_value_Score <- pchisq(Score_stat, df = 1, lower.tail = FALSE)

# Display Score Test results
cat("----- Part iii: Score Test -----\n")
cat("Score Function U(λ0):", score_U, "\n")
cat("Fisher Information I(λ0):", fisher_info, "\n")
cat("Test Statistic (Score):", Score_stat, "\n")
cat("P-value:", p_value_Score, "\n")

# Decision at 5% significance level
if(p_value_Score < 0.05){
  cat("Result: Reject H0 at 5% significance level.\n\n")
} else {
  cat("Result: Do not reject H0 at 5% significance level.\n\n")
}

# ---------------------------
# Summary of All Tests
# ---------------------------

cat("----- Summary of Tests -----\n")
cat("Likelihood Ratio Test: Statistic =", Lambda_stat, ", P-value =", p_value_LRT, "\n")
cat("Wald Test: Statistic =", Wald_stat, ", P-value =", p_value_Wald, "\n")
cat("Score Test: Statistic =", Score_stat, ", P-value =", p_value_Score, "\n\n")

if(p_value_LRT < 0.05){
  cat("Likelihood Ratio Test: Reject H0 at 5% significance level.\n")
} else {
  cat("Likelihood Ratio Test: Do not reject H0 at 5% significance level.\n")
}

if(p_value_Wald < 0.05){
  cat("Wald Test: Reject H0 at 5% significance level.\n")
} else {
  cat("Wald Test: Do not reject H0 at 5% significance level.\n")
}

if(p_value_Score < 0.05){
  cat("Score Test: Reject H0 at 5% significance level.\n")
} else {
  cat("Score Test: Do not reject H0 at 5% significance level.\n")
}