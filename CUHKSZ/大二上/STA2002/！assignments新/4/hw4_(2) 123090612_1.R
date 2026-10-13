# Set the true value of lambda
lambda_true <- 2

# Sample size
n <- 100

# Significance level
alpha <- 0.10

# Simulate data
set.seed(123)  # For reproducibility
Y <- rexp(n, rate = lambda_true)

# Compute the sum of Y_i
S <- sum(Y)

# Degrees of freedom
df <- 2 * n

# Compute the chi-squared quantiles
chi_sq_lower <- qchisq(alpha / 2, df = df)
chi_sq_upper <- qchisq(1 - alpha / 2, df = df)

# Compute the confidence interval for lambda
lambda_L <- chi_sq_lower / (2 * S)
lambda_U <- chi_sq_upper / (2 * S)

# Output the confidence interval
cat("90% Confidence Interval for lambda: [", lambda_L, ", ", lambda_U, "]\n")

# Check if the true lambda is within the interval
if (lambda_L <= lambda_true && lambda_true <= lambda_U) {
  cat("The 90% CI contains the true value of lambda.\n")
} else {
  cat("The 90% CI does NOT contain the true value of lambda.\n")
}
