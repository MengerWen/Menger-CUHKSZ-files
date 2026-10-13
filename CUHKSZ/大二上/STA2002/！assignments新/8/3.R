# Set seed for reproducibility
set.seed(1)

# Define the given x_i's
x <- c(3.4, 3.6, 3.8, 3.3, 3.4, 3.5, 3.7, 3.6, 3.7)
n <- length(x)

# Given parameters
beta <- 0
sigma <- 2

# Generate synthetic data Y_i ~ N(beta * x_i, sigma^2)
Y <- rnorm(n, mean = beta * x, sd = sigma)

# 1. Likelihood ratio test for H0: beta = 0 vs H1: beta ≠ 0

# Under H0: beta = 0
# Estimate sigma^2 under H0
sigma2_0_hat <- mean((Y - 0)^2)

# Under H1: beta is estimated from data
# Estimate beta using the formula beta_hat = (Σ x_i * Y_i) / (Σ x_i^2)
beta_hat <- sum(x * Y) / sum(x^2)

# Compute residuals under H1
e <- Y - beta_hat * x

# Estimate sigma^2 under H1
sigma2_hat <- mean(e^2)

# Compute the likelihood ratio test statistic
LR_stat <- n * log(sigma2_0_hat / sigma2_hat)

# Compute p-value (chi-squared distribution with 1 degree of freedom)
p_value <- 1 - pchisq(LR_stat, df = 1)

# Output the test statistic and p-value
cat("Likelihood Ratio Test Statistic:", LR_stat, "\n")
cat("P-value:", p_value, "\n")

# 2. Conclusion
if(p_value < 0.05){
  cat("Conclusion: Reject H0. There is evidence that beta is not equal to zero.\n")
} else {
  cat("Conclusion: Fail to reject H0. There is no evidence that beta is not equal to zero.\n")
}
