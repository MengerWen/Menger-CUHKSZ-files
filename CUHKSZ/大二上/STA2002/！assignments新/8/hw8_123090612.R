# Set the random seed
set.seed(1)

# Define parameters
beta <- 0
sigma <- 2  # Given that sigma^2 = 4

# Define x_i's
x <- c(3.4, 3.6, 3.8, 3.3, 3.4, 3.5, 3.7, 3.6, 3.7)

# Sample size
n <- length(x)

# Generate synthetic data Y_i = beta * x_i + epsilon_i
# Since beta = 0, Y_i = epsilon_i
epsilon <- rnorm(n, mean = 0, sd = sigma)
Y <- beta * x + epsilon  # Since beta = 0, this is just epsilon

# 1. Likelihood Ratio Test for H0: beta = 0 vs. H1: beta != 0

# Under H0: beta = 0
# Estimate sigma^2 under H0 using MLE
sigma2_H0 <- sum((Y - 0)^2) / n  # Since beta = 0 and mu_i = 0

# Log-likelihood under H0
logL_H0 <- -n/2 * log(2 * pi) - n/2 * log(sigma2_H0) - sum((Y - 0)^2) / (2 * sigma2_H0)

# Under H1: beta estimated from data
# Fit linear regression of Y on x without intercept
model <- lm(Y ~ x - 1)  # Removes intercept

# Extract estimated beta
beta_hat <- coef(model)[1]  # Now only beta

# Estimate sigma^2 under H1 using MLE
sigma2_H1 <- sum(residuals(model)^2) / n

# Log-likelihood under H1
logL_H1 <- -n/2 * log(2 * pi) - n/2 * log(sigma2_H1) - sum((Y - beta_hat * x)^2) / (2 * sigma2_H1)

# Likelihood Ratio Test Statistic
D <- -2 * (logL_H0 - logL_H1)

# Degrees of freedom: difference in number of parameters (beta)
df <- 1

# p-value from chi-squared distribution
p_value <- 1 - pchisq(D, df)

# 2. Conclusion
if (p_value < 0.05) {
  conclusion <- "Reject H0: There is evidence that beta is not equal to zero."
} else {
  conclusion <- "Fail to reject H0: There is not enough evidence that beta is different from zero."
}

# Print the results
cat("Likelihood Ratio Test Statistic D:", D, "\n")
cat("Degrees of Freedom:", df, "\n")
cat("p-value:", p_value, "\n")
cat("Estimated beta:", beta_hat, "\n")
cat("Conclusion:", conclusion, "\n")