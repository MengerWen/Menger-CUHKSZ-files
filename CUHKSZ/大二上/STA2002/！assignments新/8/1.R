# Set random seed
set.seed(1)

# Known x_i values
x_i <- c(3.4, 3.6, 3.8, 3.3, 3.4, 3.5, 3.7, 3.6, 3.7)

# Parameters
beta <- 0
sigma <- 2

# Generate Y_i ~ N(beta * x_i, sigma^2)
Y_i <- rnorm(length(x_i), mean = beta * x_i, sd = sigma)

# Part 1: Finding the likelihood ratio test for H0: beta = 0 vs. H1: beta != 0

# Under H0: beta = 0
beta_0 <- 0
# Calculate residuals under H0
residuals_0 <- Y_i - beta_0 * x_i  # Under H0, residuals are just Y_i
# Estimate sigma^2 under H0
sigma0_sq_hat <- sum(residuals_0^2) / length(Y_i)

# Under H1: beta is estimated from the data
# Fit linear regression model to estimate beta
model <- lm(Y_i ~ x_i)
beta_hat <- coef(model)[2]  # Estimated beta under H1
# Calculate residuals under H1
residuals_1 <- residuals(model)
# Estimate sigma^2 under H1
sigma1_sq_hat <- sum(residuals_1^2) / length(Y_i)

# Compute the likelihood ratio test statistic
lambda <- length(Y_i) * log(sigma0_sq_hat / sigma1_sq_hat)
# Under H0, lambda follows a chi-squared distribution with 1 degree of freedom

# Part 2: Conclusion
# Calculate the p-value
p_value <- 1 - pchisq(lambda, df = 1)

# Print results
cat("Likelihood ratio test statistic (lambda):", lambda, "\n")
cat("p-value:", p_value, "\n")

if(p_value < 0.05) {
  cat("Conclusion: Reject H0: beta = 0 at the 5% significance level.\n")
} else {
  cat("Conclusion: Fail to reject H0: beta = 0 at the 5% significance level.\n")
}