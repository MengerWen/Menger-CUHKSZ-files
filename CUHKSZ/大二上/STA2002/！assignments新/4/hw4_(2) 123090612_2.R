# Parameters
n <- 100
lambda_true <- 2

# Simulate Exponential data
Y <- rexp(n, rate = lambda_true)

# Sample mean
Y_bar <- mean(Y)

# Confidence level
conf_level <- 0.90
alpha <- 1 - conf_level

# Chi-squared quantiles
chi_sq_lower <- qchisq(alpha / 2, df = 2 * n)
chi_sq_upper <- qchisq(1 - alpha / 2, df = 2 * n)

# Compute confidence interval for lambda
lambda_lower <- (2 * n) / (chi_sq_upper * Y_bar)
lambda_upper <- (2 * n) / (chi_sq_lower * Y_bar)

# Display results
cat("90% Confidence Interval for lambda: [", lambda_lower, ", ", lambda_upper, "]\n")
cat("Does the interval contain the true lambda (2)? ", 
    ifelse(lambda_lower <= lambda_true & lambda_upper >= lambda_true, "Yes", "No"), "\n")
