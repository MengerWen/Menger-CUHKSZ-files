# Parameters
n <- 100
lambda_true <- 2
num_simulations <- 1000
conf_level <- 0.90
alpha <- 1 - conf_level

# Initialize counter
coverage_count <- 0

# Loop over simulations
for (i in 1:num_simulations) {
  # Simulate Exponential data
  Y <- rexp(n, rate = lambda_true)

  # Sample mean
  Y_bar <- mean(Y)

  # Chi-squared quantiles
  chi_sq_lower <- qchisq(alpha / 2, df = 2 * n)
  chi_sq_upper <- qchisq(1 - alpha / 2, df = 2 * n)

  # Compute confidence interval for lambda
  lambda_lower <- (2 * n) / (chi_sq_upper * Y_bar)
  lambda_upper <- (2 * n) / (chi_sq_lower * Y_bar)

  # Check if true lambda is within the interval
  if (lambda_lower <= lambda_true && lambda_upper >= lambda_true) {
    coverage_count <- coverage_count + 1
  }
}

# Calculate coverage rate
coverage_rate <- coverage_count / num_simulations

# Display results
cat("Coverage rate of the 90% CI over", num_simulations, "simulations:", coverage_rate * 100, "%\n")
