# Set the true value of lambda
lambda_true <- 2

# Sample size
n <- 100

# Number of simulations
num_simulations <- 1000

# Significance level
alpha <- 0.10

# Degrees of freedom
df <- 2 * n

# Initialize counter for coverage
coverage_count <- 0


for (i in 1:num_simulations) {
  # Simulate data
  Y <- rexp(n, rate = lambda_true)
  
  # Compute the sum
  S <- sum(Y)
  
  # Compute chi-squared quantiles
  chi_sq_lower <- qchisq(alpha / 2, df = df)
  chi_sq_upper <- qchisq(1 - alpha / 2, df = df)
  
  # Compute confidence interval for lambda
  lambda_L <- chi_sq_lower / (2 * S)
  lambda_U <- chi_sq_upper / (2 * S)
  
  # Check if true lambda is within the interval
  if (lambda_L <= lambda_true && lambda_true <= lambda_U) {
    coverage_count <- coverage_count + 1
  }
}

# Calculate coverage rate
coverage_rate <- coverage_count / num_simulations

# Output the coverage rate
cat("Coverage rate of the 90% CI over", num_simulations, "simulations:", coverage_rate, "\n")
