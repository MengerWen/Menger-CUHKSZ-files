# Set parameters
n <- 100 # Sample size
alpha <- 0.05 # Significance level
sigma <- 1 # Known standard deviation
iterations <- 200 # Number of iterations

# Function to perform hypothesis test
perform_test <- function(mu_0, mu_true) {
  # Generate sample data
  X <- rnorm(n, mean = mu_true, sd = sigma)

  # Calculate sample mean
  sample_mean <- mean(X)

  # Calculate Z statistic
  Z <- (sample_mean - mu_0) / (sigma / sqrt(n))

  # Calculate critical value for two-tailed test
  critical_value <- qnorm(1 - alpha)

  # Reject H0 if |Z| > critical_value
  return((Z) > critical_value)
}

# Part i & ii: Testing H0: mu = 0 vs H1: mu = 1
reject_H0_count_0 <- 0

for (i in 1:iterations) {
  if (perform_test(mu_0 = 0, mu_true = 0)) {
    reject_H0_count_0 <- reject_H0_count_0 + 1
  }
}

# Frequency of rejecting H0 for N(0, 1)
reject_H0_rate_0 <- reject_H0_count_0 / iterations
cat("Frequency of rejecting H0 for N(0, 1):", reject_H0_rate_0, "\n") # Output clue

# Part iv: Redo for N(1, 1)
reject_H0_count_1 <- 0

for (i in 1:iterations) {
  if (perform_test(mu_0 = 0, mu_true = 1)) {
    reject_H0_count_1 <- reject_H0_count_1 + 1
  }
}

# Frequency of rejecting H0 for N(1, 1)
reject_H0_rate_1 <- reject_H0_count_1 / iterations
cat("Frequency of rejecting H0 for N(1, 1):", reject_H0_rate_1, "\n") # Output clue
