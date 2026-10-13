# Setting parameters
alpha <- 0.05  # Type I error rate
sigma_squared <- 1  # Known variance
sigma <- sqrt(sigma_squared)  # Standard deviation

# Collected data
data <- c(-1.78, -0.53, -0.40, 0.15, -0.31, -1.07, -1.32, -0.80, -0.89, -0.40, 
          0.10, 0.67, -1.25, -1.29, 0.04)

# Sample size
n <- length(data)
cat("n =", n, "\n")

# Calculating sample mean
mean_x <- mean(data)
cat("Sample Mean (X̄):", mean_x, "\n")

# Test statistic calculation
z_alpha <- qnorm(1 - alpha)  # Critical value for alpha
test_statistic <- (mean_x - 0) / (sigma / sqrt(n))  # Z statistic
cat("Test Statistic (Z):", test_statistic, "\n")

# Type I Error
type_1_error <- alpha  # By definition
cat("Type I Error (α):", type_1_error, "\n")

# Type II Error
# For the alternative hypothesis H1: mu = 1
beta <- pnorm(z_alpha + 1)  # Probability of failing to reject H0 when H1 is true
cat("Type II Error (β):", beta, "\n")

# Decision on Hypothesis
if (test_statistic > z_alpha) {
  cat("Reject H0: Preference for H1 (μ = 1)\n")
} else {
  cat("Fail to reject H0: Preference for H0 (μ = 0)\n")
}
