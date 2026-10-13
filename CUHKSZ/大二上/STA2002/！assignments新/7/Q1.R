# Given values
sigma_squared <- 100       # Population variance
sigma <- sqrt(sigma_squared) # Population standard deviation
n <- 16                    # Sample size
xbar <- 113.5              # Sample mean
mu0 <- 110                 # Null hypothesis mean

# Calculate standard error
SE <- sigma / sqrt(n)

# Calculate Z-statistic
Z <- (xbar - mu0) / SE

# (a) Decision at the 5% significance level
alpha_5 <- 0.05
# Critical value for one-tailed test at 5% significance level
Z_critical_5 <- qnorm(1 - alpha_5)

# Print Z-statistic and critical value for verification
cat("Z-statistic:", Z, "\n")
cat("Critical Z-value at 5% significance level:", Z_critical_5, "\n")

# Determine whether to reject H0 at 5% level
if (Z > Z_critical_5) {
  cat("(a) Reject H0 at the 5% significance level.\n")
} else {
  cat("(a) Do not reject H0 at the 5% significance level.\n")
}

# (b) Decision at the 10% significance level
alpha_10 <- 0.10
# Critical value for one-tailed test at 10% significance level
Z_critical_10 <- qnorm(1 - alpha_10)

# Print critical value for verification
cat("Critical Z-value at 10% significance level:", Z_critical_10, "\n")

# Determine whether to reject H0 at 10% level
if (Z > Z_critical_10) {
  cat("(b) Reject H0 at the 10% significance level.\n")
} else {
  cat("(b) Do not reject H0 at the 10% significance level.\n")
}

# (c) Calculate the p-value
p_value <- 1 - pnorm(Z)

# Print the p-value
cat(sprintf("(c) The p-value is approximately %.4f.\n", p_value))