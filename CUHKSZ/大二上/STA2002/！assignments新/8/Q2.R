# Data input
n <- 1919     # Total number of deaths
X <- 922      # Number of deaths before the holiday

# Calculate the sample proportion (theta_hat)
theta_hat <- X / n

# Calculate the estimated standard error (se_hat)
se_hat <- sqrt(theta_hat * (1 - theta_hat) / n)

# Null hypothesis proportion (theta_0)
theta_0 <- 0.5

# Calculate the test statistic (z)
# For a one-sided test where H1: theta < 0.5
z <- (theta_hat - theta_0) / se_hat

# Calculate the p-value for the one-sided test
p_value <- pnorm(z)

# Define the significance level (alpha)
alpha <- 0.05

# Calculate the critical value (z_alpha) for the one-sided 95% confidence interval
z_alpha <- qnorm(1 - alpha)

# Calculate the upper limit of the one-sided confidence interval for theta
theta_upper <- theta_hat + z_alpha * se_hat

# Output the results
cat("Estimated theta (θ̂):", round(theta_hat, 3), "\n")
cat("Estimated Standard Error (SE):", round(se_hat, 4), "\n")
cat("Test Statistic (z):", round(z, 3), "\n")
cat("p-value:", round(p_value, 3), "\n")
cat("One-sided 95% Confidence Interval for θ: (0,", round(theta_upper, 3), ")\n")