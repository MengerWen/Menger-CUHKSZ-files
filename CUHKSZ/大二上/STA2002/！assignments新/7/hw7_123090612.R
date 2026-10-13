# (1) Input the FVC data
fvc_data <- c(3.4, 3.6, 3.8, 3.3, 3.4, 3.5, 3.7, 3.6, 3.7)

# Hypothesized mean (mu0)
mu0 <- 3.4

# Sample size
n <- length(fvc_data)

# Calculate the sample mean
sample_mean <- mean(fvc_data)

# Calculate the sample standard deviation
sample_sd <- sd(fvc_data)

# Calculate the standard error
SE <- sample_sd / sqrt(n)

# (1) Calculate the test statistic (t-statistic)
t_statistic <- (sample_mean - mu0) / SE

# Print the calculated values for verification
cat("Sample mean:", sample_mean, "\n")
cat("Sample standard deviation:", sample_sd, "\n")
cat("Standard error:", SE, "\n")
cat("t-statistic:", t_statistic, "\n")

# Degrees of freedom
df <- n - 1

# (2) Determine the critical t-value for a one-tailed test at alpha = 0.05
alpha <- 0.05
t_critical <- qt(1 - alpha, df)

# Print the critical t-value
cat("Critical t-value at alpha =", alpha, "and df =", df, "is:", t_critical, "\n")

# (2) Conclusion based on the test statistic and critical value
if (t_statistic > t_critical) {
  cat("(2) Conclusion: Reject the null hypothesis at the", alpha * 100, "% significance level.\n")
} else {
  cat("(2) Conclusion: Do not reject the null hypothesis at the", alpha * 100, "% significance level.\n")
}

# (3) Calculate the p-value for a one-tailed test
p_value <- 1 - pt(t_statistic, df)

# Print the p-value
cat(sprintf("(3) The p-value is approximately %.4f.\n", p_value))