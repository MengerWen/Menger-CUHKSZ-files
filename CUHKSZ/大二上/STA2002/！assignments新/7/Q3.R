# Part (a): Defining the test statistic and critical region for testing H0: mu_X = mu_Y against H1: mu_X > mu_Y
# Assuming equal variances

# Sample sizes
n_X <- 16  # Number of observations for X (males)
n_Y <- 13  # Number of observations for Y (females)

# Significance level
alpha <- 0.01  # Size of the test

# Degrees of freedom
df <- n_X + n_Y - 2  # Degrees of freedom for the pooled t-test

# Critical value for one-sided t-test at alpha = 0.01
t_critical <- qt(alpha, df, lower.tail = FALSE)

# Display the critical value
cat("Part (a):\n")
cat("Critical value for t at alpha =", alpha, "and df =", df, "is", round(t_critical, 3), "\n\n")

# Part (b): Calculating the test statistic and stating the conclusion

# Given sample means and variances
mean_X <- 415.16  # Sample mean for X
mean_Y <- 347.40  # Sample mean for Y
s2_X <- 1356.75   # Sample variance for X
s2_Y <- 692.21    # Sample variance for Y

# Pooled variance
s_p2 <- ((n_X - 1) * s2_X + (n_Y - 1) * s2_Y) / (n_X + n_Y - 2)
s_p <- sqrt(s_p2)  # Pooled standard deviation

# Standard error
SE <- s_p * sqrt(1 / n_X + 1 / n_Y)

# Test statistic
t_stat <- (mean_X - mean_Y) / SE

# Display the results
cat("Part (b):\n")
cat("Pooled variance (s_p^2):", round(s_p2, 3), "\n")
cat("Pooled standard deviation (s_p):", round(s_p, 3), "\n")
cat("Standard error (SE):", round(SE, 3), "\n")
cat("Test statistic (t):", round(t_stat, 3), "\n")
cat("Critical value (t_critical):", round(t_critical, 3), "\n")

# Conclusion
if (t_stat > t_critical) {
  cat("Conclusion: Reject the null hypothesis H0.\n")
  cat("There is sufficient evidence to conclude that mu_X > mu_Y at the 1% significance level.\n\n")
} else {
  cat("Conclusion: Fail to reject the null hypothesis H0.\n")
  cat("There is not sufficient evidence to conclude that mu_X > mu_Y at the 1% significance level.\n\n")
}

# Part (c): Using Welch's t-test without assuming equal variances

# Standard errors for each group
SE_X <- sqrt(s2_X / n_X)
SE_Y <- sqrt(s2_Y / n_Y)

# Combined standard error for Welch's t-test
SE_welch <- sqrt(s2_X / n_X + s2_Y / n_Y)

# Test statistic for Welch's t-test
t_welch <- (mean_X - mean_Y) / SE_welch

# Degrees of freedom for Welch's t-test
num_df <- (s2_X / n_X + s2_Y / n_Y)^2
den_df <- ((s2_X / n_X)^2) / (n_X - 1) + ((s2_Y / n_Y)^2) / (n_Y - 1)
df_welch <- num_df / den_df

# Critical value for Welch's t-test
t_critical_welch <- qt(alpha, df_welch, lower.tail = FALSE)

# Display the results
cat("Part (c):\n")
cat("Standard error (SE) for Welch's t-test:", round(SE_welch, 3), "\n")
cat("Test statistic (t) for Welch's t-test:", round(t_welch, 3), "\n")
cat("Degrees of freedom (df) for Welch's t-test:", round(df_welch, 3), "\n")
cat("Critical value (t_critical) for Welch's t-test:", round(t_critical_welch, 3), "\n")

# Conclusion
if (t_welch > t_critical_welch) {
  cat("Conclusion: Reject the null hypothesis H0 using Welch's t-test.\n")
  cat("There is sufficient evidence to conclude that mu_X > mu_Y at the 1% significance level without assuming equal variances.\n")
} else {
  cat("Conclusion: Fail to reject the null hypothesis H0 using Welch's t-test.\n")
  cat("There is not sufficient evidence to conclude that mu_X > mu_Y at the 1% significance level without assuming equal variances.\n")
}