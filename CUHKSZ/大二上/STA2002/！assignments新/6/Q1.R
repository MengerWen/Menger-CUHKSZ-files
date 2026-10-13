# Question i: Calculate an approximate 95% two-sided confidence interval
cat("Question i:\n")
# for the difference of the mean tensile strengths of the two rubber compounds μ_A - μ_B.

# Data for Compound A and Compound B
compound_A <- c(32, 30, 33, 32, 29, 34, 32)
compound_B <- c(33, 35, 36, 37, 35)

# Step 1: Calculate sample sizes
n_A <- length(compound_A)
n_B <- length(compound_B)
cat("Sample sizes:\n")
cat("n_A =", n_A, "\n")
cat("n_B =", n_B, "\n\n")

# Step 2: Calculate sample means
mean_A <- mean(compound_A)
mean_B <- mean(compound_B)
cat("Sample means:\n")
cat("Mean_A =", round(mean_A, 4), "\n")
cat("Mean_B =", round(mean_B, 4), "\n\n")

# Step 3: Calculate sample variances
var_A <- var(compound_A)
var_B <- var(compound_B)
cat("Sample variances:\n")
cat("Var_A =", round(var_A, 4), "\n")
cat("Var_B =", round(var_B, 4), "\n\n")

# Step 4: Calculate the standard error (SE) of the difference of means
SE <- sqrt(var_A / n_A + var_B / n_B)
cat("Standard Error (SE) of the difference of means:", round(SE, 4), "\n\n")

# Step 5: Calculate degrees of freedom using Welch-Satterthwaite equation
df_num <- (var_A / n_A + var_B / n_B)^2
df_den <- ( (var_A / n_A)^2 / (n_A - 1) ) + ( (var_B / n_B)^2 / (n_B - 1) )
df <- df_num / df_den
cat("Degrees of Freedom (df):", round(df, 4), "\n\n")

# Step 6: Calculate the critical t-value for a 95% confidence interval
alpha <- 0.05
t_critical <- qt(1 - alpha / 2, df)
cat("Critical t-value (two-tailed, α = 0.05):", round(t_critical, 4), "\n\n")

# Step 7: Calculate the margin of error
ME <- t_critical * SE
cat("Margin of Error (ME):", round(ME, 4), "\n\n")

# Step 8: Calculate the confidence interval
diff_means <- mean_A - mean_B
lower_limit <- diff_means - ME
upper_limit <- diff_means + ME
cat("Difference of means (Mean_A - Mean_B):", round(diff_means, 4), "\n")
cat("95% Confidence Interval: (", round(lower_limit, 4), ",", round(upper_limit, 4), ")\n\n\n")


# Question ii: Derive the upper-side confidence interval for μ_A - μ_B.
cat("Question ii:\n")
# Step 1: Calculate the critical t-value for a one-tailed 95% confidence interval
t_critical_one_tail <- qt(1 - alpha, df)
cat("Critical t-value (one-tailed, α = 0.05):", round(t_critical_one_tail, 4), "\n\n")

# Step 2: Calculate the margin of error for the one-tailed test
ME_one_tail <- t_critical_one_tail * SE
cat("One-Tailed Margin of Error (ME):", round(ME_one_tail, 4), "\n\n")

# Step 3: Calculate the upper confidence limit
upper_limit_one_tail <- diff_means + ME_one_tail
cat("Upper-side 95% Confidence Interval for μ_A - μ_B is ≤", round(upper_limit_one_tail, 4), "\n")