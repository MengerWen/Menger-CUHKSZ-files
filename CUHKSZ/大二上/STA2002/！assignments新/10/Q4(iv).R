# Data vectors
x <- c(6.91, 4.32, 2.38, 7.98, 8.26, 2.00, 3.10, 7.69, 2.21, 3.42,
       8.18, 5.39, 1.19, 3.21, 5.47, 7.35, 2.32, 7.54, 1.27, 7.33,
       8.41, 8.72, 6.09, 5.30, 7.30)

y <- c(17.52, 22.69, 17.61, 14.29, 10.77, 12.87, 18.63, 16.77, 14.97, 19.16,
       11.15, 22.41, 7.50, 19.06, 23.89, 16.63, 15.09, 14.75, 10.75, 17.42,
       9.40, 9.83, 22.33, 21.37, 17.36)

# Create design matrix for quadratic model: Y = beta0 + beta1*X + beta2*X^2 + error
X <- model.matrix(~ x + I(x^2))

# Assume sigma^2 = 1 (known variance)
sigma2 <- 1

# Compute (X'X)^-1
XtX_inv <- solve(t(X) %*% X)

# Calculate beta_hat (MLE for beta)
beta_hat <- XtX_inv %*% t(X) %*% y

# Extract coefficients
beta0_hat <- beta_hat[1,1]
beta1_hat <- beta_hat[2,1]
beta2_hat <- beta_hat[3,1]

# Calculate standard errors for beta coefficients
# Since sigma^2 is known, SE(beta) = sqrt(sigma^2 * diag((X'X)^-1))
se_beta0 <- sqrt(sigma2 * XtX_inv[1,1])
se_beta1 <- sqrt(sigma2 * XtX_inv[2,2])
se_beta2 <- sqrt(sigma2 * XtX_inv[3,3])

# 95% Confidence Interval for beta1
alpha <- 0.05
z_alpha <- qnorm(1 - alpha/2)  # Approximately 1.96 for 95% CI
CI_lower <- beta1_hat - z_alpha * se_beta1
CI_upper <- beta1_hat + z_alpha * se_beta1

# Check if the confidence interval covers 0
covers_zero <- (CI_lower <= 0) & (CI_upper >= 0)

# Compute P-value for H0: beta1 = 0 vs H1: beta1 > 0
# Under H0, beta1_hat ~ N(0, sigma^2 * Var(beta1))
Z <- beta1_hat / se_beta1
p_value <- 1 - pnorm(Z)  # One-sided test

# Display the results
cat("Estimate of beta1:", round(beta1_hat, 4), "\n")
cat("Standard Error of beta1:", round(se_beta1, 4), "\n")
cat("95% Confidence Interval for beta1: [", round(CI_lower, 4), ",", round(CI_upper, 4), "]\n")
cat("Does the interval cover 0?", covers_zero, "\n")
cat("P-value for H0: beta1 = 0 vs H1: beta1 > 0:", round(p_value, 6), "\n")