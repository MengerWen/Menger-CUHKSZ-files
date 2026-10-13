set.seed(123)

#############################################
# Part i: Generate data and estimate β0, β1
#############################################

# Define sample size
n <- 200  # You can choose any reasonable sample size, e.g., 100

# Generate X_i ~ N(0,1)
X <- rnorm(n, mean = 0, sd = 1)

# Generate epsilon_i ~ N(0,1)
epsilon <- rnorm(n, mean = 0, sd = 1)

# Generate Y_i = 0.5 * X_i + 0.1 + epsilon_i
Y <- 0.5 * X + 0.1 + epsilon

# Estimate β0 and β1 using OLS formulas
# β1_hat = cov(X, Y) / var(X)
beta1_hat <- cov(X, Y) / var(X)

# β0_hat = mean(Y) - β1_hat * mean(X)
beta0_hat <- mean(Y) - beta1_hat * mean(X)

# Report the estimated β0 and β1
cat("Part i:\n")
cat("Estimated β0:", beta0_hat, "\n")
cat("Estimated β1:", beta1_hat, "\n\n")

#############################################
# Part ii: Repeat 200 times and plot histograms
#############################################

# Number of simulations
num_sim <- 200

# Initialize vectors to store estimates
beta0_estimates <- numeric(num_sim)
beta1_estimates <- numeric(num_sim)

# Loop to perform simulations
for (i in 1:num_sim) {
  # Generate X_i ~ N(0,1)
  X_sim <- rnorm(n, mean = 0, sd = 1)
  
  # Generate epsilon_i ~ N(0,1)
  epsilon_sim <- rnorm(n, mean = 0, sd = 1)
  
  # Generate Y_i
  Y_sim <- 0.5 * X_sim + 0.1 + epsilon_sim
  
  # Estimate β1_hat and β0_hat
  beta1_sim <- cov(X_sim, Y_sim) / var(X_sim)
  beta0_sim <- mean(Y_sim) - beta1_sim * mean(X_sim)
  
  # Store the estimates
  beta0_estimates[i] <- beta0_sim
  beta1_estimates[i] <- beta1_sim
}

# Plot histogram for β0_hat
hist(beta0_estimates, breaks = 20, main = expression(paste("Histogram of ", hat(beta)[0])), 
     xlab = expression(hat(beta)[0]), col = "lightblue", border = "black")

# Plot histogram for β1_hat
hist(beta1_estimates, breaks = 20, main = expression(paste("Histogram of ", hat(beta)[1])), 
     xlab = expression(hat(beta)[1]), col = "lightgreen", border = "black")

#############################################
# Part iii: Hypothesis testing and P-value histogram
#############################################

# Initialize vector to store p-values
p_values <- numeric(num_sim)

# Loop to perform hypothesis testing
for (i in 1:num_sim) {
  # Generate X_i ~ N(0,1)
  X_sim <- rnorm(n, mean = 0, sd = 1)
  
  # Generate epsilon_i ~ N(0,1)
  epsilon_sim <- rnorm(n, mean = 0, sd = 1)
  
  # Generate Y_i
  Y_sim <- 0.5 * X_sim + 0.1 + epsilon_sim
  
  # Estimate β1_hat and β0_hat
  beta1_sim <- cov(X_sim, Y_sim) / var(X_sim)
  beta0_sim <- mean(Y_sim) - beta1_sim * mean(X_sim)
  
  # Estimate standard error of β1_hat
  # SE(beta1) = sqrt[ σ^2 / sum((X_i - mean(X))^2) ]
  # Here, σ^2 is estimated by residual variance
  residuals <- Y_sim - beta0_sim - beta1_sim * X_sim
  sigma2_hat <- sum(residuals^2) / (n - 2)
  SE_beta1 <- sqrt(sigma2_hat / sum( (X_sim - mean(X_sim))^2 ))
  
  # Compute test statistic for H0: beta1 = 0 vs H1: beta1 < 0
  t_stat <- (beta1_sim - 0) / SE_beta1
  
  # Compute one-sided p-value
  p_val <- pt(t_stat, df = n - 2)
  
  # Store the p-value
  p_values[i] <- p_val
}

# Plot histogram of p-values
hist(p_values, breaks = 20, main = "Histogram of P-values", 
     xlab = "P-value", col = "salmon", border = "black")