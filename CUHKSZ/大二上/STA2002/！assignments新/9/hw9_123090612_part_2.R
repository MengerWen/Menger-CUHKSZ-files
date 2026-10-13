# Part ii: Investigate the distribution of p-values under H0

# Define parameters
mu_true <- 1        # True mean (mu) of the distribution
sigma_squared <- 2  # Known variance (sigma^2)
n <- 100            # Sample size
num_sim <- 500      # Number of simulations
mu0 <- mu_true      # Null hypothesis value (H0: mu = mu0)

# Initialize a vector to store p-values
p_values <- numeric(num_sim)

# Perform num_sim simulations
for (i in 1:num_sim) {
  # Step 1: Generate a sample of size n from N(mu_true, sigma^2)
  X <- rnorm(n, mean = mu_true, sd = sqrt(sigma_squared))
  
  # Step 2: Calculate the sample mean
  X_bar <- mean(X)
  
  # Step 3: Compute the test statistic (Z-score)
  Z <- (X_bar - mu0) / (sqrt(sigma_squared) / sqrt(n))
  
  # Step 4: Calculate the p-value for the one-sided test H1: mu > mu0
  p_values[i] <- 1 - pnorm(Z)
}

# Plot the histogram of p-values (Counts)
hist(p_values, 
     breaks = 20, 
     main = "Histogram of p-values under H0 (Counts)",
     xlab = "p-value",
     col = "lightblue",
     border = "black")

# Plot the histogram of p-values (Proportions)
hist(p_values, 
     breaks = 20, 
     probability = TRUE,  # Set to TRUE to plot proportions
     main = "Histogram of p-values under H0 (Proportions)",
     xlab = "p-value",
     col = "lightgreen",
     border = "black")

# Add the expected uniform density line
abline(h = 1, col = "blue", lwd = 2, lty = 2)
legend("topright", legend = "Uniform Density (1)", 
       col = "blue", lwd = 2, lty = 2, bty = "n")