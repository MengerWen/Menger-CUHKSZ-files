# Part iii: Investigate the distribution of p-values under H1

# Define parameters
n <- 100            # Sample size
sigma_squared <- 2
sigma <- sqrt(sigma_squared)
mu_true <- 1.0      # True mean under H1
mu0 <- -1.0         # Hypothesized mean under H0
num_sim <- 500      # Number of simulations

# Initialize a vector to store p-values
p_values <- numeric(num_sim)

# Run simulations
for (i in 1:num_sim) {
  # Generate random sample from N(mu_true, sigma^2)
  sample_data <- rnorm(n, mean = mu_true, sd = sigma)
  
  # Calculate the sample mean
  sample_mean <- mean(sample_data)
  
  # Calculate the standard error of the mean
  standard_error <- sigma / sqrt(n)
  
  # Compute the test statistic (z-value)
  z_value <- (sample_mean - mu0) / standard_error
  
  # Compute the p-value for a one-sided test (right-tailed)
  p_values[i] <- 1 - pnorm(z_value)
}

# Plot histogram of p-values with adjusted axis range and better binning
hist(p_values, breaks = 50, main = "Histogram of p-values", 
     xlab = "p-value", col = "lightblue", border = "black",
     xlim = c(-1, 1), ylim = c(0, 70))