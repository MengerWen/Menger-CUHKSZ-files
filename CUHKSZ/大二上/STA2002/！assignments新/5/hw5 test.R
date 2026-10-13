# Set parameters
n <- 40
p_true <- 0.3
alpha <- 0.05
num_simulations <- 100

# Compute z-values for confidence interval
z_lower <- qnorm(alpha / 2)       # z_{2.5%}
z_upper <- qnorm(1 - alpha / 2)   # z_{97.5%}

# Initialize vectors to store results
p_hats <- numeric(num_simulations)
lower_bounds <- numeric(num_simulations)
upper_bounds <- numeric(num_simulations)
covers_p <- logical(num_simulations)

# Set seed for reproducibility
set.seed(123)

# Loop over simulations
for (i in 1:num_simulations) {
  # Generate Bernoulli random variables
  X <- rbinom(n, size = 1, prob = p_true)
  
  # Compute sample proportion
  p_hat <- mean(X)
  
  # Compute standard error
  SE <- sqrt(p_hat * (1 - p_hat) / n)
  
  # Compute confidence interval
  lower_bound <- p_hat + z_lower * SE
  upper_bound <- p_hat + z_upper * SE
  
  # Store results
  p_hats[i] <- p_hat
  lower_bounds[i] <- lower_bound
  upper_bounds[i] <- upper_bound
  
  # Check if confidence interval covers true p
  covers_p[i] <- (lower_bound <= p_true) && (p_true <= upper_bound)
}

# Calculate how many intervals cover the true p
num_cover <- sum(covers_p)
cat("Number of intervals covering true p:", num_cover, "\n")

# Plotting the confidence intervals using base R
# Set up the plot area
plot(1:num_simulations, p_hats, ylim=c(0,1), pch=19, xlab="Simulation Number", ylab="Sample Proportion", main="95% Confidence Intervals for p")

# Add confidence intervals
for (i in 1:num_simulations) {
  segments(x0 = i, y0 = lower_bounds[i], x1 = i, y1 = upper_bounds[i], col = ifelse(covers_p[i], "black", "blue"))
}

# Add a horizontal line for true p
abline(h = p_true, lty = 2, col = "red")

# Add a legend
legend("topright", legend=c("CI covers true p", "CI does not cover true p", "True p"), col=c("black", "blue", "red"), lty=c(1,1,2), pch=c(19,19,NA), bty="n")
