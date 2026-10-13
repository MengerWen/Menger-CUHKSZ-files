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
# Set up the plot area with horizontal intervals
plot(NULL, xlim = c(min(lower_bounds) - 0.05, max(upper_bounds) + 0.05), ylim = c(1, num_simulations), 
     xlab = "p", ylab = "Simulations", main = "Confidence Intervals")

# Add horizontal confidence intervals
for (i in 1:num_simulations) {
  color <- ifelse(covers_p[i], "gray", "red")
  segments(x0 = lower_bounds[i], y0 = i, x1 = upper_bounds[i], y1 = i, col = color, lwd = 2)
}

# Add a vertical line for the true p value
abline(v = p_true, col = "green", lwd = 2)

# Add a legend
legend("topright", legend=c("CI covers true p", "CI does not cover true p", "True p"), 
       col=c("gray", "red", "green"), lty=1, lwd=2, bty="n")
