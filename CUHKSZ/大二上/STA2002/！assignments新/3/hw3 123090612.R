# Set the number of experiments and sample size
num_experiments <- 10
sample_size <- 500

# Initialize a matrix to store running means
running_means <- matrix(0, nrow = sample_size, ncol = num_experiments)

# Perform the experiments
for (exp in 1:num_experiments) {
  # Generate 500 i.i.d. Uniform[0,1] random variables
  X <- runif(sample_size)
  
  # Compute the running means
  running_means[, exp] <- cumsum(X) / (1:sample_size)
}

# Plot the running means
plot(1:sample_size, running_means[,1], type = "l", ylim = c(0,1),
     xlab = "t", ylab = "Running Sample Mean",
     main = "Running Sample Means of Uniform[0,1] Random Variables")

# Add the rest of the running means to the plot
for (exp in 2:num_experiments) {
  lines(1:sample_size, running_means[, exp], col = exp)
}

# Add a horizontal line at the expected mean 0.5
abline(h = 0.5, col = "red", lty = 2)

# Add legend
legend("right", legend = paste("Experiment", 1:num_experiments),
       col = 1:num_experiments, lty = 1, cex = 0.6)
