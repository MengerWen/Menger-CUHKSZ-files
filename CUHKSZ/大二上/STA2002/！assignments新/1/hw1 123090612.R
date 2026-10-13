# Set parameters
n_reps <- 300    # Number of repetitions
n_samples <- 500 # Sample size

# Initialize vectors to store statistics
sample_means <- numeric(n_reps)
sample_variances <- numeric(n_reps)
sample_skewnesses <- numeric(n_reps)
sample_kurtoses <- numeric(n_reps)

# Run the experiment
for (i in 1:n_reps) {
  # Generate 500 Uniform[0,1] random variables
  x <- runif(n_samples, min = 0, max = 1)
  
  # Compute statistics
  sample_means[i] <- mean(x)
  sample_variances[i] <- var(x)
  sample_skewnesses[i] <- skewness(x)
  sample_kurtoses[i] <- kurtosis(x)
}

# Theoretical values
theoretical_mean <- 0.5
theoretical_variance <- 1/12
theoretical_skewness <- 0
theoretical_kurtosis <- 1.8

# Plot histograms with theoretical values
par(mfrow = c(2, 2))  # Set up a 2x2 plotting space

# Histogram of sample means
hist(sample_means, main = 'Histogram of Sample Means', xlab = 'Sample Mean', col = 'lightblue', breaks = 20)
abline(v = theoretical_mean, col = 'red', lwd = 2)
legend('topright', legend = paste('Theoretical Mean =', theoretical_mean), col = 'red', lwd = 2)

# Histogram of sample variances
hist(sample_variances, main = 'Histogram of Sample Variances', xlab = 'Sample Variance', col = 'lightgreen', breaks = 20)
abline(v = theoretical_variance, col = 'red', lwd = 2)
legend('topright', legend = paste('Theoretical Variance =', round(theoretical_variance, 4)), col = 'red', lwd = 2)

# Histogram of sample skewnesses
hist(sample_skewnesses, main = 'Histogram of Sample Skewnesses', xlab = 'Sample Skewness', col = 'lightcoral', breaks = 20)
abline(v = theoretical_skewness, col = 'red', lwd = 2)
legend('topright', legend = paste('Theoretical Skewness =', theoretical_skewness), col = 'red', lwd = 2)

# Histogram of sample kurtoses
hist(sample_kurtoses, main = 'Histogram of Sample Kurtoses', xlab = 'Sample Kurtosis', col = 'lightyellow', breaks = 20)
abline(v = theoretical_kurtosis, col = 'red', lwd = 2)
legend('topright', legend = paste('Theoretical Kurtosis =', theoretical_kurtosis), col = 'red', lwd = 2)

# Reset plotting parameters
par(mfrow = c(1, 1))

# Compare sample averages with theoretical values
cat("Comparison of Sample Averages with Theoretical Values:\n")
cat("------------------------------------------------------\n")
cat(sprintf("Theoretical Mean:        %f\n", theoretical_mean))
cat(sprintf("Average of Sample Means: %f\n\n", mean(sample_means)))

cat(sprintf("Theoretical Variance:        %f\n", theoretical_variance))
cat(sprintf("Average of Sample Variances: %f\n\n", mean(sample_variances)))

cat(sprintf("Theoretical Skewness:        %f\n", theoretical_skewness))
cat(sprintf("Average of Sample Skewness:  %f\n\n", mean(sample_skewnesses)))

cat(sprintf("Theoretical Kurtosis:        %f\n", theoretical_kurtosis))
cat(sprintf("Average of Sample Kurtoses:  %f\n", mean(sample_kurtoses)))
