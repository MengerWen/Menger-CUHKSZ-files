# Define the sample sizes
n_values <- c(5, 10, 20, 50)

# Set up the plotting area: 4 rows and 2 columns for 8 histograms
par(mfrow = c(4, 2), mar = c(4, 4, 2, 1))

# Loop over each sample size
for (n in n_values) {
  A_values <- numeric(200)  # Initialize vector to store A statistics
  B_values <- numeric(200)  # Initialize vector to store B statistics
  
  # Repeat the experiment 200 times
  for (i in 1:200) {
    Z <- rpois(n, lambda = 1)        # Generate n Poisson(1) random variables
    A <- mean(Z)                     # Compute A = (1/n) * sum(Z_i)
    B <- sqrt(n) * (A - 1)           # Compute B = sqrt(n) * (A - 1)
    A_values[i] <- A                 # Store A
    B_values[i] <- B                 # Store B
  }
  
  # Plot histogram for A
  hist(A_values, breaks = 30, col = 'skyblue', border = 'white',
       main = paste("Histogram of A for n =", n),
       xlab = "A", ylab = "Frequency")
  
  # Plot histogram for B
  hist(B_values, breaks = 30, col = 'salmon', border = 'white',
       main = paste("Histogram of B for n =", n),
       xlab = "B", ylab = "Frequency")
}
