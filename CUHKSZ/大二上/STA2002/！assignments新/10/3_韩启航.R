# Set seed for reproducibility
set.seed(123)

# Number of samples and simulations
n <- 100
simulations <- 200

# Function to generate data
generate_data <- function(n) {
  X <- rnorm(n, mean = 0, sd = 1)
  epsilon <- rnorm(n, mean = 0, sd = 1)
  Y <- 0.5 * X + 0.1 + epsilon
  data.frame(X = X, Y = Y)
}

# Function to estimate coefficients
estimate_coefficients <- function(data) {
  model <- lm(Y ~ X, data = data)
  return(coef(model))
}

# Function to calculate p-value
calculate_p_value <- function(data) {
  # Fit the model
  model <- lm(Y ~ X, data = data)
  summary_model <- summary(model)
  
  # Extract p-value for the hypothesis H0: β1 = 0
  p_value <- summary_model$coefficients[2, 4]
  return(p_value)
}

# Initialize storage for results
beta0_hats <- numeric(simulations)
beta1_hats <- numeric(simulations)
p_values <- numeric(simulations)

# Run simulations
for (i in 1:simulations) {
  data <- generate_data(n)
  coeffs <- estimate_coefficients(data)
  beta0_hats[i] <- coeffs[1]
  beta1_hats[i] <- coeffs[2]
  p_values[i] <- calculate_p_value(data)
}

# Output estimated coefficients from one run
cat("Estimated β0 from first simulation:", beta0_hats[1], "\n")
cat("Estimated β1 from first simulation:", beta1_hats[1], "\n")

# Plot histograms for β0 and β1 estimates
hist(beta0_hats, breaks = 20, main = "Histogram of β0 Estimates", xlab = "β0")
hist(beta1_hats, breaks = 20, main = "Histogram of β1 Estimates", xlab = "β1")

# Plot histogram for p-values
hist(p_values, breaks = 20, main = "Histogram of P-values", xlab = "P-value")
