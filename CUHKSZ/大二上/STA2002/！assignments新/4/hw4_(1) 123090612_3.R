# Time points
t <- c(1, 2, 3, 4, 5)

# Attendance data
X <- c(85, 73, 68, 56, 43)

neg_loglik <- function(params) {
  alpha <- params[1]
  beta <- params[2]

  mu <- alpha + beta * t

  # Return a large number if mu <= 0 to penalize invalid parameters
  if (any(mu <= 0)) {
    return(1e10)
  }

  # Compute the negative log-likelihood
  loglik <- -sum(-mu + X * log(mu) - lgamma(X + 1))
  return(loglik)
}

# Initial guesses for alpha and beta
start_params <- c(alpha = 10, beta = -10)

# Perform the optimization
result <- optim(start_params, neg_loglik)

# Extract the estimated parameters
alpha_hat <- result$par[1]
beta_hat <- result$par[2]

# Display the results
cat("Estimated alpha:", alpha_hat, "\n")
cat("Estimated beta:", beta_hat, "\n")
