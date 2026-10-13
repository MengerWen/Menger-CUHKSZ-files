# Observed data
t <- c(1, 2, 3, 4, 5)
x <- c(85, 73, 68, 56, 43)

# Negative log-likelihood function
neg_log_likelihood <- function(params) {
  alpha <- params[1]
  beta <- params[2]

  # Ensure lambda_t is positive
  lambda_t <- alpha + beta * t
  if (any(lambda_t <= 0)) {
    return(Inf)
  }

  # Calculate negative log-likelihood
  nll <- -sum(dpois(x, lambda_t, log = TRUE))
  return(nll)
}

# Initial guesses for alpha and beta
init_params <- c(alpha = 90, beta = -10)

# Optimize the negative log-likelihood
result <- optim(par = init_params, fn = neg_log_likelihood, method = "L-BFGS-B",
                lower = c(0, -Inf), upper = c(Inf, Inf))

# Extract the estimates
alpha_hat <- result$par[1]
beta_hat <- result$par[2]

# Output the estimates
cat("The MLE of alpha is:", alpha_hat, "\n")
cat("The MLE of beta is:", beta_hat, "\n")
