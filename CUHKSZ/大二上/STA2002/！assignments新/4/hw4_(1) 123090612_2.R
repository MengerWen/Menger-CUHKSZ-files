# Attendance data
t <- c(1, 2, 3, 4, 5)
X <- c(85, 73, 68, 56, 43)

# Negative log-likelihood function
neg_log_likelihood <- function(params) {
  alpha <- params[1]
  beta <- params[2]

  # Ensure lambda_t > 0
  lambda_t <- alpha + beta * t
  if (any(lambda_t <= 0)) return(Inf)

  nll <- -sum(-lambda_t + X * log(lambda_t))
  return(nll)
}

# Initial parameter estimates
init_params <- c(alpha = 50, beta = -5)

# Perform optimization
result <- optim(par = init_params, fn = neg_log_likelihood, method = "L-BFGS-B",
                lower = c(0, -Inf), upper = c(Inf, Inf))

# Extract estimated parameters
estimated_alpha <- result$par[1]
estimated_beta <- result$par[2]

# Display results
cat("Estimated alpha:", estimated_alpha, "\n")
cat("Estimated beta:", estimated_beta, "\n")
