# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define parameters
n = 100          # Number of data points for each simulation
num_simulations = 200  # Number of simulations

# Initialize lists to store estimated coefficients and p-values
beta0_estimates = []
beta1_estimates = []
p_values = []

# Function to perform linear regression and hypothesis testing
def linear_regression(X, Y):
    """
    Perform linear regression to estimate beta0 and beta1.
    Returns estimated coefficients and p-value for testing beta1 < 0.
    """
    # Calculate means
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    
    # Calculate beta1 using the formula: Cov(X,Y) / Var(X)
    covariance = np.sum((X - X_mean) * (Y - Y_mean))
    variance = np.sum((X - X_mean)**2)
    beta1_hat = covariance / variance
    
    # Calculate beta0 using the formula: Y_mean - beta1 * X_mean
    beta0_hat = Y_mean - beta1_hat * X_mean
    
    # Calculate residuals
    residuals = Y - (beta0_hat + beta1_hat * X)
    
    # Estimate variance of residuals
    sigma_squared = np.sum(residuals**2) / (n - 2)
    
    # Calculate standard error of beta1_hat
    SE_beta1 = np.sqrt(sigma_squared / variance)
    
    # Calculate t-statistic for H0: beta1 = 0 vs H1: beta1 < 0
    t_stat = (beta1_hat - 0) / SE_beta1
    
    # Since H1 is beta1 < 0, calculate one-tailed p-value
    p_val = stats.t.cdf(t_stat, df=n-2)
    
    return beta0_hat, beta1_hat, p_val

# Part i: Generate data and estimate beta0 and beta1 once
# --------------------------------------------------------
# Generate X ~ N(0,1)
X = np.random.normal(0, 1, n)

# Generate epsilon ~ N(0,1)
epsilon = np.random.normal(0, 1, n)

# Generate Y = 0.5 * X + 0.1 + epsilon
Y = 0.5 * X + 0.1 + epsilon

# Perform linear regression
beta0_i, beta1_i, p_value_i = linear_regression(X, Y)

# Report the estimated beta0 and beta1
print("Part i: Single Simulation")
print(f"Estimated β0: {beta0_i:.4f}")
print(f"Estimated β1: {beta1_i:.4f}\n")

# Part ii: Repeat step i for 200 times and plot histograms
# --------------------------------------------------------
for _ in range(num_simulations):
    # Generate X ~ N(0,1)
    X_sim = np.random.normal(0, 1, n)
    
    # Generate epsilon ~ N(0,1)
    epsilon_sim = np.random.normal(0, 1, n)
    
    # Generate Y = 0.5 * X + 0.1 + epsilon
    Y_sim = 0.5 * X_sim + 0.1 + epsilon_sim
    
    # Perform linear regression
    beta0_hat, beta1_hat, _ = linear_regression(X_sim, Y_sim)
    
    # Store the estimates
    beta0_estimates.append(beta0_hat)
    beta1_estimates.append(beta1_hat)

# Plot histogram for beta0 estimates
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(beta0_estimates, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Estimated β0')
plt.xlabel('β0 Estimates')
plt.ylabel('Frequency')

# Plot histogram for beta1 estimates
plt.subplot(1, 2, 2)
plt.hist(beta1_estimates, bins=20, color='salmon', edgecolor='black')
plt.title('Histogram of Estimated β1')
plt.xlabel('β1 Estimates')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Part iii: Hypothesis testing and plot histogram of p-values
# -----------------------------------------------------------
for _ in range(num_simulations):
    # Generate X ~ N(0,1)
    X_sim = np.random.normal(0, 1, n)
    
    # Generate epsilon ~ N(0,1)
    epsilon_sim = np.random.normal(0, 1, n)
    
    # Generate Y = 0.5 * X + 0.1 + epsilon
    Y_sim = 0.5 * X_sim + 0.1 + epsilon_sim
    
    # Perform linear regression and get p-value
    _, _, p_val = linear_regression(X_sim, Y_sim)
    
    # Store the p-value
    p_values.append(p_val)

# Plot histogram for p-values
plt.figure(figsize=(8, 6))
plt.hist(p_values, bins=20, color='lightgreen', edgecolor='black')
plt.title('Histogram of P-values for H₀: β₁ = 0 vs H₁: β₁ < 0')
plt.xlabel('P-value')
plt.ylabel('Frequency')
plt.show()