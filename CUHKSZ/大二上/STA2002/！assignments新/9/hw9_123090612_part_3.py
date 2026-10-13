import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set parameters
mu_0 = -1.0
sigma_2 = 2
sigma = np.sqrt(sigma_2)
n = 100
num_sim = 500
mu = 1.0  # True mean under H1

# Initialize array to store p-values
p_values = np.zeros(num_sim)

# Simulation loop
for i in range(num_sim):
    # Generate data under H1 (mu > mu_0)
    X = np.random.normal(loc=mu, scale=sigma, size=n)
    # Compute sample mean
    X_bar = np.mean(X)
    # Compute test statistic
    Z = (X_bar - mu_0) / (sigma / np.sqrt(n))
    # Compute p-value for one-sided test
    p_value = 1 - stats.norm.cdf(Z)
    # Store p-value
    p_values[i] = p_value

# Plot histogram of p-values
plt.hist(p_values, bins=20, edgecolor='black')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.title('Histogram of p-values under H1')
plt.show()