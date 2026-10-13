import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu_0 = -1.0
sigma_2 = 2
sigma = np.sqrt(sigma_2)
n = 100
num_sim = 500

p_values = []

for _ in range(num_sim):
    data = np.random.normal(mu_0, sigma, n)
    sample_mean = np.mean(data)
    z = (sample_mean - mu_0) / (sigma / np.sqrt(n))
    p_value = 1 - norm.cdf(z)
    p_values.append(p_value)

plt.hist(p_values, bins=30, density=True, alpha=0.7, color = 'red')
plt.title('Histogram of p-values under H1(mu=1.0, mu_0=-1.0)')
plt.xlabel('p-values')
plt.ylabel('Frequency')
plt.grid()
plt.show()