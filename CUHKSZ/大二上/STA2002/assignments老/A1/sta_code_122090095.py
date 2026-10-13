import numpy as np
import pandas as pd

# Load the dataset from GMM.csv
data = pd.read_csv("GMM.csv")

# Separate data into groups based on K
group_0 = data[data['K'] == 0]
group_1 = data[data['K'] == 1]

# Calculate sample means and variances for each group
mu_0 = group_0['X'].mean()
sigma_0 = group_0['X'].var()
mu_1 = group_1['X'].mean()
sigma_1 = group_1['X'].var()

# Calculate sample proportions for pi_0 and pi_1
n_0 = len(group_0)
n_1 = len(group_1)
n_total = len(data)
pi_0 = n_0 / n_total
pi_1 = n_1 / n_total

# These are the MLEs for the parameters
print("MLE for pi_0:", pi_0)
print("MLE for mu_0:", mu_0)
print("MLE for sigma^2_0:", sigma_0)
print("MLE for mu_1:", mu_1)
print("MLE for sigma^2_1:", sigma_1)