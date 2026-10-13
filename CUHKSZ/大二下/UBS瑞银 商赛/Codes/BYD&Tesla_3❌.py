# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the data
file_path = r'D:\MG\！CUHKSZ\~！大二 下\UBS瑞银 商赛\data\BYD&Tesla.xlsx'
data = pd.read_excel(
    file_path,
    header=0,
    skiprows=[0],
    index_col='日期',
    parse_dates=['日期'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)
data = data[['比亚迪股份', '特斯拉(TESLA)']]
data.columns = ['BYD', 'TESLA']

# %%
# Preprocess the data
print(data.isnull().sum())  # Check for missing values
data = data.loc[~data.duplicated(keep='first')]  # Remove duplicates
prices_normed = pd.DataFrame(data.apply(scale), index=data.index, columns=['BYD', 'TESLA'])

# %%
# Apply rolling regression
model_randomwalk = pm.Model()
with model_randomwalk:
    sigma_alpha = pm.Exponential('sigma_alpha', 50.)
    alpha = pm.GaussianRandomWalk('alpha', sigma=sigma_alpha, shape=len(prices_normed))
    sigma_beta = pm.Exponential('sigma_beta', 50.)
    beta = pm.GaussianRandomWalk('beta', sigma=sigma_beta, shape=len(prices_normed))
    regression = alpha + beta * prices_normed['TESLA']
    sigma = pm.HalfNormal('sigma', sigma=0.1)
    likelihood = pm.Normal('y', mu=regression, sigma=sigma, observed=prices_normed['BYD'])
    trace_rw = pm.sample(tune=2000, draws=200, cores=1, target_accept=0.9)

# %%
# Visualize results
sns.set_style('whitegrid')

# Plot coefficients
fig, axes = plt.subplots(figsize=(15, 5), ncols=2, sharex=True)
axes[0].plot(trace_rw['alpha'].T, 'r', alpha=0.05)
axes[0].set_xlabel('Time'); axes[0].set_ylabel('Alpha'); axes[0].set_title('Intercept')
axes[1].plot(trace_rw['beta'].T, 'b', alpha=0.05)
axes[1].set_xlabel('Time'); axes[1].set_ylabel('Beta'); axes[1].set_title('Slope')
date_labels = [str(p.date()) for p in prices_normed.index[::len(prices_normed)//9]]
for ax in axes:
    ax.set_xticks(range(0, len(prices_normed), len(prices_normed)//9))
    ax.set_xticklabels(date_labels, rotation=45)
fig.suptitle('Changes of Coefficients Over Time (BYD vs TESLA)', fontsize=14)
sns.despine(); fig.tight_layout(); fig.subplots_adjust(top=0.9)
plt.show()

# Plot posterior predictive lines
x = np.linspace(prices_normed['TESLA'].min(), prices_normed['TESLA'].max(), 100)
dates = [str(p.year) for p in prices_normed.index[::len(prices_normed)//9]]
colors = np.linspace(0.1, 1, len(prices_normed))
colors_sc = np.linspace(0.1, 1, len(trace_rw[::10]['alpha'].T))
cmap = plt.get_cmap('winter')
fig, ax = plt.subplots(figsize=(14, 8))
for i, (alpha, beta) in enumerate(zip(trace_rw[::25]['alpha'].T, trace_rw[::25]['beta'].T)):
    for a, b in zip(alpha[::25], beta[::25]):
        ax.plot(x, a + b * x, alpha=0.01, lw=0.5, c=cmap(colors_sc[i]))
points = ax.scatter(prices_normed['TESLA'], prices_normed['BYD'], c=colors, s=5, cmap=cmap)
cbar = plt.colorbar(points); cbar.ax.set_yticklabels(dates)
ax.set_xlabel('TESLA (Normalized Price)'); ax.set_ylabel('BYD (Normalized Price)')
ax.set_title('Posterior Predictive Regression Lines (BYD vs TESLA)')
sns.despine(); fig.tight_layout()
plt.show()