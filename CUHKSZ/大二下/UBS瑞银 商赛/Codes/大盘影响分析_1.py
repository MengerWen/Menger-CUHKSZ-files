# %% [markdown]
# # Data Preparation

# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

# %%
# File paths (adjust as needed if running locally)
file_path_stocks = r'D:\MG\！CUHKSZ\~！大二 下\UBS瑞银 商赛\Data\BYD&Tesla.xlsx'
file_path_indices = r'D:\MG\！CUHKSZ\~！大二 下\UBS瑞银 商赛\Data\恒生&纳斯达克.xlsx'

# Read stock prices (BYD and Tesla)
stocks = pd.read_excel(
    file_path_stocks,
    header=0,
    skiprows=[0],
    index_col='日期',
    parse_dates=['日期'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)
stocks = stocks[['比亚迪股份', '特斯拉(TESLA)']]
stocks.columns = ['BYD', 'TESLA']
stocks = stocks[(stocks['BYD'] > 0) & (stocks['TESLA'] > 0)]

# Read index prices (HSI and IXIC)
indices = pd.read_excel(
    file_path_indices,
    header=0,
    skiprows=[0],
    index_col='日期',
    parse_dates=['日期'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)
indices = indices[['恒生指数', '纳斯达克指数']]
indices.columns = ['HSI', 'IXIC']
indices = indices[(indices['HSI'] > 0) & (indices['IXIC'] > 0)]

# Align dates by taking the intersection
common_dates = stocks.index.intersection(indices.index)
stocks = stocks.loc[common_dates]
indices = indices.loc[common_dates]

# Compute log prices
log_stocks = np.log(stocks)
log_indices = np.log(indices)

# Compute daily log returns and drop NaN values
returns = pd.DataFrame({
    'BYD': log_stocks['BYD'].diff(),
    'TESLA': log_stocks['TESLA'].diff(),
    'HSI': log_indices['HSI'].diff(),
    'IXIC': log_indices['IXIC'].diff()
}).dropna()

# %% [markdown]
# # 1. CAPM Analysis

# %% [markdown]
# CAPM models the expected return of an asset as:
# $$R_{\text{asset}} = \alpha + \beta \cdot R_{\text{market}} + \epsilon$$
# where:
# - $R_{\text{asset}}$: Asset return (e.g., BYD or Tesla).
# - $R_{\text{market}}$: Market return (e.g., HSI or IXIC).
# - $\beta$: Sensitivity of the asset to the market.
# - $\alpha$: Intercept (not typically interpreted in basic CAPM without a risk-free rate).
# - $\epsilon$: Error term.

# %% [markdown]
# ### 1.1 BYD vs. HSI

# %%
X = returns['HSI']
y = returns['BYD']
X = sm.add_constant(X)  # Add intercept
model_byd = sm.OLS(y, X).fit()
print("BYD vs. HSI:")
print(model_byd.summary())
beta_byd = model_byd.params['HSI']
p_value_byd = model_byd.pvalues['HSI']
r_squared_byd = model_byd.rsquared

# %% [markdown]
# ### 1.2 Tesla vs. IXIC

# %%
X = returns['IXIC']
y = returns['TESLA']
X = sm.add_constant(X)
model_tesla = sm.OLS(y, X).fit()
print("Tesla vs. IXIC:")
print(model_tesla.summary())
beta_tesla = model_tesla.params['IXIC']
p_value_tesla = model_tesla.pvalues['IXIC']
r_squared_tesla = model_tesla.rsquared

# %% [markdown]
# # 2. Effect of the General Market on Stock Prices

# %%
# Define the dependent and independent variables
spread_return = returns['BYD'] - returns['TESLA']
market_spread_return = (returns['HSI'] - returns['IXIC']).rename('market_spread')

# Prepare the regression inputs
X = sm.add_constant(market_spread_return)
y = spread_return

# Fit the OLS model
model_spread = sm.OLS(y, X).fit()

# Output results
print("Spread (BYD - Tesla) vs. Spread (HSI - IXIC):")
print(model_spread.summary())

# Access parameters using the known name
beta_spread = model_spread.params['market_spread']
p_value_spread = model_spread.pvalues['market_spread']
r_squared_spread = model_spread.rsquared

# Optional: Print to verify
print(f"Beta coefficient: {beta_spread}")
print(f"P-value: {p_value_spread}")
print(f"R-squared: {r_squared_spread}")