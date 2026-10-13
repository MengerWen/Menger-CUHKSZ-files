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
# - $\alpha$: Intercept.
# - $\epsilon$: Error term.

# %%
def run_capm_analysis(asset_return, market_return, asset_name, market_name):
    """Run CAPM regression and return formatted results."""
    X = sm.add_constant(market_return)
    y = asset_return
    model = sm.OLS(y, X).fit()
    
    return {
        'Asset': asset_name,
        'Market Index': market_name,
        'Alpha': model.params['const'],
        'Alpha P-value': model.pvalues['const'],
        'Beta': model.params[market_name],
        'Beta P-value': model.pvalues[market_name],
        'R-squared': model.rsquared
    }

# Run CAPM for BYD vs HSI and Tesla vs IXIC
capm_results = [
    run_capm_analysis(returns['BYD'], returns['HSI'], 'BYD', 'HSI'),
    run_capm_analysis(returns['TESLA'], returns['IXIC'], 'TESLA', 'IXIC')
]

# Create styled table
capm_df = pd.DataFrame(capm_results)
capm_styled = capm_df.style.format({
    'Alpha': '{:.4f}',
    'Alpha P-value': '{:.4f}',
    'Beta': '{:.4f}',
    'Beta P-value': '{:.4f}',
    'R-squared': '{:.4f}'
}).set_caption("CAPM Regression Results").hide_index()

# Highlight significant p-values
def highlight_p(val):
    return 'background-color: yellow' if val < 0.05 else ''

capm_styled = capm_styled.applymap(highlight_p, subset=['Alpha P-value', 'Beta P-value'])

# %% [markdown]
# # 2. Market Spread Analysis

# %%
# Define the dependent and independent variables
spread_return = returns['BYD'] - returns['TESLA']
market_spread_return = (returns['HSI'] - returns['IXIC']).rename('market_spread')

# Prepare and run regression
X = sm.add_constant(market_spread_return)
y = spread_return
model_spread = sm.OLS(y, X).fit()

# Format spread results
spread_results = [{
    'Model': 'Spread (BYD - TESLA) vs (HSI - IXIC)',
    'Alpha': model_spread.params['const'],
    'Alpha P-value': model_spread.pvalues['const'],
    'Beta': model_spread.params['market_spread'],
    'Beta P-value': model_spread.pvalues['market_spread'],
    'R-squared': model_spread.rsquared
}]

spread_df = pd.DataFrame(spread_results)
spread_styled = spread_df.style.format({
    'Alpha': '{:.4f}',
    'Alpha P-value': '{:.4f}',
    'Beta': '{:.4f}',
    'Beta P-value': '{:.4f}',
    'R-squared': '{:.4f}'
}).set_caption("Market Spread Analysis").hide_index().applymap(highlight_p, subset=['Alpha P-value', 'Beta P-value'])

# %% [markdown]
# # Final Results

# %%
# Display all results
print("\n=== CAPM Analysis ===")
display(capm_styled)

print("\n=== Market Spread Analysis ===")
display(spread_styled)