# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# %%
# Step 1: Load and filter Fama-French factor data
factors_df_path = 'D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A4/STK_MKT_THRFACMONTH.xlsx'

factors_df = pd.read_excel(factors_df_path, header=0, skiprows=[0, 2], dtype={'Market Type Code': str})
factors_df = factors_df[factors_df['Market Type Code'] == 'P9706']  # SSE-SZSE A share market
factors_df['Trading Month'] = pd.to_datetime(factors_df['Trading Month'], format='%Y-%m')
factors_df.set_index('Trading Month', inplace=True)
factors = factors_df[['Market Risk Premium Factor (Weighted Average Value of Negotiable Shares)', 
                      'Book-to-Market Ratio Factor (Weighted Average Value of Negotiable Shares)']]
factors.columns = ['Market Risk', 'BM Ratio']
factors = factors.loc['2001-01':'2024-12']

# %%
# Step 2: Load and filter stock trading data
stock_df_path = 'D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A4/TRD_Mnth.xlsx'

stock_df = pd.read_excel(stock_df_path, header=0, skiprows=[0, 2], dtype={'Stock Code': str})
stock_df = stock_df[stock_df['Market Type'].isin([1, 4])]  # SSE (1) and SZSE (4) mainboard
stock_df['Stock Code'] = stock_df['Stock Code'].astype(str).str.zfill(6)  # Ensure 6-digit codes
stock_df = stock_df[~((stock_df['Market Type'] == 1) & stock_df['Stock Code'].str.startswith('688'))]  # Exclude STAR Market
stock_df = stock_df[~((stock_df['Market Type'] == 4) & stock_df['Stock Code'].str.startswith('300'))]  # Exclude ChiNext
stock_df['Trading Month'] = pd.to_datetime(stock_df['Trading Month'], format='%Y-%m')

# %%
# Step 3: Create lagged market capitalization
stock_df = stock_df.sort_values(by=['Stock Code', 'Trading Month'])
stock_df['Market Cap Lag1'] = stock_df.groupby('Stock Code')['Market Value of Tradable Shares'].shift(1)

# %%
# Step 4: Assign deciles based on lagged market capitalization
def assign_decile(group):
    try:
        group['Decile'] = pd.qcut(group['Market Cap Lag1'], 10, labels=False, duplicates='drop')
    except ValueError:
        group['Decile'] = np.nan
    return group

stock_df = stock_df.groupby('Trading Month').apply(assign_decile).reset_index(drop=True)

# %%
# Step 5: Calculate equal-weighted portfolio returns
portfolio_returns = stock_df.groupby(['Trading Month', 'Decile'])['Monthly Return Without Cash Dividend Reinvested'].mean().unstack()
portfolio_returns = portfolio_returns.loc['2001-01':'2024-12']

# %%
# Step 6: Perform regression for each portfolio and store results
models = {}
for decile in range(10):
    y = portfolio_returns[decile].dropna()
    X = factors.loc[y.index].dropna()
    y = y.loc[X.index]  # Align dates
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    models[decile] = model

# Create a summary table for all models
summary_table = summary_col(
    [models[i] for i in range(10)],
    stars=True,
    float_format='%.4f',
    model_names=[f'Portfolio {i}' for i in range(10)],
    info_dict={'N': lambda x: f"{x.nobs:.0f}", 'R2': lambda x: f"{x.rsquared:.4f}"}
)

print("\nRegression Summaries for All Portfolios:")
print(summary_table)

# %%
# Step 7: Summarize regression coefficients
results = {}
for decile in range(10):
    model = models[decile]
    results[decile] = {
        'Intercept': model.params['const'],
        'Market Risk Beta': model.params['Market Risk'],
        'BM Ratio Beta': model.params['BM Ratio'],
        'Intercept p-value': model.pvalues['const'],
        'Market Risk p-value': model.pvalues['Market Risk'],
        'BM Ratio p-value': model.pvalues['BM Ratio'],
        'R-squared': model.rsquared
    }

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
results_df.index.name = 'Portfolio Decile'

# Highlight significant p-values (p < 0.05)
def highlight_significant(s):
    is_significant = s < 0.05
    return ['background-color: yellow' if v else '' for v in is_significant]

styled_results_df = results_df.style.apply(highlight_significant, subset=['Intercept p-value', 'Market Risk p-value', 'BM Ratio p-value'])
print("\nSummary of Regression Coefficients:")
display(styled_results_df)

# %%
# Step 8: Plot betas across deciles
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df['Market Risk Beta'], label='Market Risk Beta', marker='o', color='blue')
plt.plot(results_df.index, results_df['BM Ratio Beta'], label='BM Ratio Beta', marker='o', color='green')
plt.xlabel('Portfolio Decile (0 = Smallest, 9 = Largest)', fontsize=12)
plt.ylabel('Beta Coefficient', fontsize=12)
plt.title('Beta Coefficients Across Market Cap Deciles', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Step 10: Discussion template
print("\n**Discussion of Findings:**")
print("1. **Market Risk Exposure**: Analyze how Market Risk Beta varies from decile 0 (smallest cap) to 9 (largest cap). Typically, smaller firms may exhibit higher betas due to greater sensitivity to market movements.")
print("2. **Book-to-Market Sensitivity**: Examine BM Ratio Beta trends. Higher BM ratios in smaller deciles might indicate value characteristics.")
print("3. **Statistical Significance**: Assess p-values to determine the significance of each factor across portfolios (highlighted in yellow if p < 0.05).")
print("4. **Model Fit**: Evaluate R-squared values to understand explanatory power, expecting variation across deciles.")
print("**Note**: Full analysis requires running the code with complete data from 2001-2024.")

# %%
# Step 10: Discussion template (to be expanded with actual data)
print("\nDiscussion of Findings:")
print("1. **Market Risk Exposure**: Analyze how Market Risk Beta varies from decile 0 (smallest cap) to 9 (largest cap). Typically, smaller firms may exhibit higher betas due to greater sensitivity to market movements.")
print("2. **Book-to-Market Sensitivity**: Examine BM Ratio Beta trends. Higher BM ratios in smaller deciles might indicate value characteristics.")
print("3. **Statistical Significance**: Assess p-values to determine the significance of each factor across portfolios.")
print("4. **Model Fit**: Evaluate R-squared values to understand explanatory power, expecting variation across deciles.")
print("Note: Full analysis requires running the code with complete data from 2001-2024.")


