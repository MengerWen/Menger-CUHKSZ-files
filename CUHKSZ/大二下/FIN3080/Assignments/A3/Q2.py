import pandas as pd
import numpy as np
import statsmodels.api as sm

# File paths
stock_data1 = r"D:\MG\！CUHKSZ\~！大二 下\FIN3080\Assignments\A3\TRD_Week_1.xlsx"
stock_data2 = r"D:\MG\！CUHKSZ\~！大二 下\FIN3080\Assignments\A3\TRD_Week_2.xlsx"
rf_path = r"D:\MG\！CUHKSZ\~！大二 下\FIN3080\Assignments\A3\weekly_risk_free_rate.xlsx"

# Load stock return data (assuming columns: Stkcd, Trdwnt, Wretnd)
weekly_returns1 = pd.read_excel(stock_data1, dtype={'Stkcd': str})
weekly_returns2 = pd.read_excel(stock_data2, dtype={'Stkcd': str})
weekly_returns = pd.concat([weekly_returns1, weekly_returns2], ignore_index=True)
weekly_returns = weekly_returns[weekly_returns['Trdwnt'] < '2023-01'].reset_index(drop=True)

# Load risk-free return data
df = pd.read_excel(rf_path, sheet_name='Sheet1')
df['trading_date_yw'] = pd.to_datetime(df['trading_date_yw'])
df.set_index('trading_date_yw', inplace=True)
df['Trdwnt'] = df.index.to_period('W').strftime('%Y-%U')

# Merge stock and risk-free data
final_df = pd.merge(weekly_returns, df, on='Trdwnt', how='left')

# Calculate market returns
market_returns = final_df.groupby('Trdwnt')['Wretnd'].mean().reset_index()
market_returns.rename(columns={'Wretnd': 'Market_Return'}, inplace=True)
final_df = pd.merge(final_df, market_returns, on='Trdwnt', how='left')

# Sort by trading week
final_df.sort_values(by=['Trdwnt'], inplace=True)

# Display basic info
print("First few rows of merged DataFrame:")
print(final_df.head())