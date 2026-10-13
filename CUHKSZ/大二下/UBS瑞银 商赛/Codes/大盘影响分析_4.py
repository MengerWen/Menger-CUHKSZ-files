# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm

# %%
# Read and process BYD & Tesla data
byd_tesla_path = r'D:\MG\！CUHKSZ\~！大二 下\UBS瑞银 商赛\Data\BYD&Tesla.xlsx'
byd_tesla = pd.read_excel(
    byd_tesla_path,
    header=0,
    skiprows=[0],
    index_col='日期',
    parse_dates=['日期'],
)
byd_tesla = byd_tesla[['比亚迪股份', '特斯拉(TESLA)']].dropna()
byd_monthly = byd_tesla['比亚迪股份'].resample('M').last().pct_change().dropna().to_frame(name='BYD')
tesla_monthly = byd_tesla['特斯拉(TESLA)'].resample('M').last().pct_change().dropna().to_frame(name='Tesla')

# Read and process HSI & Nasdaq data
hsi_nasdaq_path = r'D:\MG\！CUHKSZ\~！大二 下\UBS瑞银 商赛\Data\恒生&纳斯达克.xlsx'
hsi_nasdaq = pd.read_excel(
    hsi_nasdaq_path,
    header=0,
    skiprows=[0],
    index_col='日期',
    parse_dates=['日期'],
)
hsi_nasdaq = hsi_nasdaq[['恒生指数', '纳斯达克指数']].dropna()
hsi_monthly = hsi_nasdaq['恒生指数'].resample('M').last().pct_change().dropna().to_frame(name='HSI')
nasdaq_monthly = hsi_nasdaq['纳斯达克指数'].resample('M').last().pct_change().dropna().to_frame(name='Nasdaq')

# Read and process HongKong factors
hongkong_path = r'D:\MG\！CUHKSZ\~！大二 下\UBS瑞银 商赛\Data\HongKong.xlsx'
hongkong = pd.read_excel(hongkong_path)
hongkong['Date'] = pd.to_datetime(hongkong['YearMonth'], format='%Y%m') + pd.offsets.MonthEnd(1)
hongkong.set_index('Date', inplace=True)
for col in ['Mkt', 'BE/ME High', 'BE/ME Low', 'E/P High', 'E/P Low', 'CE/P High', 'CE/P Low', 'Yld High', 'Yld Low']:
    hongkong[col] /= 100  # Convert percentages to decimals
hongkong['BE/ME_factor'] = hongkong['BE/ME High'] - hongkong['BE/ME Low']
hongkong['E/P_factor'] = hongkong['E/P High'] - hongkong['E/P Low']
hongkong['CE/P_factor'] = hongkong['CE/P High'] - hongkong['CE/P Low']
hongkong['Yld_factor'] = hongkong['Yld High'] - hongkong['Yld Low']
hongkong_factors = hongkong[['Mkt', 'BE/ME_factor', 'E/P_factor', 'CE/P_factor', 'Yld_factor']]

# Read and process Nasdaq factors
nasdaq_factors_path = r'D:\MG\！CUHKSZ\~！大二 下\UBS瑞银 商赛\Data\Nasdaq.xlsx'
nasdaq_factors = pd.read_excel(nasdaq_factors_path)
nasdaq_factors['Date'] = pd.to_datetime(nasdaq_factors['YearMonth'], format='%Y%m') + pd.offsets.MonthEnd(1)
nasdaq_factors.set_index('Date', inplace=True)
for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
    nasdaq_factors[col] /= 100  # Convert percentages to decimals

# Merge data for analysis
merged_byd = pd.merge(byd_monthly, hongkong_factors, left_index=True, right_index=True)
merged_tesla = pd.merge(tesla_monthly, nasdaq_factors, left_index=True, right_index=True)
merged_tesla['Tesla_excess'] = merged_tesla['Tesla'] - merged_tesla['RF']

# %%
# Task 1.1: APT for BYD
X_byd = sm.add_constant(merged_byd[['Mkt', 'BE/ME_factor', 'E/P_factor', 'CE/P_factor', 'Yld_factor']])
model_byd = sm.OLS(merged_byd['BYD'], X_byd).fit()

# %%
# Task 1.2: APT for Tesla
X_tesla = sm.add_constant(merged_tesla[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
model_tesla = sm.OLS(merged_tesla['Tesla_excess'], X_tesla).fit()

# %%
# Task 2.1: BYD beta to HSI
merged_byd_hsi = pd.merge(byd_monthly, hsi_monthly, left_index=True, right_index=True)
model_byd_hsi = sm.OLS(merged_byd_hsi['BYD'], sm.add_constant(merged_byd_hsi['HSI'])).fit()


# %%
# Task 2.2: Tesla beta to Nasdaq
merged_tesla_nasdaq = pd.merge(tesla_monthly, nasdaq_monthly, left_index=True, right_index=True)
model_tesla_nasdaq = sm.OLS(merged_tesla_nasdaq['Tesla'], sm.add_constant(merged_tesla_nasdaq['Nasdaq'])).fit()

# %%
# Task 2.3: Spread analysis
merged_spread = pd.merge(merged_byd_hsi, merged_tesla_nasdaq, left_index=True, right_index=True)
merged_spread['stock_spread'] = merged_spread['BYD'] - merged_spread['Tesla']
merged_spread['index_spread'] = merged_spread['HSI'] - merged_spread['Nasdaq']
model_spread = sm.OLS(merged_spread['stock_spread'], sm.add_constant(merged_spread['index_spread'])).fit()

# %%
# Output results
print("Task 1.1: BYD APT Analysis")
print(model_byd.summary())
print("\nTask 1.2: Tesla APT Analysis")
print(model_tesla.summary())
print("\nTask 2.1: BYD Beta to HSI:", model_byd_hsi.params['HSI'])
print("Task 2.2: Tesla Beta to Nasdaq:", model_tesla_nasdaq.params['Nasdaq'])
print("Task 2.3: Spread Beta:", model_spread.params['index_spread'])


