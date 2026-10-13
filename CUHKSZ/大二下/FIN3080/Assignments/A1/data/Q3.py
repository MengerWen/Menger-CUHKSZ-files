import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = r'D:\MG\！CUHKSZ\~！大二 下\FIN3080\Assignments\A1\problem 3_data.csv'
df = pd.read_csv(file_path, parse_dates=['EndDate'])
df['Year'] = df['EndDate'].dt.year

# Identify valid symbols with complete data from 2011-2020 for ROE and TotalRevenue, and 2010 TotalRevenue
valid_symbols = []
for symbol, group in df.groupby('Symbol'):
    years_present = group['Year'].unique()
    required_years = set(range(2011, 2021))
    if not required_years.issubset(years_present):
        continue
    mask_2011_2020 = group['Year'].between(2011, 2020)
    if not (group.loc[mask_2011_2020, 'ROEC'].notnull().all() and 
            group.loc[mask_2011_2020, 'TotalRevenue'].notnull().all()):
        continue
    if not group[group['Year'] == 2010]['TotalRevenue'].notnull().any():
        continue
    valid_symbols.append(symbol)

subsample = df[df['Symbol'].isin(valid_symbols)].copy()

# Calculate TotalRevenue growth rate
subsample_sorted = subsample.sort_values(['Symbol', 'Year'])
subsample_sorted['PrevTotalRevenue'] = subsample_sorted.groupby('Symbol')['TotalRevenue'].shift(1)
subsample_sorted['GrowthRate'] = (subsample_sorted['TotalRevenue'] - subsample_sorted['PrevTotalRevenue']) / subsample_sorted['PrevTotalRevenue']
subsample_sorted = subsample_sorted[subsample_sorted['Year'].between(2011, 2020)]

# Compute annual medians
annual_median_roe = subsample_sorted.groupby('Year')['ROEC'].median().rename('ROE_Median')
annual_median_growth = subsample_sorted.groupby('Year')['GrowthRate'].median().rename('Growth_Median')

# Create DataFrame of medians and save to Excel
medians_df = pd.concat([annual_median_roe, annual_median_growth], axis=1).reset_index()
medians_df.to_excel('annual_medians.xlsx', index=False)  # Saves to current working directory

# Merge medians back to subsample_sorted for analysis
subsample_sorted = subsample_sorted.merge(annual_median_roe, on='Year')
subsample_sorted = subsample_sorted.merge(annual_median_growth, on='Year')

# Determine above median status
subsample_sorted['Above_ROE'] = subsample_sorted['ROEC'] > subsample_sorted['ROE_Median']
subsample_sorted['Above_Growth'] = subsample_sorted['GrowthRate'] > subsample_sorted['Growth_Median']

# Pivot tables for above status
roe_pivot = subsample_sorted.pivot(index='Symbol', columns='Year', values='Above_ROE')
growth_pivot = subsample_sorted.pivot(index='Symbol', columns='Year', values='Above_Growth')

# Calculate cumulative percentages
roe_percent = pd.Series()
growth_percent = pd.Series()

for year in range(2011, 2021):
    years_needed = list(range(2011, year + 1))
    roe_current = roe_pivot[years_needed].all(axis=1).mean() * 100
    growth_current = growth_pivot[years_needed].all(axis=1).mean() * 100
    roe_percent[year] = roe_current
    growth_percent[year] = growth_current

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(roe_percent.index, roe_percent.values, marker='o', label='ROE')
plt.title('Percentage of Companies with Consistently Above-Median ROE (2011-2020)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(range(2011, 2021))
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(growth_percent.index, growth_percent.values, marker='o', color='orange', label='Revenue Growth Rate')
plt.title('Percentage of Companies with Consistently Above-Median Revenue Growth Rate (2011-2020)')
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.xticks(range(2011, 2021))
plt.grid(True)
plt.show()