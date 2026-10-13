# Import necessary libraries (already imported in preprocessing, but repeated for completeness)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

# Load TRD_Mnth data with monthly returns (modify the preprocessing step)
trd_mnth_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A2/A2_TRD_Mnth.xlsx"
trd_mnth = pd.read_excel(trd_mnth_path, header=0, skiprows=[1, 2], dtype={'Stkcd': str})
trd_mnth = trd_mnth[['Stkcd', 'Trdmnt', 'Mclsprc', 'Mretnd']]
trd_mnth = trd_mnth.rename(columns={'Mclsprc': 'close', 'Mretnd': 'return'})
trd_mnth['Trdmnt'] = pd.to_datetime(trd_mnth['Trdmnt'], format='%Y-%m') + pd.offsets.MonthEnd(0)

# Assuming the rest of the preprocessing code runs as provided to generate final_df
# For brevity, I'll proceed assuming final_df is available, but we need 'return' in it
# Let's redefine the necessary steps to include 'return'

# Define the function to find the latest quarterly date (from preprocessing)
def get_latest_quarter_date(group):
    stock_quarter = fi_t5[fi_t5['Stkcd'] == group.name]
    group['latest_quarter'] = [
        stock_quarter[stock_quarter['Accper'] <= t].sort_values('Accper').tail(1)['Accper'].iloc[0]
        if not stock_quarter[stock_quarter['Accper'] <= t].empty else pd.NaT
        for t in group['Trdmnt']
    ]
    return group

# Apply the function and merge data
month_merged = trd_mnth.groupby('Stkcd').apply(get_latest_quarter_date).reset_index(drop=True)
df = pd.merge(
    month_merged,
    fi_t5,
    how='left',
    left_on=['Stkcd', 'latest_quarter'],
    right_on=['Stkcd', 'Accper']
)
df['monthly_PB'] = df['close'] / df['NPS']
df['monthly_PB'] = df['monthly_PB'].replace([np.inf, -np.inf], np.nan)

# Filter extreme P/B values
def filter_extreme_values(group):
    pb_5 = group['monthly_PB'].quantile(0.05, interpolation='linear')
    pb_95 = group['monthly_PB'].quantile(0.95, interpolation='linear')
    return group[(group['monthly_PB'].notna()) & 
                 (group['monthly_PB'] >= pb_5) & 
                 (group['monthly_PB'] <= pb_95)]

final_df = df.groupby('Trdmnt').apply(filter_extreme_values).reset_index(drop=True)

# Now, proceed with TASK (**)

# Step 1: Add previous month's trading date
final_df['prev_Trdmnt'] = final_df['Trdmnt'] - MonthEnd(1)

# Step 2: Create a DataFrame with lagged P/B ratios
lagged_df = final_df[['Stkcd', 'Trdmnt', 'monthly_PB']].rename(columns={'monthly_PB': 'lagged_PB'})

# Step 3: Merge to get last-month P/B ratios
final_df = pd.merge(
    final_df,
    lagged_df[['Stkcd', 'Trdmnt', 'lagged_PB']],
    how='left',
    left_on=['Stkcd', 'prev_Trdmnt'],
    right_on=['Stkcd', 'Trdmnt'],
    suffixes=('', '_lagged')
)

# Drop the redundant Trdmnt_lagged column
final_df = final_df.drop(columns=['Trdmnt_lagged'])

# Step 4: Assign deciles based on lagged_PB for each month
def assign_decile(group):
    try:
        group['decile'] = pd.qcut(group['lagged_PB'], 10, labels=False, duplicates='drop')
    except ValueError:
        # Handle cases with insufficient unique values
        group['decile'] = np.nan
    return group

final_df = final_df.groupby('Trdmnt').apply(assign_decile).reset_index(drop=True)

# Step 5: Filter for the required period (Jan 2010 to Dec 2024) and calculate portfolio returns
final_df = final_df[(final_df['Trdmnt'] >= '2010-01-31') & (final_df['Trdmnt'] <= '2024-12-31')]
portfolio_returns = final_df.groupby(['Trdmnt', 'decile'])['return'].mean().reset_index()

# Step 6: Calculate average monthly returns for each decile
average_returns = portfolio_returns.groupby('decile')['return'].mean().reset_index()

# Step 7: Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(average_returns['decile'], average_returns['return'], color='skyblue')
plt.xlabel('P/B Ratio Decile (0: Lowest, 9: Highest)')
plt.ylabel('Average Monthly Return')
plt.title('Average Monthly Returns by P/B Ratio Decile (Jan 2010 - Dec 2024)')
plt.xticks(range(10))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Display average returns for reference
print(average_returns)