import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fi_t5_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A2/A2_FI_T5(Merge Query).xlsx"
trd_mnth_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A2/A2_TRD_Mnth.xlsx"

# Load FI_T5 data, skipping the first two rows (units info) and setting stock code as string
fi_t5 = pd.read_excel(fi_t5_path, header=0, skiprows=[1, 2], dtype={'FI_T5.Stkcd': str})

# Filter for consolidated statements (Typrep == 'A') and select relevant columns
fi_t5 = fi_t5[fi_t5['FI_T5.Typrep'] == 'A']
fi_t5 = fi_t5[['FI_T5.Stkcd', 'FI_T5.Accper', 'FI_T9.F091001A']]

# Rename columns for simplicity
fi_t5 = fi_t5.rename(columns={
    'FI_T5.Stkcd': 'Stkcd',        # Stock code
    'FI_T5.Accper': 'Accper',      # Accounting period end date
    'FI_T9.F091001A': 'NPS'        # Net Assets per Share
})

# Convert Accper to datetime, dropping rows with invalid dates
fi_t5['Accper'] = pd.to_datetime(fi_t5['Accper'], errors='coerce')
fi_t5 = fi_t5.dropna(subset=['Accper'])

# Sort by stock code and accounting period, then remove duplicates
fi_t5 = fi_t5.sort_values(by=['Stkcd', 'Accper'])
fi_t5 = fi_t5.drop_duplicates(subset=['Stkcd', 'Accper'])

# Load TRD_Mnth data, skipping the first two rows and setting stock code as string
trd_mnth = pd.read_excel(trd_mnth_path, header=0, skiprows=[1, 2], dtype={'Stkcd': str})

# Select relevant columns and rename for clarity
trd_mnth = trd_mnth[['Stkcd', 'Trdmnt', 'Mclsprc']]
trd_mnth = trd_mnth.rename(columns={'Mclsprc': 'close'})  # Monthly closing price

# Convert Trdmnt to datetime, adjusting to the last day of the month
trd_mnth['Trdmnt'] = pd.to_datetime(trd_mnth['Trdmnt'], format='%Y-%m') + pd.offsets.MonthEnd(0)

# Define a function to find the latest quarterly date for each trading month
def get_latest_quarter_date(group):
    """
    For each trading month in a stock's group, find the most recent quarterly accounting date.
    Args:
        group: A DataFrame group of monthly data for a single stock.
    Returns:
        The group with an added 'latest_quarter' column.
    """
    stock_quarter = fi_t5[fi_t5['Stkcd'] == group.name]
    group['latest_quarter'] = [
        stock_quarter[stock_quarter['Accper'] <= t].sort_values('Accper').tail(1)['Accper'].iloc[0]
        if not stock_quarter[stock_quarter['Accper'] <= t].empty else pd.NaT
        for t in group['Trdmnt']
    ]
    return group

# Apply the function to each stock group to match quarterly data
month_merged = trd_mnth.groupby('Stkcd').apply(get_latest_quarter_date)
month_merged = month_merged.reset_index(drop=True)

# Merge monthly and quarterly data
df = pd.merge(
    month_merged,
    fi_t5,
    how='left',                          # Left join to keep all monthly records
    left_on=['Stkcd', 'latest_quarter'], # Match on stock code and latest quarter date
    right_on=['Stkcd', 'Accper']         # Corresponding columns in quarterly data
)

# Calculate the monthly P/B ratio
df['monthly_PB'] = df['close'] / df['NPS']

# Replace infinite values (e.g., from division by zero) with NaN
df['monthly_PB'] = df['monthly_PB'].replace([np.inf, -np.inf], np.nan)

# Define a function to filter out extreme P/B values (below 5th or above 95th percentile)
def filter_extreme_values(group):
    """
    Filter P/B ratios within a month to exclude values below 5% or above 95% percentiles.
    Args:
        group: A DataFrame group of data for a single month.
    Returns:
        Filtered group with P/B ratios in the desired range.
    """
    pb_5 = group['monthly_PB'].quantile(0.05, interpolation='linear')  # 5th percentile
    pb_95 = group['monthly_PB'].quantile(0.95, interpolation='linear') # 95th percentile
    return group[(group['monthly_PB'].notna()) & 
                 (group['monthly_PB'] >= pb_5) & 
                 (group['monthly_PB'] <= pb_95)]

# Apply the filter by month
final_df = df.groupby('Trdmnt').apply(filter_extreme_values).reset_index(drop=True)