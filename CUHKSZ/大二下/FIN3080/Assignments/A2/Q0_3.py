# Import necessary libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Define file paths (update these to your local paths if needed)
financial_file = r'D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A2/A2_FI_T5(Merge Query).xlsx'
trading_file = r'D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A2/A2_TRD_Mnth.xlsx'

# Step 1: Load and prepare financial data from FI_T5(Merge Query).xlsx
# Load the financial data
financial_df = pd.read_excel(financial_file)

# Filter for Typrep 'A' (assumed to be annual/consolidated reports)
financial_df = financial_df[financial_df['FI_T5.Typrep'] == 'A']

# Select relevant columns: Stock Code, Accounting Period, Net Assets per Share
financial_df = financial_df[['FI_T5.Stkcd', 'FI_T5.Accper', 'FI_T9.F091001A']]

# Rename columns for clarity
financial_df.columns = ['Stkcd', 'Accper', 'Net_Assets_per_Share']

# Convert Stkcd to string and Accper to datetime
financial_df['Stkcd'] = financial_df['Stkcd'].astype(str)
financial_df['Accper'] = pd.to_datetime(financial_df['Accper'], format='%Y-%m-%d')

# Drop rows with missing Net Assets per Share
financial_df = financial_df.dropna(subset=['Net_Assets_per_Share'])

# Step 2: Load and prepare trading data from TRD_Mnth.xlsx
# Load the trading data
trading_df = pd.read_excel(trading_file)

# Select relevant columns: Stock Code, Trading Month, Monthly Closing Price
trading_df = trading_df[['Stkcd', 'Trdmnt', 'Mclsprc']]

# Rename columns for clarity
trading_df.columns = ['Stkcd', 'Trdmnt', 'Monthly_Closing_Price']

# Convert Stkcd to string and Trdmnt to datetime (end of month)
trading_df['Stkcd'] = trading_df['Stkcd'].astype(str)
trading_df['Trdmnt'] = pd.to_datetime(trading_df['Trdmnt'], format='%Y-%m') + pd.offsets.MonthEnd(0)

# Drop rows with missing Monthly Closing Price
trading_df = trading_df.dropna(subset=['Monthly_Closing_Price'])

# Step 3: Merge financial and trading data using an as-of join
# Sort both dataframes by date for as-of merge
financial_df = financial_df.sort_values('Accper')
trading_df = trading_df.sort_values('Trdmnt')

# Perform the as-of merge to match each trading month with the latest financial data
merged_df = pd.merge_asof(
    trading_df,
    financial_df,
    left_on='Trdmnt',
    right_on='Accper',
    by='Stkcd',
    direction='backward'  # Ensures we get the latest financial data before or on the trading date
)

# Drop rows where no prior financial data exists (Accper is NaT or missing)
merged_df = merged_df.dropna(subset=['Accper', 'Net_Assets_per_Share'])

# Step 4: Calculate P/B ratios
# P/B = Monthly Closing Price / Net Assets per Share
merged_df['PB_Ratio'] = merged_df['Monthly_Closing_Price'] / merged_df['Net_Assets_per_Share']

# Drop rows where PB_Ratio is NaN, infinite, or negative (invalid cases)
merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
merged_df = merged_df.dropna(subset=['PB_Ratio'])
merged_df = merged_df[merged_df['PB_Ratio'] > 0]  # Ensure positive P/B ratios

# Step 5: Filter for the required time period (Jan 2010 to Dec 2024)
start_date = pd.to_datetime('2010-01-01')
end_date = pd.to_datetime('2024-12-31')
merged_df = merged_df[(merged_df['Trdmnt'] >= start_date) & (merged_df['Trdmnt'] <= end_date)]

# Step 6: Exclude P/B ratios outside the 5th and 95th percentiles
# Calculate the percentiles
pb_5th_percentile = merged_df['PB_Ratio'].quantile(0.05)
pb_95th_percentile = merged_df['PB_Ratio'].quantile(0.95)

# Filter out records outside the 5th to 95th percentile range
final_df = merged_df[
    (merged_df['PB_Ratio'] >= pb_5th_percentile) & 
    (merged_df['PB_Ratio'] <= pb_95th_percentile)
]

# Step 7: Finalize the output
# Select relevant columns for the final output
final_df = final_df[['Stkcd', 'Trdmnt', 'Monthly_Closing_Price', 'Net_Assets_per_Share', 'PB_Ratio']]

# Sort by stock code and trading month for readability
final_df = final_df.sort_values(['Stkcd', 'Trdmnt'])

# Display the first few rows of the final dataframe
print("Final DataFrame with Monthly P/B Ratios (Jan 2010 - Dec 2024, 5th-95th percentile):")
print(final_df.head())

# Optional: Save the final dataframe to a CSV file
final_df.to_csv('monthly_pb_ratios_2010_2024.csv', index=False)
print("\nData saved to 'monthly_pb_ratios_2010_2024.csv'")

# Display summary statistics of the P/B ratios
print("\nSummary Statistics of P/B Ratios:")
print(final_df['PB_Ratio'].describe())