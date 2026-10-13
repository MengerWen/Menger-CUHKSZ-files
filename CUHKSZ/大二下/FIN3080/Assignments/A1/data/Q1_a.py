import pandas as pd

# Load data from Excel files, skipping the Chinese column names and unit rows
fi_t5_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A1/data/FI_T5(Merge Query).xlsx"
trd_mnth_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A1/data/TRD_Mnth(Merge Query).xlsx"

# Read FI_T5 data (financial statements), specifying stock code as string
df_fi = pd.read_excel(fi_t5_path, header=0, skiprows=[1, 2], dtype={'FI_T5.Stkcd': str})
# Read TRD_Mnth data (trading data), specifying stock code as string
df_trd = pd.read_excel(trd_mnth_path, header=0, skiprows=[1, 2], dtype={'TRD_Mnth.Stkcd': str})

# Pad stock codes to 6 characters with leading zeros
df_fi['FI_T5.Stkcd'] = df_fi['FI_T5.Stkcd'].str.zfill(6)
df_trd['TRD_Mnth.Stkcd'] = df_trd['TRD_Mnth.Stkcd'].str.zfill(6)

# Process FI_T5: Filter for financial statement type 'A' (assumed to be consolidated statements)
df_fi_filtered = df_fi[df_fi['FI_T5.Typrep'] == 'A'].copy()

# Convert accounting period end date to datetime and drop invalid dates
df_fi_filtered['Accper_date'] = pd.to_datetime(df_fi_filtered['FI_T5.Accper'], errors='coerce')
df_fi_filtered = df_fi_filtered.dropna(subset=['Accper_date'])

# Rename establishment date column to avoid conflicts during merge
df_fi_filtered = df_fi_filtered.rename(columns={'csmar_listedcoinfo.Estbdt': 'Estbdt'})

# Process TRD_Mnth: Convert trading month to end-of-month date
# Extract YYYY-MM format and convert to end-of-month datetime
df_trd['TRD_Mnth.Trdmnt'] = df_trd['TRD_Mnth.Trdmnt'].str.extract(r'(\d{4}-\d{2})')[0]
df_trd['Trdmnt_date'] = pd.to_datetime(df_trd['TRD_Mnth.Trdmnt'] + '-01') + pd.offsets.MonthEnd(1)

# Extract year and month from trading month for quarter-end calculation
df_trd['Year'] = df_trd['TRD_Mnth.Trdmnt'].str[:4].astype(int)
df_trd['Month'] = df_trd['TRD_Mnth.Trdmnt'].str[5:7].astype(int)

# Function to determine the quarter-end date based on the month
def get_quarter_end_date(year, month):
    if month <= 3:
        return pd.Timestamp(year, 3, 31)
    elif month <= 6:
        return pd.Timestamp(year, 6, 30)
    elif month <= 9:
        return pd.Timestamp(year, 9, 30)
    else:
        return pd.Timestamp(year, 12, 31)

# Apply the function to get the quarter-end date for each trading month
df_trd['Quarter_End_Date'] = df_trd.apply(lambda row: get_quarter_end_date(row['Year'], row['Month']), axis=1)

# Sort trading data by stock code and trading date
df_trd_sorted = df_trd.sort_values(['TRD_Mnth.Stkcd', 'Trdmnt_date'])

# Merge trading and financial data using asof merge (backward direction)
# This ensures each trading month uses the most recent financial data available
merged = pd.merge_asof(
    df_trd_sorted.sort_values('Trdmnt_date'),
    df_fi_filtered.sort_values('Accper_date'),
    left_on='Trdmnt_date',
    right_on='Accper_date',
    left_by='TRD_Mnth.Stkcd',
    right_by='FI_T5.Stkcd',
    direction='backward'
)

# Calculate P/E ratio: Monthly Closing Price / Earnings per Share (TTM2)
merged['PE_ratio'] = merged['TRD_Mnth.Mclsprc'] / merged['FI_T9.F090102C']

# Calculate P/B ratio
# Total Equity = Total Assets - Total Liabilities
merged['Total_Equity'] = merged['FS_Combas.A001000000'] - merged['FS_Combas.A002000000']
# Market Value in CNY (convert from thousands to match units)
merged['Market_Value_CNY'] = merged['TRD_Mnth.Msmvosd'] * 1000
# Number of shares = Market Value / Closing Price
merged['Shares'] = merged['Market_Value_CNY'] / merged['TRD_Mnth.Mclsprc']
# Book Value per Share = Total Equity / Shares
merged['BVPS'] = merged['Total_Equity'] / merged['Shares']
# P/B Ratio = Closing Price / Book Value per Share
merged['PB_ratio'] = merged['TRD_Mnth.Mclsprc'] / merged['BVPS']

# Calculate R&D Expenses / Total Assets (quarterly data from FI_T5)
merged['RD_TA'] = merged['FS_Comins.B001216000'] / merged['FS_Combas.A001000000']

# Convert establishment date to datetime for age calculation
merged['Estbdt'] = pd.to_datetime(merged['Estbdt'], errors='coerce')

# Calculate Firm Age (in years) based on the quarter-end date
merged['Firm_Age'] = (merged['Quarter_End_Date'] - merged['Estbdt']).dt.days / 365.25

# Select and rename final output columns for clarity
result = merged[[
    'TRD_Mnth.Stkcd', 'TRD_Mnth.Trdmnt', 'TRD_Mnth.Mclsprc',
    'PE_ratio', 'PB_ratio', 'RD_TA', 'Firm_Age'
]]
result.columns = [
    'Stock Code', 'Trading Month', 'Closing Price',
    'P/E Ratio', 'P/B Ratio', 'R&D/Total Assets', 'Firm Age (Years)'
]

# Optional: Verify stock code format before saving
print("Stock Code dtype:", result['Stock Code'].dtype)
print("Sample Stock Codes:\n", result['Stock Code'].head())

# Save to Excel with proper formatting to preserve leading zeros
writer = pd.ExcelWriter('Calculated_Ratios.xlsx', engine='xlsxwriter')
result.to_excel(writer, index=False, sheet_name='Sheet1')
workbook = writer.book
worksheet = writer.sheets['Sheet1']
text_format = workbook.add_format({'num_format': '@'})  # Set format to text
worksheet.set_column('A:A', None, text_format)  # Apply to 'Stock Code' column (first column)
writer.close()

# Print a confirmation message
print("Calculations completed. Results saved to 'Calculated_Ratios.xlsx'.")