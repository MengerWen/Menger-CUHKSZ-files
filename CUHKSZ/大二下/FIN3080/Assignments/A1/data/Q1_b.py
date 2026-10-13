import pandas as pd

# Load data from Excel files
fi_t5_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A1/data/FI_T5(Merge Query).xlsx"
trd_mnth_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A1/data/TRD_Mnth(Merge Query).xlsx"

# Read FI_T5 data
df_fi = pd.read_excel(fi_t5_path, header=0, skiprows=[1, 2], dtype={'FI_T5.Stkcd': str})
# Read TRD_Mnth data
df_trd = pd.read_excel(trd_mnth_path, header=0, skiprows=[1, 2], dtype={'TRD_Mnth.Stkcd': str})

# Pad stock codes to 6 characters
df_fi['FI_T5.Stkcd'] = df_fi['FI_T5.Stkcd'].str.zfill(6)
df_trd['TRD_Mnth.Stkcd'] = df_trd['TRD_Mnth.Stkcd'].str.zfill(6)

# Filter FI_T5 for consolidated statements (Type A)
df_fi_filtered = df_fi[df_fi['FI_T5.Typrep'] == 'A'].copy()

# Convert dates to datetime
df_fi_filtered['Accper_date'] = pd.to_datetime(df_fi_filtered['FI_T5.Accper'], errors='coerce')
df_fi_filtered.dropna(subset=['Accper_date'], inplace=True)
df_fi_filtered.rename(columns={'csmar_listedcoinfo.Estbdt': 'Estbdt'}, inplace=True)

# Process TRD_Mnth dates
df_trd['TRD_Mnth.Trdmnt'] = df_trd['TRD_Mnth.Trdmnt'].str.extract(r'(\d{4}-\d{2})')[0]
df_trd['Trdmnt_date'] = pd.to_datetime(df_trd['TRD_Mnth.Trdmnt'] + '-01') + pd.offsets.MonthEnd(1)

# Determine quarter-end dates
df_trd['Year'] = df_trd['TRD_Mnth.Trdmnt'].str[:4].astype(int)
df_trd['Month'] = df_trd['TRD_Mnth.Trdmnt'].str[5:7].astype(int)
df_trd['Quarter_End_Date'] = df_trd.apply(
    lambda row: pd.Timestamp(row['Year'], min(row['Month']//3*3 + 3, 12), 30 if min(row['Month']//3*3 + 3, 12) in [3,6,9,12] else 31), axis=1
)

# Merge data
merged = pd.merge_asof(
    df_trd.sort_values('Trdmnt_date'),
    df_fi_filtered.sort_values('Accper_date'),
    left_on='Trdmnt_date',
    right_on='Accper_date',
    left_by='TRD_Mnth.Stkcd',
    right_by='FI_T5.Stkcd',
    direction='backward'
)

# Calculate financial ratios
merged['PE_ratio'] = merged['TRD_Mnth.Mclsprc'] / merged['FI_T9.F090102C']
merged['Total_Equity'] = merged['FS_Combas.A001000000'] - merged['FS_Combas.A002000000']
merged['Market_Value_CNY'] = merged['TRD_Mnth.Msmvosd'] * 1000
merged['Shares'] = merged['Market_Value_CNY'] / merged['TRD_Mnth.Mclsprc']
merged['BVPS'] = merged['Total_Equity'] / merged['Shares']
merged['PB_ratio'] = merged['TRD_Mnth.Mclsprc'] / merged['BVPS']
merged['RD_TA'] = merged['FS_Comins.B001216000'] / merged['FS_Combas.A001000000']
merged['Estbdt'] = pd.to_datetime(merged['Estbdt'], errors='coerce')
merged['Firm_Age'] = (merged['Quarter_End_Date'] - merged['Estbdt']).dt.days / 365.25

# Classify market groups
merged['Market_Group'] = merged['TRD_Mnth.Stkcd'].str[:3].apply(
    lambda x: 'GEM Board' if x in ['300', '301', '688'] else 'Main Board'
)

# Define variables and aggregation
monthly_vars = ['TRD_Mnth.Mretwd', 'PE_ratio', 'PB_ratio']
quarterly_vars = ['FI_T5.F050204C', 'FI_T5.F050504C', 'RD_TA', 'Firm_Age']
agg_funcs = ['count', 'mean', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'std']

# Compute monthly stats
monthly_stats = merged.groupby('Market_Group')[monthly_vars].agg(agg_funcs)

# Compute quarterly stats (deduplicate)
df_quarterly = merged.drop_duplicates(subset=['FI_T5.Stkcd', 'Accper_date'])
quarterly_stats = df_quarterly.groupby('Market_Group')[quarterly_vars].agg(agg_funcs)

# Rename columns for clarity
monthly_stats.columns = [f"{var}_{stat}" for var in monthly_vars for stat in ['count', 'mean', 'median', 'p25', 'p75', 'std']]
quarterly_stats.columns = [f"{var}_{stat}" for var in quarterly_vars for stat in ['count', 'mean', 'median', 'p25', 'p75', 'std']]

# Save to Excel
with pd.ExcelWriter('Summary_Statistics.xlsx') as writer:
    monthly_stats.T.to_excel(writer, sheet_name='Monthly Stats')
    quarterly_stats.T.to_excel(writer, sheet_name='Quarterly Stats')

print("Summary statistics saved to Summary_Statistics.xlsx")