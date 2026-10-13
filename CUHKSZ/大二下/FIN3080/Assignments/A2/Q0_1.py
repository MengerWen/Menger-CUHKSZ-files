import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
fi_t5_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A2/A2_FI_T5(Merge Query).xlsx"
trd_mnth_path = "D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A2/A2_TRD_Mnth.xlsx"

# Load the financial indicators and trading data
fi_t5 = pd.read_excel(fi_t5_path, header=0, skiprows=[1, 2], dtype={'FI_T5.Stkcd': str})
trd_mnth = pd.read_excel(trd_mnth_path, header=0, skiprows=[1, 2], dtype={'Stkcd': str})

'''
Prepare the Financial Indicators Data
'''

# Filter for consolidated statements (Typrep == 'A') and select relevant columns
fi_t5 = fi_t5[fi_t5['FI_T5.Typrep'] == 'A']
fi_t5 = fi_t5[['FI_T5.Stkcd', 'FI_T5.Accper', 'FI_T9.F091001A']]

# Rename columns for simplicity
fi_t5 = fi_t5.rename(columns={
    'FI_T5.Stkcd': 'Stkcd',
    'FI_T5.Accper': 'Accper',
    'FI_T9.F091001A': 'Net_Assets_Per_Share'
})

# Convert Accper to datetime and drop invalid dates
fi_t5['Accper'] = pd.to_datetime(fi_t5['Accper'], errors='coerce')
fi_t5 = fi_t5.dropna(subset=['Accper'])

# Sort by stock code and accounting period
fi_t5 = fi_t5.sort_values(by=['Stkcd', 'Accper'])

# Remove duplicates
fi_t5 = fi_t5.drop_duplicates(subset=['Stkcd', 'Accper'])

'''
Prepare the Trading Data
'''

# Convert Trdmnt to datetime (last day of the month)
trd_mnth['Trdmnt'] = trd_mnth['Trdmnt'].str.extract(r'(\d{4}-\d{2})')[0]
trd_mnth['Trdmnt'] = pd.to_datetime(trd_mnth['Trdmnt'] + '-01', errors='coerce') + pd.offsets.MonthEnd(1)

# Drop rows with invalid Trdmnt dates (NaT)
trd_mnth = trd_mnth.dropna(subset=['Trdmnt'])

# Extract year and month for reference (optional)
trd_mnth['Year'] = trd_mnth['Trdmnt'].dt.year
trd_mnth['Month'] = trd_mnth['Trdmnt'].dt.month

# Sort by stock code and trading month
trd_mnth = trd_mnth.sort_values(by=['Stkcd', 'Trdmnt'])

# Remove duplicates
trd_mnth = trd_mnth.drop_duplicates(subset=['Stkcd', 'Trdmnt'])

'''
Merge Data to Match Latest Financial Indicators
'''

# Ensure both DataFrames are sorted for merge_asof
trd_mnth = trd_mnth.sort_values(by=['Stkcd', 'Trdmnt'])
fi_t5 = fi_t5.sort_values(by=['Stkcd', 'Accper'])

# Perform as-of merge (backward) to get the latest financial data for each trading month
merged = pd.merge_asof(
    trd_mnth,
    fi_t5,
    left_on='Trdmnt',
    right_on='Accper',
    by='Stkcd',
    direction='backward'
)

'''
Calculate P/B Ratio and Filter
'''

# Calculate P/B ratio: Monthly Closing Price / Net Assets per Share
merged['PB_Ratio'] = merged['Mclsprc'] / merged['Net_Assets_Per_Share']

# Filter data between Jan 2010 and Dec 2024
merged = merged[(merged['Trdmnt'] >= '2010-01-31') & (merged['Trdmnt'] <= '2024-12-31')]

# Exclude P/B ratios outside the 5th to 95th percentile range
pb_5th = merged['PB_Ratio'].quantile(0.05)
pb_95th = merged['PB_Ratio'].quantile(0.95)
filtered = merged[(merged['PB_Ratio'] >= pb_5th) & (merged['PB_Ratio'] <= pb_95th)]

# Display the first few rows of the filtered result
print("First few rows of the filtered data:")
print(filtered[['Stkcd', 'Trdmnt', 'Mclsprc', 'Net_Assets_Per_Share', 'PB_Ratio']].head())

# Optional: Save to Excel for further analysis
filtered.to_excel("PB_Ratios_Filtered.xlsx", index=False)
print("Filtered P/B ratios saved to 'PB_Ratios_Filtered.xlsx'")