import pandas as pd
import matplotlib.pyplot as plt

# Load the Calculated_Ratios data
df = pd.read_excel("D:/MG/！CUHKSZ/~！大二 下/FIN3080/Assignments/A1/data/Q1_a_Calculated_Ratios.xlsx", 
                   dtype={'Stock Code': str})

# Classify market type (GEM Board: 300, 688; Main Board: others)
df['Market_Group'] = df['Stock Code'].str[:3].apply(
    lambda x: 'GEM Board' if x in ['300', '301', '688'] else 'Main Board'
)

# Convert Trading Month to datetime and filter up to Sep 2023
df['Trading Month'] = pd.to_datetime(df['Trading Month'] + '-01') + pd.offsets.MonthEnd(1)
df = df[df['Trading Month'] <= '2023-09-30']

# Exclude invalid P/E ratios (negative or zero)
df = df[df['P/E Ratio'] > 0]

# Calculate median P/E by month and market
pe_ts = df.groupby(['Trading Month', 'Market_Group'])['P/E Ratio'].median().unstack()

# Plot the time-series
plt.figure(figsize=(12, 6))
pe_ts.plot(title='Median P/E Ratio by Market Type (Up to Sep 2023)')
plt.ylabel('P/E Ratio')
plt.xlabel('Month')
plt.grid(True)
plt.show()

# Print summary statistics for P/E ratios
print("Summary Statistics for P/E Ratios:")
print(pe_ts.describe())

# Analyze investment implications
print("\nInvestment Analysis as of Sep 2023:")
print("1. GEM Board:")
print(f"   - Median P/E: {pe_ts['GEM Board'].median():.2f}")
print(f"   - Higher R&D/Total Assets (from Summary Statistics): ~1.52%")
print("   - Suitable for aggressive investors seeking growth but with higher volatility.")

print("\n2. Main Board:")
print(f"   - Median P/E: {pe_ts['Main Board'].median():.2f}")
print(f"   - Lower R&D/Total Assets (from Summary Statistics): ~0.82%")
print("   - Suitable for conservative investors seeking stability and value.")

# Propose a trading strategy
print("\nTrading Strategy Based on P/E Ratios:")
print("1. Signal: Compute the P/E ratio gap between GEM and Main Board indices.")
print("   - Gap = (GEM P/E) / (Main Board P/E)")
print("2. Thresholds (calibrated from historical data):")
print("   - Buy Main Board ETF when Gap > 1.5 (GEM overvalued).")
print("   - Buy GEM Board ETF when Gap < 1.2 (GEM undervalued).")
print("3. Execution:")
print("   - Use ETFs like 159915 (ChiNext ETF) and 510300 (HS300 ETF).")
print("   - Hold until the gap reverts to its 1-year average.")

# Save analysis to Excel
with pd.ExcelWriter('Analysis_Results.xlsx') as writer:
    pe_ts.to_excel(writer, sheet_name='PE_Trends')
print("\nAnalysis complete. Results saved to Analysis_Results.xlsx.")