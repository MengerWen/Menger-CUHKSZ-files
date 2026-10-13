import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

month_close_cap_return = pd.read_csv('TRD_Mnth.csv',dtype={'Stkcd': str})
columns = ['Stkcd', 'Trdmnt','Mclsprc','Msmvosd','Mretnd']
new_name = ['Stkcd', 'Trdmnt','close','market_cap','return']
month_close_cap_return = month_close_cap_return[columns]
month_close_cap_return.columns = new_name
month_close_cap_return['Trdmnt'] = pd.to_datetime(month_close_cap_return['Trdmnt'], format='%Y-%m') + pd.offsets.MonthEnd(0)

nps= pd.read_csv('FI_T9.csv',dtype={'Stkcd': str})
columns = ['Stkcd', 'Accper','Typrep','F091002A']
nps =nps[columns]
nps = nps[nps['Typrep'] == 'A']
nps['NPS'] = nps['F091002A']
nps.drop(columns=['Typrep', 'F091002A'], inplace=True)

roe = pd.read_csv('FI_T5.csv',dtype={'Stkcd': str})
columns = ['Stkcd', 'Accper','Typrep','F050504C']
roe =roe[columns]
roe = roe[roe['Typrep'] == 'A']
roe['ROE'] = roe['F050504C']
roe.drop(columns=['Typrep', 'F050504C'], inplace=True)

quarter_roe_nps = pd.merge(nps, roe, on=['Stkcd', 'Accper'])
quarter_roe_nps

asset_lia = pd.read_csv('FS_Combas.csv',dtype={'Stkcd': str})
columns = ['Stkcd', 'Accper','Typrep','A001000000','A002000000']
asset_lia=asset_lia[columns]
asset_lia = asset_lia[asset_lia['Typrep'] == 'A']
asset_lia['total_asset'] =asset_lia['A001000000']
asset_lia['total_liability'] =asset_lia['A002000000']
asset_lia.drop(columns=['Typrep', 'A001000000', 'A002000000'], inplace=True)

quarter_data = pd.merge(quarter_roe_nps, asset_lia, on=['Stkcd', 'Accper'])

daily_vola = pd.read_csv('STK_MKT_STKBTAL.csv',dtype={'Symbol': str})

quarter_data['Accper'] = pd.to_datetime(quarter_data['Accper']) 
month_close_cap_return.set_index('Stkcd', inplace=True)
quarter_data.set_index('Stkcd', inplace=True)

#combine monthly and quarterly data
def get_latest_quarter_date(group, quarter_data):
    # 筛选出相同股票代码的季度数据
    stock_data = quarter_data[quarter_data.index == group.name]
    
    # 找到该月之前最近的季度日期
    latest_quarters = []
    for row in group.itertuples():
        
        latest_quarter = stock_data[stock_data['Accper'] <= getattr(row, 'Trdmnt')].sort_values('Accper', ascending=False).head(1)
        
        if not latest_quarter.empty:
            latest_quarters.append(latest_quarter['Accper'].iloc[0])
        else:
            latest_quarters.append(pd.NaT)
    
    group['latest_quarter_date'] = latest_quarters
    return group

month_merged =month_close_cap_return.groupby(month_close_cap_return.index).apply(lambda group: get_latest_quarter_date(group, quarter_data))

month_merged = month_merged.droplevel(level=0)
month_merged.reset_index(inplace= True)

df= pd.merge(month_merged, quarter_data, how='left', left_on=['Stkcd', 'latest_quarter_date'], right_on=['Stkcd', 'Accper'])

#calculate PB
final_df = df.copy()
final_df.set_index(['Stkcd','Trdmnt'], inplace=True)
total_share = final_df['market_cap']*10**3/final_df['close']
BVPS = (final_df['total_asset'] - final_df['total_liability'])/total_share
final_df['monthly_PB'] = final_df['close']/BVPS
final_df['monthly_PB'] = final_df.apply(
    lambda row: float('nan') if  row['monthly_PB'] == float('inf') else row['monthly_PB'], axis=1
)

#delete extreme pb 
def filter_extreme_values(group):
    # 计算当月所有股票的 5% 和 95% 分位数
    pb_5 = group['monthly_PB'].quantile(0.05)
    pb_95 = group['monthly_PB'].quantile(0.95)
    # 只保留 P/B 在 [5%, 95%] 范围内的股票
    return group[(group['monthly_PB'] >= pb_5) & (group['monthly_PB'] <= pb_95)]

final_df = final_df.groupby('Trdmnt').apply(filter_extreme_values)

final_df = final_df .droplevel(level=0)

daily_vola.set_index(['Symbol','TradingDate'], inplace=True)

'''
Q2
'''

rolling_df = final_df.loc[:,['monthly_PB','return']]

rolling_df = final_df.loc[:,['monthly_PB','return']]
trading_date = rolling_df.index.get_level_values(1).unique()
portfolio_returns = pd.DataFrame()
for i in range(1, len(trading_date)):
    last_month = trading_date[i-1]
    current_month = trading_date[i]

    last_month_data = rolling_df.xs(last_month, level=1)
    last_month_data['decile'] = pd.qcut(last_month_data['monthly_PB'], 10, labels=False)
    current_month_data = rolling_df.xs(current_month, level=1)
    returns = []
    for decile in range(10):
        stocks_in_decile = last_month_data[last_month_data['decile'] == decile].index.tolist()
            
        decile_stocks_returns = current_month_data.loc[current_month_data.index.isin(stocks_in_decile)]

        if not decile_stocks_returns.empty and 'return' in decile_stocks_returns.columns:
                avg_return = decile_stocks_returns['return'].mean()
                returns.append(avg_return)
    portfolio_returns[current_month] = returns    
portfolio_returns = portfolio_returns.T

avg_portfolio_returns = portfolio_returns.mean()
plt.figure(figsize=(12, 6))
bars = plt.bar(avg_portfolio_returns.index, avg_portfolio_returns.values * 100)  # 转换为百分比


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom')

plt.title('Average Monthly Returns of P/B-Sorted Portfolios (2010-2024)')
plt.xlabel('Portfolio (P1: Lowest P/B, P10: Highest P/B)')
plt.ylabel('Average Monthly Return (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 计算累积收益率
# 从第一个月开始，累积乘积（1 + 月度收益率）
cumulative_returns = (1 + portfolio_returns).cumprod()

# 使用更加可辨别的颜色方案
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# 绘制累积收益率折线图
plt.figure(figsize=(14, 8))

# 遍历每个投资组合并绘制其累积收益率
for i, col in enumerate(cumulative_returns.columns):
    plt.plot(cumulative_returns.index, cumulative_returns[col], 
             label=col, color=colors[i], linewidth=1.5)

# 添加图表标题和标签
plt.title('Cumulative Returns of P/B-Sorted Portfolios (2010-2024)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return (Initial Investment = 1)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# 优化x轴日期显示
plt.xticks(rotation=45)
# 设置适当的时间间隔，例如每年
years = pd.date_range(start=min(cumulative_returns.index), 
                      end=max(cumulative_returns.index), 
                      freq='YS')  # Year Start
plt.xticks(years, [d.strftime('%Y') for d in years])

# 添加图表说明
plt.figtext(0.02, 0.02, 'P1: Lowest P/B ratio, P10: Highest P/B ratio', 
            fontsize=10, style='italic')

plt.tight_layout()
plt.show()

# 打印最终累积收益
final_returns = cumulative_returns.iloc[-1]
print("最终累积收益 (初始投资 = 1):")
for col in cumulative_returns.columns:
    print(f"{col}: {final_returns[col]:.4f}")

# 计算年化收益率
years = len(cumulative_returns) / 12  # 假设是月度数据
annualized_returns = (final_returns ** (1/years) - 1) * 100
print("\n年化收益率 (%):")
for col in cumulative_returns.columns:
    print(f"{col}: {annualized_returns[col]:.2f}%")