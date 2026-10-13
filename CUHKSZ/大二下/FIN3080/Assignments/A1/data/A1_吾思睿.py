import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROE_ROA_df = pd.read_excel('FI_T5.xlsx')

columns = ['Stkcd', 'Accper','Typrep','F050202B', 'F050502B']
ROE_ROA_df = ROE_ROA_df[columns]
ROE_ROA_df = ROE_ROA_df[ROE_ROA_df['Typrep'] == 'A']
ROE_ROA_df = ROE_ROA_df.drop(columns=['Typrep'])
ROE_ROA_df['ROA'] = ROE_ROA_df['F050202B']
ROE_ROA_df['ROE'] = ROE_ROA_df['F050502B']
ROE_ROA_df.drop(columns=['F050202B', 'F050502B'], inplace=True)

EPS_df = pd.read_excel('FI_T9.xlsx')
columns = ['Stkcd', 'Accper','Typrep','F090102C']
EPS_df = EPS_df[columns]
EPS_df = EPS_df[EPS_df['Typrep'] == 'A']
EPS_df['EPS'] = EPS_df['F090102C']
EPS_df.drop(columns=['Typrep', 'F090102C'], inplace=True)
data = pd.merge(ROE_ROA_df, EPS_df, on=['Stkcd', 'Accper'])

data['Accper'] = pd.to_datetime(data['Accper'], format='%Y-%m-%d')

R_D_expense = pd.read_csv('FS_Comins.csv',dtype={'Stkcd': str})

columns = ['Stkcd', 'Accper','Typrep','B001216000']
R_D_expense =R_D_expense[columns]
R_D_expense = R_D_expense[R_D_expense['Typrep'] == 'A']
R_D_expense['RDexpense'] =R_D_expense['B001216000']
R_D_expense.drop(columns=['Typrep', 'B001216000'], inplace=True)
R_D_expense['Accper'] = pd.to_datetime(R_D_expense['Accper'], format='%Y-%m-%d')
data = pd.merge(data, R_D_expense, on=['Stkcd', 'Accper'])

asset_lia_df = pd.read_csv('FS_Combas.csv',dtype={'Stkcd': str})
columns = ['Stkcd', 'Accper','Typrep','A001000000','A002000000']
asset_lia_df =asset_lia_df[columns]
asset_lia_df = asset_lia_df[asset_lia_df['Typrep'] == 'A']
asset_lia_df['total_asset'] =asset_lia_df['A001000000']
asset_lia_df['total_lia'] =asset_lia_df['A002000000']
asset_lia_df.drop(columns=['Typrep','A001000000','A002000000'], inplace=True)
asset_lia_df['Accper'] = pd.to_datetime(asset_lia_df['Accper'], format='%Y-%m-%d')
quarter_data = pd.merge(data, asset_lia_df, on=['Stkcd', 'Accper'])

month_data = pd.read_csv('TRD_Mnth.csv',dtype={'Stkcd': str})
columns = ['Stkcd', 'Trdmnt','Mclsprc','Msmvosd','Mretnd']
new_name = ['Stkcd', 'Trdmnt','close','market_cap','return']
month_data = month_data[columns]
month_data.columns = new_name

month_data['Trdmnt'] = pd.to_datetime(month_data['Trdmnt'], format='%Y-%m') + pd.offsets.MonthEnd(0)

company_basic = pd.read_csv('TRD_Co.csv',dtype={'Stkcd': str})
columns = ['Stkcd','Listdt','Estbdt','Markettype']
company_basic = company_basic[columns]

stock = set(company_basic['Stkcd'].tolist()).intersection(month_data['Stkcd'].tolist()).intersection(quarter_data['Stkcd'].tolist())
company_basic = company_basic[company_basic['Stkcd'].isin(stock)]
month_data = month_data[month_data['Stkcd'].isin(stock)]
quarter_data = quarter_data[quarter_data['Stkcd'].isin(stock)]

month_data['Trdmnt'] = pd.to_datetime(month_data['Trdmnt'])
quarter_data['Accper'] = pd.to_datetime(quarter_data['Accper'])
month_data.set_index('Stkcd', inplace=True)
quarter_data.set_index('Stkcd', inplace=True)

def get_latest_quarter_date(group, quarter_data):
    # Screening out quarterly data with the same ticker symbol
    stock_data = quarter_data[quarter_data.index == group.name]
    
    # Find the most recent quarterly date prior to that month
    latest_quarters = []
    for row in group.itertuples():
        # 使用getattr()访问Trdmnt字段
        latest_quarter = stock_data[stock_data['Accper'] <= getattr(row, 'Trdmnt')].sort_values('Accper', ascending=False).head(1)
        
        if not latest_quarter.empty:
            latest_quarters.append(latest_quarter['Accper'].iloc[0])
        else:
            latest_quarters.append(pd.NaT)
    
    group['latest_quarter_date'] = latest_quarters
    return group

# Use groupby to apply to each stock
month_merged = month_data.groupby(month_data.index).apply(lambda group: get_latest_quarter_date(group, quarter_data))

month_merged = month_merged.droplevel(level=0)
month_merged.reset_index(inplace= True)

df_merged = pd.merge(month_merged, quarter_data, how='left', left_on=['Stkcd', 'latest_quarter_date'], right_on=['Stkcd', 'Accper'])
df_merged = pd.merge(df_merged, company_basic, on='Stkcd')

final_df = df_merged.copy()
final_df.set_index(['Stkcd','Trdmnt'], inplace=True)
final_df['monthly_PE'] = final_df['close'] / final_df['EPS']
total_share = final_df['market_cap']*10**3/final_df['close']
BVPS = (final_df['total_asset'] - final_df['total_lia'])/total_share
final_df['monthly_PB'] = final_df['close']/BVPS
final_df['monthly_PB'] = final_df.apply(
    lambda row: float('nan') if  row['monthly_PB'] == float('inf') else row['monthly_PB'], axis=1
)
final_df['RD/assets'] = final_df['RDexpense']/final_df['total_asset']
final_df['Estbdt'] = pd.to_datetime(final_df['Estbdt'], format='%Y-%m-%d')
final_df['Listdt'] = pd.to_datetime(final_df['Listdt'], format='%Y-%m-%d')
final_df['quarter_age'] = (final_df.index.get_level_values('Trdmnt') - final_df['Estbdt'])