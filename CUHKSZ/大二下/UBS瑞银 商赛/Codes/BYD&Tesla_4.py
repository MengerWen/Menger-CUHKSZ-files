# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from hurst import compute_Hc
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")

# %%
def parse_pair(pair):
    """Parse pair string S1-S2 into tickers S1 and S2."""
    dp = pair.find('-')
    s1 = pair[:dp]
    s2 = pair[dp+1:]
    return s1, s2

def calculate_halflife(spread):
    """Calculate half-life of mean reversion of the spread."""
    ylag = spread.shift()
    deltay = spread - ylag
    ylag.dropna(inplace=True)
    deltay.dropna(inplace=True)
    res = OLS(deltay, add_constant(ylag)).fit()
    halflife = -np.log(2) / res.params[0]
    return halflife

def calculate_metrics(pairs, cumret, pairs_df):
    """Calculate metrics for pairs using data in cumret, including additional statistics."""
    cols = pd.MultiIndex.from_tuples([
        ('Hedge ratio', ''),
        ('CADF Test', 'Test Statistic'),
        ('CADF Test', 'p-value'),
        ('CADF Test', 'Critical Value 1%'),
        ('CADF Test', 'Critical Value 5%'),
        ('CADF Test', 'Critical Value 10%'),
        ('ADF Test', 'Test Statistic'),
        ('ADF Test', 'p-value'),
        ('ADF Test', 'Number of Lags Used'),
        ('ADF Test', 'Number of Observations Used'),
        ('ADF Test', 'Critical Value 1%'),
        ('ADF Test', 'Critical Value 5%'),
        ('ADF Test', 'Critical Value 10%'),
        ('Spread SD', ''),
        ('Pearson r', ''),
        ('Num zero-crossings', ''),
        ('Hurst Exponent', ''),
        ('Half-life of mean reversion', ''),
        ('% days within historical 2-SD band', '')
    ])
    results = pd.DataFrame(index=pairs, columns=cols)
    
    for pair in pairs:
        s1, s2 = parse_pair(pair)
        hedge_ratio = pairs_df.loc[pair]['Hedge ratio']
        spread = cumret[s1] - hedge_ratio * cumret[s2]
        results.loc[pair, ('Hedge ratio', '')] = hedge_ratio
        
        coint_result = coint(cumret[s1], cumret[s2])
        results.loc[pair, ('CADF Test', 'Test Statistic')] = coint_result[0]
        results.loc[pair, ('CADF Test', 'p-value')] = coint_result[1]
        results.loc[pair, ('CADF Test', 'Critical Value 1%')] = coint_result[2][0]
        results.loc[pair, ('CADF Test', 'Critical Value 5%')] = coint_result[2][1]
        results.loc[pair, ('CADF Test', 'Critical Value 10%')] = coint_result[2][2]
        
        adf_result = adfuller(spread)
        results.loc[pair, ('ADF Test', 'Test Statistic')] = adf_result[0]
        results.loc[pair, ('ADF Test', 'p-value')] = adf_result[1]
        results.loc[pair, ('ADF Test', 'Number of Lags Used')] = adf_result[2]
        results.loc[pair, ('ADF Test', 'Number of Observations Used')] = adf_result[3]
        results.loc[pair, ('ADF Test', 'Critical Value 1%')] = adf_result[4]['1%']
        results.loc[pair, ('ADF Test', 'Critical Value 5%')] = adf_result[4]['5%']
        results.loc[pair, ('ADF Test', 'Critical Value 10%')] = adf_result[4]['10%']
        
        hist_mu = pairs_df.loc[pair]['Spread mean']
        hist_sd = pairs_df.loc[pair]['Spread SD']
        results.loc[pair, ('Spread SD', '')] = hist_sd
        results.loc[pair, ('Pearson r', '')] = np.corrcoef(cumret[s1], cumret[s2])[0][1]
        spread_nm = spread - hist_mu
        results.loc[pair, ('Num zero-crossings', '')] = ((spread_nm[1:].values * spread_nm[:-1].values) < 0).sum()
        results.loc[pair, ('Hurst Exponent', '')] = compute_Hc(spread)[0]
        results.loc[pair, ('Half-life of mean reversion', '')] = calculate_halflife(spread)
        results.loc[pair, ('% days within historical 2-SD band', '')] = (abs(spread - hist_mu) < 2 * hist_sd).sum() / len(spread) * 100
    return results

def plot_spread(pair, cumret, pairs_df):
    """Spread visualization."""
    s1, s2 = parse_pair(pair)
    hedge_ratio = pairs_df.loc[pair]['Hedge ratio']
    spread = cumret[s1] - hedge_ratio * cumret[s2]
    spread_mean = pairs_df.loc[pair]['Spread mean']
    spread_sd = pairs_df.loc[pair]['Spread SD']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(spread, color='#1f77b4', linewidth=1.5, label='Normalized Spread')
    ax.axhline(spread_mean, color='#2ca02c', linestyle='--', linewidth=1.5, label='Historical Mean')
    ax.fill_between(spread.index, spread_mean - 2*spread_sd, spread_mean + 2*spread_sd,
                    color='#ff7f0e', alpha=0.15, label='±2σ Historical Band')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=60)
    last_val = spread[-1]
    ax.annotate(f'Current Value: {last_val:.3f}',
                xy=(spread.index[-1], last_val),
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black'))
    ax.set_title(f'{pair} Spread Analysis\n(March 31, 2020 - April 17, 2025)', fontsize=14, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Spread Value', fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()

def plot_metrics(metrics):
    """Generate an enhanced table for Pair Analysis Metrics with grouped sections including ARIMA."""
    fig, ax = plt.subplots(figsize=(8, 13))
    ax.axis('off')

    cell_text = []
    cell_colours = []
    metric_groups = [
        ('Hedge ratio', [('Hedge ratio', '')]),
        ('CADF Test', [
            ('CADF Test', 'Test Statistic'),
            ('CADF Test', 'p-value'),
            ('CADF Test', 'Critical Value 1%'),
            ('CADF Test', 'Critical Value 5%'),
            ('CADF Test', 'Critical Value 10%')
        ]),
        ('ADF Test', [
            ('ADF Test', 'Test Statistic'),
            ('ADF Test', 'p-value'),
            ('ADF Test', 'Number of Lags Used'),
            ('ADF Test', 'Number of Observations Used'),
            ('ADF Test', 'Critical Value 1%'),
            ('ADF Test', 'Critical Value 5%'),
            ('ADF Test', 'Critical Value 10%')
        ]),
        ('Spread SD', [('Spread SD', '')]),
        ('Pearson r', [('Pearson r', '')]),
        ('Num zero-crossings', [('Num zero-crossings', '')]),
        ('Hurst Exponent', [('Hurst Exponent', '')]),
        ('Half-life of mean reversion', [('Half-life of mean reversion', '')]),
        ('% days within historical 2-SD band', [('% days within historical 2-SD band', '')]),
        ('ARIMA Model', [
            ('ARIMA Model', 'Order'),
            ('ARIMA Model', 'AIC'),
            ('ARIMA Model', 'BIC')
        ])
    ]

    for group_name, sub_metrics in metric_groups:
        cell_text.append([group_name, ''])
        cell_colours.append(['#d3d3d3', '#d3d3d3'])
        for metric in sub_metrics:
            val = metrics.iloc[0][metric]
            metric_name = metric[1] if metric[1] else metric[0]
            if 'Order' in metric_name:
                formatted_val = val
            elif 'p-value' in metric_name:
                formatted_val = f"{val:.2e}"
            elif 'Test Statistic' in metric_name or 'Critical Value' in metric_name:
                formatted_val = f"{val:.3f}"
            elif 'Number of Lags Used' in metric_name or 'Number of Observations Used' in metric_name or metric_name == 'Num zero-crossings':
                formatted_val = f"{int(val)}"
            elif '%' in metric_name:
                formatted_val = f"{val:.1f}%"
            elif metric_name == 'Half-life of mean reversion':
                formatted_val = f"{val:.1f}"
            elif metric_name in ['Hurst Exponent', 'Hedge ratio', 'AIC', 'BIC']:
                formatted_val = f"{val:.3f}"
            else:
                formatted_val = f"{val:.3f}"
            cell_text.append([f"  {metric_name}", formatted_val])
            cell_colours.append(['#f8f9fa', '#ffffff'])

    table = ax.table(cellText=cell_text, colLabels=['Metric', 'Value'], loc='center',
                     cellLoc='left', colColours=['#40466e', '#40466e'], cellColours=cell_colours)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    for (row, col), cell in table.get_celld().items():
        if col == 0 and cell_text[row-1][1] == '':
            cell.set_text_props(weight='bold')
            cell.set_height(0.05)
    plt.title('Pair Analysis Metrics', fontsize=15, pad=20)
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(spread):
    """Plot ACF and PACF of the spread with confidence intervals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(spread, ax=ax1, lags=40)
    plot_pacf(spread, ax=ax2, lags=40)
    plt.tight_layout()
    plt.show()

def fit_arima_model(spread, test_size=30):
    """Fit the best ARIMA model to the spread and predict the test set."""
    train = spread[:-test_size]
    test = spread[-test_size:]
    
    adf_result = adfuller(train)
    d = 0 if adf_result[1] < 0.05 else 1
    
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for p in range(6):
        for q in range(6):
            try:
                model = ARIMA(train, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
                    best_model = model
            except:
                continue
    
    predictions = best_model.forecast(steps=test_size)
    return best_order, best_model.aic, best_model.bic, predictions, test

def plot_arima_predictions(test, predictions):
    """Plot actual vs predicted spread for the test period."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test, label='Actual Spread', color='blue')
    ax.plot(test.index, predictions, label='Predicted Spread', color='red')
    ax.set_title('ARIMA Model Predictions vs Actual Spread')
    ax.set_xlabel('Date')
    ax.set_ylabel('Spread')
    ax.legend()
    plt.show()

def plot_spread_with_forecast(spread, future_forecast, future_dates):
    """Plot historical spread with future forecast."""
    spread_mean = spread.mean()
    spread_sd = spread.std()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(spread.index, spread, label='Historical Spread', color='blue')
    ax.plot(future_dates, future_forecast, label='Forecasted Spread', color='red')
    ax.axvline(x=spread.index[-1], color='gray', linestyle='--', label='Forecast Start')
    ax.axhline(spread_mean, color='green', linestyle='--', label='Historical Mean')
    ax.fill_between(spread.index, spread_mean - 2*spread_sd, spread_mean + 2*spread_sd, 
                    color='orange', alpha=0.1, label='±2σ Historical Band')
    ax.fill_between(future_dates, spread_mean - 2*spread_sd, spread_mean + 2*spread_sd, 
                    color='orange', alpha=0.1)
    ax.set_title('Spread with Future Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Spread')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=60)
    plt.show()

# %%
# Read the data
file_path = r'D:\MG\！CUHKSZ\~！大二 下\UBS瑞银 商赛\data\BYD&Tesla.xlsx'
prices = pd.read_excel(
    file_path,
    header=0,
    skiprows=[0],
    index_col='日期',
    parse_dates=['日期'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)
prices = prices[['比亚迪股份', '特斯拉(TESLA)']]
prices.columns = ['BYD', 'TESLA']
prices = prices[(prices['BYD'] > 0) & (prices['TESLA'] > 0)]

log_prices = np.log(prices)
log_prices = log_prices.replace([np.inf, -np.inf], np.nan).dropna()

pair = 'BYD-TESLA'
s1, s2 = parse_pair(pair)

res = OLS(log_prices[s1], add_constant(log_prices[s2])).fit()
hedge_ratio = res.params[s2]

spread = log_prices[s1] - hedge_ratio * log_prices[s2]

spread_mean = spread.mean()
spread_sd = spread.std()

pairs_df = pd.DataFrame(index=[pair], columns=['Hedge ratio', 'Spread mean', 'Spread SD'])
pairs_df.loc[pair] = [hedge_ratio, spread_mean, spread_sd]

# Plot ACF and PACF
print("\n\033[1mAutocorrelation and Partial Autocorrelation Plots\033[0m")
plot_acf_pacf(spread)

# Compute metrics
metrics = calculate_metrics([pair], log_prices, pairs_df)

# Fit ARIMA model and predict spread movement
arima_order, aic, bic, predictions, test = fit_arima_model(spread)
metrics.loc[pair, ('ARIMA Model', 'Order')] = str(arima_order)
metrics.loc[pair, ('ARIMA Model', 'AIC')] = aic
metrics.loc[pair, ('ARIMA Model', 'BIC')] = bic

# Display results
print("Metrics for BYD-TESLA pair:")
print(metrics)

print("\n\033[1mPair Analysis Metrics\033[0m")
plot_metrics(metrics)

print("\n\033[1mSpread Visualization with Statistical Bands\033[0m")
plot_spread(pair, log_prices, pairs_df)

print("\n\033[1mARIMA Model Predictions\033[0m")
plot_arima_predictions(test, predictions)

# Future forecast using the entire dataset
print("\n\033[1mFuture Spread Forecast for the Next 30 Days\033[0m")
full_model = ARIMA(spread, order=arima_order).fit()
future_forecast = full_model.forecast(steps=30)
last_date = spread.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
plot_spread_with_forecast(spread, future_forecast, future_dates)


