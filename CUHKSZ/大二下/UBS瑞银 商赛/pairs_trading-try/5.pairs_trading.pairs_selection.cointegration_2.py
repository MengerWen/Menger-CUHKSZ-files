"""
# Pairs trading. Pairs selection. Cointegration (Part 2)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

prices_train = pd.read_csv('vbr13_16.csv', index_col='Date')
prices_test = pd.read_csv('vbr16_19.csv', index_col='Date')

# calculate cumulative returns
cumret_train = np.log(prices_train).diff().cumsum()+1
cumret_train.dropna(how='all', inplace=True)
cumret_train.dropna(axis=1, inplace=True)
cumret_test = np.log(prices_test).diff().cumsum()+1
cumret_test.dropna(how='all', inplace=True)

# divide by first row so that all prices start at 1
cumret_train = cumret_train / cumret_train.iloc[0]
cumret_test = cumret_test / cumret_test.iloc[0]



"""
# Functions to calculate different metrics
"""
def parse_pair(pair):
    '''
    parse pair string S1-S2
    return tickers S1, S2
    '''
    dp = pair.find('-')
    s1 = pair[:dp]
    s2 = pair[dp+1:]
    
    return s1,s2

def cadf_pvalue(s1, s2, cumret):
    '''
    perform CADF cointegration tests
    since it is sensitive to the order of stocks in the pair, perform both tests (s1-2 and s2-s1)
    return the smallest p-value of two tests
    '''
    from statsmodels.tsa.stattools import coint
    
    p1 = coint(cumret[s1], cumret[s2])[1]
    p2 = coint(cumret[s2], cumret[s1])[1]
    
    return min(p1,p2)

def calculate_halflife(spread):
    '''
    calculate half-life of mean reversion of the spread
    '''
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    
    ylag = spread.shift()
    deltay = spread - ylag
    ylag.dropna(inplace=True)
    deltay.dropna(inplace=True)

    res = OLS(deltay, add_constant(ylag)).fit()
    halflife = -np.log(2)/res.params[0]
    
    return halflife

def calculate_metrics(pairs, cumret, pairs_df):
    '''
    calculate metrics for pairs using data in cumret
    return dataframe of results
    '''
    from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.stattools import coint
    
    cols = ['Distance', 'CADF p-value', 'ADF p-value', 'Spread SD', 'Pearson r',
        'Num zero-crossings', 'Hurst Exponent', 'Half-life of mean reversion', '% days within historical 2-SD band']
    results = pd.DataFrame(index=pairs, columns=cols)
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        hedge_ratio = pairs_df.loc[pair]['Hedge ratio']
        spread = cumret[s1] - hedge_ratio*cumret[s2]
        results.loc[pair]['CADF p-value'] = coint(cumret[s1], cumret[s2])[1]
        results.loc[pair]['ADF p-value'] = adfuller(spread)[1]
        hist_mu = pairs_df.loc[pair]['Spread mean'] # historical mean
        hist_sd = pairs_df.loc[pair]['Spread SD'] # historical standard deviation
        results.loc[pair]['Spread SD'] = hist_sd
        results.loc[pair]['Pearson r'] = np.corrcoef(cumret[s1], cumret[s2])[0][1]
        # subtract the mean to calculate distances and num_crossings
        spread_nm = spread - hist_mu
        results.loc[pair]['Distance'] = np.sqrt(np.sum((spread_nm)**2))
        results.loc[pair]['Num zero-crossings'] = ((spread_nm[1:].values * spread_nm[:-1].values) < 0).sum()
        results.loc[pair]['Hurst Exponent'] = compute_Hc(spread)[0]
        results.loc[pair]['Half-life of mean reversion'] = calculate_halflife(spread)
        results.loc[pair]['% days within historical 2-SD band'] = (abs(spread-hist_mu) < 2*hist_sd).sum() / len(spread) * 100
        
    return results

def plot_pairs(pairs, cumret_train, cumret_test):
    '''
    plot cumulative returns of the spread for each pair in pairs
    '''
    
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        res = OLS(cumret_train[s1], add_constant(cumret_train[s2])).fit()
        spread_train = cumret_train[s1] - res.params[s2]*cumret_train[s2]
        spread_test = cumret_test[s1] - res.params[s2]*cumret_test[s2]
        spread_mean = spread_train.mean() # historical mean
        spread_std = spread_train.std() # historical standard deviation

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,4))
        fig.suptitle(f'Spread of {pair} pair', fontsize=16)
        ax1.plot(spread_train, label='spread')
        ax1.set_title('Formation period')
        ax1.axhline(y=spread_mean, color='g', linestyle='dotted', label='mean')
        ax1.axhline(y=spread_mean+2*spread_std, color='r', linestyle='dotted', label='2-SD band')
        ax1.axhline(y=spread_mean-2*spread_std, color='r', linestyle='dotted')
        ax1.legend()
        ax2.plot(spread_test, label='spread')
        ax2.set_title('Trading period')
        ax2.axhline(y=spread_mean, color='g', linestyle='dotted', label='mean')
        ax2.axhline(y=spread_mean+2*spread_std, color='r', linestyle='dotted', label='2-SD band')
        ax2.axhline(y=spread_mean-2*spread_std, color='r', linestyle='dotted')
        ax2.legend()
        
def select_pairs(train):
    '''
    select pairs using data from train dataframe
    return dataframe of selected pairs
    '''
    tested = []

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import coint

    cols = ['Distance', 'Num zero-crossings', 'Pearson r', 'Spread mean', 
            'Spread SD', 'Hurst Exponent', 'Half-life of mean reversion', 'Hedge ratio']
    pairs = pd.DataFrame(columns=cols)

    for s1 in train.columns:
        for s2 in train.columns:
            if s1!=s2 and (f'{s1}-{s2}' not in tested):
                tested.append(f'{s1}-{s2}')
                cadf_p = coint(train[s1], train[s2])[1]
                if cadf_p<0.01 and (f'{s2}-{s1}' not in pairs.index): # stop if pair already added as s2-s1
                    res = OLS(train[s1], add_constant(train[s2])).fit()
                    hedge_ratio = res.params[s2]
                    if hedge_ratio > 0: # hedge ratio should be posititve
                        spread = train[s1] - hedge_ratio*train[s2]
                        hurst = compute_Hc(spread)[0]
                        if hurst<0.5:
                            halflife = calculate_halflife(spread)
                            if halflife>1 and halflife<30:
                                # subtract the mean to calculate distances and num_crossings
                                spread_nm = spread - spread.mean() 
                                num_crossings = (spread_nm.values[1:] * spread_nm.values[:-1] < 0).sum()
                                if num_crossings>len(train.index)/252*12: 
                                    distance = np.sqrt(np.sum(spread_nm**2))
                                    pearson_r = np.corrcoef(train[s1], train[s2])[0][1]
                                    pairs.loc[f'{s1}-{s2}'] = [distance, num_crossings, pearson_r, spread.mean(),
                                                               spread.std(), hurst, halflife, hedge_ratio]
                                
    return pairs

# prepare data
train_form = cumret_train.loc['2013-01-01':'2015-12-31'] # formation period (train data)
train_trade = cumret_train.loc['2016-01-01':] # trading period (train data)
test_form = cumret_test.loc['2016-07-01':'2019-06-31'] # formation period (test data)
test_trade = cumret_test.loc['2019-07-01':] # trading period (test data)

train_form

pairs_train = select_pairs(train_form)

pairs_train

pairs_train.index.rename('Pair', inplace=True)
pairs_train.to_csv('pairs13_16.csv')

pairs_train = pd.read_csv('pairs13_16.csv', index_col='Pair')

pairs_test = select_pairs(test_form)

pairs_test

pairs_test.index.rename('Pair', inplace=True)
pairs_test.to_csv('pairs16_19.csv')

pairs_test = pd.read_csv('pairs16_19.csv', index_col='Pair')

# calculate metrics for selected pairs
metrics_train_form = calculate_metrics(pairs_train.index, train_form, pairs_train) # in formation period
metrics_train_trade = calculate_metrics(pairs_train.index, train_trade, pairs_train) # in trading period

# add hedge ratio to metrics just in case
metrics_train_form['Hedge ratio'] = pairs_train['Hedge ratio']
metrics_train_form

metrics_train_trade

# repeat the procedure for test data
metrics_test_form = calculate_metrics(pairs_test.index, test_form, pairs_test)
metrics_test_trade = calculate_metrics(pairs_test.index, test_trade, pairs_test)
metrics_test_form['Hedge ratio'] = pairs_test['Hedge ratio']



"""
# Data preparation
"""
data_train = metrics_train_form.copy()
data_train['Num zero-crossings trade'] = metrics_train_trade['Num zero-crossings']
data_train

data_test = metrics_test_form.copy()
data_test['Num zero-crossings trade'] = metrics_test_trade['Num zero-crossings']
data_test


"""
# Data exploration
"""
data_train.dtypes

# convert data to numeric types
data_train = data_train.apply(pd.to_numeric, errors='raise')
data_test = data_test.apply(pd.to_numeric, errors='raise')

data_test

data_train.dtypes

data_train.describe()

# remove pairs with too big or too small hedge ratio
data_train = data_train[(data_train['Hedge ratio']>0.2) & (data_train['Hedge ratio']<5)]
data_test = data_test[(data_test['Hedge ratio']>0.2) & (data_test['Hedge ratio']<5)]

data_train.corr()



"""
# Visualize data
"""
_ = data_train.hist(figsize=(18,12))

_ = data_train.plot(kind='density', subplots=True, layout=(4,3), sharex=False, figsize=(18,12))


"""
## Classification
"""
X_train = data_train.values[:,:10]
X_test = data_test.values[:,:10]
y_train = data_train.values[:,10]
y_test = data_test.values[:,10]

# what fraction of samples has more than 6 crossings?
len(y_train[y_train>6]) / len(y_train)

from sklearn.preprocessing import Binarizer

# binarize dependent variable
binarizer = Binarizer(threshold=6).fit(y_train.reshape(-1,1))
y_train_bin = binarizer.transform(y_train.reshape(-1,1))
binarizer = Binarizer(threshold=6).fit(y_test.reshape(-1,1))
y_test_bin = binarizer.transform(y_test.reshape(-1,1))

from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def top10_accuracy(estimator, X, y):
    '''
    compute accuracy of top 10 predictions:
    select top 10 samples with the highest probability of belonging to class 1
    return the fraction of samples actually belonging to class1
    '''
    pred_prob = estimator.predict_proba(X)
    top10_ind = np.argsort(pred_prob[:,1])[-10:]
    score = sum(y[top10_ind] == np.ones(10))/10
    
    return score

def test_models_class(models, X=X_train, y=y_train_bin.flatten(), n_folds=10, scoring=top10_accuracy, seed=45):
    '''
    cross-validate each model in models using n_folds and scoring
    '''
    
    results = []
    names = []
    for name,model in models:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print(f'{name}: {cv_results.mean()} ({cv_results.std()})')
        
    return results, names

models = []
models.append(('LR', LogisticRegression(max_iter=300, class_weight='balanced')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(class_weight='balanced', probability=True)))

results,names = test_models_class(models, y=y_train_bin.flatten())

pipelines = []
for name,model in models:
    pipelines.append((f'scaled{name}', Pipeline([('scaler', PowerTransformer(method='box-cox')),
                                                 (name,model)])))
    
results,names = test_models_class(pipelines, y=y_train_bin.flatten())

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

pipelines = []
for name,model in models:
    pipelines.append((f'scaled{name}_4ft', Pipeline([('scaler', PowerTransformer(method='box-cox')),
                                                 ('KBest', SelectKBest(score_func=f_classif, k=4)),
                                                 (name,model)])))
    
results,names = test_models_class(pipelines, y=y_train_bin.flatten())

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

ensembles = []
ensembles.append(('ADA', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier(class_weight='balanced')))
ensembles.append(('ET', ExtraTreesClassifier(class_weight='balanced')))

results,names = test_models_class(ensembles, y=y_train_bin.flatten())

from sklearn.model_selection import GridSearchCV

n_estimators=[10,50,100,150]
max_depth = [5,7,10,15]
max_leaf_nodes = [50,100,150]
min_samples_leaf = [1,3,5,7,10,15]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, 
                  max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
model = ExtraTreesClassifier(class_weight='balanced')
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=45)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=top10_accuracy, cv=kfold)
grid_result = grid.fit(X_train, y_train_bin.flatten())

print(grid_result.best_score_)
print(grid_result.best_params_)

from sklearn.linear_model import LogisticRegressionCV

models = [('scaledLR_4ft_cv', Pipeline([('scaler', PowerTransformer(method='box-cox')), 
                            ('KBest', SelectKBest(score_func=f_classif, k=4)), 
                            ('LR', LogisticRegressionCV(max_iter=300, class_weight='balanced', 
                                                        scoring=top10_accuracy))]))]

results,names = test_models_class(models, y=y_train_bin.flatten())

scaler = PowerTransformer(method='box-cox').fit(X_train)
scaledX = scaler.transform(X_train)
scaledX_4ft = SelectKBest(score_func=f_classif, k=4).fit_transform(X_train, y_train_bin.flatten())
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC(class_weight='balanced', probability=True)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=45)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=top10_accuracy, cv=kfold)
grid_result = grid.fit(scaledX_4ft, y_train_bin.flatten())

print(grid_result.best_score_)
print(grid_result.best_params_)

scaler = PowerTransformer(method='box-cox').fit(X_train)
scaledX = scaler.transform(X_train)
scaledX_4ft = SelectKBest(score_func=f_classif, k=4).fit_transform(X_train, y_train_bin.flatten())
param_grid = dict(n_neighbors=np.arange(1,22,2))
model = KNeighborsClassifier()
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=45)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=top10_accuracy, cv=kfold)
grid_result = grid.fit(scaledX_4ft, y_train_bin.flatten())

print(grid_result.best_score_)
print(grid_result.best_params_)

# train ET using parameters obtained in grid search
model = ExtraTreesClassifier(class_weight='balanced', max_depth=10, max_leaf_nodes=50, 
                             min_samples_leaf=15, n_estimators=10)
model.fit(X_train, y_train_bin.flatten())

# accuracy on train data
top10_accuracy(model, X_train, y_train_bin.flatten())

# accuracy on test data
top10_accuracy(model, X_test, y_test_bin.flatten())

pred_prob = model.predict_proba(X_test) # predict probabilities
# select top 10 pairs with highest probability of belonging to class 1 (sort in descending order)
top10_ind = np.argsort(pred_prob[:,1])[-10:][::-1] 
data_test.iloc[top10_ind,:] # show data

top50_ind = np.argsort(pred_prob[:,1])[-50:][::-1]
top50_pairs = data_test.iloc[top50_ind,:].index

selected_pairs = []
selected_stocks = []
for pair in top50_pairs:
    s1,s2 = parse_pair(pair)
    if (s1 not in selected_stocks) and (s2 not in selected_stocks):
        selected_stocks.append(s1)
        selected_stocks.append(s2)
        selected_pairs.append(pair)
        
        if len(selected_pairs)==5:
            break

selected_pairs

plot_pairs(selected_pairs, test_form, test_trade)

import json

def convert_ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r',encoding='utf-8') as f:
        notebook = json.load(f)

    with open(py_file, 'w',encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                f.write(''.join(cell['source']) + '\n\n')

convert_ipynb_to_py('4.pairs_trading.pairs_selection.cointegration_1.ipynb', '4.pairs_trading.pairs_selection.cointegration_1.py')

