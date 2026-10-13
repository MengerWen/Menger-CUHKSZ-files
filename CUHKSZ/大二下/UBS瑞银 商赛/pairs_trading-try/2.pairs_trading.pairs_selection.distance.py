import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

symbols = ['IEX', 'VICI', 'NUAN', 'MOH', 'NVAX', 'SBNY', 'PWR', 'WSM', 'FANG', 'BRO', 'STLD', 'ALLE', 'PNR', 'ATO', 'KIM', 
           'APO', 'JLL', 'HWM', 'WTRG', 'MPW', 'ARES', 'TPR', 'LPLA', 'BAH', 'RHI', 'BLDR', 'CSL', 'MOS', 'GLPI', 'RPM', 
           'CLF', 'IVZ', 'EWBC', 'CMA', 'AIZ', 'BG', 'TRGP', 'SCI', 'DVN', 'AFG', 'OC', 'LAD', 'NWL', 'UGI', 'CG', 'UTHR', 
           'ACM', 'NCLH', 'LSI', 'RS', 'ON', 'ZION', 'FRT', 'ST', 'DXC', 'SEE', 'MRO', 'FHN', 'KSS', 'ARW', 'FSLR', 'ATR', 
           'OGN', 'CPRI', 'DCI', 'LB', 'NNN', 'ATH', 'LECO', 'DKS', 'AGCO', 'JBL', 'ITT', 'HII', 'THC', 'AIRC', 'GXO', 
           'OHI', 'VOYA', 'RGA', 'AA', 'OSK', 'MRTX', 'EHC', 'TTEK', 'RNR', 'BJ', 'KNX', 'NLSN', 'VNO', 'JAZZ', 'BC', 'ORI', 
           'GPS', 'TOL', 'JHG', 'PVH', 'AN', 'MAT', 'CBSH', 'APA', 'FAF', 'IPG', 'FR', 'CNXC', 'EGP', 'NYT', 'ALK', 'OLN', 
           'OVV', 'OGE', 'STWD', 'PBCT', 'M', 'PII', 'BRX', 'LDOS', 'RXN', 'SF', 'NVST', 'JEF', 'USFD', 'AXTA', 'PFGC', 'AMG', 
           'SYNA', 'EME', 'MAN', 'XEC', 'AYI', 'SON', 'CFR', 'HBI', 'X', 'LPX', 'GME', 'MDU', 'FFIN', 'AAL', 'SRCL', 'LNC', 
           'PB', 'VNT', 'LEG', 'BPOP', 'SMG', 'CFX', 'DAR', 'SNV', 'HOG', 'RBC', 'FFIV', 'VAC', 'CACI', 'SLM', 'PRI', 'THO', 
           'ACHC', 'PACB', 'DECK', 'WCC', 'FL', 'HELE', 'UHAL', 'GPK', 'VER', 'BBIO', 'NVT', 'MSA', 'CUZ', 'OMF', 'TDC', 
           'HUBB', 'NRG', 'INGR', 'NCR', 'VSCO', 'CC', 'ASH', 'MHK', 'KBR', 'DEI', 'CIT', 'PRGO', 'CHK', 'VVV', 'HLF', 'NYCB',
           'CHNG', 'TMX', 'IRDM', 'IDA', 'EQT', 'VLY', 'VMI', 'HUN', 'SNX', 'OZK', 'MTG', 'NOV', 'FCNCA', 'RYN', 'MTZ', 
           'MIDD', 'GBCI', 'ADC', 'LAZ', 'FLS', 'CR', 'TKR', 'SIGI', 'CW', 'PACW', 'LAMR', 'UNM', 'SAIC', 'SSB', 'CF', 'EVR', 
           'SLG', 'NRZ', 'HRC', 'CRUS', 'LEA', 'JBLU', 'FCN', 'ESI', 'AVNT', 'HIW', 'JNPR', 'WTS', 'HFC', 'UBSI', 'RLI', 
           'HRB', 'REZI', 'GRA', 'POR', 'FLO', 'AEO', 'BXMT', 'WBS', 'THG', 'WAL', 'RDN', 'HE', 'CAR', 'ARMK', 'BKH', 'GT', 
           'UFPI', 'STL', 'NFG', 'UAA', 'ATKR', 'PNW', 'PINC', 'AL', 'PK', 'UMPQ', 'TAP', 'CRI', 'AGNC', 'WTFC', 'HR', 'PNM', 
           'XPO', 'KMPR', 'BHF', 'UMBF', 'SPR', 'CRNC', 'WEN', 'R', 'ALSN', 'SEIC', 'AVT', 'MTH', 'UNVR', 'HWC', 'CBU', 'UA', 
           'HPP', 'QRTEA', 'MUSA', 'RCII', 'LXP', 'NAVI', 'SIG', 'CMC', 'SUM', 'BKU', 'TGNA', 'AXS', 'BCO', 'SWX', 'OGS', 
           'KFY', 'MC', 'EPR', 'AIMC', 'KBH', 'MSM', 'SEM', 'FNB', 'VIAV', 'TRTN', 'SXT', 'WWD', 'FHB', 'NSIT', 'ENS', 'OUT', 
           'ABG', 'NJR', 'SABR', 'ETRN', 'AGO', 'CIM', 'AR', 'FUL', 'ALE', 'SLGN', 'UNF', 'TRIP', 'XRX', 'SBRA', 'AIT', 
           'TGTX', 'TMHC', 'MATX', 'TCBI', 'APG', 'RRC', 'ARNC', 'FLOW', 'AWR', 'WH', 'BOH', 'TEX', 'DAN', 'VG', 'ADS', 
           'MAC', 'FCFS', 'EVTC', 'ABM', 'EBC', 'HRI', 'PCH', 'WTM', 'HL', 'HOMB', 'SR', 'ITGR', 'GATX', 'COMM', 'NWE', 
           'KEX', 'CNO', 'PRG', 'APAM', 'ISBC', 'JWN', 'WD', 'HAIN', 'FHI', 'OFC', 'DNB', 'SPB', 'MLHR', 'HI', 'APLE', 
           'IOVA', 'MUR', 'KMT', 'SWN', 'EQC', 'CBT', 'CATY', 'ESGR', 'COTY', 'ASB', 'SFNC', 'AM', 'WWW', 'MIC', 'SFM', 
           'MDC', 'AUB', 'BXS', 'VSH', 'MGEE', 'PAG', 'AVA', 'KOD', 'SITC', 'CLR', 'BECN', 'H', 'PBH', 'GPI', 'RPAI', 
           'TPTX', 'RETA', 'TPH', 'KTB', 'SJI', 'BBBY', 'UFS', 'FBP', 'AEL', 'WERN', 'GDOT', 'TRN', 'ENR', 'SPXC', 'RMBS', 
           'PDCO', 'OMI', 'SAVE', 'RL', 'MTX', 'MD', 'CVBF', 'KRTX', 'ONB', 'MYGN', 'PLXS', 'UCBI', 'KW', 'VRNT', 'HP', 
           'COLB', 'NHI', 'FBC', 'FULT', 'CRC', 'BDC', 'CADE', 'SC', 'JELD', 'NUS', 'CWEN', 'EGHT', 'SCL', 'ESNT', 'SHO', 
           'AWI', 'PIPR', 'BRC', 'ACA', 'JW.A', 'AX', 'SANM', 'BOKF', 'EAT', 'TROX', 'MNTV', 'CWK', 'MLI', 'INDB', 'WAFD', 
           'OI', 'IBOC', 'BLMN', 'RLJ', 'BDN', 'HXL', 'GHC', 'FLR', 'TNL', 'SKYW', 'ODP', 'MOG.A', 'CVET', 'HLI', 'IOSP', 
           'EPC', 'B', 'OTTR', 'BCC', 'ATI', 'CMP', 'VRTS', 'HUBG', 'DTM', 'XPER', 'GO', 'FRME', 'ESE', 'IDCC', 'MGLN', 
           'PFSI', 'ANF', 'PDM', 'CAKE', 'REGI', 'AI', 'WSBC', 'FFBC', 'ARI', 'MANT', 'OM', 'WSFS', 'FMBI', 'WRE', 'CNX', 
           'AAWW', 'MFA', 'NMRK', 'KAR', 'UE', 'THS', 'SBH', 'DNLI', 'RIG', 'CORE', 'UNFI', 'HTH', 'TWO', 'TDS', 'MDRX', 
           'RLGY', 'PRGS', 'ROIC', 'SPT', 'KALU', 'GMS', 'DOC', 'TWNK', 'TOWN', 'XHR', 'WOR', 'VGR', 'BGS', 'TTEC', 'AMKR', 
           'SAFM', 'HCSG', 'PSMT', 'URBN', 'INT', 'CXP', 'NTCT', 'CNMD', 'GNW', 'DRH', 'PMT', 'SVC', 'RNST', 'RUSHA', 'BANR', 
           'JBGS', 'SPWR', 'GVA', 'KYMR', 'KN', 'EGBN', 'STAR', 'AKR', 'PRAA', 'CNK', 'LAZR', 'TRMK', 'FROG', 'AMWL', 'ILPT', 
           'SUMO', 'ESRT', 'NPO', 'AAT', 'ARGO', 'BGCP', 'ATGE', 'FUBO', 'FIBK', 'PRK', 'MTOR', 'AVIR', 'FOE', 'LANC', 'BIG', 
           'CENTA', 'HMN', 'ALGM', 'BLKB', 'GWB', 'CBRL', 'MEI', 'OSIS', 'MGRC', 'GOLF', 'LAUR', 'DBRG', 'WIRE', 'MDP', 
           'MCY', 'ANAT', 'CDE', 'LZB', 'SGRY', 'AVYA', 'DLX', 'HOPE', 'GNL', 'CWT', 'CRS', 'ARCB', 'SKT', 'GEF', 'AVNS', 
           'NWBI', 'NWN', 'HNI', 'SDGR', 'RAMP', 'SMCI', 'PFS', 'UIS', 'MCFE', 'CFFN', 'UNIT', 'ECPG', 'CSGS', 'SHOO', 'LC', 
           'WBT', 'MTRN', 'NBTB', 'REYN', 'CLI', 'ASO', 'OXM', 'WABC', 'PEB', 'HSC', 'ALEX', 'CALM', 'TTMI', 'STEM', 'WLK', 
           'CNNE', 'PTEN', 'TRS', 'PRA', 'IHRT', 'RGR', 'PPC', 'LTC', 'BKD', 'CDLX', 'AZZ', 'CSR', 'BRMK', 'ASTE', 'PRVA', 
           'AMCX', 'PRIM', 'FIX', 'ADT', 'SPNT', 'LADR', 'GFF', 'FCF', 'LGF.B', 'OPI', 'KRA', 'TEN', 'CLB', 'BANF', 'GIII', 
           'ALXO', 'AIN', 'BHLB', 'PBI', 'GBX', 'OFG', 'SXI', 'GCP', 'SWM', 'GTES', 'GTY', 'DDS', 'GPRO', 'VSTO', 'AROC', 
           'UVV', 'STBA', 'SNDR', 'CXW', 'FDP', 'CHCO', 'CNDT', 'SCHN', 'KFRC', 'BKE', 'SAFT', 'HCC', 'VREX', 'SCS', 'DK', 
           'AIR', 'SM', 'HURN', 'POLY', 'EIG', 'S', 'MATW', 'BRKL', 'NNI', 'SATS', 'DY', 'ARCH', 'KAMN', 'MAXR', 'DOCS', 
           'RPT', 'CWEN.A', 'HEES', 'APOG', 'SKIN', 'SAH', 'MGY', 'NTGR', 'AIV', 'FSR', 'TMP', 'MPLN', 'ADTN', 'PBF', 'SEB', 
           'HA', 'TVTY', 'TBI', 'RYAN', 'AXL', 'MDGL', 'AFIN', 'BJRI', 'GES', 'RAD', 'JAMF', 'CCO', 'WMK', 'STEP', 'BHE', 
           'BRSP', 'NHC', 'ANDE', 'SCSC', 'AAN', 'GEO', 'DHC', 'DBI', 'HTLF', 'STFC', 'IVR', 'ZUMZ', 'VIVO', 'BFLY', 'ACCO', 
           'TILE', 'DNOW', 'WRLD', 'KREF', 'CARS', 'NXGN', 'CHPT', 'WETF', 'NX', 'CERE', 'CLBK', 'ARR', 'LGF.A', 'WNC', 
           'EPAC', 'BCOR', 'NFBK', 'SBGI', 'TRTX', 'ETWO', 'AXSM', 'CIR', 'OPK', 'ELF', 'RMR', 'PQG', 'GLT', 'RWT', 'AERI', 
           'BTRS', 'ADV', 'NWLI', 'CPF', 'LRN', 'DIN', 'PUMP', 'AMRX', 'KOP', 'KELYA', 'PWSC', 'NUVB', 'AMBC', 'BV', 'CFLT', 
           'MRC', 'IPAR', 'CMO', 'ZIP', 'DNMR', 'FMTX', 'UFCS', 'SWBI', 'HY', 'PRAX', 'PLCE', 'PCT', 'TPC', 'DM', 'SKLZ', 
           'GRPN', 'MCW', 'WW', 'TGI', 'BFS', 'SCHL', 'CNM', 'REVG', 'OLO', 'AEVA', 'USM', 'GOEV', 'VVNT', 'ENDP', 'TR', 
           'ALVR', 'PAHC', 'STOK', 'OLMA', 'CLOV', 'SQSP', 'OUST', 'FIGS', 'INN', 'FSP', 'HYLN', 'SRG', 'CAL', 'NPK', 'CVI', 
           'GCO', 'VZIO', 'HIMS', 'TASK', 'GEF.B', 'ZGNX', 'CRCT', 'DBD', 'CENT', 'HTLD', 'RXRX', 'RUSHB', 'GPMT', 'TG', 
           'XL', 'RIDE', 'GRC', 'EBIX', 'FOR', 'KNSA', 'AGTI', 'IMVT', 'YOU', 'SWI', 'DUOL', 'LOCO', 'INSW', 'GIC', 'PYCR', 
           'VLDR', 'KRO', 'COOK', 'UBA', 'ADGI', 'DNUT', 'HLX', 'NABL', 'SWIM', 'LFST', 'FA', 'GTHX', 'APPH', 'NGM', 'RES', 
           'PECO', 'LE', 'SBTX', 'ANAB', 'ULCC', 'VIEW', 'HRMY', 'AMK', 'LTCH', 'APEI', 'KNBE', 'AVAH', 'FLYW', 'INTA', 
           'XMTR', 'ERAS', 'RMO', 'TDUP', 'LYEL', 'GCMG', 'MILE', 'NAPA', 'BRDG', 'IAS', 'VERX', 'MLNK', 'CTKB', 'LAW', 
           'WEBR', 'TMCI', 'SEMR', 'HNST', 'DAWN', 'PAY', 'ZY', 'CXM', 'ABSI', 'BDTX', 'DSGN', 'FF', 'OMIC', 'ZETA', 'AC', 
           'OWL', 'BHG', 'UBP', 'LZ', 'LDI', 'INST', 'DV', 'COMP', 'CURV', 'TXG', 'MRVI', 'EVCM', 'WK', 'CHX', 'LPRO', 'PKI', 
           'ALGT', 'PCOR', 'BLND', 'SAGE', 'ALKT', 'SWTX', 'OII', 'HAYW', 'DRQ', 'COUR', 'BPYPM', 'SGYPQ', 'FTBK']

start = '2016-07-01'
end = '2019-12-31'

# create dataframe for storing prices
tmpdf = yf.download(symbols[0], start, end)
prices = pd.DataFrame(index = tmpdf.index)

for symbol in symbols:
    tmpdf = yf.download(symbol, start, end)
    if (tmpdf['Volume'] < 1000).sum() == 0:   # check that volume at any trading day exceeds 1000
        prices[symbol] = tmpdf['Adj Close']
        
# delete all columns with NA values
prices = prices.dropna(axis=1)

# save prices to file
prices.to_csv('vbr16_19.csv')

# read prices from csv file
prices = pd.read_csv('D:/MG/！CUHKSZ/~！大二 下/UBS瑞银 商赛/pairs_trading-main/vbr16_19.csv', index_col='Date')

prices

cumret = np.log(prices).diff().cumsum()+1 # calculate cumulative returns
cumret.dropna(inplace=True) 

cumret

def calculate_distances(cumret):
    '''
    calculate Euclidean distance for each pair of stocks in the dataframe
    return sorted dictionary (in ascending order)
    '''
    distances = {} # dictionary with distance for each pair
    
    # calculate distances
    for s1 in cumret.columns:
        for s2 in cumret.columns:
            if s1!=s2 and (f'{s1}-{s2}' not in distances.keys()) and (f'{s2}-{s1}' not in distances.keys()):
                dist = np.sqrt(np.sum((cumret[s1] - cumret[s2])**2)) # Euclidean distance
                distances[f'{s1}-{s2}'] = dist
    
    # sort dictionary
    sorted_distances = {k:v for k,v in sorted(distances.items(), key = lambda item: item[1])}
    
    return sorted_distances

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

def calculate_metrics(sorted_distances, cumret, N=5):
    '''
    calculate metrics for N pairs with the smallest Euclidean distance
    return dataframe of results
    '''
    from hurst import compute_Hc
    from statsmodels.tsa.stattools import adfuller
    
    pairs = [k for k,v in sorted_distances.items()][:N]
    
    cols = ['Euclidean distance', 'CADF p-value', 'ADF p-value', 'Spread SD', 
        'Num zero-crossings', 'Hurst Exponent', 'Half-life of mean reversion', '% days within 2-SD band']
    results = pd.DataFrame(index=pairs, columns=cols)
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        spread = cumret[s1] - cumret[s2]
        results.loc[pair]['Euclidean distance'] = np.sqrt(np.sum((spread)**2))
        results.loc[pair]['CADF p-value'] = cadf_pvalue(s1, s2, cumret)
        results.loc[pair]['ADF p-value'] = adfuller(spread)[1]
        results.loc[pair]['Spread SD'] = spread.std()
        results.loc[pair]['Num zero-crossings'] = ((spread[1:].values * spread[:-1].values) < 0).sum()
        results.loc[pair]['Hurst Exponent'] = compute_Hc(spread)[0]
        results.loc[pair]['Half-life of mean reversion'] = calculate_halflife(spread)
        results.loc[pair]['% days within 2-SD band'] = (abs(spread) < 2*spread.std()).sum() / len(spread) * 100
        
    return results

def plot_pairs(sorted_distances, cumret_train, cumret_test, N=5):
    '''
    plot cumulative returns of the spread for each of N pairs with smallest Euclidean distance
    '''
    pairs = [k for k,v in sorted_distances.items()][:N]
    
    for pair in pairs:
        s1,s2 = parse_pair(pair)
        spread_train = cumret_train[s1] - cumret_train[s2]
        spread_test = cumret_test[s1] - cumret_test[s2]
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

# prepare data
cumret12_6 = cumret.loc['2018-07-01':]
cumret12_6 = cumret12_6 / cumret12_6.iloc[0] # divide by first row so that all prices start at 1
train12_6 = cumret12_6.loc['2018-07-01':'2019-06-31'] # formation period
test12_6 = cumret12_6.loc['2019-07-01':'2019-12-31'] # trading period

# calculate Euclidean distances for each pair
sorted_distances12_6 = calculate_distances(train12_6)

# print top 10 pairs with smallest distance
[[k,v] for k,v in sorted_distances12_6.items()][:10]

plot_pairs(sorted_distances12_6, train12_6, test12_6)

calculate_metrics(sorted_distances12_6, train12_6)

calculate_metrics(sorted_distances12_6, test12_6)

# prepare data
cumret24_6 = cumret.loc['2017-07-01':]
cumret24_6 = cumret24_6 / cumret24_6.iloc[0] # divide by first row so that all prices start at 1
train24_6 = cumret24_6.loc['2017-07-01':'2019-06-31'] # formation period
test24_6 = cumret24_6.loc['2019-07-01':'2019-12-31'] # trading period

# calculate Euclidean distances for each pair
sorted_distances24_6 = calculate_distances(train24_6)

# print top 10 pairs with smallest distance
[[k,v] for k,v in sorted_distances24_6.items()][:10]

plot_pairs(sorted_distances24_6, train24_6, test24_6)

calculate_metrics(sorted_distances24_6, train24_6)

calculate_metrics(sorted_distances24_6, test24_6)

# prepare data
cumret36_6 = cumret.loc['2016-07-01':]
cumret36_6 = cumret36_6 / cumret36_6.iloc[0] # divide by first row so that all prices start at 1
train36_6 = cumret36_6.loc['2016-07-01':'2019-06-31'] # formation period
test36_6 = cumret36_6.loc['2019-07-01':'2019-12-31'] # trading period

# calculate Euclidean distances for each pair
sorted_distances36_6 = calculate_distances(train36_6)

# print top 10 pairs with smallest distance
[[k,v] for k,v in sorted_distances36_6.items()][:10]

plot_pairs(sorted_distances36_6, train36_6, test36_6)

calculate_metrics(sorted_distances36_6, train36_6)

calculate_metrics(sorted_distances36_6, test36_6)

import json

def convert_ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r',encoding='utf-8') as f:
        notebook = json.load(f)

    with open(py_file, 'w',encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                f.write(''.join(cell['source']) + '\n\n')

convert_ipynb_to_py('2.pairs_trading.pairs_selection.distance.ipynb', '2.pairs_trading.pairs_selection.distance.py')

