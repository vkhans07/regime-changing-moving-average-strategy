import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score



# regime-filtered moving average model
def get_data(ticker='SPY', start_date='2010-01-01', end_date='2025-01-01'):
    df = yf.download(ticker, start=start_date, end=end_date)
    vix = yf.download('^VIX', start=start_date, end=end_date)
    data = pd.DataFrame()
    data['Price'] = df['Close']
    data['VIX'] = vix['Close']
    data.dropna(inplace=True)

    return data

# idea is that we'll use moving averages to determine trend direction, but only trade
# when VIX is below a certain threshold to avoid high volatility periods
def apply_strategy(data, shorter_window=50, longer_window=200, vix_threshold=20):
    # log returns are easy
    data['Daily Log Return'] = np.log(data['Price'] / data['Price'].shift(1))
    # i think this is fine? it just drops off the first monday when we don't have a return
    data.dropna(inplace=True)
    data['Shorter MA'] = data['Price'].rolling(window=shorter_window).mean()
    data['Longer MA'] = data['Price'].rolling(window=longer_window).mean()

    # generate signals: we want to long if shorter ma > longer than ma and vix < threshold
    # otherwise, we stay neutral
    data['Adv Signal'] = np.where((data['Shorter MA'] > data['Longer MA']) &
                              (data['VIX'] < vix_threshold), 1, 0)
    
    data['Bsc Signal'] = np.where(data['Shorter MA'] > data['Longer MA'], 1, 0)

    return data

def backtest(data):
    # you're not clairvoyant; you calculate signals at close so
    # you can only act on them the next day
    # so gng you gotta shift everything back by 1
    # assume your df is well-formatted
    data['Adv Strategy Return'] = data['Adv Signal'].shift(1) * data['Daily Log Return']
    data['Bsc Strategy Return'] = data['Bsc Signal'].shift(1) * data['Daily Log Return']
    data.dropna(inplace=True)
    data['Long Strategy Return'] = data['Daily Log Return'].cumsum().apply(np.exp)
    data['Adv MA Strategy Return'] = data['Adv Strategy Return'].cumsum().apply(np.exp)
    data['Bsc MA Strategy Return'] = data['Bsc Strategy Return'].cumsum().apply(np.exp)

# i kinda wanna find what the optimal vix threshold is
data = get_data(ticker='AAPL')
apply_strategy(data)
backtest(data)
print(data[['Long Strategy Return', 'Adv MA Strategy Return', 'Bsc MA Strategy Return']].tail())
print('Sharpe Ratios:')
print(f'Long Only: {data["Long Strategy Return"].mean() / data["Long Strategy Return"].std() * np.sqrt(252)}')
print(f'Adv MA Only: {data["Adv MA Strategy Return"].mean() / data["Adv MA Strategy Return"].std() * np.sqrt(252)}')
print(f'Bsc MA Only: {data["Bsc MA Strategy Return"].mean() / data["Bsc MA Strategy Return"].std() * np.sqrt(252)}')