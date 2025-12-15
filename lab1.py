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
def get_data(ticker='SPY', start_date='2010-01-01', end_date='2024-01-01'):
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

data = get_data()
apply_strategy(data)
print(data.head())
