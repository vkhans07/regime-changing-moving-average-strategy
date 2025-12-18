import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
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
def fixed_vix_ma_strategy(data, shorter_window=50, longer_window=200, vix_threshold=20):
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
    data['Total Long Strategy Return'] = data['Daily Log Return'].cumsum().apply(np.exp)
    data['Total Adv MA Strategy Return'] = data['Adv Strategy Return'].cumsum().apply(np.exp)
    data['Total Bsc MA Strategy Return'] = data['Bsc Strategy Return'].cumsum().apply(np.exp)

def optimize_vix_threshold_gd(data, learning_rate=0.5, epochs=50, initial_threshold=20):
    """
    Optimizes the VIX threshold using Gradient Descent to maximize the Sharpe Ratio.
    Since the trading signal is a step function (VIX < Threshold), the gradient is zero almost everywhere.
    We approximate the step function with a Sigmoid to make it differentiable for the optimizer.
    """
    # Create a local copy to avoid messing up the main dataframe during optimization loops
    opt_data = data.copy()
    
    # Sigmoid parameters
    k = 1  # Steepness of the sigmoid (higher = closer to a step function)
    
    theta = initial_threshold
    
    print(f"\nStarting Gradient Descent from VIX Threshold: {theta}")
    
    history = []

    for i in range(epochs):
        # 1. Calculate 'Soft' Signal using Sigmoid: 1 / (1 + e^(-k * (Threshold - VIX)))
        # This approximates 1 when VIX < Threshold and 0 when VIX > Threshold
        # Note: We want (Threshold - VIX) to be positive for the signal to be active
        sigmoid_input = k * (theta - opt_data['VIX'])
        soft_signal = 1 / (1 + np.exp(-sigmoid_input))
        
        # Combine with MA signal (Hard constraint for MA, Soft for VIX)
        # We treat MA signal as a constant mask here (1 or 0)
        ma_signal = np.where(opt_data['Shorter MA'] > opt_data['Longer MA'], 1, 0)
        combined_signal = soft_signal * ma_signal
        
        # Calculate Returns (Shifted by 1 as per backtest logic)
        strategy_returns = combined_signal.shift(1) * opt_data['Daily Log Return']
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            break

        # Calculate Sharpe Ratio (Annualized)
        mu = strategy_returns.mean()
        sigma = strategy_returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma != 0 else 0
        
        # --- Gradient Approximation (Finite Difference) ---
        # While we have a sigmoid, calculating the exact derivative of the Sharpe Ratio 
        # w.r.t theta is complex. Numerical gradient (finite difference) on the 
        # smoothed sigmoid objective is robust and effectively 'Gradient Descent'.
        
        epsilon = 0.01
        
        # Perturb theta + epsilon
        input_plus = k * ((theta + epsilon) - opt_data['VIX'])
        signal_plus = (1 / (1 + np.exp(-input_plus))) * ma_signal
        ret_plus = signal_plus.shift(1) * opt_data['Daily Log Return']
        sharpe_plus = (ret_plus.mean() / ret_plus.std()) * np.sqrt(252)
        
        # Gradient = d(Sharpe)/d(Theta)
        gradient = (sharpe_plus - sharpe) / epsilon
        
        # Update Theta (Ascent because we want to maximize Sharpe)
        theta = theta + learning_rate * gradient
        
        history.append(sharpe)

    print(f"Optimization Complete. Optimal VIX Threshold: {theta:.2f}")
    return theta

def rolling_vix_ma_strategy(data, shorter_window=50, longer_window=200, vix_rolling_window=60):
    """
    Strategy using a rolling mean of the VIX as the dynamic threshold.
    """
    # Ensure MAs exist (re-calculating to be safe or assuming they exist from previous runs)
    data['Shorter MA'] = data['Price'].rolling(window=shorter_window).mean()
    data['Longer MA'] = data['Price'].rolling(window=longer_window).mean()
    
    # Calculate Rolling VIX Threshold
    data['Rolling VIX Mean'] = data['VIX'].rolling(window=vix_rolling_window).mean()
    
    # Generate Signal: Long if MA bullish AND Current VIX < Rolling VIX Mean
    # This implies we trade when volatility is lower than it has been recently
    data['Rolling VIX Signal'] = np.where(
        (data['Shorter MA'] > data['Longer MA']) & 
        (data['VIX'] < data['Rolling VIX Mean']), 
        1, 0
    )
    
    return data

def backtest_rolling(data):
    # Specialized backtest append for the rolling strategy
    data['Rolling VIX Strategy Return'] = data['Rolling VIX Signal'].shift(1) * data['Daily Log Return']
    # Handle NaNs created by the new rolling window
    mask = ~np.isnan(data['Rolling VIX Strategy Return'])
    data.loc[mask, 'Total Rolling VIX Strategy Return'] = data.loc[mask, 'Rolling VIX Strategy Return'].cumsum().apply(np.exp)


# --- MAIN EXECUTION ---

# 1. Run Standard & Fixed VIX Strategies
data = get_data(ticker='SPY')
fixed_vix_ma_strategy(data)
backtest(data)

# 2. Optimize VIX Threshold (Gradient Descent)
optimal_threshold = optimize_vix_threshold_gd(data)

# Apply Optimal Threshold to create a new equity curve for comparison
data['Opt Signal'] = np.where((data['Shorter MA'] > data['Longer MA']) &
                              (data['VIX'] < optimal_threshold), 1, 0)
data['Opt Strategy Return'] = data['Opt Signal'].shift(1) * data['Daily Log Return']
data['Total Opt Strategy Return'] = data['Opt Strategy Return'].cumsum().apply(np.exp)

# 3. Run Rolling VIX Strategy
rolling_vix_ma_strategy(data)
backtest_rolling(data)

# --- REPORTING ---

print(data[['Total Long Strategy Return', 'Total Adv MA Strategy Return', 'Total Bsc MA Strategy Return', 'Total Opt Strategy Return']].tail())
print('\nSharpe Ratios:')
print(f'Long Only: {data["Daily Log Return"].mean() / data["Daily Log Return"].std() * np.sqrt(252):.4f}')
print(f'Fixed VIX (20): {data["Adv Strategy Return"].mean() / data["Adv Strategy Return"].std() * np.sqrt(252):.4f}')
print(f'Uncontrolled MA: {data["Bsc Strategy Return"].mean() / data["Bsc Strategy Return"].std() * np.sqrt(252):.4f}')
print(f'Optimized VIX ({optimal_threshold:.2f}): {data["Opt Strategy Return"].mean() / data["Opt Strategy Return"].std() * np.sqrt(252):.4f}')
print(f'Rolling VIX: {data["Rolling VIX Strategy Return"].mean() / data["Rolling VIX Strategy Return"].std() * np.sqrt(252):.4f}')

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Total Long Strategy Return'], label='Long Only', color='blue', alpha=0.3)
plt.plot(data.index, data['Total Bsc MA Strategy Return'], label='Uncontrolled MA', color='gray', alpha=0.5)
plt.plot(data.index, data['Total Adv MA Strategy Return'], label='Fixed VIX (20)', color='red')
plt.plot(data.index, data['Total Opt Strategy Return'], label=f'Optimized VIX ({optimal_threshold:.1f})', color='purple', linestyle='--')
plt.plot(data.index, data['Total Rolling VIX Strategy Return'], label='Rolling VIX Threshold', color='green')

plt.title('Strategy Comparison: Fixed vs Optimized vs Rolling VIX')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.savefig('Strategy_Comparison.png')
plt.show()
plt.clf()