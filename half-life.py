import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Load your clean features
df = pd.read_parquet("data/processed/top100af.parquet")

# 2. Dictionary to hold our dynamic horizons
ticker_half_lives = {}


for ticker, group in df.groupby('Ticker'):
    # Extract the Z-Score series
    z = group['Z_Score'].dropna()
    
    if len(z) < 50:
        ticker_half_lives[ticker] = 4 # Fallback for missing data
        continue
        
    # Calculate Delta Z (Current - Previous)
    z_lag = z.shift(1).dropna()
    z_curr = z.loc[z_lag.index]
    delta_z = z_curr - z_lag
    
    # Linear Regression: Delta_Z vs Z_lag
    X = z_lag.values.reshape(-1, 1)
    y = delta_z.values
    
    model = LinearRegression().fit(X, y)
    lam = model.coef_[0]
    
    # If lambda is >= 0, the stock is trending, not mean-reverting.
    # We assign a max timeout of 24 hours.
    if lam >= -1e-5:
        ticker_half_lives[ticker] = 24
    else:
        # Calculate Half-Life
        hl = -np.log(2) / lam
        
        
        hl = max(1, min(int(np.round(hl)), 24))
        ticker_half_lives[ticker] = hl


hl_df = pd.DataFrame(list(ticker_half_lives.items()), columns=['Ticker', 'Half_Life'])

hl_df.to_csv("data/processed/ou_half_lives.csv", index=False)