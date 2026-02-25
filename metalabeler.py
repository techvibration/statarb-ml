import pandas as pd 
import numpy as np
from numba import njit
import time

@njit
def triple_barrier(p, vl, h, pt, sl):
    n = len(p)
    labels = np.zeros(n)
    for i in range(n-h):
        ep = p[i]
        vol = vl[i]
        # making the dynamic barriers
        ub = ep + (ep * vol * pt)
        lb = ep - (ep * vol * sl)
        for j in range(i+1, i+h+1):
            cp = p[j]
            if cp >= ub:         # Profitable trade
                labels[i] = 1
                break
            elif cp <= lb:       # Stop loss
                labels[i] = 0
                break 
            else:                # Time barrier
                labels[i] = 0
    return labels

df = pd.read_parquet("data/processed/top100af.parquet")

start_time = time.time()
labeled_df = []
for ticker, group in df.groupby('Ticker'):
    p = group['Close'].values
    vl = group['VOL_24H'].values
    
    # 1. Apply Triple Barrier Meta-Labels
    group['Meta'] = triple_barrier(p, vl, h=12, pt=2.0, sl=1.0)
    
    # 2. Safely calculate rolling features inside the ticker group
    group['BBW_MA20'] = group['BBW'].rolling(window=20).mean()
    
    labeled_df.append(group)

# Rebuild the dataframe
df_final = pd.concat(labeled_df, ignore_index=True)
print(f"Labeling complete in {time.time() - start_time:.2f} seconds!")

# 3. Apply the combined logic filter
entry = -2.0

# Filter for deeply oversold AND contracting volatility
train_df = df_final[
    (df_final['Z_Score'] < entry) & 
    (df_final['BBW'] < df_final['BBW_MA20']) &
    (df_final['ADX'] < 25)
].copy()

# Drop rows where the rolling MA was NaN (the first 19 days of each ticker)
train_df = train_df.dropna(subset=['BBW_MA20'])

print("-" * 30)
print("--- STRATEGY AUDIT ---")
print(f"Total Oversold Signals (Z < {entry} & Contracting BBW): {len(train_df)}")
print(f"Trades that hit Profit Target: {train_df['Meta'].sum()}")
print(f"Base Win Rate: {train_df['Meta'].mean() * 100:.2f}%")
print("-" * 30)

train_df.to_parquet("data/processed/training_set.parquet", index=False)