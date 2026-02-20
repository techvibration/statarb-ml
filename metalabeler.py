import pandas as pd 
import numpy as np
from numba import njit
import time
@njit
def triple_barrier(p,vl,h,pt,sl):
    n = len(p)
    labels = np.zeros(n)
    for i in range(n-h):
        ep = p[i]
        vol = vl[i]
        #making the dynamic barriers
        ub = ep + (ep*vol*pt)
        lb = ep - (ep*vol*sl)
        for j in range(i+1,i+h+1):
            cp = p[j]
            if cp >= ub:         #Profitable trade
                labels[i] =1
                break
            elif cp <= lb:        #Stop loss
                labels[i] =0
                break 
            else:                 #Time barrier
                labels[i] =0
    return labels
df = pd.read_parquet("data/processed/top100af.parquet")



start_time = time.time()
labeled_df = []
for ticker,group in df.groupby('Ticker'):
    p = group['Close'].values
    vl = group['VOL_24H'].values
    
    group['Meta'] = triple_barrier(p,vl,h=8,pt=1.5,sl=1.0)
    labeled_df.append(group)
df_final = pd.concat(labeled_df)
print(f"Labeling complete in {time.time() - start_time:.2f} seconds!")

entry = -2.0
train_df = df_final[df_final['Z_Score'] < entry].copy()
print("-" * 30)
print("--- STRATEGY AUDIT ---")
print(f"Total Oversold Signals (Z < {entry}): {len(train_df)}")
print(f"Trades that hit Profit Target: {train_df['Meta'].sum()}")
print(f"Win Rate: {train_df['Meta'].mean() * 100:.2f}%")
print("-" * 30)
train_df.to_parquet("data/processed/training_set.parquet",index = False)



   



