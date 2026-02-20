import pandas as pd 
import numpy as np
import importlib.metadata  # Explicitly import the sub-module
dist = importlib.metadata.distribution("pandas-ta-openbb")
import pandas_ta as ta
df = pd.read_parquet("data/processed/top_100_features.parquet")
df = df.sort_values(['Ticker','Datetime'])


def calc_features(tdf):
    tdf['RSI'] = ta.rsi(tdf['Close'],length=14)
    adx_df = ta.adx(tdf['High'],tdf['Low'],tdf['Close'],length=14)
    if adx_df is not None:
        tdf['ADX'] = adx_df.iloc[:,0]
    bb_df = ta.bbands(tdf['Close'],length=20,std=2)
    if bb_df is not None:
        tdf['BBW'] = bb_df.iloc[:,3]
    tdf['LOG_RET'] = np.log(tdf['Close']/tdf['Close'].shift(1))
    tdf['VOL_24H'] = tdf['LOG_RET'].rolling(24).std()
    return tdf
if 'index' in df.columns:
    df = df.drop(columns=['index'])
features = []
for ticker,group in df.groupby('Ticker'):
    fg = calc_features(group)
    features.append(fg)

features_df = pd.concat(features,ignore_index=True)


features_df = features_df.dropna(subset=['RSI', 'ADX', 'BBW', 'VOL_24H', 'Z_Score'])
features_df.to_parquet("data/processed/top100af.parquet",index=False)

    

    
   