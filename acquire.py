import pandas as pd 
import yfinance as yf
top_100 = pd.read_csv('top_100_ticker.csv')
tickers = [str(t).replace('.', '-') for t in top_100['Ticker'].tolist()]
df_raw = yf.download(
    tickers = tickers,
    period="2y",
    interval="1h"
)
df = df_raw.stack(level=1,future_stack=True).reset_index()


f_path = "data/raw/top_100_hourly.parquet"
df.to_parquet(f_path,index=True)



