import pandas as pd 
import yfinance as yf 
import requests
import io
from concurrent.futures import ThreadPoolExecutor
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(url, headers=headers)

table = pd.read_html(io.StringIO(response.text))
df = table[0]
ticker_list = df['Symbol'].tolist()
def market_cap(ticker):
    try:
        t = yf.Ticker(ticker.replace('.','-'))
        return ticker, t.fast_info['market_cap']
    except:
        return ticker,0
with ThreadPoolExecutor(max_workers =20) as executor:
    results = list(executor.map(market_cap,ticker_list))

df_cap = pd.DataFrame(results,columns = ['Ticker','MarketCap'])
df_cap = df_cap.sort_values(by = 'MarketCap',ascending=False).reset_index(drop=True)

#Taking the Top 100 stocks by Marketcap
top_100 = df_cap.head(100)
top_100_ticker = top_100.merge(
    df[['Symbol','GICS Sector','GICS Sub-Industry']],
    left_on='Ticker', 
    right_on='Symbol',
    how='left'
).drop(columns=['Symbol'])
top_100_ticker.to_csv('top_100_ticker.csv',index=False)
print(top_100_ticker.head(10))


