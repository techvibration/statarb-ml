import pandas as pd


df = pd.read_parquet("data/raw/top_100_hourly.parquet")
df = df.sort_values(['Ticker', 'Datetime'])


df['Return'] = df.groupby('Ticker')['Close'].pct_change()
df['Fwd_Return'] = df.groupby('Ticker')['Close'].pct_change().shift(-1)

#Cross-Sectional Z-Score for every hour
df['Z_Score'] = df.groupby('Datetime')['Return'].transform(lambda x: (x - x.mean()) / x.std())


df_clean = df.dropna(subset=['Z_Score', 'Fwd_Return'])
#Calculate the Spearman Rank Correlation (The IC) for every single hour
ic_series = df_clean.groupby('Datetime').apply(
    lambda x: x['Z_Score'].corr(x['Fwd_Return'], method='spearman'), 
    include_groups=False
)
df = df.reset_index()
print(df.columns.tolist())

# The Verdict
print(f"Average Hourly IC: {ic_series.mean():.4f}")
print(f"Percentage of Hours with Negative IC: {(ic_series < 0).mean() * 100:.2f}%")
df.to_parquet("data/processed/top_100_features.parquet", index=False)
ic_df = ic_series.reset_index()
ic_df.columns = ['Datetime', 'IC']
ic_df.to_csv("data/processed/hourly_ic_baseline.csv", index=False)
