import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


bot_df = pd.read_csv("data/processed/bot_daily_equity.csv", index_col=0, parse_dates=True)
# Force index to be sorted just in case CSV parsing shuffled it
bot_df = bot_df.sort_index()
bot_equity = bot_df['Bot_Equity']

start_date = bot_equity.index.min()
end_date = bot_equity.index.max()

print(f"Bot traded from {start_date.date()} to {end_date.date()}. Slicing local data to match...")

#Load Your Local Market Data (No Internet Required)
raw_df = pd.read_parquet("data/processed/top100af.parquet")
raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])

# Filter raw data to match the exact timeframe
market_df = raw_df[(raw_df['Datetime'] >= start_date) & (raw_df['Datetime'] <= end_date)].copy()
market_df = market_df.sort_values('Datetime')

# Create the Equal-Weight Benchmark (Your specific 100-ticker universe)
market_df['Daily_Return'] = market_df.groupby('Ticker')['Close'].pct_change()
# Average the returns across all tickers for each day
daily_market_returns = market_df.groupby('Datetime')['Daily_Return'].mean().fillna(0)

# Resample to match the bot's index exactly to prevent date misalignment
daily_market_returns = daily_market_returns.resample('B').mean().fillna(0)

# Align the return streams perfectly side-by-side
combined_returns = pd.concat([bot_equity.pct_change(), daily_market_returns], axis=1).dropna()
combined_returns.columns = ['Bot_Returns', 'Market_Returns']

# Rebuild the equity curves from the perfectly aligned returns
STARTING_CAPITAL = bot_equity.iloc[0]
aligned_bot_equity = STARTING_CAPITAL * (1 + combined_returns['Bot_Returns']).cumprod()
aligned_mkt_equity = STARTING_CAPITAL * (1 + combined_returns['Market_Returns']).cumprod()

# Calculate Head-to-Head Metrics
def calc_metrics(equity_series, daily_returns):
    net_return = ((equity_series.iloc[-1] - STARTING_CAPITAL) / STARTING_CAPITAL) * 100
    peak = equity_series.cummax()
    drawdown = ((peak - equity_series) / peak).max() * 100
    sharpe = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
    return net_return, drawdown, sharpe

bot_ret, bot_dd, bot_sharpe = calc_metrics(aligned_bot_equity, combined_returns['Bot_Returns'])
mkt_ret, mkt_dd, mkt_sharpe = calc_metrics(aligned_mkt_equity, combined_returns['Market_Returns'])

# Calculate Beta and Correlation
cov_matrix = np.cov(combined_returns['Bot_Returns'], combined_returns['Market_Returns'])
beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
correlation = combined_returns['Bot_Returns'].corr(combined_returns['Market_Returns'])

# 7. Print the Meta Report

print("--- ALPHAFINDER VS LOCAL UNIVERSE BENCHMARK ---")
print(f"Timeframe:          {combined_returns.index[0].date()} to {combined_returns.index[-1].date()}")

print(f"Net Return:         Bot: {bot_ret:>8.2f}%  |  Market: {mkt_ret:>8.2f}%")
print(f"Max Drawdown:       Bot: {bot_dd:>8.2f}%  |  Market: {mkt_dd:>8.2f}%")
print(f"Sharpe Ratio:       Bot: {bot_sharpe:>8.2f}   |  Market: {mkt_sharpe:>8.2f}")

print(f"Portfolio Beta:     {beta:.3f} (1.0 = tracks your universe perfectly)")
print(f"Correlation:        {correlation:.3f} (1.0 = identical movement)")


# 8. Generate Presentation Chart
plt.figure(figsize=(12, 6))
plt.plot(aligned_bot_equity.index, aligned_bot_equity.values, color='teal', linewidth=2, label=f'AlphaFinder (+{bot_ret:.2f}%)')
plt.plot(aligned_mkt_equity.index, aligned_mkt_equity.values, color='orange', linewidth=2, alpha=0.8, label=f'Universe Benchmark (+{mkt_ret:.2f}%)')

plt.title('AlphaFinder Performance vs. Universe Benchmark')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()