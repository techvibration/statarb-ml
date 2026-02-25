import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/predictions.csv")

# Ensure Datetime is properly formatted for time-series math
df['Datetime'] = pd.to_datetime(df['Datetime'])

# --- CORRECTED BACKTEST PARAMETERS ---
STARTING_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01      # Risking 1% of portfolio per trade
FEE_PER_TRADE = 0.0005     # 5 bps broker fee per transaction 

# These MUST match your triple_barrier function 
PROFIT_TARGET_R = 2.0      
STOP_LOSS_R = 1.0          
CONFIDENCE_THRESHOLD = 0.65 
AVG_STOP_LOSS_PCT = 0.05   
RISK_FREE_RATE = 0.0       # Assumed risk-free rate for Sharpe

# We only simulate the timeline of trades we took
trades_taken = df[df['AI_Probability'] > CONFIDENCE_THRESHOLD].copy()
trades_taken = trades_taken.sort_values('Datetime').reset_index(drop=True)

print(f"Total Market Setups: {len(df)}")
print(f"Trades Approved by AI: {len(trades_taken)}")

# --- THE SIMULATION ENGINE ---
portfolio_value = STARTING_CAPITAL
equity_curve = [portfolio_value]

# Start the equity curve timeline on the date of the very first trade
trade_dates = [trades_taken.loc[0, 'Datetime'] if not trades_taken.empty else pd.Timestamp.now()]

peak_value = portfolio_value
max_drawdown = 0.0

winning_trades = 0
losing_trades = 0
trade_returns = [] 

for i in range(len(trades_taken)):
    actual_outcome = trades_taken.loc[i, 'Actual_Target']
    trade_time = trades_taken.loc[i, 'Datetime']
    
    # 1. Calculate Risk and True Position Size
    trade_risk_amount = portfolio_value * RISK_PER_TRADE
    position_size = trade_risk_amount / AVG_STOP_LOSS_PCT
    
    # 2. Calculate Round-Trip Fees (Entry + Exit)
    total_fees = (position_size * FEE_PER_TRADE) * 2
    
    # 3. Apply Profit Target or Stop Loss, factoring in Slippage/Fees
    if actual_outcome == 1:
        net_profit = (trade_risk_amount * PROFIT_TARGET_R) - total_fees
        portfolio_value += net_profit
        winning_trades += 1
        trade_returns.append(net_profit / portfolio_value)
    else:
        net_loss = (trade_risk_amount * STOP_LOSS_R) + total_fees
        portfolio_value -= net_loss
        losing_trades += 1
        trade_returns.append(-net_loss / portfolio_value)
        
    # 4. Drawdown Tracking
    if portfolio_value > peak_value:
        peak_value = portfolio_value
        
    current_drawdown = (peak_value - portfolio_value) / peak_value
    if current_drawdown > max_drawdown:
        max_drawdown = current_drawdown
        
    equity_curve.append(portfolio_value)
    trade_dates.append(trade_time)

# --- CALCULATE FINAL METRICS ---
total_return_pct = ((portfolio_value - STARTING_CAPITAL) / STARTING_CAPITAL) * 100

# Trade Stability Expectancy
returns_series = pd.Series(trade_returns)
trade_expectancy_ratio = returns_series.mean() / returns_series.std() if returns_series.std() != 0 else 0

# --- ANNUALIZED SHARPE RATIO CALCULATION ---
if len(trades_taken) > 0:
    # 1. Map equity values to exact dates
    equity_ts = pd.Series(equity_curve, index=trade_dates)
    
    # 2. Resample to Business Days ('B'), keeping the last value if multiple trades occurred, 
    # and forward-filling days where no trades occurred.
    daily_equity = equity_ts.resample('B').last().ffill()
    
    # 3. Calculate Daily Returns
    daily_returns = daily_equity.pct_change().dropna()
    
    # 4. Calculate Sharpe Ratio
    if daily_returns.std() != 0:
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    else:
        sharpe_ratio = 0.0
else:
    sharpe_ratio = 0.0

print("-" * 30)
print(f"Starting Capital:   ${STARTING_CAPITAL:,.2f}")
print(f"Ending Capital:     ${portfolio_value:,.2f}")
print(f"Net Return:         {total_return_pct:.2f}%")
print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
print(f"Trade Stability:    {trade_expectancy_ratio:.3f} (Mean Return / Std Dev)")
print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
print(f"Win/Loss Ratio:     {winning_trades}W / {losing_trades}L ({(winning_trades/len(trades_taken))*100:.2f}%)")
print("-" * 30)
# Save the daily equity curve for the analytics script
daily_equity.to_csv("data/processed/bot_daily_equity.csv", header=['Bot_Equity'])

# --- PRESENTATION CHART ---
plt.figure(figsize=(12, 6))
plt.plot(equity_curve, color='teal', linewidth=2)
plt.title(f'AlphaFinder V2 | Net Return: {total_return_pct:.2f}% | Max DD: {max_drawdown*100:.2f}% | Sharpe: {sharpe_ratio:.2f}')
plt.xlabel('Number of Trades Executed (Sequential)')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, alpha=0.3)
plt.axhline(STARTING_CAPITAL, color='black', linestyle='--', linewidth=1)
plt.fill_between(range(len(equity_curve)), equity_curve, STARTING_CAPITAL, where=(np.array(equity_curve) >= STARTING_CAPITAL), color='green', alpha=0.1)
plt.fill_between(range(len(equity_curve)), equity_curve, STARTING_CAPITAL, where=(np.array(equity_curve) < STARTING_CAPITAL), color='red', alpha=0.1)
plt.show()