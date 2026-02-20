import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("--- ALPHAFINDER V2: INSTITUTIONAL BACKTEST ---")

# 1. Load the AI's predictions
df = pd.read_csv("data/processed/predictions.csv")

# 2. Backtest Parameters
STARTING_CAPITAL = 10000.0
RISK_PER_TRADE = 0.01      # Risk 1% of current account balance per trade
FEE_PER_TRADE = 0.001      # 0.1% transaction cost (slippage + commission)
PROFIT_TARGET = 2.0        # 1.5x Reward
STOP_LOSS = 1.0            # 1.0x Risk
CONFIDENCE_THRESHOLD = 0.65 # The AI "Bouncer" strictness

# 3. Filter for trades the AI actually approved
# We only simulate the timeline of trades we took
trades_taken = df[df['AI_Probability'] > CONFIDENCE_THRESHOLD].copy()
trades_taken = trades_taken.reset_index(drop=True)

print(f"Total Market Setups: {len(df)}")
print(f"Trades Approved by AI: {len(trades_taken)}")

# 4. The Simulation Engine
portfolio_value = STARTING_CAPITAL
equity_curve = [portfolio_value]
peak_value = portfolio_value
max_drawdown = 0.0

winning_trades = 0
losing_trades = 0

for i in range(len(trades_taken)):
    actual_outcome = trades_taken.loc[i, 'Actual_Target']
    
    # Calculate Risk (1% of current capital)
    trade_risk_amount = portfolio_value * RISK_PER_TRADE
    
    # Pay the broker fee up front
    portfolio_value -= (portfolio_value * FEE_PER_TRADE)
    
    # Did the trade win or lose?
    if actual_outcome == 1:
        portfolio_value += (trade_risk_amount * PROFIT_TARGET)
        winning_trades += 1
    else:
        portfolio_value -= (trade_risk_amount * STOP_LOSS)
        losing_trades += 1
        
    # Drawdown Tracking
    if portfolio_value > peak_value:
        peak_value = portfolio_value
        
    current_drawdown = (peak_value - portfolio_value) / peak_value
    if current_drawdown > max_drawdown:
        max_drawdown = current_drawdown
        
    equity_curve.append(portfolio_value)

# 5. Calculate Final Metrics
total_return_pct = ((portfolio_value - STARTING_CAPITAL) / STARTING_CAPITAL) * 100

# Rough Sharpe Ratio (Using the trades as steps)
returns = pd.Series(equity_curve).pct_change().dropna()
# Assuming roughly 252 trading days, but scaled by trade frequency
sharpe_ratio = np.sqrt(len(trades_taken)) * (returns.mean() / returns.std()) if returns.std() != 0 else 0

# 6. Final Printout
print("-" * 40)
print(f"Starting Capital:   ${STARTING_CAPITAL:,.2f}")
print(f"Ending Capital:     ${portfolio_value:,.2f}")
print(f"Net Return:         {total_return_pct:.2f}%")
print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
print(f"Win/Loss Ratio:     {winning_trades}W / {losing_trades}L")
print("-" * 40)

# 7. The Presentation Chart
plt.figure(figsize=(12, 6))
plt.plot(equity_curve, color='teal', linewidth=2)
plt.title(f'AlphaFinder V2 Equity Curve | Net Return: {total_return_pct:.2f}% | Max DD: {max_drawdown*100:.2f}%')
plt.xlabel('Number of Trades Executed')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, alpha=0.3)
plt.axhline(STARTING_CAPITAL, color='black', linestyle='--', linewidth=1)
plt.fill_between(range(len(equity_curve)), equity_curve, STARTING_CAPITAL, where=(np.array(equity_curve) >= STARTING_CAPITAL), color='green', alpha=0.1)
plt.fill_between(range(len(equity_curve)), equity_curve, STARTING_CAPITAL, where=(np.array(equity_curve) < STARTING_CAPITAL), color='red', alpha=0.1)
plt.show()