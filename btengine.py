import backtrader as bt
import pandas as pd
import numpy as np
import math

# --- 1. DEFINE THE CUSTOM DATA FEED ---
class AlphaData(bt.feeds.PandasData):
    # Tell Backtrader to expect our specific machine learning columns
    lines = ('ai_prob', 'vol_24h',)
    params = (
        ('ai_prob', -1),
        ('vol_24h', -1),
    )

class MarketNeutralStrategy(bt.Strategy):
    params = (
        ('num_positions', 5),            # Top 5 Long, Bottom 5 Short
        ('profit_target_mult', 2.0),     
        ('stop_loss_mult', 1.0),         
        ('time_barrier', 12),            # 12-hour holding period
    )

    def __init__(self):
        self.entry_bars = {}
        self.active_orders = {}

    def next(self):
        portfolio_value = self.broker.getvalue()
        
        # 1. MANAGE EXISTING POSITIONS & GHOST ORDERS
        for data in self.datas:
            ticker = data._name
            position = self.getposition(data)
            
            if position.size != 0:
                bars_held = len(data) - self.entry_bars.get(ticker, 0)
                if bars_held >= self.params.time_barrier:
                    self.close(data=data) # Liquidate
                    
                    # Kill the ghost orders
                    if ticker in self.active_orders:
                        for order in self.active_orders[ticker]:
                            if order and order.alive():
                                self.cancel(order)
                        del self.active_orders[ticker]

        # 2. CROSS-SECTIONAL RANKING (The Magic)
        # Only rank stocks if we aren't currently holding them and they have valid data
        available_tickers = []
        for data in self.datas:
            if self.getposition(data).size == 0 and len(data) > 0:
                # Store the probability, the ticker name, and the data object
                available_tickers.append((data.ai_prob[0], data._name, data))

        # Sort the list based on AI Probability (Highest to Lowest)
        available_tickers.sort(key=lambda x: x[0], reverse=True)

        # 3. SELECT THE EXTREMES
        # We need at least enough tickers to fill our long and short buckets
        if len(available_tickers) >= (self.params.num_positions * 2):
            top_longs = available_tickers[:self.params.num_positions]
            bottom_shorts = available_tickers[-self.params.num_positions:]
            
            # 4. DOLLAR-NEUTRAL ALLOCATION
            # We divide 90% of our total portfolio value evenly across the 10 positions
            allocation_per_trade = (portfolio_value * 0.90) / (self.params.num_positions * 2)

            # --- EXECUTE LONGS ---
            for prob, ticker, data in top_longs:
                if prob == 0.0: continue # Skip if no ML prediction exists
                
                stop_dist = data.close[0] * data.vol_24h[0] * self.params.stop_loss_mult
                target_dist = data.close[0] * data.vol_24h[0] * self.params.profit_target_mult
                
                if stop_dist > 0:
                    ideal_shares = math.floor(allocation_per_trade / data.close[0])
                    
                    if ideal_shares >= 1:
                        orders = self.buy_bracket(
                            data=data, size=ideal_shares,
                            stopprice=data.close[0] - stop_dist,
                            limitprice=data.close[0] + target_dist
                        )
                        self.entry_bars[ticker] = len(data)
                        self.active_orders[ticker] = [orders[1], orders[2]]

            # --- EXECUTE SHORTS ---
            for prob, ticker, data in bottom_shorts:
                if prob == 0.0: continue 
                
                stop_dist = data.close[0] * data.vol_24h[0] * self.params.stop_loss_mult
                target_dist = data.close[0] * data.vol_24h[0] * self.params.profit_target_mult
                
                if stop_dist > 0:
                    ideal_shares = math.floor(allocation_per_trade / data.close[0])
                    
                    if ideal_shares >= 1:
                        orders = self.sell_bracket(
                            data=data, size=ideal_shares,
                            stopprice=data.close[0] + stop_dist,  # Stop loss is ABOVE
                            limitprice=data.close[0] - target_dist # Target is BELOW
                        )
                        self.entry_bars[ticker] = len(data)
                        self.active_orders[ticker] = [orders[1], orders[2]]
# --- 3. SPIN UP THE ENGINE ---
if __name__ == '__main__':
    cerebro = bt.Cerebro()

    
    df = pd.read_parquet("data/processed/main.parquet")
    
    # Backtrader hates timezone-aware datetimes, so we strip the UTC +00:00 just to be safe
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
    
    # Inject each ticker as its own independent data stream
    tickers = df['Ticker'].unique()
    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker].copy()
        ticker_df.set_index('Datetime', inplace=True)
        ticker_df.sort_index(inplace=True)
        
        data = AlphaData(dataname=ticker_df,
                        ai_prob='AI_Probability', 
                        vol_24h='VOL_24H')
        cerebro.adddata(data, name=ticker)

    # Attach our Strategy
    cerebro.addstrategy(MarketNeutralStrategy)

    # Set Institutional Rules
    STARTING_CAPITAL = 1000000.0
    cerebro.broker.setcash(STARTING_CAPITAL)
    cerebro.broker.setcommission(commission=0.0005) # 5 basis points round-trip friction

    # Attach Institutional Analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    print("Running Simulation. This may take a minute as it walks forward tick-by-tick...")
    
    # Run it!
    results = cerebro.run()
    strat = results[0]

    # --- THE MOMENT OF TRUTH ---
   # --- THE MOMENT OF TRUTH ---
    final_value = cerebro.broker.getvalue()
    print("\n" + "="*45)
    print("--- ALPHAFINDER STRICT BACKTEST RESULTS ---")
    print(f"Ending Portfolio Value: ${final_value:,.2f}")
    print(f"Net Return:             {((final_value - STARTING_CAPITAL) / STARTING_CAPITAL) * 100:.2f}%")
    
    dd_info = strat.analyzers.drawdown.get_analysis()
    print(f"Max Drawdown:           {dd_info.get('max', {}).get('drawdown', 0.0):.2f}%")
    
    trade_info = strat.analyzers.trades.get_analysis()
    total_closed = trade_info.get('total', {}).get('closed', 0)

    print("-" * 45)
    if total_closed > 0:
        wins = trade_info.get('won', {}).get('total', 0)
        losses = trade_info.get('lost', {}).get('total', 0)
        win_rate = (wins / total_closed) * 100
        
        # THE AUTOPSY METRICS
        avg_win = trade_info.get('won', {}).get('pnl', {}).get('average', 0.0)
        avg_loss = trade_info.get('lost', {}).get('pnl', {}).get('average', 0.0)
        worst_loss = trade_info.get('lost', {}).get('pnl', {}).get('max', 0.0)
        
        print(f"Total Trades Executed:  {total_closed}")
        print(f"Win/Loss Ratio:         {wins}W / {losses}L ({win_rate:.2f}%)")
        print("-" * 45)
        print(f"Average Win (Profit):   ${avg_win:,.2f}")
        print(f"Average Loss:           ${avg_loss:,.2f}")
        print(f"Worst Single Loss:      ${worst_loss:,.2f}")
        
        if avg_loss != 0:
            realized_ratio = abs(avg_win / avg_loss)
            print(f"Realized Reward Ratio:  {realized_ratio:.2f}x (Target was 2.0x)")
    else:
        print("Total Trades Executed:  0")
    print("="*45)