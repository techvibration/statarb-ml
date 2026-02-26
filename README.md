AlphaFinder: Market-Neutral ML Trading Bot
AlphaFinder is an end-to-end algorithmic trading system that leverages Gradient Boosted Decision Trees (LightGBM) to execute market-neutral statistical arbitrage. The bot is designed to identify high-conviction intraday signals on large-cap equities and execute trades autonomously via the Alpaca Markets API.

🛠 Features
Predictive Engine: Powered by LightGBM, trained on historical hourly OHLCV data.

Market-Neutral Strategy: Simultaneously evaluates Long and Short opportunities to hedge against broad market volatility.

Feature Engineering: Custom pipeline calculating Z-Scores, Bollinger Band Width (BBW), ADX, and Relative Strength (Ticker vs. Universe).

Live Execution: Real-time trade execution using the alpaca-py SDK with a secure .env credential vault.

Walk-Forward Optimization: Backtested using a rolling window to prevent data leakage and ensure temporal validity.

📊 Technical Stack
Language: Python 3.11.9 

ML Library: LightGBM

Data Source: Yahoo Finance (yfinance)

Brokerage API: Alpaca Markets

Indicators: pandas_ta

Environment Management: python-dotenv
