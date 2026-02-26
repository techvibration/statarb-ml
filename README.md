AlphaFinder: Market-Neutral ML Trading Bot
AlphaFinder is a fully autonomous, multi-class algorithmic trading system designed to exploit short-term market inefficiencies across a 100-stock universe. Built with a strict focus on capital preservation, the engine utilizes a LightGBM classifier to identify high-conviction setups and executes them using dynamic, volatility-adjusted risk parameters.


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

## Installation

Clone the repository:

git clone https://github.com/techvibration/statarb-ml.git  
cd AlphaFinder  

Create a virtual environment:

python -m venv venv  

Activate the virtual environment:

Windows:
venv\Scripts\activate  

Linux / Mac:
source venv/bin/activate  

Install dependencies:

pip install -r requirements.txt  

Create a .env file in the root directory and add your Alpaca credentials:

ALPACA_API_KEY=your_api_key  
ALPACA_SECRET_KEY=your_secret_key  

# Boot it up
python alphafinder_live.py



