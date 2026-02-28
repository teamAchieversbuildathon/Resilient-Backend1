import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["SPY", "QQQ", "IWM", "EFA", "TLT"]

all_prices = []

for ticker in tickers:
    print(f"Downloading {ticker}...")
    data = yf.download(
        ticker,
        start="2015-01-01",
        end="2024-01-01",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    data = data[["Close"]].rename(columns={"Close": ticker})
    all_prices.append(data)

prices = pd.concat(all_prices, axis=1)
prices = prices.dropna()

print(prices.head())

prices.to_csv("data/prices.csv")

prices.plot(figsize=(10, 5))
plt.title("ETF Price History")
plt.show()
