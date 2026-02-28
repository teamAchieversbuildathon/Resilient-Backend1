import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load price data
prices = pd.read_csv(
    "data/prices.csv",
    index_col=0,
    parse_dates=True
)

# FORCE all columns to be numeric
prices = prices.apply(pd.to_numeric, errors="coerce")


# 2. Compute daily log returns
returns = np.log(prices / prices.shift(1))

# 3. Remove first row (NaN)
returns = returns.dropna()

# 4. Preview returns
print(returns.head())

# 5. Save returns
returns.to_csv("data/returns.csv")

# 6. Plot returns
returns.plot(figsize=(10, 6), title="Daily Log Returns")
plt.show()
