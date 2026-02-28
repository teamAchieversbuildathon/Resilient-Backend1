import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load returns data
returns = pd.read_csv(
    "data/returns.csv",
    index_col=0,
    parse_dates=True
)

# 2. Basic statistics
mean_returns = returns.mean()
volatility = returns.std()

print("\nAverage Daily Returns:")
print(mean_returns)

print("\nDaily Volatility:")
print(volatility)

# 3. Covariance matrix
cov_matrix = returns.cov()

print("\nCovariance Matrix:")
print(cov_matrix)

# 4. Correlation matrix
corr_matrix = returns.corr()

# 5. Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0
)
plt.title("Asset Correlation Heatmap")
plt.show()
