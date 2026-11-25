"""
Prepare GLD historical prices for Bayesian risk modeling.

Extract adjusted closing prices and calculate daily log returns.

Input: data/GLD.csv
Output: data/GLD_returns.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Find the shared data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load GLD prices
df = pd.read_csv(DATA_DIR / "GLD.csv", header=[0, 1], index_col=0)

# Extract closing prices
prices = pd.to_numeric(df["Close"].iloc[:, 0])

# Calc daily log returns
log_returns = np.log(prices / prices.shift(1)).dropna()

# Display first five returns
print(log_returns.head())

# Save daily log returns
log_returns.to_csv(DATA_DIR / "GLD_returns.csv", header=["Log_Return"])


# print("Number of returns:", len(log_returns))
# print("First date:", log_returns.index.min())
# print("Last date:", log_returns.index.max())
# print("Missing values:", log_returns.isna().sum())
# print("Duplicate dates:", log_returns.index.duplicated().sum())