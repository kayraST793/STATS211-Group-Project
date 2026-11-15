from pathlib import Path
import pandas as pd
import numpy as np

# Find the data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load GLD prices
df = pd.read_csv(DATA_DIR / "GLD.csv", header=[0, 1], index_col=0)

# Extract closing prices
prices = pd.to_numeric(df["Close"].iloc[:, 0])

# Calculate daily log returns
log_returns = np.log(prices / prices.shift(1)).dropna()

# Display first five returns
print(log_returns.head())

# Save daily log returns
log_returns.to_csv(DATA_DIR / "GLD_returns.csv", header=["Log_Return"])