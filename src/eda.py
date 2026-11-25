"""
Exploratory Data Analysis (EDA) of GLD Daily Log Returns

This script analyzes three years of GLD daily log returns
(November 2023 - November 2026).

It calculates summary statistics and examines the return
distribution to help assess whether a Student-t model is
appropriate for the data. (We will use the Student-t model
regardless, but we shouldn't assume it's the right fit)

Input:
    data/GLD_returns.csv

Outputs:
    figures/eda/returns_time_series.png
    figures/eda/returns_histogram.png
    figures/eda/normal_qq_plot.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

#1. Load Data---------------------------------------------
# Find the project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define input and output directories
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures" / "eda"

# Create the figures directory if needed
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load daily log returns calculated in data_prep.py
returns = pd.read_csv(
    DATA_DIR / "GLD_returns.csv",
    index_col=0,
    parse_dates=True
)["Log_Return"]


#2.  Summary Statistics-----------------------------------
# Count, mean, std, quartiles, min, max
print(returns.describe())

# Measure distribution asymmetry and tail heaviness
print("Skewness:", returns.skew())
print("Excess kurtosis:", returns.kurt())

# Identify the most extreme negative daily log return and its date to check for potential data errors.
# NOTE: The largest daily loss occurred on January 30, 2026,
# coinciding with a major gold market selloff. This suggests
# the extreme observation reflects a genuine market event
# rather than a data error.
print("Worst return date:", returns.idxmin())
print("Worst log return:", returns.min())

"""
Initial observations:
- Negative skewness suggests a more pronounced left tail.
- High excess kurtosis suggests heavier tails than a normal distribution.
- These findings support investigating a Student-t distribution.
- However, the standard Student-t distribution is symmetric and may
  not fully capture the negative skewness observed in GLD returns.
"""


# 3. Time Series of Daily Log Returns-----------------------
# Examine how returns and volatility change over time
plt.figure(figsize=(12, 5))

plt.plot(returns.index, returns, linewidth=0.8)

plt.axhline(y=0, color="black", linestyle="--")

plt.title("Time Series of GLD Daily Log Returns")
plt.xlabel("Date")
plt.ylabel("Daily Log Return")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "returns_time_series.png", dpi=300)
plt.show()
plt.close()

"""
Time Series Observations:
- Daily log returns generally fluctuate around zero.
- Larger fluctuations appear more frequently in 2025 and 2026.
- Periods of large fluctuations suggest potential volatility clustering.
- The largest negative spike corresponds to January 30, 2026.
- Changing volatility may be a limitation of the Student-t model,
  which assumes a constant scale parameter over time.
"""

# 4. Histogram and Normal Distribution Comparison---------------------
# Fit a normal distribution using the sample mean and std
mu, sigma = norm.fit(returns)

plt.figure(figsize=(8, 5))

# Plot the distribution of GLD daily log returns
plt.hist(
    returns,
    bins=40,
    density=True,
    alpha=0.6,
    label="GLD Returns"
)

# Overlay the fitted normal distribution
x = np.linspace(returns.min(), returns.max(), 500)
plt.plot(
    x,
    norm.pdf(x, mu, sigma),
    label="Fitted Normal Distribution"
)

plt.title("Distribution of GLD Daily Log Returns")
plt.xlabel("Daily Log Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(FIGURES_DIR / "returns_histogram.png", dpi=300)
plt.show()
plt.close()

"""
Graph Observations:
- Returns are more concentrated near zero than the fitted normal predicts.
- Extreme observations appear in both tails, especially the negative tail. (Around −0.10, −0.07, and +0.06)
- The normal distribution may underestimate the likelihood of extreme returns.
- A Student-t distribution may provide a better fit due to its heavier tails
"""


# 5. Normal Q-Q Plot-----------------------------------------
plt.figure(figsize=(8, 5))

# Compare observed GLD return quantiles with normal quantiles
stats.probplot(returns, dist="norm", plot=plt)

plt.title("Normal Q-Q Plot of GLD Daily Log Returns")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Observed Quantiles")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "normal_qq_plot.png", dpi=300)
plt.show()
plt.close()

# How to interpret it: If the returns follow a normal distribution,
# the points should fall approximately along a straight line. If they
# curve away from the line at the ends, that suggests the tails differ
# from those of a normal distribution.

"""
Q-Q Plot Observations:
- Most returns follow the reference line near the center.
- Both tails deviate from normality, especially the left tail.
- Extreme negative returns are more severe than a normal model predicts.
- These deviations suggest heavy tails and support evaluating
  the Student-t model, although its suitability requires validation.
"""

