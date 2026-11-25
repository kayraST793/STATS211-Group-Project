"""
Download three years of daily GLD price data
from Yahoo Finance using the yfinance Python package.

Output: data/GLD.csv
"""

import yfinance as yf

gold = yf.download(
    "GLD",
    start="2023-11-13",
    end="2026-11-14",
    interval="1d",
    auto_adjust=True # Use adjusted prices to account for stock splits and dividends, ensuring daily returns reflect price changes consistently
)

gold.to_csv("GLD.csv")