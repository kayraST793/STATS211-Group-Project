import yfinance as yf

gold = yf.download(
    "GLD",
    start="2023-11-13",
    end="2026-11-14",
    interval="1d"
)

print(gold.head())