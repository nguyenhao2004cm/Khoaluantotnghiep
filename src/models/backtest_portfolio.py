# src/models/backtest_portfolio.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "data_processed"
PORTFOLIO_DIR = os.path.join(DATA_DIR, "portfolio")
PRICE_DIR = os.path.join(DATA_DIR, "prices")
OUTPUT_DIR = os.path.join(DATA_DIR, "backtest")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. LOAD DATA
# =========================

print(" Loading data...")

prices = pd.read_csv(
    os.path.join(PRICE_DIR, "prices.csv"),
    parse_dates=["date"]
)

risk_portfolio = pd.read_csv(
    os.path.join(PORTFOLIO_DIR, "risk_based_portfolio.csv"),
    parse_dates=["date"]
)

# =========================
# 2. COMPUTE RETURNS
# =========================

prices = prices.sort_values(["symbol", "date"])
prices["return"] = prices.groupby("symbol")["close"].pct_change()

# shift return backward → return(t+1)
prices["return_fwd"] = prices.groupby("symbol")["return"].shift(-1)

# =========================
# 3. MERGE WEIGHT + RETURN
# =========================

merged = risk_portfolio.merge(
    prices[["date", "symbol", "return_fwd"]],
    on=["date", "symbol"],
    how="inner"
)

merged["weighted_return"] = merged["weight"] * merged["return_fwd"]

portfolio_returns = (
    merged.groupby("date")["weighted_return"]
    .sum()
    .reset_index(name="risk_return")
)

# =========================
# 4. EQUAL WEIGHT PORTFOLIO
# =========================

print("⚖️ Computing equal-weight portfolio...")

eq = prices.dropna(subset=["return_fwd"]).copy()

eq["weight"] = 1 / eq.groupby("date")["symbol"].transform("count")
eq["weighted_return"] = eq["weight"] * eq["return_fwd"]

equal_returns = (
    eq.groupby("date")["weighted_return"]
    .sum()
    .reset_index(name="equal_return")
)

# =========================
# 5. COMBINE RETURNS
# =========================

returns = portfolio_returns.merge(equal_returns, on="date", how="inner")
returns = returns.sort_values("date")

returns["risk_cum"] = (1 + returns["risk_return"]).cumprod()
returns["equal_cum"] = (1 + returns["equal_return"]).cumprod()

returns.to_csv(
    os.path.join(OUTPUT_DIR, "portfolio_returns.csv"),
    index=False
)

# =========================
# 6. METRICS
# =========================

def sharpe_ratio(r):
    return np.mean(r) / np.std(r)

def max_drawdown(cum):
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

metrics = pd.DataFrame({
    "Portfolio": ["Risk-based", "Equal-weight"],
    "Return_mean": [
        returns["risk_return"].mean(),
        returns["equal_return"].mean()
    ],
    "Volatility": [
        returns["risk_return"].std(),
        returns["equal_return"].std()
    ],
    "Sharpe": [
        sharpe_ratio(returns["risk_return"]),
        sharpe_ratio(returns["equal_return"])
    ],
    "Max_Drawdown": [
        max_drawdown(returns["risk_cum"]),
        max_drawdown(returns["equal_cum"])
    ]
})

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "portfolio_metrics.csv"),
    index=False
)

print(" Metrics:")
print(metrics)

# =========================
# 7. PLOT
# =========================

plt.figure(figsize=(10, 5))
plt.plot(returns["date"], returns["risk_cum"], label="Risk-based")
plt.plot(returns["date"], returns["equal_cum"], label="Equal-weight")
plt.legend()
plt.title("Portfolio Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()

print(" Backtest completed successfully")
