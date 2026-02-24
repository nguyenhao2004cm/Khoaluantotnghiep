import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
TICKER = "AAA"
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

# =========================
# LOAD PRICE DATA
# prices.csv có schema: date, close, symbol
# =========================
price_all = pd.read_csv("data_processed/prices/prices.csv", parse_dates=["date"])
price = price_all[price_all["symbol"] == TICKER].copy()

# =========================
# LOAD RISK / WEIGHT DATA
# =========================
risk = pd.read_csv(
    f"data_processed/decision/{TICKER}_weight.csv",
    parse_dates=["date"]
)

# =========================
# MERGE
# =========================
df = price.merge(risk, on="date", how="inner")

# =========================
# CRISIS WINDOW
# =========================
df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
df = df.sort_values("date").reset_index(drop=True)

# =========================
# RETURNS
# =========================
df["ret"] = df["close"].pct_change()
df["bh_ret"] = df["ret"]
df["risk_ret"] = df["ret"] * df["weight"]

# =========================
# CUMULATIVE RETURNS
# =========================
df["bh_cum"] = (1 + df["bh_ret"]).cumprod()
df["risk_cum"] = (1 + df["risk_ret"]).cumprod()

# =========================
# MAX DRAWDOWN
# =========================
def max_dd(x):
    return (x / x.cummax() - 1).min()

print("Buy & Hold Max Drawdown:", round(max_dd(df["bh_cum"]), 4))
print("Risk-based Max Drawdown:", round(max_dd(df["risk_cum"]), 4))

# =========================
# OPTIONAL: SUMMARY
# =========================
print("\nFinal cumulative return:")
print("Buy & Hold:", round(df["bh_cum"].iloc[-1] - 1, 4))
print("Risk-based:", round(df["risk_cum"].iloc[-1] - 1, 4))
