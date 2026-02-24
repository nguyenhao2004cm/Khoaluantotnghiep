import numpy as np
import pandas as pd

ALPHA = 0.05

# =========================
# LOAD DATA
# =========================
port_df = pd.read_csv("data_processed/powerbi/portfolio_timeseries.csv")
bench_df = pd.read_csv("data_processed/benchmark/equal_weight_portfolio.csv")

# Merge theo date
df = port_df.merge(
    bench_df,
    on="date",
    how="inner"
)

# =========================
# DEFINE RISK REGIME
# =========================
# Dùng rolling volatility làm proxy risk regime
df["rolling_vol"] = df["portfolio_return"].rolling(60).std()

threshold = df["rolling_vol"].quantile(0.75)
df["high_risk"] = df["rolling_vol"] >= threshold

# =========================
# CVaR FUNCTION
# =========================
def cvar(x, alpha=0.05):
    var = np.quantile(x, alpha)
    return x[x <= var].mean()

# =========================
# COMPUTE CVaR BY REGIME
# =========================
high = df[df["high_risk"]]

cvar_port = cvar(high["portfolio_return"].values, ALPHA)
cvar_bench = cvar(high["benchmark_return"].values, ALPHA)

reduction = (abs(cvar_bench) - abs(cvar_port)) / abs(cvar_bench)

# =========================
# PRINT
# =========================
print("\nCVaR Comparison in HIGH-RISK REGIME (5%)")
print("=" * 55)
print(f"Portfolio CVaR : {cvar_port:.4f}")
print(f"Benchmark CVaR : {cvar_bench:.4f}")
print("-" * 55)
print(f"CVaR Reduction : {reduction*100:.2f}%")

if reduction > 0:
    print("Conclusion: AI portfolio REDUCES tail risk in high-risk regimes")
else:
    print("Conclusion: AI portfolio DOES NOT reduce tail risk in high-risk regimes")
