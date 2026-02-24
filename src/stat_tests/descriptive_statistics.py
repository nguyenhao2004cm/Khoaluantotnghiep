import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# =========================
# CONFIG
# =========================
PORT_FILE  = "data_processed/powerbi/portfolio_timeseries.csv"
BENCH_FILE = "data_processed/benchmark/equal_weight_portfolio.csv"

OUT_FILE = "data_processed/reporting/descriptive_statistics.csv"

# =========================
# LOAD DATA
# =========================
port  = pd.read_csv(PORT_FILE, parse_dates=["date"])
bench = pd.read_csv(BENCH_FILE, parse_dates=["date"])

returns = pd.DataFrame({
    "portfolio": port["portfolio_return"].dropna(),
    "benchmark": bench["benchmark_return"].dropna()
}).dropna()

# =========================
# HELPER FUNCTIONS
# =========================
def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def cvar(x, alpha=0.05):
    var = np.quantile(x, alpha)
    return x[x <= var].mean()

# =========================
# DESCRIPTIVE STATISTICS
# =========================
stats = []

for col in returns.columns:
    r = returns[col]

    stats.append({
        "Asset": col,
        "Observations": len(r),
        "Mean": r.mean(),
        "Std Dev": r.std(ddof=1),
        "Min": r.min(),
        "Max": r.max(),
        "Skewness": skew(r),
        "Excess Kurtosis": kurtosis(r),
        "VaR 5%": np.quantile(r, 0.05),
        "CVaR 5%": cvar(r, 0.05),
        "Max Drawdown": max_drawdown(r)
    })

df_stats = pd.DataFrame(stats)

# =========================
# FORMAT FOR REPORTING
# =========================
numeric_cols = df_stats.columns.drop("Asset")
df_stats[numeric_cols] = df_stats[numeric_cols].astype(float)

df_stats.to_csv(OUT_FILE, index=False)

# =========================
# PRINT SUMMARY
# =========================
print("\nDESCRIPTIVE STATISTICS SUMMARY")
print("=" * 80)
print(df_stats.to_string(index=False, float_format="%.4f"))
print("\nSaved to:", OUT_FILE)
