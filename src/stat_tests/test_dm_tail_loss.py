import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05

PORT_FILE = "data_processed/powerbi/portfolio_timeseries.csv"
BENCH_FILE = "data_processed/benchmark/equal_weight_portfolio.csv"

def tail_loss(r, var):
    return np.maximum(0, var - r)

# =========================
# LOAD DATA
# =========================
port = pd.read_csv(PORT_FILE, parse_dates=["date"])
bench = pd.read_csv(BENCH_FILE, parse_dates=["date"])

df = port.merge(bench, on="date").dropna()

# =========================
# COMPUTE VAR
# =========================
var_p = np.quantile(df["portfolio_return"], ALPHA)
var_b = np.quantile(df["benchmark_return"], ALPHA)

loss_p = tail_loss(df["portfolio_return"], var_p)
loss_b = tail_loss(df["benchmark_return"], var_b)

d = loss_p - loss_b
d_mean = d.mean()

# =========================
# DM TEST
# =========================
T = len(d)
gamma0 = np.var(d, ddof=1)
dm_stat = d_mean / np.sqrt(gamma0 / T)
p_value = stats.norm.cdf(dm_stat)

print("\nDiebold–Mariano Test (Tail Loss)")
print("=" * 60)
print(f"DM Statistic : {dm_stat:.4f}")
print(f"p-value (1-sided): {p_value:.4f}")

if p_value < 0.05:
    print("Conclusion: Portfolio has significantly lower tail loss.")
else:
    print("Conclusion: No significant tail loss improvement.")
