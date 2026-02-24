import numpy as np
import pandas as pd
from scipy import stats

# =========================
# LOAD DATA
# =========================
PORT_FILE = "data_processed/powerbi/portfolio_timeseries.csv"
BENCH_FILE = "data_processed/benchmark/equal_weight_portfolio.csv"

port_df = pd.read_csv(PORT_FILE)
bench_df = pd.read_csv(BENCH_FILE)

# Align length
returns_p = port_df["portfolio_return"].dropna().values
returns_b = bench_df["benchmark_return"].dropna().values

T = min(len(returns_p), len(returns_b))
returns_p = returns_p[:T]
returns_b = returns_b[:T]

# =========================
# SHARPE RATIOS
# =========================
rf = 0.03 / 252  # daily risk-free rate (3% annually)

mean_p = returns_p.mean() - rf
mean_b = returns_b.mean() - rf

std_p = returns_p.std(ddof=1)
std_b = returns_b.std(ddof=1)

sharpe_p = mean_p / std_p
sharpe_b = mean_b / std_b

# =========================
# JOBSON–KORKIE TEST (Memmel)
# =========================
cov_pb = np.cov(returns_p, returns_b)[0, 1]

var_diff = (
    (1 / T)
    * (
        (std_p**2 + std_b**2 - 2 * cov_pb)
        / (std_p * std_b)
    )
)

jk_stat = (sharpe_p - sharpe_b) / np.sqrt(var_diff)

p_value = 1 - stats.norm.cdf(jk_stat)  # one-sided

# =========================
# PRINT RESULTS
# =========================
print("\nSharpe Ratio Comparison Test (Jobson–Korkie)")
print("=" * 55)
print(f"Portfolio Sharpe : {sharpe_p:.4f}")
print(f"Benchmark Sharpe : {sharpe_b:.4f}")
print("-" * 55)
print(f"JK Statistic     : {jk_stat:.4f}")
print(f"p-value (1-sided): {p_value:.6f}")

print("\nConclusion:")
if p_value < 0.05:
    print("  Portfolio Sharpe is statistically HIGHER than benchmark.")
else:
    print("  No statistical evidence that portfolio Sharpe exceeds benchmark.")
