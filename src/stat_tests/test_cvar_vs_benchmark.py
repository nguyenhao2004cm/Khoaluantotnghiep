import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ALPHA = 0.05

PORT_FILE = "data_processed/powerbi/portfolio_timeseries.csv"
BENCH_FILE = "data_processed/benchmark/equal_weight_portfolio.csv"

# =========================
# LOAD DATA
# =========================
port_df = pd.read_csv(PORT_FILE)
bench_df = pd.read_csv(BENCH_FILE)

port_ret = port_df["portfolio_return"].dropna().values
bench_ret = bench_df["benchmark_return"].dropna().values

# =========================
# HISTORICAL CVaR FUNCTION
# =========================
def historical_cvar(returns, alpha=0.05):
    var = np.quantile(returns, alpha)
    return returns[returns <= var].mean()

# =========================
# COMPUTE CVaR
# =========================
cvar_port = historical_cvar(port_ret, ALPHA)
cvar_bench = historical_cvar(bench_ret, ALPHA)

reduction_ratio = (abs(cvar_bench) - abs(cvar_port)) / abs(cvar_bench)

# =========================
# PRINT RESULTS
# =========================
print("\nCVaR Comparison vs Benchmark (5%)")
print("=" * 55)
print(f"Portfolio CVaR : {cvar_port:.4f}")
print(f"Benchmark CVaR : {cvar_bench:.4f}")
print("-" * 55)
print(f"CVaR Reduction : {reduction_ratio*100:.2f}%")

if reduction_ratio > 0:
    print("Conclusion: Portfolio REDUCES tail risk vs benchmark")
else:
    print("Conclusion: Portfolio DOES NOT reduce tail risk vs benchmark")
