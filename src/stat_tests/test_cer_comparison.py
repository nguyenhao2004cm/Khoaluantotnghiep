import numpy as np
import pandas as pd

# =========================
# LOAD DATA
# =========================
PORT_FILE = "data_processed/powerbi/portfolio_timeseries.csv"
BENCH_FILE = "data_processed/benchmark/equal_weight_portfolio.csv"

port = pd.read_csv(PORT_FILE)["portfolio_return"].dropna().values
bench = pd.read_csv(BENCH_FILE)["benchmark_return"].dropna().values

T = min(len(port), len(bench))
port = port[:T]
bench = bench[:T]

# =========================
# PARAMETERS
# =========================
gamma = 5  # risk aversion (Q1 standard)

# =========================
# CER CALCULATION
# =========================
def CER(r, gamma):
    mu = r.mean()
    var = r.var(ddof=1)
    return mu - 0.5 * gamma * var

cer_port = CER(port, gamma)
cer_bench = CER(bench, gamma)

# =========================
# PRINT RESULTS
# =========================
print("\nCertainty Equivalent Return (CER)")
print("=" * 50)
print(f"Risk aversion γ = {gamma}")
print("-" * 50)
print(f"Portfolio CER : {cer_port:.6f}")
print(f"Benchmark CER : {cer_bench:.6f}")
print("-" * 50)
print(f"Δ CER (P - B): {cer_port - cer_bench:.6f}")

if cer_port > cer_bench:
    print("Conclusion: AI portfolio preferred by risk-averse investors.")
else:
    print("Conclusion: Benchmark preferred by risk-averse investors.")
