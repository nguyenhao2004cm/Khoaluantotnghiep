import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ALPHA = 0.05
WINDOWS = [10, 20, 30, 60]

PORT_FILE = "data_processed/powerbi/portfolio_timeseries.csv"
BENCH_FILE = "data_processed/benchmark/equal_weight_portfolio.csv"
RISK_FILE = "data_processed/reporting/risk_regime_timeseries.csv"

# =========================
# CVaR FUNCTION
# =========================
def cvar(x, alpha=0.05):
    var = np.quantile(x, alpha)
    return x[x <= var].mean()

# =========================
# LOAD DATA
# =========================
port = pd.read_csv(PORT_FILE, parse_dates=["date"])
bench = pd.read_csv(BENCH_FILE, parse_dates=["date"])
risk = pd.read_csv(RISK_FILE, parse_dates=["date"])

df = (
    port[["date", "portfolio_return"]]
    .merge(bench[["date", "benchmark_return"]], on="date")
    .merge(risk[["date", "portfolio_risk_regime"]], on="date")
).dropna()

# =========================
# DETECT REGIME SWITCH → HIGH
# =========================
df["regime_shift"] = (
    (df["portfolio_risk_regime"].shift(1) != "HIGH") &
    (df["portfolio_risk_regime"] == "HIGH")
)

switch_dates = df.loc[df["regime_shift"], "date"]

print("\nCVaR Sensitivity to Window Length")
print("=" * 70)
print(f"Detected regime switches: {len(switch_dates)}\n")

results = []

# =========================
# SENSITIVITY LOOP
# =========================
for W in WINDOWS:
    deltas = []

    for d in switch_dates:
        window_df = df[
            (df["date"] > d) &
            (df["date"] <= d + pd.Timedelta(days=W))
        ]

        if len(window_df) < 10:
            continue

        cvar_p = cvar(window_df["portfolio_return"].values, ALPHA)
        cvar_b = cvar(window_df["benchmark_return"].values, ALPHA)
        deltas.append(cvar_p - cvar_b)

    if deltas:
        results.append({
            "Window (days)": W,
            "Mean ΔCVaR": np.mean(deltas),
            "Better Ratio (%)": 100 * (np.array(deltas) < 0).mean(),
            "Observations": len(deltas)
        })

res = pd.DataFrame(results)

print(res.to_string(index=False))

print("\nConclusion:")
print("  Stable negative ΔCVaR across windows ⇒ robust tail-risk reduction.")
