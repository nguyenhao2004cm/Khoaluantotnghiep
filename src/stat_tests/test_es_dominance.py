import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ALPHAS = np.arange(0.01, 0.11, 0.01)   # 1% → 10%
WINDOW = 30

PORT_FILE  = "data_processed/powerbi/portfolio_timeseries.csv"
BENCH_FILE = "data_processed/benchmark/equal_weight_portfolio.csv"
RISK_FILE  = "data_processed/reporting/risk_regime_timeseries.csv"

# =========================
# LOAD DATA
# =========================
port  = pd.read_csv(PORT_FILE,  parse_dates=["date"])
bench = pd.read_csv(BENCH_FILE, parse_dates=["date"])
risk  = pd.read_csv(
    RISK_FILE,
    parse_dates=["date"],
    dayfirst=True
)

df = (
    port[["date", "portfolio_return"]]
    .merge(bench[["date", "benchmark_return"]], on="date")
    .merge(risk[["date", "portfolio_risk_regime"]], on="date")
).dropna()

# =========================
# DETECT REGIME SWITCH (→ HIGH)
# =========================
df["regime_shift"] = (
    (df["portfolio_risk_regime"].shift(1) != "HIGH") &
    (df["portfolio_risk_regime"] == "HIGH")
)

switch_dates = df.loc[df["regime_shift"], "date"]
print(f"\nDetected regime switches: {len(switch_dates)}")

# =========================
# EXPECTED SHORTFALL FUNCTION
# =========================
def expected_shortfall(x, alpha):
    var = np.quantile(x, alpha)
    return x[x <= var].mean()

# =========================
# COMPUTE ES DOMINANCE
# =========================
records = []

for alpha in ALPHAS:
    es_p_list, es_b_list = [], []

    for d in switch_dates:
        window_df = df[
            (df["date"] > d) &
            (df["date"] <= d + pd.Timedelta(days=WINDOW))
        ]

        if len(window_df) < 10:
            continue

        es_p_list.append(
            expected_shortfall(window_df["portfolio_return"].values, alpha)
        )
        es_b_list.append(
            expected_shortfall(window_df["benchmark_return"].values, alpha)
        )

    if len(es_p_list) == 0:
        continue

    records.append({
        "alpha": alpha,
        "portfolio_es": np.mean(es_p_list),
        "benchmark_es": np.mean(es_b_list),
        "delta_es": np.mean(es_p_list) - np.mean(es_b_list),
        "better_ratio": (np.array(es_p_list) < np.array(es_b_list)).mean() * 100,
        "observations": len(es_p_list)
    })

res = pd.DataFrame(records)

# =========================
# RESULTS
# =========================
print("\nExpected Shortfall Dominance Test")
print("=" * 70)

if res.empty:
    print("No valid regime switches for ES dominance test.")
else:
    print(res.to_string(index=False, float_format="%.4f"))

# =========================
# CONCLUSION
# =========================
print("\nConclusion:")
if res.empty:
    print("  ES dominance test not applicable (no regime switches).")
else:
    if (res["delta_es"] < 0).all():
        print("  Portfolio exhibits FIRST-ORDER Expected Shortfall dominance.")
    else:
        print("  No uniform Expected Shortfall dominance detected.")

    stable = (res["better_ratio"] > 50).mean() * 100
    print(f"  Dominance holds in {stable:.1f}% of tail levels.")
