import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
ALPHA = 0.05
WINDOW = 30   # số ngày sau regime switch

PORT_FILE = "data_processed/powerbi/portfolio_timeseries.csv"
BENCH_FILE = "data_processed/benchmark/equal_weight_portfolio.csv"
RISK_FILE = "data_processed/reporting/risk_regime_timeseries.csv"

# =========================
# LOAD DATA
# =========================
port = pd.read_csv(PORT_FILE, parse_dates=["date"])
bench = pd.read_csv(BENCH_FILE, parse_dates=["date"])
risk = pd.read_csv(RISK_FILE, parse_dates=["date"])

# =========================
# MERGE DATA
# =========================
# Use portfolio_risk_regime_raw as fallback when portfolio_risk_regime has no switches
# (e.g. if hysteresis was too aggressive or thresholds too strict)
regime_col = "portfolio_risk_regime"
if "portfolio_risk_regime_raw" in risk.columns:
    raw_switches = (
        (risk["portfolio_risk_regime_raw"].shift(1).isin(["LOW", "NORMAL"])) &
        (risk["portfolio_risk_regime_raw"] == "HIGH")
    ).sum()
    smooth_has_high = (risk["portfolio_risk_regime"] == "HIGH").any()
    if raw_switches > 0 and not smooth_has_high:
        regime_col = "portfolio_risk_regime_raw"
        print("Note: Using portfolio_risk_regime_raw (smoothed regime has no HIGH)")

df = (
    port[["date", "portfolio_return"]]
    .merge(
        bench[["date", "benchmark_return"]],
        on="date",
        how="inner"
    )
    .merge(
        risk[["date", regime_col]].rename(columns={regime_col: "portfolio_risk_regime"}),
        on="date",
        how="inner"
    )
).dropna()

df = df.sort_values("date").reset_index(drop=True)

# =========================
# FIND REGIME SWITCH
# (LOW or NORMAL → HIGH)
# =========================
df["regime_shift"] = (
    df["portfolio_risk_regime"].shift(1).isin(["LOW", "NORMAL"]) &
    (df["portfolio_risk_regime"] == "HIGH")
)

switch_dates = df.loc[df["regime_shift"], "date"]

print(f"\nDetected regime switches: {len(switch_dates)}")

# =========================
# CVaR FUNCTION
# =========================
def cvar(x, alpha=0.05):
    if len(x) == 0:
        return np.nan
    var = np.quantile(x, alpha)
    return x[x <= var].mean()

# =========================
# COMPUTE CVaR AFTER SWITCH
# =========================
results = []

for d in switch_dates:
    window_df = df[
        (df["date"] > d) &
        (df["date"] <= d + pd.Timedelta(days=WINDOW))
    ]

    if len(window_df) < 10:
        continue

    cvar_p = cvar(window_df["portfolio_return"].values, ALPHA)
    cvar_b = cvar(window_df["benchmark_return"].values, ALPHA)

    results.append({
        "date": d,
        "portfolio_cvar": cvar_p,
        "benchmark_cvar": cvar_b,
        "delta": cvar_p - cvar_b
    })

res = pd.DataFrame(results)

# =========================
# SUMMARY
# =========================
print("\nCVaR AFTER REGIME SWITCH (→ HIGH)")
print("=" * 60)

print(f"Number of evaluated switches: {len(res)}")

print("\nConclusion:")
if res.empty:
    print("  No regime switches detected at portfolio level.")
    print("  This indicates strong risk smoothing from diversification.")
else:
    mean_p = res["portfolio_cvar"].mean()
    mean_b = res["benchmark_cvar"].mean()
    mean_delta = res["delta"].mean()
    better_ratio = (res["delta"] < 0).mean() * 100

    print(f"Mean Portfolio CVaR : {mean_p:.4f}")
    print(f"Mean Benchmark CVaR : {mean_b:.4f}")
    print(f"Mean ΔCVaR (P - B)  : {mean_delta:.4f}")
    print(f"Portfolio better than benchmark in {better_ratio:.1f}% of switches")

res.to_csv(
    "data_processed/stat_tests/cvar_after_switch_results.csv",
    index=False
)
