# =====================================================
# ADF STATIONARITY TEST
# Augmented Dickey-Fuller: H0 = unit root (non-stationary)
# p < 0.05 → reject H0 → stationary
# =====================================================

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RISK_FEATURE_DIR = PROJECT_DIR / "data_processed" / "risk_features"
LATENT_DIR = PROJECT_DIR / "data_processed" / "latent"
OUT_DIR = PROJECT_DIR / "data_processed" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["vol_20", "vol_60"]
LATENT_DIM = 3


def load_cross_sectional_mean_ts(feature_dir, cols, pattern="*_risk_features.csv"):
    """Cross-sectional mean per date → univariate time series."""
    dfs = []
    for f in feature_dir.glob(pattern):
        df = pd.read_csv(f, parse_dates=["date"])
        if all(c in df.columns for c in cols):
            dfs.append(df[["date"] + cols])

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    agg = combined.groupby("date")[cols].mean().reset_index()
    return agg.sort_values("date")


def load_latent_mean_ts():
    """Cross-sectional mean of latent per date."""
    dfs = []
    for f in LATENT_DIR.glob("*_latent.csv"):
        df = pd.read_csv(f, parse_dates=["date"])
        dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    latent_cols = [f"latent_{i}" for i in range(LATENT_DIM)]
    agg = combined.groupby("date")[latent_cols].mean().reset_index()
    return agg.sort_values("date")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" ADF STATIONARITY TEST")
    print("=" * 60)

    results = []

    # Volatility
    vol_df = load_cross_sectional_mean_ts(RISK_FEATURE_DIR, FEATURE_COLS)
    if vol_df is not None:
        for col in FEATURE_COLS:
            series = vol_df[col].dropna()
            if len(series) > 20:
                adf_result = adfuller(series, autolag="AIC")
                results.append({
                    "Variable": col,
                    "ADF_stat": adf_result[0],
                    "pvalue": adf_result[1],
                    "Stationary": adf_result[1] < 0.05
                })

    # Latent
    latent_df = load_latent_mean_ts()
    if latent_df is not None:
        for i in range(LATENT_DIM):
            col = f"latent_{i}"
            series = latent_df[col].dropna()
            if len(series) > 20:
                adf_result = adfuller(series, autolag="AIC")
                results.append({
                    "Variable": f"Z{i+1}",
                    "ADF_stat": adf_result[0],
                    "pvalue": adf_result[1],
                    "Stationary": adf_result[1] < 0.05
                })

    if not results:
        print("\n⚠ No data. Run build_risk_features and encode_latent first.")
    else:
        res_df = pd.DataFrame(results)
        out_file = OUT_DIR / "adf_stationarity_results.csv"
        res_df.to_csv(out_file, index=False)

        print("\n" + res_df.round(4).to_string())
        print(f"\nSaved to: {out_file}")
        print("\n(p < 0.05 → stationary)")

    print("\n" + "=" * 60 + "\n")
