# =====================================================
# 4.1 DESCRIPTIVE STATISTICS OF RISK FEATURES
# =====================================================

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

FEATURE_DIR = PROJECT_DIR / "data_processed" / "features"
OUT_DIR = PROJECT_DIR / "data_processed" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "vol_20",
    "vol_60",
    "vol_change",
    "max_drawdown_60"
]


def load_all_features():
    dfs = []

    files = list(FEATURE_DIR.glob("*_features.csv"))
    print(f"Total risk feature files detected: {len(files)}")

    for file in files:
        df = pd.read_csv(file)

        # đảm bảo đủ cột
        if set(FEATURE_COLS).issubset(df.columns):
            dfs.append(df[FEATURE_COLS])

    if not dfs:
        raise ValueError("No valid risk feature files found")

    return pd.concat(dfs, ignore_index=True)

def compute_descriptive_stats(df):
    results = []

    for col in FEATURE_COLS:
        series = df[col].dropna()

        results.append({
            "Variable": col,
            "Observations": len(series),
            "Mean": series.mean(),
            "Std Dev": series.std(),
            "Min": series.min(),
            "Max": series.max(),
            "Skewness": skew(series),
            "Excess Kurtosis": kurtosis(series)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("\nRunning Descriptive Statistics for Risk Features...\n")
    df = load_all_features()
    stats_df = compute_descriptive_stats(df)

    out_file = OUT_DIR / "risk_features_descriptive_statistics.csv"
    stats_df.to_csv(out_file, index=False)

    print(stats_df.round(4))
    print(f"\nSaved to: {out_file}\n")
