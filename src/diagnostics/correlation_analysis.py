# =====================================================
# 4.2 CORRELATION STRUCTURE & MULTICOLLINEARITY
# =====================================================

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

def compute_vif(df):
    vif_data = []
    X = df.dropna().values

    for i in range(X.shape[1]):
        vif_data.append({
            "Variable": FEATURE_COLS[i],
            "VIF": variance_inflation_factor(X, i)
        })

    return pd.DataFrame(vif_data)


if __name__ == "__main__":
    print("\nRunning Correlation & VIF Analysis...\n")

    df = load_all_features().dropna()

    # Correlation matrix
    corr = df.corr()
    corr_file = OUT_DIR / "risk_features_correlation_matrix.csv"
    corr.to_csv(corr_file)

    print("Correlation matrix saved.")

    # VIF
    vif_df = compute_vif(df)
    vif_file = OUT_DIR / "risk_features_vif.csv"
    vif_df.to_csv(vif_file, index=False)

    print(vif_df.round(3))
    print(f"\nSaved to: {vif_file}\n")

    # Heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(FEATURE_COLS)), FEATURE_COLS, rotation=45)
    plt.yticks(range(len(FEATURE_COLS)), FEATURE_COLS)
    plt.colorbar()
    plt.title("Risk Features Correlation Matrix")
    plt.tight_layout()

    heatmap_path = OUT_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    print(f"Heatmap saved to: {heatmap_path}")
