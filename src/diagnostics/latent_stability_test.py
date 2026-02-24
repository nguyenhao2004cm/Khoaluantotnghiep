# =====================================================
# LATENT STABILITY TEST
# 1. Rolling mean stability (252-day)
# 2. Correlation persistence (Period 1 vs Period 2)
# =====================================================

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

LATENT_DIR = PROJECT_DIR / "data_processed" / "latent"
RISK_FEATURE_DIR = PROJECT_DIR / "data_processed" / "risk_features"
OUT_DIR = PROJECT_DIR / "data_processed" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINDOW = 252
LATENT_DIM = 3


def load_latent_panel():
    """Load latent + vol_20 for correlation. Cross-sectional mean per date."""
    latent_dfs = []
    for f in LATENT_DIR.glob("*_latent.csv"):
        df = pd.read_csv(f, parse_dates=["date"])
        latent_dfs.append(df)

    if not latent_dfs:
        return None, None

    latent = pd.concat(latent_dfs, ignore_index=True)
    latent_agg = latent.groupby("date")[[f"latent_{i}" for i in range(LATENT_DIM)]].mean().reset_index()

    # Vol from risk_features (cross-sectional mean)
    vol_dfs = []
    for f in RISK_FEATURE_DIR.glob("*_risk_features.csv"):
        df = pd.read_csv(f, parse_dates=["date"])
        if "vol_20" in df.columns:
            vol_dfs.append(df[["date", "vol_20"]])

    if not vol_dfs:
        return latent_agg.sort_values("date"), None

    vol = pd.concat(vol_dfs, ignore_index=True)
    vol_agg = vol.groupby("date")["vol_20"].mean().reset_index()

    return latent_agg.sort_values("date"), vol_agg.sort_values("date")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" LATENT STABILITY TEST")
    print("=" * 60)

    latent_ts, vol_ts = load_latent_panel()

    if latent_ts is None or len(latent_ts) < 100:
        print("\n⚠ Insufficient data. Run encode_latent first.")
    else:
        # 1. Rolling mean
        roll = latent_ts.set_index("date").rolling(ROLLING_WINDOW, min_periods=50).mean()

        fig, axes = plt.subplots(LATENT_DIM, 1, figsize=(10, 8), sharex=True)
        axes = np.atleast_1d(axes)
        for i in range(LATENT_DIM):
            axes[i].plot(roll.index, roll[f"latent_{i}"], label=f"Z{i+1} rolling mean")
            axes[i].set_ylabel(f"Z{i+1}")
            axes[i].legend(loc="upper right")
        axes[-1].set_xlabel("Date")
        plt.suptitle(f"Latent Factor Rolling Mean ({ROLLING_WINDOW}d)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "latent_rolling_stability.png", dpi=150)
        plt.close()
        print(f"\n1. Rolling mean plot saved: latent_rolling_stability.png")

        # 2. Correlation persistence
        merged = latent_ts.merge(vol_ts, on="date", how="inner") if vol_ts is not None else None

        if merged is not None and len(merged) >= 100:
            mid = len(merged) // 2
            p1 = merged.iloc[:mid]
            p2 = merged.iloc[mid:]

            corr_results = []
            for i in range(LATENT_DIM):
                c1 = p1[f"latent_{i}"].corr(p1["vol_20"])
                c2 = p2[f"latent_{i}"].corr(p2["vol_20"])
                corr_results.append({
                    "Latent": f"Z{i+1}",
                    "Corr_P1_vol20": c1,
                    "Corr_P2_vol20": c2,
                    "Delta": abs(c2 - c1)
                })

            corr_df = pd.DataFrame(corr_results)
            corr_file = OUT_DIR / "latent_correlation_persistence.csv"
            corr_df.to_csv(corr_file, index=False)
            print("\n2. Correlation persistence:")
            print(corr_df.round(4).to_string())
            print(f"\nSaved to: {corr_file}")

    print("\n" + "=" * 60 + "\n")
