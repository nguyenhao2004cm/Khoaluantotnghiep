# =====================================================
# FILE: src/reporting/build_portfolio_risk_regime.py
# Module A – Market Risk Aggregation Engine
# =====================================================

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[2]

RISK_DIR = PROJECT_DIR / "data_processed" / "risk_normalized"
OUT_DIR  = PROJECT_DIR / "data_processed" / "reporting"
OUT_DIR.mkdir(exist_ok=True)

OUT_FILE = OUT_DIR / "risk_regime_timeseries.csv"

# =========================
# LOAD ALL ASSET RISK SIGNALS
# =========================
frames = []

for f in RISK_DIR.glob("*_risk_norm.csv"):
    symbol = f.stem.replace("_risk_norm", "")
    df = pd.read_csv(f, parse_dates=["date"])

    #  CRITICAL: dùng risk_z liên tục, không chỉ risk_regime
    frames.append(
        df[["date", "risk_z"]].assign(symbol=symbol)
    )

risk = pd.concat(frames, ignore_index=True)

# =========================
# CROSS-SECTIONAL AGGREGATION
# =========================
summary = (
    risk
    .groupby("date")
    .agg(
        mean_risk_z=("risk_z", "mean"),
        high_intensity=("risk_z", lambda x: (x > 1).mean()),
        low_intensity =("risk_z", lambda x: (x < -1).mean()),
        dispersion=("risk_z", "std"),
        total_assets=("risk_z", "count")
    )
    .reset_index()
)

# =========================
# TEMPORAL SMOOTHING (KEY FIX)
# =========================
# Smooth market stress to avoid regime flickering
summary["mean_risk_z_smooth"] = (
    summary["mean_risk_z"]
    .rolling(window=5, min_periods=3)
    .mean()
)

summary["high_intensity_smooth"] = (
    summary["high_intensity"]
    .rolling(window=5, min_periods=3)
    .mean()
)

# =========================
# REGIME CLASSIFICATION (INERTIAL)
# =========================
def classify(row):
    """
    Regime logic:
    - HIGH: sustained positive stress (mean_risk_z > 0.25, >15% assets in stress)
    - LOW: sustained calm
    - NORMAL: transitional
    """
    mz = row["mean_risk_z_smooth"]
    hi = row["high_intensity_smooth"]
    li = row["low_intensity"]
    if pd.notna(mz) and pd.notna(hi) and mz > 0.25 and hi > 0.15:
        return "HIGH"
    elif pd.notna(mz) and pd.notna(li) and mz < -0.25 and li > 0.15:
        return "LOW"
    else:
        return "NORMAL"

summary["portfolio_risk_regime_raw"] = summary.apply(classify, axis=1)

# =========================
# REGIME INERTIA (HYSTERESIS)
# Require 2 consecutive days in new regime before switching
# =========================
final_regime = []
prev = "NORMAL"
pending = None  # (regime, consecutive_count)

for r in summary["portfolio_risk_regime_raw"]:
    if r == prev:
        pending = None
        final_regime.append(r)
        prev = r
    elif pending is not None and pending[0] == r:
        count = pending[1] + 1
        if count >= 2:
            final_regime.append(r)
            prev = r
            pending = None
        else:
            final_regime.append(prev)
            pending = (r, count)
    else:
        # New regime, first day
        final_regime.append(prev)
        pending = (r, 1)

summary["portfolio_risk_regime"] = final_regime

# =========================
# SAVE
# =========================
summary.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

print("\n Portfolio risk regime built (latent-stress based)")
print(summary["portfolio_risk_regime"].value_counts())
print("Saved to:", OUT_FILE)
