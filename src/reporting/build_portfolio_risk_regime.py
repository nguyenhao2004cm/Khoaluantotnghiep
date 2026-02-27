# build_portfolio_risk_regime.py - Market risk aggregation, regime classification

import pandas as pd
import numpy as np
from pathlib import Path

import os

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_CUTOFF = os.environ.get("DATA_CUTOFF_DATE")

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
    if DATA_CUTOFF:
        df = df[df["date"] <= pd.to_datetime(DATA_CUTOFF)]
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
# REGIME CLASSIFICATION — QUANTILE-BASED (bắt buộc 3 pha)
# Thay threshold cố định bằng quantile → phù hợp emerging market
# =========================
def classify_quantile(series):
    """
    Quantile-based: 33% LOW, 34% NORMAL, 33% HIGH.
    Dùng expanding quantiles để tránh look-ahead.
    """
    out = []
    for i in range(len(series)):
        if i < 20:  # Warm-up
            out.append("NORMAL")
            continue
        hist = series.iloc[: i + 1].dropna()
        if len(hist) < 20:
            out.append("NORMAL")
            continue
        low_th = hist.quantile(0.33)
        high_th = hist.quantile(0.66)
        val = series.iloc[i]
        if pd.isna(val):
            out.append("NORMAL")
        elif val <= low_th:
            out.append("LOW")
        elif val >= high_th:
            out.append("HIGH")
        else:
            out.append("NORMAL")
    return out

summary["portfolio_risk_regime_raw"] = classify_quantile(summary["mean_risk_z_smooth"])

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
