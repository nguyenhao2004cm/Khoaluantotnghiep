# =====================================================
# FILE: src/decision/build_signal.py
# PURPOSE:
#   Build cross-sectional allocation signal
#   based on latent risk (NOT return prediction)
#
# PHILOSOPHY:
#   - Signal = relative safety under market regime
#   - Cross-sectional (same day, all assets)
#   - Fast reaction to regime switch
#
# REFERENCES:
#   Ma et al. (2020), Bertani (2021), Aithal et al. (2023)
# =====================================================

import os
import pandas as pd
import numpy as np

BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "data_processed", "risk_normalized")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed", "decision")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD ALL RISK FILES
# =====================================================
frames = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith("_risk_norm.csv"):
        continue

    symbol = file.replace("_risk_norm.csv", "")
    df = pd.read_csv(
        os.path.join(INPUT_DIR, file),
        parse_dates=["date"]
    )

    df["symbol"] = symbol
    frames.append(df[["date", "symbol", "risk_z"]])

risk_df = pd.concat(frames, ignore_index=True)

# =====================================================
# CORE IDEA:
# Signal is CROSS-SECTIONAL, not per-asset normalized
# =====================================================

def compute_signal(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signal for one date across all assets
    """

    # 1 Cross-sectional normalization (KEY FIX)
    z = group["risk_z"]

    # Lower risk_z → better signal
    # Rank-based is more robust than min-max
    group["signal_raw"] = (-z).rank(method="average")

    # Normalize to [0, 1]
    group["signal_raw"] = (
        group["signal_raw"] - group["signal_raw"].min()
    ) / (
        group["signal_raw"].max() - group["signal_raw"].min() + 1e-8
    )

    return group


# Apply cross-sectional signal
signal_df = (
    risk_df
    .groupby("date", group_keys=False)
    .apply(compute_signal)
    .sort_values(["date", "signal_raw"], ascending=[True, False])
)

# =====================================================
# OPTIONAL: Mild temporal smoothing (NOT momentum)
# =====================================================
# Purpose: reduce day-to-day noise, NOT delay regime reaction

signal_df["signal"] = (
    signal_df
    .groupby("symbol")["signal_raw"]
    .transform(lambda x: x.ewm(span=5, adjust=False).mean())
)

# Ensure non-negative
signal_df["signal"] = signal_df["signal"].clip(lower=0)

# =====================================================
# FINAL EXPORT
# =====================================================
out = signal_df[["date", "symbol", "signal"]]

out.to_csv(
    os.path.join(OUTPUT_DIR, "signal.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(" signal.csv created (cross-sectional, regime-aware)")
print("   Philosophy: relative safety under latent market risk")
