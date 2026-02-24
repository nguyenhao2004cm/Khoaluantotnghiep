# =====================================================
# BUILD FEATURES – KEEP ALL DATA VERSION
# =====================================================

import os
import pandas as pd
import numpy as np

RAW_DIR = "data_raw/stocks"
OUT_DIR = "data_processed/features"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-8

def build_features(symbol):

    df = pd.read_csv(f"{RAW_DIR}/{symbol}.csv")

    df.columns = [c.lower().strip() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.sort_values("date")
    df = df[df["close"] > 0]

    # =========================
    # LOG RETURN
    # =========================
    df["return"] = np.log(df["close"]).diff()

    # =========================
    # RISK FEATURES
    # =========================
    df["vol_20"] = df["return"].rolling(20).std()
    df["vol_60"] = df["return"].rolling(60).std()

    rolling_max = df["close"].rolling(60).max()
    df["max_drawdown_60"] = df["close"] / rolling_max - 1

    df["vol_change"] = np.log(
        (df["vol_20"] + EPS) /
        (df["vol_20"].shift(1) + EPS)
    )

    feature_cols = [
        "vol_20",
        "vol_60",
        "vol_change",
        "max_drawdown_60"
    ]

    # =========================
    # SAFE CLEAN (NO DROP)
    # =========================
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # KHÔNG drop
    df[feature_cols] = df[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].bfill()

    # Include log_return for LSTM training target
    df_out = df[["date", "return"] + feature_cols].rename(columns={"return": "log_return"})

    df_out.to_csv(
        f"{OUT_DIR}/{symbol}_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f" Features built for {symbol} ({len(df_out)} rows)")


def run():
    print("\nBuilding features (keeping all data)...\n")
    for file in sorted(os.listdir(RAW_DIR)):
        if file.endswith(".csv"):
            symbol = file.replace(".csv", "")
            build_features(symbol)


if __name__ == "__main__":
    run()