# =====================================================
# BUILD PURE RISK FEATURES – RESEARCH GRADE
# =====================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_PRICE_FILE = PROJECT_DIR / "data_processed/prices/prices.csv"
OUT_DIR = PROJECT_DIR / "data_processed/risk_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8
WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99


def winsorize_series(s):
    lower = s.quantile(WINSOR_LOW)
    upper = s.quantile(WINSOR_HIGH)
    return s.clip(lower, upper)


def build_risk_features(symbol):

    df = pd.read_csv(str(RAW_PRICE_FILE))

    df["symbol"] = df["symbol"].str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["symbol"] == symbol]
    df = df.sort_values("date")

    # Remove invalid prices
    df = df[df["close"] > 0]

    if len(df) < 120:
        return

    # =========================
    # LOG RETURNS (stable)
    # =========================
    df["return"] = np.log(df["close"]).diff()

    # =========================
    # RISK FEATURES
    # =========================
    df["vol_20"] = df["return"].rolling(20).std()
    df["vol_60"] = df["return"].rolling(60).std()

    rolling_max = df["close"].rolling(60).max()
    df["max_drawdown_60"] = df["close"] / rolling_max - 1

    # Volatility acceleration (log ratio – stable)
    df["vol_change"] = np.log(
        (df["vol_20"] + EPS) /
        (df["vol_20"].shift(1) + EPS)
    )

    # Drop NA from rolling
    df = df.dropna()

    feature_cols = [
        "vol_20",
        "vol_60",
        "vol_change",
        "max_drawdown_60"
    ]

    # Winsorization
    for col in feature_cols:
        df[col] = winsorize_series(df[col])

    # Log transform volatility (reduces skew massively)
    df["vol_20"] = np.log1p(df["vol_20"])
    df["vol_60"] = np.log1p(df["vol_60"])

    out = df[["date"] + feature_cols]

    out_path = OUT_DIR / f"{symbol}_risk_features.csv"
    out.to_csv(str(out_path), index=False, encoding="utf-8-sig")

    print(f" Clean risk features built for {symbol}")


def run():
    price_df = pd.read_csv(str(RAW_PRICE_FILE))
    symbols = price_df["symbol"].unique()

    for symbol in symbols:
        build_risk_features(symbol)


if __name__ == "__main__":
    run()