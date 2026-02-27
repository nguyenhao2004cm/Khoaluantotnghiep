# build_signal.py - Cross-sectional allocation signal from latent risk (top-K selection only)

import os
import pandas as pd
import numpy as np

BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "data_processed", "risk_normalized")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed", "decision")
DATA_CUTOFF = os.environ.get("DATA_CUTOFF_DATE")  # Walk-forward
os.makedirs(OUTPUT_DIR, exist_ok=True)

frames = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith("_risk_norm.csv"):
        continue

    symbol = file.replace("_risk_norm.csv", "")
    df = pd.read_csv(
        os.path.join(INPUT_DIR, file),
        parse_dates=["date"]
    )

    if DATA_CUTOFF:
        df = df[df["date"] <= pd.to_datetime(DATA_CUTOFF)]
    df["symbol"] = symbol
    frames.append(df[["date", "symbol", "risk_z"]])

risk_df = pd.concat(frames, ignore_index=True)

def compute_signal(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signal for one date across all assets
    """

    z = group["risk_z"]
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

signal_df["signal"] = (
    signal_df
    .groupby("symbol")["signal_raw"]
    .transform(lambda x: x.ewm(span=5, adjust=False).mean())
)

signal_df["signal"] = signal_df["signal"].clip(lower=0)
out = signal_df[["date", "symbol", "signal"]]

out.to_csv(
    os.path.join(OUTPUT_DIR, "signal.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(" signal.csv created")
