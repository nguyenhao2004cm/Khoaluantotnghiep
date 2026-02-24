# =====================================================
# FILE: src/models/encode_latent.py
# Encode PURE RISK FEATURES -> Latent Risk Factors
# Module A – Market Risk Engine
# =====================================================

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import torch
import pandas as pd
import numpy as np
import joblib

from src.models.autoencoder import MarketAutoencoder

# ================================
# CONFIG
# ================================
FEATURE_DIR = PROJECT_DIR / "data_processed" / "risk_features"
OUT_DIR = PROJECT_DIR / "data_processed" / "latent"
MODEL_PATH = PROJECT_DIR / "models" / "market_autoencoder.pt"
SCALER_PATH = PROJECT_DIR / "models" / "risk_scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# RISK FEATURE SET (MUST MATCH TRAINING)
# ================================
RISK_FEATURE_COLS = [
    "vol_20",
    "vol_60",
    "vol_change",
    "max_drawdown_60"
]

LATENT_DIM = 3


# ================================
# LOAD MODEL & SCALER
# ================================
def load_model(input_dim: int):
    model = MarketAutoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM
    )
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            " risk_scaler.pkl not found. "
            "You MUST save scaler during training."
        )
    return joblib.load(str(SCALER_PATH))


# ================================
# ENCODE ONE FILE
# ================================
def encode_file(path: str, model, scaler):
    df = pd.read_csv(path)

    if not set(RISK_FEATURE_COLS).issubset(df.columns):
        raise ValueError(
            f" Missing risk features in {path}"
        )

    df = df.dropna(subset=RISK_FEATURE_COLS)
    if len(df) == 0:
        return None

    dates = df["date"]

    X = df[RISK_FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if len(X) == 0:
        return None

    #  CRITICAL: APPLY SAME SCALER AS TRAINING
    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(
        X_scaled, dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        _, latent = model(X_tensor)
        latent = latent.cpu().numpy()

    latent_df = pd.DataFrame(
        latent,
        columns=[f"latent_{i}" for i in range(latent.shape[1])]
    )

    latent_df.insert(0, "date", dates.values)

    return latent_df


# ================================
# RUN PIPELINE
# ================================
def run():
    print("\n" + "=" * 70)
    print("🔹 ENCODING LATENT RISK FACTORS")
    print("   Role: Market Risk Representation")
    print("=" * 70 + "\n")

    files = [
        f.name for f in FEATURE_DIR.glob("*_risk_features.csv")
    ]

    if len(files) == 0:
        print(" No risk feature files found")
        return

    # Load model & scaler
    model = load_model(input_dim=len(RISK_FEATURE_COLS))
    scaler = load_scaler()

    for file in files:
        path = FEATURE_DIR / file

        latent_df = encode_file(str(path), model, scaler)
        if latent_df is None:
            print(f"  Skipping {file} (no valid data)")
            continue

        print(f"Encoding {file}")
        symbol = file.replace("_risk_features.csv", "")
        out_path = OUT_DIR / f"{symbol}_latent.csv"

        latent_df.to_csv(
            str(out_path),
            index=False,
            encoding="utf-8-sig"
        )

    print("\n Latent risk encoding completed")
    print("=" * 70 + "\n")


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    run()
