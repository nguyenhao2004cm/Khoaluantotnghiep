# =====================================================
# 4.3 LATENT RISK FACTOR DISTRIBUTION ANALYSIS
# =====================================================

import os
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import joblib

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.models.autoencoder import MarketAutoencoder

FEATURE_DIR = PROJECT_DIR / "data_processed" / "risk_features"
MODEL_PATH = PROJECT_DIR / "models" / "market_autoencoder.pt"
SCALER_PATH = PROJECT_DIR / "models" / "risk_scaler.pkl"
OUT_DIR = PROJECT_DIR / "data_processed" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "vol_20",
    "vol_60",
    "vol_change",
    "max_drawdown_60"
]

LATENT_DIM = 3


def load_all_features():
    dfs = []

    files = list(FEATURE_DIR.glob("*_risk_features.csv"))
    print(f"Total risk feature files detected: {len(files)}")

    for file in files:
        df = pd.read_csv(file)

        # đảm bảo đủ cột
        if set(FEATURE_COLS).issubset(df.columns):
            dfs.append(df[FEATURE_COLS])

    if not dfs:
        raise ValueError("No valid risk feature files found")

    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    print("\nRunning Latent Factor Analysis...\n")

    df = load_all_features().dropna()

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(df.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = MarketAutoencoder(input_dim=len(FEATURE_COLS), latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        _, z = model(X_tensor)

    z = z.numpy()

    latent_stats = []

    for i in range(LATENT_DIM):
        series = z[:, i]
        latent_stats.append({
            "Latent Factor": f"Z{i+1}",
            "Mean": np.mean(series),
            "Std Dev": np.std(series),
            "Skewness": skew(series),
            "Excess Kurtosis": kurtosis(series)
        })

        plt.figure()
        plt.hist(series, bins=50)
        plt.title(f"Latent Factor Z{i+1} Distribution")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"latent_Z{i+1}_hist.png", dpi=150)
        plt.close()

    latent_df = pd.DataFrame(latent_stats)
    latent_df.to_csv(OUT_DIR / "latent_factor_statistics.csv", index=False)

    print(latent_df.round(4))
    print("\nLatent analysis completed.\n")
