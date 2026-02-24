# src/models/train_autoencoder.py

import os
import sys
import torch
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

sys.path.append(os.path.abspath("."))

from src.models.autoencoder import MarketAutoencoder

# =====================================================
# CONFIG
# =====================================================
FEATURE_DIR = "data_processed/risk_features"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# RISK-ONLY FEATURE SET
# =========================
FEATURE_COLS = [
    "vol_20",
    "vol_60",
    "vol_change",
    "max_drawdown_60"
]



LATENT_DIM = 3
BATCH_SIZE = 256
EPOCHS = 80
LR = 1e-3
LATENT_REG = 1e-3   # regularization strength


# =====================================================
# LOAD FEATURE DATA
# =====================================================
def load_feature_data():
    """
    Load ALL feature files and stack cross-sectionally.
    Autoencoder learns GLOBAL market risk structure.
    """
    dfs = []

    for file in os.listdir(FEATURE_DIR):
        if file.endswith("_risk_features.csv"):
            path = os.path.join(FEATURE_DIR, file)
            df = pd.read_csv(path)

            # Only keep risk features
            dfs.append(df[FEATURE_COLS])

    data = pd.concat(dfs, ignore_index=True)
    return data


# =====================================================
# RISK-WEIGHTED LOSS
# =====================================================
def risk_weighted_mse(x_hat, x):
    """
    Stress-aware reconstruction loss
    - Penalize errors more during high-volatility states
    """
    # Use vol_20 as stress proxy (first feature)
    vol_proxy = x[:, 0]

    # High-stress = top 20% volatility
    stress_threshold = torch.quantile(vol_proxy, 0.80)
    weights = torch.ones_like(vol_proxy)
    weights[vol_proxy > stress_threshold] = 4.0

    loss = ((x_hat - x) ** 2).mean(dim=1)
    return (loss * weights).mean()


# =====================================================
# TRAINING
# =====================================================
def train():
    print("\n" + "=" * 70)
    print(" TRAINING MARKET RISK AUTOENCODER")
    print("   Role: Latent Risk Structure Learning")
    print("=" * 70 + "\n")

    df = load_feature_data()

    # Robust scaling: preserves tails
    scaler = RobustScaler()
    X = scaler.fit_transform(df.values)
    
    joblib.dump(scaler, "models/risk_scaler.pkl")

    X_tensor = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)

    # DO NOT SHUFFLE time series risk data
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = MarketAutoencoder(
        input_dim=len(FEATURE_COLS),
        latent_dim=LATENT_DIM
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for (x,) in loader:
            optimizer.zero_grad()

            x_hat, z = model(x)

            # Reconstruction loss (stress-aware)
            recon_loss = risk_weighted_mse(x_hat, x)

            # Latent space regularization (stability)
            z_reg = torch.mean(z ** 2)

            loss = recon_loss + LATENT_REG * z_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:03d} | Loss = {total_loss:.6f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "market_autoencoder.pt")
    torch.save(model.state_dict(), model_path)

    print("\n Training completed")
    print(f" Model saved to: {model_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train()
