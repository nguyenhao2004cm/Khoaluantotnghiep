#C:\Users\ASUS\fintech-project\src\models\compute_risk_score.py
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch

from src.models.autoencoder import MarketAutoencoder
from src.models.train_lstm_latent import LatentLSTM

# ==============================
# CONFIG
# ==============================
LATENT_DIR = PROJECT_DIR / "data_processed" / "latent"
MODEL_PATH = PROJECT_DIR / "models" / "lstm_latent.pt"
OUT_DIR = PROJECT_DIR / "data_processed" / "risk"

SEQ_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)


# ==============================
# LOAD MODEL
# ==============================
def load_model(input_dim):
    model = LatentLSTM(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ==============================
# COMPUTE RISK
# ==============================
def _for_file(path):
    df = pd.read_csv(path)

    latent_cols = [c for c in df.columns if c.startswith("latent")]
    df_latent = df[latent_cols].dropna()

    if len(df_latent) <= SEQ_LEN:
        return None

    data = df_latent.values
    model = load_model(input_dim=data.shape[1])

    risks = []
    dates = []

    with torch.no_grad():
        for i in range(SEQ_LEN, len(data)):
            x = data[i - SEQ_LEN:i]
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            pred_latent = model(x).cpu().numpy()[0]

            # Risk score = composite latent (L2 norm: sqrt(Z1²+Z2²+Z3²))
            risk = np.sqrt(np.sum(pred_latent ** 2))

            risks.append(risk)
            dates.append(df.loc[df_latent.index[i], "date"])

    return pd.DataFrame({
        "date": dates,
        "risk_score": risks
    })


# ==============================
# RUN ALL
# ==============================
def run():
    for file in os.listdir(LATENT_DIR):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace("_latent.csv", "")
        print(f"Computing risk: {symbol}")

        df_risk = _for_file(str(LATENT_DIR / file))

        if df_risk is None:
            continue

        df_risk.to_csv(
            str(OUT_DIR / f"{symbol}_risk.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print(" Hoàn thành tính Risk Score")


if __name__ == "__main__":
    run()

