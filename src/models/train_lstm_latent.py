import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ==============================
# CONFIG
# ==============================
LATENT_DIR = PROJECT_DIR / "data_processed" / "latent"
FEATURE_DIR = PROJECT_DIR / "data_processed" / "features"
MODEL_PATH = PROJECT_DIR / "models" / "lstm_latent.pt"

SEQ_LEN = 20          # Bertani (2021)
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# MODEL
# ==============================
class LatentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)  # predict return

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ==============================
# DATA
# ==============================
def load_symbol(symbol):
    latent_path = LATENT_DIR / f"{symbol}_latent.csv"
    feat_path = FEATURE_DIR / f"{symbol}_features.csv"

    if not latent_path.exists() or not feat_path.exists():
        return None, None

    latent = pd.read_csv(str(latent_path))
    feat = pd.read_csv(str(feat_path))

    df = pd.merge(latent, feat[["date", "log_return"]], on="date")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    latent_cols = [c for c in df.columns if c.startswith("latent")]

    if len(df) <= SEQ_LEN:
        return None, None

    X, y = [], []

    for i in range(len(df) - SEQ_LEN):
        X.append(df[latent_cols].iloc[i:i+SEQ_LEN].values)
        y.append(df["log_return"].iloc[i+SEQ_LEN])

    return np.array(X), np.array(y)


def load_all():
    X_all, y_all = [], []

    for file in os.listdir(str(LATENT_DIR)):
        symbol = file.replace("_latent.csv", "")
        X, y = load_symbol(symbol)

        if X is None:
            continue

        X_all.append(X)
        y_all.append(y)

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print(f" Total samples: {X.shape[0]}")
    return X, y


# ==============================
# TRAIN
# ==============================
def train():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_all()

    #  Normalize latent
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(DEVICE)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LatentLSTM(input_dim=X.shape[2]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f" Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
