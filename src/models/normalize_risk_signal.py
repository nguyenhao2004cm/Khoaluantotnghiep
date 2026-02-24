#C:\Users\ASUS\fintech-project\src\models\normalize_risk_signal.py
import os
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
IN_DIR = "data_processed/risk"
OUT_DIR = "data_processed/risk_normalized"
WINDOW = 60

os.makedirs(OUT_DIR, exist_ok=True)

def normalize(df):
    df = df.sort_values("date").copy()

    EPS = 1e-8
    mu = df["risk_score"].rolling(WINDOW).mean()
    sigma = df["risk_score"].rolling(WINDOW).std() + EPS

    df["risk_z"] = (df["risk_score"] - mu) / sigma
    df["risk_z"] = df["risk_z"].clip(-3, 3)

    df["risk_regime"] = pd.cut(
        df["risk_z"],
        bins=[-np.inf, -1, 1, np.inf],
        labels=["LOW", "NORMAL", "HIGH"]
    )

    # Avoid dropping early dates
    df["risk_z"] = df["risk_z"].fillna(0)
    df["risk_regime"] = df["risk_regime"].fillna("NORMAL")

    return df

def run():
    for file in os.listdir(IN_DIR):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace("_risk.csv", "")
        print(f"🔧 Normalizing {symbol}")

        path = os.path.join(IN_DIR, file)
        df = pd.read_csv(path, parse_dates=["date"])

        out_df = normalize(df)

        out_path = os.path.join(OUT_DIR, f"{symbol}_risk_norm.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(" All risk signals normalized")

if __name__ == "__main__":
    run()
