#C:\Users\ASUS\fintech-project\src\models\build_portfolio.py
import os
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
RISK_DIR = "data_processed/risk"
OUT_DIR = "data_processed/portfolio"

os.makedirs(OUT_DIR, exist_ok=True)


# ==============================
# LOAD ALL RISK
# ==============================
def load_all_risk():
    data = []

    for file in os.listdir(RISK_DIR):
        if not file.endswith("_risk.csv"):
            continue

        symbol = file.replace("_risk.csv", "")
        df = pd.read_csv(os.path.join(RISK_DIR, file))

        df["symbol"] = symbol
        data.append(df)

    return pd.concat(data, ignore_index=True)


# ==============================
# BUILD PORTFOLIO
# ==============================
def build_inverse_risk_portfolio(df):
    df = df.dropna()

    portfolio = []

    for date, g in df.groupby("date"):
        g = g.copy()

        # tránh chia cho 0
        g["risk_score"] = g["risk_score"].replace(0, np.nan)
        g = g.dropna()

        inv_risk = 1.0 / g["risk_score"]
        g["weight"] = inv_risk / inv_risk.sum()

        portfolio.append(g[["date", "symbol", "weight"]])

    return pd.concat(portfolio, ignore_index=True)


# ==============================
# RUN
# ==============================
def run():
    print(" Building risk-based portfolio...")

    df_risk = load_all_risk()
    df_portfolio = build_inverse_risk_portfolio(df_risk)

    df_portfolio.to_csv(
        f"{OUT_DIR}/risk_based_portfolio.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(" Danh mục đã được xây dựng")


if __name__ == "__main__":
    run()
