import os
import pandas as pd

DECISION_DIR = "data_processed/decision"
OUT_DIR = "data_processed/portfolio"
OUT_FILE = "portfolio_weights.csv"

os.makedirs(OUT_DIR, exist_ok=True)

all_rows = []

for file in os.listdir(DECISION_DIR):
    if not file.endswith("_weight.csv"):
        continue

    symbol = file.replace("_weight.csv", "")
    path = os.path.join(DECISION_DIR, file)

    df = pd.read_csv(path, parse_dates=["date"])
    df["symbol"] = symbol

    all_rows.append(df[["date", "symbol", "weight"]])

portfolio = pd.concat(all_rows, ignore_index=True)

portfolio.to_csv(
    os.path.join(OUT_DIR, OUT_FILE),
    index=False,
    encoding="utf-8-sig"
)

print("✅ Created portfolio_weights.csv")
