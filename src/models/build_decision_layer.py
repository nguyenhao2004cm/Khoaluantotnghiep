import os
import pandas as pd

RISK_DIR = "data_processed/risk_normalized"
OUT_DIR = "data_processed/decision"

os.makedirs(OUT_DIR, exist_ok=True)

def risk_to_weight(r):
    if r > 0.8:
        return 0.2
    elif r < 0.2:
        return 1.3
    else:
        return 1.0


def run():
    for file in os.listdir(RISK_DIR):
        if not file.endswith("_risk_norm.csv"):
            continue

        symbol = file.replace("_risk_norm.csv", "")
        df = pd.read_csv(os.path.join(RISK_DIR, file), parse_dates=["date"])

        df["weight"] = df["risk_z"].apply(risk_to_weight)

        out_path = os.path.join(OUT_DIR, f"{symbol}_weight.csv")
        df[["date", "weight"]].to_csv(out_path, index=False)

        print(f" Decision saved: {symbol}")


if __name__ == "__main__":
    run()
