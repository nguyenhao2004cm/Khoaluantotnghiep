import os
import pandas as pd

# ==============================
# CONFIG
# ==============================
RISK_DIR = "data_processed/risk_normalized"
OUT_DIR = "data_processed/decision"

os.makedirs(OUT_DIR, exist_ok=True)


def risk_to_weight(risk):
    """
    Rule-based decision layer
    """
    if risk >= 0.8:
        return 0.0        # high risk → exit
    elif risk <= 0.2:
        return 1.0        # low risk → full allocation
    else:
        return 1 - risk   # linear decay


def run():
    for file in os.listdir(RISK_DIR):
        if not file.endswith("_risk_norm.csv"):
            continue

        symbol = file.replace("_risk_norm.csv", "")
        print(f" Build weight for {symbol}")

        df = pd.read_csv(os.path.join(RISK_DIR, file), parse_dates=["date"])
        df["weight"] = df["risk_z"].apply(risk_to_weight)

        out_path = os.path.join(OUT_DIR, f"{symbol}_weight.csv")
        df[["date", "weight"]].to_csv(out_path, index=False)

        print(f" Saved {out_path}")


if __name__ == "__main__":
    run()
