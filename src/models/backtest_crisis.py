import os
import numpy as np
import pandas as pd

# ==============================
# CONFIG
# ==============================
FEATURE_DIR = "data_processed/features"
DECISION_DIR = "data_processed/decision"

CRISIS_START = "2022-01-01"
CRISIS_END   = "2023-12-31"


# ==============================
# METRIC
# ==============================
def max_drawdown(cum_return):
    peak = np.maximum.accumulate(cum_return)
    drawdown = (cum_return - peak) / peak
    return drawdown.min()


# ==============================
# MAIN
# ==============================
def run():
    all_rows = []

    for file in os.listdir(FEATURE_DIR):
        if not file.endswith("_features.csv"):
            continue

        symbol = file.replace("_features.csv", "")
        feat_path = os.path.join(FEATURE_DIR, file)
        dec_path = os.path.join(DECISION_DIR, f"{symbol}_weight.csv")

        if not os.path.exists(dec_path):
            continue

        feat = pd.read_csv(feat_path, parse_dates=["date"])
        dec = pd.read_csv(dec_path, parse_dates=["date"])

        df = pd.merge(
            feat[["date", "log_return"]],
            dec[["date", "weight"]],
            on="date",
            how="inner"
        )

        df = df[
            (df["date"] >= CRISIS_START) &
            (df["date"] <= CRISIS_END)
        ]

        if len(df) < 30:
            continue

        df["symbol"] = symbol
        df["eq_ret"] = df["log_return"]
        df["rb_ret"] = df["log_return"] * df["weight"]

        all_rows.append(df[["date", "symbol", "eq_ret", "rb_ret"]])

    if len(all_rows) == 0:
        raise ValueError(" No valid crisis data")

    panel = pd.concat(all_rows, ignore_index=True)

    # ==============================
    # PORTFOLIO BY DATE (KEY FIX)
    # ==============================
    daily = panel.groupby("date").mean(numeric_only=True)

    eq_cum = (1 + daily["eq_ret"]).cumprod()
    rb_cum = (1 + daily["rb_ret"]).cumprod()

    eq_dd = max_drawdown(eq_cum.values)
    rb_dd = max_drawdown(rb_cum.values)

    print(" Max Drawdown (Crisis 2022–2023)")
    print(f"Equal-weight: {eq_dd:.4f}")
    print(f"Risk-based : {rb_dd:.4f}")


if __name__ == "__main__":
    run()
