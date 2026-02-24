import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
PRICE_DIR = "data_processed/features"
DECISION_DIR = "data_processed/decision"   # nếu chưa có, dùng equal-weight
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

OUT_FIG = "outputs/crisis_cumulative_return.png"
os.makedirs("outputs", exist_ok=True)


# ==============================
# UTIL
# ==============================
def max_drawdown(cum_ret):
    peak = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - peak) / peak
    return drawdown.min()


# ==============================
# MAIN
# ==============================
def run():
    eq_returns_all = []
    risk_returns_all = []

    for file in os.listdir(PRICE_DIR):
        if not file.endswith("_features.csv"):
            continue

        symbol = file.replace("_features.csv", "")
        df = pd.read_csv(os.path.join(PRICE_DIR, file), parse_dates=["date"])

        # Filter crisis period
        df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
        if len(df) < 50:
            continue

        returns = df["log_return"].values

        # ==================
        # Equal-weight
        # ==================
        eq_returns_all.append(returns)

        # ==================
        # Risk-based
        # ==================
        decision_path = os.path.join(DECISION_DIR, f"{symbol}_weight.csv")

        if os.path.exists(decision_path):
            w = pd.read_csv(decision_path)
            w["date"] = pd.to_datetime(w["date"])
            df = df.merge(w, on="date", how="left")
            df["weight"] = df["weight"].fillna(1.0)
            risk_returns_all.append(df["log_return"].values * df["weight"].values)
        else:
            # fallback nếu chưa có decision layer
            risk_returns_all.append(returns)

    # ==============================
    # Aggregate portfolio
    # ==============================
    min_len = min(map(len, eq_returns_all))

    eq_port = np.mean([r[:min_len] for r in eq_returns_all], axis=0)
    risk_port = np.mean([r[:min_len] for r in risk_returns_all], axis=0)

    eq_cum = np.cumprod(1 + eq_port)
    risk_cum = np.cumprod(1 + risk_port)

    # ==============================
    # Plot
    # ==============================
    plt.figure(figsize=(10, 5))
    plt.plot(eq_cum, label="Equal-weight", linewidth=2)
    plt.plot(risk_cum, label="Risk-based", linewidth=2)

    plt.title("Crisis Cumulative Return (2022–2023)")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.show()

    # ==============================
    # Metrics
    # ==============================
    print(" Max Drawdown (Crisis 2022–2023)")
    print(f"Equal-weight: {max_drawdown(eq_cum):.4f}")
    print(f"Risk-based : {max_drawdown(risk_cum):.4f}")

    print(f" Figure saved to {OUT_FIG}")


if __name__ == "__main__":
    run()
