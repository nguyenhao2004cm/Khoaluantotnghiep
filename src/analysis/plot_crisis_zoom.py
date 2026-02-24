import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
risk = pd.read_csv(
    "data_processed/risk_normalized/AAA_risk_norm.csv",
    parse_dates=["date"]
)

feat = pd.read_csv(
    "data_processed/features/AAA_features.csv",
    parse_dates=["date"]
)

# Merge
df = risk.merge(feat, on="date", how="inner")

# =========================
# CRISIS WINDOW (GẦN NHẤT)
# =========================
crisis_df = df[
    (df["date"] >= "2022-01-01") &
    (df["date"] <= "2023-12-31")
]

# =========================
# PLOT
# =========================
fig, ax1 = plt.subplots(figsize=(12, 5))

# Latent risk
ax1.plot(
    crisis_df["date"],
    crisis_df["risk_z"],
    label="Latent Risk (Normalized)",
    linewidth=2
)
ax1.set_ylabel("Latent Risk")
ax1.legend(loc="upper left")

# Volatility (secondary axis)
ax2 = ax1.twinx()
ax2.plot(
    crisis_df["date"],
    crisis_df["vol_20"],
    linestyle="--",
    label="Rolling Volatility (20d)"
)
ax2.set_ylabel("Volatility")
ax2.legend(loc="upper right")

plt.title("Crisis Zoom (2022–2023): Latent Risk vs Volatility")
plt.tight_layout()
plt.show()
