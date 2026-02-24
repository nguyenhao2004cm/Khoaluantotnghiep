import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
RISK_PATH = "data_processed/risk_normalized/AAA_risk_norm.csv"
FEATURE_PATH = "data_processed/features/AAA_features.csv"
ROLLING_WIN = 30

# =========================
# LOAD DATA
# =========================
risk = pd.read_csv(RISK_PATH, parse_dates=["date"])
feat = pd.read_csv(FEATURE_PATH, parse_dates=["date"])

df = pd.merge(
    risk[["date", "risk_z"]],
    feat[["date", "log_return"]],
    on="date",
    how="inner"
)

# =========================
# COMPUTE VOLATILITY
# =========================
df["volatility"] = (
    df["log_return"]
    .rolling(ROLLING_WIN)
    .std()
)

df = df.dropna()

# =========================
# CORRELATION
# =========================
corr = df["risk_z"].corr(df["volatility"])
print(f"📊 Correlation (Risk vs Volatility): {corr:.4f}")

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["risk_z"], label="Latent Risk (z-score)")
plt.plot(df["date"], df["volatility"], label="Rolling Volatility (30d)")
plt.legend()
plt.title("Risk Signal vs Traditional Volatility")
plt.tight_layout()
plt.show()
