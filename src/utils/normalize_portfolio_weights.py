import os
import pandas as pd

BASE_DIR = os.getcwd()

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data_processed",
    "portfolio",
    "portfolio_weights.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data_processed",
    "portfolio",
    "portfolio_weights_allocation.csv"
)

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# Normalize weight theo NGÀY
df["allocation_weight"] = (
    df["weight"] /
    df.groupby("date")["weight"].transform("sum")
)

# Check: mỗi ngày phải ~ 1.0
check = df.groupby("date")["allocation_weight"].sum()
print("Check allocation sum (first 5 days):")
print(check.head())

df_out = df[["date", "symbol", "allocation_weight"]]
df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("✅ Normalized portfolio weights saved")
print("📁 Output:", OUTPUT_FILE)
