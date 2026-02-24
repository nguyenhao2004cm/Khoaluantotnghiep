# src/reporting/compute_annual_returns_table.py
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

PORT_FILE = PROJECT_DIR / "data_processed/powerbi/portfolio_timeseries.csv"
OUT_DIR = PROJECT_DIR / "data_processed/reporting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "annual_returns_table.csv"

df = pd.read_csv(PORT_FILE, parse_dates=["date"])
df = df.dropna(subset=["portfolio_return"])

# =========================
# MONTHLY COMPOUNDED RETURN
# =========================
monthly = (
    df.set_index("date")["portfolio_return"]
      .resample("ME")
      .apply(lambda x: (1 + x).prod() - 1)
      .to_frame("monthly_return")
      .reset_index()
)

monthly["Năm"] = monthly["date"].dt.year
monthly["Tháng"] = monthly["date"].dt.month

# =========================
# PIVOT TABLE
# =========================
table = monthly.pivot(
    index="Năm",
    columns="Tháng",
    values="monthly_return"
)

# =========================
# ANNUAL COMPOUND RETURN
# =========================
table["Cả năm"] = (1 + table).prod(axis=1) - 1

table = table.round(4)
table.to_csv(OUT_FILE, encoding="utf-8-sig")

print(" Saved:", OUT_FILE)
