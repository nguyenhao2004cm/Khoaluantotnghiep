#C:\Users\ASUS\fintech-project\src\portfolio_engine\build_correlation_monthly.py
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[2]

PRICE_FILE = PROJECT_DIR / "data_processed" / "prices" / "prices.csv"
OUT_DIR = PROJECT_DIR / "data_processed" / "powerbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "correlation_monthly.csv"


# =========================
# BUILD MONTHLY CORRELATION
# =========================
def build_monthly_correlation():
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])

    # Chuẩn hóa
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df = df.dropna(subset=["date", "symbol", "close"])

    # Pivot giá
    price_matrix = (
        df.pivot(index="date", columns="symbol", values="close")
          .sort_index()
    )

    # Lợi nhuận ngày
    daily_returns = price_matrix.pct_change()

    # Lợi nhuận tháng (geometric)
    monthly_returns = (
        (1 + daily_returns)
        .resample("M")
        .prod()
        - 1
    )

    # Correlation matrix
    corr = monthly_returns.corr()

    corr.to_csv(OUT_FILE, encoding="utf-8-sig")

    print(" Created correlation_monthly.csv")


if __name__ == "__main__":
    build_monthly_correlation()
