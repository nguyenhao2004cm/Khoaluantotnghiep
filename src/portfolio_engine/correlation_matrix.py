import pandas as pd
import numpy as np
from pathlib import Path

# =====================================
# CONFIG
# =====================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_DIR / "data_processed" / "prices" / "prices.csv"
OUT_DIR = PROJECT_DIR / "data_processed/reporting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "correlation_summary.csv"

# =====================================
# LOAD CLEAN PRICE DATA
# =====================================
def load_prices(symbols):
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])

    df["symbol"] = df["symbol"].str.upper()
    df = df[df["symbol"].isin(symbols)]
    df = df.dropna(subset=["date", "symbol", "close"])

    # Fix duplicate date-symbol (standard practice)
    df = (
        df.sort_values("date")
          .groupby(["date", "symbol"], as_index=False)
          .last()
    )

    prices = (
        df.pivot(index="date", columns="symbol", values="close")
          .sort_index()
    )

    return prices


# =====================================
# COMPUTE MONTHLY RETURNS
# =====================================
def compute_monthly_returns(prices):
    daily_returns = prices.pct_change(fill_method=None)

    monthly_returns = (
        (1 + daily_returns)
        .resample("ME")
        .prod() - 1
    )

    return monthly_returns.dropna(how="all")


# =====================================
# PLOT CORRELATION HEATMAP
# =====================================
def plot_correlation_heatmap(monthly_returns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr = monthly_returns.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.5
    )

    plt.title("Monthly Asset Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=100)
    plt.close()

def build_correlation_summary(monthly_returns):
    corr = monthly_returns.corr()

    # Lấy tam giác trên, bỏ diagonal (n=2: 1 giá trị; n=1: rỗng)
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_vals = corr.where(mask).stack()

    if corr_vals.empty:
        return {
            "mean_correlation": np.nan,
            "min_correlation": np.nan,
            "max_correlation": np.nan,
            "low_corr_ratio": np.nan,
        }

    summary = {
        "mean_correlation": float(corr_vals.mean()),
        "min_correlation": float(corr_vals.min()),
        "max_correlation": float(corr_vals.max()),
        "low_corr_ratio": float((corr_vals.abs() < 0.3).mean()),
    }

    return summary
if __name__ == "__main__":
    from holdings import current_holdings

    _, holdings = current_holdings()
    symbols = holdings["symbol"].tolist()

    prices = load_prices(symbols)
    monthly_returns = compute_monthly_returns(prices)

    # Plot (giữ nếu muốn)
    plot_correlation_heatmap(monthly_returns)

    #  BUILD SUMMARY
    summary = build_correlation_summary(monthly_returns)

    pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    ).to_csv(OUT_FILE, index=False)

    # print("Saved correlation summary:", OUT_FILE)
