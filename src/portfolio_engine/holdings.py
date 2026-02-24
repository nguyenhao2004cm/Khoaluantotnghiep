import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

ALLOCATION_FILE = (
    PROJECT_DIR
    / "data_processed"
    / "portfolio"
    / "portfolio_allocation_final.csv"
)

def current_holdings():
    alloc = pd.read_csv(ALLOCATION_FILE, parse_dates=["date"])

    latest_date = alloc["date"].max()

    holdings = (
        alloc[alloc["date"] == latest_date]
        .sort_values("allocation_weight", ascending=False)
        .reset_index(drop=True)
    )

    return latest_date, holdings

def average_holdings():
    alloc = pd.read_csv(ALLOCATION_FILE, parse_dates=["date"])

    avg_alloc = (
        alloc.groupby("symbol")["allocation_weight"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    return avg_alloc





if __name__ == "__main__":
    date, holdings = current_holdings()

    print(f"\nCURRENT PORTFOLIO HOLDINGS ({date.date()}):\n")
    print(holdings[["symbol", "allocation_weight"]])
    avg = average_holdings()

    print("\nAVERAGE PORTFOLIO HOLDINGS:\n")
    print(avg)