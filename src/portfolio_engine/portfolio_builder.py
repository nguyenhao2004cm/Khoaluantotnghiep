# portfolio_builder.py – đọc web_run_config để lọc theo start_date, end_date
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================
# CONFIG
# =====================================
PROJECT_DIR = Path(__file__).resolve().parents[2]

PRICE_FILE = PROJECT_DIR / "data_processed" / "prices" / "prices.csv"
ALLOCATION_FILE = (
    PROJECT_DIR
    / "data_processed"
    / "portfolio"
    / "portfolio_allocation_final.csv"
)

# =====================================
# LOAD PRICE DATA (PANEL FORMAT)
# =====================================
def load_price_data(symbols):
    """
    Price file format:
        date | symbol | close

    Return:
        DataFrame
        index   = date
        columns = symbols
        values  = close price
    """
    df = pd.read_csv(PRICE_FILE)

    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"])

    symbols = [s.upper() for s in symbols]
    df = df[df["symbol"].isin(symbols)]

    if df.empty:
        raise ValueError(" No price data after filtering symbols")

    # Nếu trùng date–symbol → lấy giá cuối cùng
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
# BUILD PORTFOLIO (NO LOOK-AHEAD)
# =====================================

def build_portfolio(
    initial_capital=10000,
    start_date=None,
    end_date=None,
):
    # Nếu có web_run_config, dùng start_date/end_date/initial_capital từ đó (ưu tiên tham số)
    try:
        from src.utils.web_run_config import load_web_run_config
        cfg = load_web_run_config()
        if cfg:
            if start_date is None and cfg.get("start_date"):
                start_date = cfg["start_date"]
            if end_date is None and cfg.get("end_date"):
                end_date = cfg["end_date"]
            if cfg.get("initial_capital") is not None:
                initial_capital = float(cfg["initial_capital"])
    except Exception:
        pass

    # -------------------------------
    # Load allocation (FINAL OUTPUT)
    # -------------------------------
    alloc = pd.read_csv(ALLOCATION_FILE)

    alloc["symbol"] = alloc["symbol"].str.strip().str.upper()
    alloc["date"] = pd.to_datetime(alloc["date"], errors="coerce")
    alloc = alloc.dropna(subset=["date", "symbol", "allocation_weight"])

    symbols = alloc["symbol"].unique().tolist()

    # -------------------------------
    # Load prices
    # -------------------------------
    prices = load_price_data(symbols)

    # -------------------------------
    # Compute asset returns
    # -------------------------------
    asset_returns = prices.pct_change(fill_method=None)
    asset_returns = asset_returns.dropna(how="all")

    # -------------------------------
    # Build weight matrix
    # -------------------------------
    alloc = (
        alloc
        .groupby(["date", "symbol"], as_index=False)
        .agg({
            "allocation_weight": "mean"
        })
    )


    weight_matrix = alloc.pivot(
        index="date",
        columns="symbol",
        values="allocation_weight"
    )

    # Align dates
    common_dates = asset_returns.index.intersection(weight_matrix.index)
    asset_returns = asset_returns.loc[common_dates]
    weight_matrix = weight_matrix.loc[common_dates].fillna(0)

    # -------------------------------
    #  CRITICAL: LAG WEIGHTS (NO LOOK-AHEAD)
    # -------------------------------
    weight_matrix = weight_matrix.sort_index().shift(1)

    valid_dates = weight_matrix.dropna().index
    weight_matrix = weight_matrix.loc[valid_dates]
    asset_returns = asset_returns.loc[valid_dates]

    # -------------------------------
    # Portfolio return (Markowitz / PV)
    # -------------------------------
    portfolio_returns = (weight_matrix * asset_returns).sum(axis=1)

    # -------------------------------
    # Portfolio value
    # -------------------------------
    portfolio_value = (1 + portfolio_returns).cumprod() * initial_capital

    result = pd.DataFrame({
        "date": portfolio_returns.index,
        "portfolio_return": portfolio_returns.values,
        "portfolio_value": portfolio_value.values
    })
    result["date"] = pd.to_datetime(result["date"])

    # Lọc theo start_date / end_date (web run config)
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        result = result[result["date"] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        result = result[result["date"] <= end_date]

    return result


# =====================================
# RUN TEST
# =====================================
if __name__ == "__main__":
    df_port = build_portfolio(initial_capital=10000)

    print("\nHEAD:")
    print(df_port.head())

    print("\nTAIL:")
    print(df_port.tail())

    print("\nRETURN STATS:")
    print(df_port["portfolio_return"].describe())

