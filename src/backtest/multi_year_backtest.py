# =====================================================
# PHASE 1 — MULTI-YEAR BACKTEST
# Mục tiêu: Chứng minh hệ thống tạo danh mục vượt trội
# =====================================================

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import numpy as np

# =====================================
# CONFIG
# =====================================
PRICE_FILE = PROJECT_DIR / "data_processed" / "prices" / "prices.csv"
ALLOCATION_FILE = PROJECT_DIR / "data_processed" / "portfolio" / "portfolio_allocation_final.csv"
REGIME_FILE = PROJECT_DIR / "data_processed" / "reporting" / "risk_regime_timeseries.csv"
OUT_DIR = PROJECT_DIR / "data_processed" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
TRADING_DAYS = 252


def load_prices_pivot(symbols):
    """Load prices as pivot: index=date, columns=symbols."""
    df = pd.read_csv(PRICE_FILE)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"])
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        raise ValueError("No price data for symbols")
    df = df.sort_values("date").groupby(["date", "symbol"], as_index=False).last()
    prices = df.pivot(index="date", columns="symbol", values="close").sort_index()
    return prices


def compute_metrics(returns: pd.Series, initial_capital: float = 10000.0) -> dict:
    """CAGR, MDD, CVaR, Sharpe cho chuỗi returns."""
    returns = returns.dropna()
    if len(returns) < 2:
        return {"CAGR": np.nan, "MDD": np.nan, "CVaR_5": np.nan, "Sharpe": np.nan}
    values = (1 + returns).cumprod() * initial_capital
    n_days = len(values)
    years = n_days / TRADING_DAYS
    cagr = (values.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    mdd = drawdown.min()
    var_5 = np.quantile(returns, 0.05)
    cvar_5 = returns[returns <= var_5].mean() if (returns <= var_5).any() else var_5
    sharpe = (returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)) if returns.std() > 0 else np.nan
    return {"CAGR": cagr, "MDD": mdd, "CVaR_5": cvar_5, "Sharpe": sharpe}


def run_multi_year_backtest():
    """Backtest theo từng năm: đầu năm tối ưu, hold 1 năm."""
    print("\n" + "=" * 60)
    print(" PHASE 1 — MULTI-YEAR BACKTEST")
    print("=" * 60)

    alloc = pd.read_csv(ALLOCATION_FILE)
    alloc["symbol"] = alloc["symbol"].str.strip().str.upper()
    alloc["date"] = pd.to_datetime(alloc["date"], errors="coerce")
    alloc = alloc.dropna(subset=["date", "symbol", "allocation_weight"])

    regime_df = pd.read_csv(REGIME_FILE, parse_dates=["date"]) if REGIME_FILE.exists() else None

    symbols = alloc["symbol"].unique().tolist()
    prices = load_prices_pivot(symbols)
    asset_returns = prices.pct_change(fill_method=None).dropna(how="all")

    weight_matrix = (
        alloc.groupby(["date", "symbol"])["allocation_weight"]
        .mean()
        .unstack(fill_value=0)
    )
    weight_matrix = weight_matrix.reindex(columns=symbols).fillna(0)

    yearly_results = []
    allocation_by_year = []

    for year in YEARS:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")

        # Lấy allocation tại ngày đầu năm (hoặc gần nhất trước đó)
        alloc_dates = weight_matrix.index[weight_matrix.index <= year_start]
        if len(alloc_dates) == 0:
            print(f"  Skip {year}: no allocation before {year_start.date()}")
            continue
        alloc_date = alloc_dates.max()

        # Regime trung bình năm
        regime = "NORMAL"
        if regime_df is not None:
            yr_regime = regime_df[
                (regime_df["date"] >= year_start) & (regime_df["date"] <= year_end)
            ]
            if not yr_regime.empty and "portfolio_risk_regime" in yr_regime.columns:
                regime = yr_regime["portfolio_risk_regime"].mode().iloc[0] if len(yr_regime) > 0 else "NORMAL"

        weights = weight_matrix.loc[alloc_date]
        weights = weights / weights.sum()

        # Returns trong năm (lag 1 ngày: dùng weight ngày trước)
        mask = (asset_returns.index > alloc_date) & (asset_returns.index <= year_end)
        yr_returns = asset_returns.loc[mask]
        if yr_returns.empty:
            print(f"  Skip {year}: no returns in period")
            continue

        # Portfolio return = sum(weight * asset_return)
        aligned = yr_returns[yr_returns.columns.intersection(weights.index)].copy()
        for c in aligned.columns:
            if c not in weights.index:
                aligned[c] = 0
            else:
                aligned[c] = aligned[c] * weights[c]
        portfolio_returns = aligned.sum(axis=1)

        metrics = compute_metrics(portfolio_returns)

        yearly_results.append({
            "year": year,
            "regime": regime,
            "CAGR": metrics["CAGR"],
            "MDD": metrics["MDD"],
            "CVaR_5": metrics["CVaR_5"],
            "Sharpe": metrics["Sharpe"],
        })

        for sym, w in weights.items():
            if w > 0.001:
                allocation_by_year.append({
                    "year": year,
                    "regime": regime,
                    "symbol": sym,
                    "allocation_weight": round(w, 4),
                })

    # Cash = 1 - sum(weights)
    alloc_wide = pd.DataFrame(allocation_by_year)
    if not alloc_wide.empty:
        cash_by_year = 1 - alloc_wide.groupby("year")["allocation_weight"].sum()
        for y, cash in cash_by_year.items():
            if cash > 0.001:
                allocation_by_year.append({
                    "year": int(y),
                    "regime": alloc_wide[alloc_wide["year"] == y]["regime"].iloc[0],
                    "symbol": "Cash",
                    "allocation_weight": round(cash, 4),
                })

    # Export
    yearly_df = pd.DataFrame(yearly_results)
    alloc_df = pd.DataFrame(allocation_by_year)

    yearly_path = OUT_DIR / "yearly_results.csv"
    alloc_path = OUT_DIR / "allocation_by_year.csv"
    yearly_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    alloc_df.to_csv(alloc_path, index=False, encoding="utf-8-sig")

    print(f"\n Saved: {yearly_path}")
    print(f" Saved: {alloc_path}")
    print("\n Yearly results:")
    print(yearly_df.to_string(index=False))
    return yearly_df, alloc_df


if __name__ == "__main__":
    run_multi_year_backtest()
