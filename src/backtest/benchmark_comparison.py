# =====================================================
# PHASE 2 — BENCHMARK COMPARISON
# Mục tiêu: Trả lời "có outperform không?"
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
OUT_DIR = PROJECT_DIR / "data_processed" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RANDOM_SEED = 42


def load_prices_pivot(symbols):
    df = pd.read_csv(PRICE_FILE)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"])
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        raise ValueError("No price data")
    df = df.sort_values("date").groupby(["date", "symbol"], as_index=False).last()
    return df.pivot(index="date", columns="symbol", values="close").sort_index()


def compute_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> dict:
    returns = returns.dropna()
    if len(returns) < 2:
        return {"CAGR": np.nan, "MDD": np.nan, "CVaR_5": np.nan, "Sharpe": np.nan, "Information_Ratio": np.nan}
    values = (1 + returns).cumprod()
    n_days = len(values)
    years = n_days / TRADING_DAYS
    cagr = (values.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    cummax = values.cummax()
    mdd = ((values - cummax) / cummax).min()
    var_5 = np.quantile(returns, 0.05)
    cvar_5 = returns[returns <= var_5].mean() if (returns <= var_5).any() else var_5
    sharpe = (returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)) if returns.std() > 0 else np.nan
    ir = np.nan
    if benchmark_returns is not None:
        common = returns.index.intersection(benchmark_returns.index)
        if len(common) > 1:
            ret_a = returns.loc[common]
            ret_b = benchmark_returns.loc[common]
            diff = ret_a - ret_b
            if diff.std() > 0:
                ir = diff.mean() / diff.std() * np.sqrt(TRADING_DAYS)
    return {"CAGR": cagr, "MDD": mdd, "CVaR_5": cvar_5, "Sharpe": sharpe, "Information_Ratio": ir}


def run_benchmark_comparison():
    """So sánh: Regime-aware vs Equal-weight vs VNIndex proxy vs Random."""
    print("\n" + "=" * 60)
    print(" PHASE 2 — BENCHMARK COMPARISON")
    print("=" * 60)

    alloc = pd.read_csv(ALLOCATION_FILE)
    alloc["symbol"] = alloc["symbol"].str.strip().str.upper()
    alloc["date"] = pd.to_datetime(alloc["date"], errors="coerce")
    symbols = alloc["symbol"].unique().tolist()

    prices = load_prices_pivot(symbols)
    asset_returns = prices.pct_change(fill_method=None).dropna(how="all")

    # Weight matrix regime-aware
    weight_matrix = (
        alloc.groupby(["date", "symbol"])["allocation_weight"]
        .mean()
        .unstack(fill_value=0)
    )
    weight_matrix = weight_matrix.reindex(columns=symbols).fillna(0).sort_index().shift(1)
    common_dates = asset_returns.index.intersection(weight_matrix.dropna().index)
    asset_ret = asset_returns.loc[common_dates]
    weight_mat = weight_matrix.loc[common_dates].fillna(0)

    # 1. Regime-aware
    regime_returns = (weight_mat * asset_ret).sum(axis=1)

    # 2. Equal-weight
    n_sym = len(symbols)
    eq_weights = pd.DataFrame(1.0 / n_sym, index=common_dates, columns=symbols)
    eq_returns = (eq_weights * asset_ret).sum(axis=1)

    # 3. VN-Index proxy = equal-weight của universe (không có VNINDEX)
    vnindex_returns = eq_returns.copy()

    # 4. Random (fixed seed, rebalance daily random weights)
    np.random.seed(RANDOM_SEED)
    random_returns = []
    for d in common_dates:
        w = np.random.dirichlet(np.ones(n_sym))
        r = (asset_ret.loc[d] * w).sum()
        random_returns.append(r)
    random_returns = pd.Series(random_returns, index=common_dates)

    # Metrics
    results = []
    for name, rets in [
        ("Regime-aware", regime_returns),
        ("Equal-weight", eq_returns),
        ("VNIndex_proxy", vnindex_returns),
        ("Random", random_returns),
    ]:
        m = compute_metrics(rets, benchmark_returns=vnindex_returns)
        results.append({
            "Strategy": name,
            "CAGR": m["CAGR"],
            "MDD": m["MDD"],
            "CVaR_5": m["CVaR_5"],
            "Sharpe": m["Sharpe"],
            "Information_Ratio": m["Information_Ratio"],
        })

    out_df = pd.DataFrame(results)
    out_path = OUT_DIR / "backtest_summary_comparison.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n Saved: {out_path}")
    print("\n Comparison:")
    print(out_df.to_string(index=False))
    return out_df


if __name__ == "__main__":
    run_benchmark_comparison()
