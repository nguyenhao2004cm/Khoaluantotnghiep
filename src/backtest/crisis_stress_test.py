# =====================================================
# PHASE 4 — CRISIS STRESS TEST
# Mục tiêu: Chứng minh hệ thống chống chịu khủng hoảng
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

CRISIS_PERIODS = {
    "2020_COVID": ("2020-01-01", "2020-12-31"),
    "2022_Thanh_khoan": ("2022-01-01", "2022-12-31"),
}


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


def max_drawdown(returns: pd.Series) -> float:
    values = (1 + returns).cumprod()
    cummax = values.cummax()
    dd = (values - cummax) / cummax
    return dd.min()


def run_crisis_stress_test():
    """So sánh Drawdown: Regime-aware vs VNIndex proxy trong khủng hoảng."""
    print("\n" + "=" * 60)
    print(" PHASE 4 — CRISIS STRESS TEST")
    print("=" * 60)

    alloc = pd.read_csv(ALLOCATION_FILE)
    alloc["symbol"] = alloc["symbol"].str.strip().str.upper()
    alloc["date"] = pd.to_datetime(alloc["date"], errors="coerce")
    symbols = alloc["symbol"].unique().tolist()

    prices = load_prices_pivot(symbols)
    asset_returns = prices.pct_change(fill_method=None).dropna(how="all")

    weight_matrix = (
        alloc.groupby(["date", "symbol"])["allocation_weight"]
        .mean()
        .unstack(fill_value=0)
    )
    weight_matrix = weight_matrix.reindex(columns=symbols).fillna(0).sort_index().shift(1)
    common_dates = asset_returns.index.intersection(weight_matrix.dropna().index)
    asset_ret = asset_returns.loc[common_dates]
    weight_mat = weight_matrix.loc[common_dates].fillna(0)

    regime_returns = (weight_mat * asset_ret).sum(axis=1)
    n_sym = len(symbols)
    eq_weights = pd.DataFrame(1.0 / n_sym, index=common_dates, columns=symbols)
    vnindex_returns = (eq_weights * asset_ret).sum(axis=1)

    results = []
    for name, (start, end) in CRISIS_PERIODS.items():
        mask = (regime_returns.index >= start) & (regime_returns.index <= end)
        r_regime = regime_returns.loc[mask]
        r_vn = vnindex_returns.loc[mask]
        if len(r_regime) < 5:
            print(f"  Skip {name}: insufficient data")
            continue
        dd_regime = max_drawdown(r_regime)
        dd_vn = max_drawdown(r_vn)
        results.append({
            "Crisis": name,
            "Regime_aware_Drawdown": round(dd_regime, 4),
            "VNIndex_proxy_Drawdown": round(dd_vn, 4),
            "Outperform": "Yes" if dd_regime > dd_vn else "No",
        })
        print(f"\n {name}:")
        print(f"   Regime-aware Drawdown: {dd_regime:.4f}")
        print(f"   VNIndex proxy Drawdown: {dd_vn:.4f}")

    out_df = pd.DataFrame(results)
    out_path = OUT_DIR / "crisis_stress_test.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n Saved: {out_path}")
    return out_df


if __name__ == "__main__":
    run_crisis_stress_test()
