# =====================================================
# ALLOCATION VALIDATION - Empirical Comparison
# Fair validation: monthly rebalance, transaction cost 0.2%
#
# So sanh 3 mo hinh: Softmax | ERC | MinVar
# Metrics: CAGR, Sharpe, MDD, CVaR, Turnover, CAGR_adj (after cost)
# =====================================================

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import numpy as np
from src.decision.build_portfolio_allocation import (
    allocate_defensive,
    allocate_balanced,
    allocate_aggressive,
    SIGNAL_FILE,
    RISK_REGIME_FILE,
    PRICE_FILE,
)
from src.portfolio_engine.risk_based_optimizer import (
    allocate_risk_based,
    load_returns,
)

OUT_DIR = PROJECT_DIR / "data_processed" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRADING_DAYS = 252
TRANSACTION_COST = 0.002  # 0.2% per weight change


def load_prices_pivot(symbols):
    df = pd.read_csv(PRICE_FILE)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return None
    df = df.sort_values("date").groupby(["date", "symbol"], as_index=False).last()
    return df.pivot(index="date", columns="symbol", values="close").sort_index()


def compute_metrics(returns: pd.Series, initial_capital: float = 10000.0) -> dict:
    """CAGR, MDD, CVaR, Sharpe."""
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


def compute_turnover(weight_matrix: pd.DataFrame) -> float:
    """Turnover trung bình mỗi lần rebalance."""
    diff = weight_matrix.diff().abs()
    return diff.sum(axis=1).mean()


def run_backtest_with_allocation(allocation_records, symbols, prices, transaction_cost=0.0):
    """Portfolio return from allocation. If transaction_cost>0, subtract cost from returns."""
    alloc_df = pd.DataFrame(allocation_records)
    alloc_df["date"] = pd.to_datetime(alloc_df["date"])
    weight_matrix = (
        alloc_df.pivot_table(index="date", columns="symbol", values="allocation_weight")
        .reindex(columns=symbols)
        .fillna(0)
    )
    weight_matrix = weight_matrix.reindex(columns=symbols).fillna(0)

    ret = prices.pct_change(fill_method=None).dropna(how="all")
    ret = ret.reindex(columns=symbols).fillna(0)
    common = weight_matrix.index.intersection(ret.index)
    common = common.sort_values()
    w = weight_matrix.reindex(common).ffill().fillna(0)
    r = ret.loc[common].reindex(columns=symbols).fillna(0)
    port_ret = (w.shift(1) * r).sum(axis=1)

    if transaction_cost > 0:
        turnover = w.diff().abs().sum(axis=1)
        cost_per_day = turnover * transaction_cost
        port_ret = port_ret - cost_per_day.reindex(port_ret.index).fillna(0)

    return port_ret.dropna(), weight_matrix


def build_allocation_softmax(df, assets, regime_series, monthly_only=True):
    """Softmax allocation. monthly_only=True for fair comparison with ERC/MinVar."""
    records = []
    last_weights_by_month = {}
    for date, g in df.groupby("date"):
        if date not in regime_series.index:
            continue
        if monthly_only and date.day > 1:
            month_key = (date.year, date.month)
            prev_key = (date.year, date.month - 1) if date.month > 1 else (date.year - 1, 12)
            wdict = last_weights_by_month.get(month_key) or last_weights_by_month.get(prev_key)
            if wdict:
                for sym, w in wdict.items():
                    records.append({"date": date, "symbol": sym, "allocation_weight": w})
            continue
        symbols = g["symbol"].tolist()
        signals = g["signal"].values
        regime = regime_series.loc[date]
        if regime == "HIGH":
            weights = allocate_defensive(symbols, signals)
        elif regime == "LOW":
            weights = allocate_aggressive(symbols, signals)
        else:
            weights = allocate_balanced(symbols, signals)
        last_weights_by_month[(date.year, date.month)] = weights
        for sym, w in weights.items():
            records.append({"date": date, "symbol": sym, "allocation_weight": w})
    return records


def build_allocation_risk_based(df, assets, regime_series, method="erc", monthly_only=True):
    """Allocation theo ERC hoặc MinVar. monthly_only=True: chi optimize dau thang (nhanh)."""
    records = []
    df["date"] = pd.to_datetime(df["date"])
    last_weights_by_month = {}
    for date, g in df.groupby("date"):
        if date not in regime_series.index:
            continue
        if monthly_only and date.day > 1:
            month_key = (date.year, date.month)
            prev_key = (date.year, date.month - 1) if date.month > 1 else (date.year - 1, 12)
            wdict = last_weights_by_month.get(month_key) or last_weights_by_month.get(prev_key)
            if wdict:
                for sym, w in wdict.items():
                    records.append({"date": date, "symbol": sym, "allocation_weight": w})
            continue
        symbols = g["symbol"].tolist()
        signals = g["signal"].values
        regime = regime_series.loc[date]
        weights = allocate_risk_based(symbols, signals, regime, method=method, as_of_date=date)
        last_weights_by_month[(date.year, date.month)] = weights
        for sym, w in weights.items():
            records.append({"date": date, "symbol": sym, "allocation_weight": w})
    return records


def run_validation():
    """So sánh Softmax vs MV vs ERC."""
    print("\n" + "=" * 60)
    print(" ALLOCATION VALIDATION — Softmax vs MV vs ERC")
    print("=" * 60)

    df = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])
    assets = df["symbol"].unique().tolist()
    if len(assets) > 15:
        assets = assets[:15]
    df = df[df["symbol"].isin(assets)]
    # Gioi han 1 nam gan nhat de validation nhanh hon (co the tang len)
    cutoff = df["date"].max() - pd.Timedelta(days=365)
    df = df[df["date"] >= cutoff]

    regime_series = pd.read_csv(RISK_REGIME_FILE, parse_dates=["date"]).set_index("date")["portfolio_risk_regime"]
    prices = load_prices_pivot(assets)
    if prices is None:
        print("  [!] No price data")
        return

    results = []
    strategies = [
        ("Softmax", lambda: build_allocation_softmax(df, assets, regime_series)),
        ("ERC", lambda: build_allocation_risk_based(df, assets, regime_series, method="erc")),
        ("MinVar", lambda: build_allocation_risk_based(df, assets, regime_series, method="minvar")),
    ]

    for name, build_fn in strategies:
        print(f"\n  Running {name}...")
        records = build_fn()
        if not records:
            continue
        port_ret, w_matrix = run_backtest_with_allocation(
            records, assets, prices, transaction_cost=TRANSACTION_COST
        )
        metrics = compute_metrics(port_ret)
        turnover = compute_turnover(w_matrix)
        metrics["Turnover"] = turnover
        metrics["Strategy"] = name
        metrics["CAGR_adj"] = metrics["CAGR"]  # After cost
        results.append(metrics)

    res_df = pd.DataFrame(results)
    res_df = res_df[["Strategy", "CAGR", "CAGR_adj", "Sharpe", "MDD", "CVaR_5", "Turnover"]]
    print("\n" + "=" * 60)
    print(" KET QUA SO SANH")
    print("=" * 60)
    print(res_df.to_string(index=False))

    out_path = OUT_DIR / "allocation_strategy_comparison.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    best_sharpe = res_df.loc[res_df["Sharpe"].idxmax(), "Strategy"]
    print(f"\n  Best Sharpe: {best_sharpe}")

    return res_df


def run_robustness_test():
    """Test 2, 3, 5, 10 assets - ensure allocation stable."""
    print("\n" + "=" * 60)
    print(" ROBUSTNESS TEST - 2, 3, 5, 10 assets")
    print("=" * 60)

    df = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])
    all_assets = df["symbol"].unique().tolist()
    regime_series = pd.read_csv(RISK_REGIME_FILE, parse_dates=["date"]).set_index("date")["portfolio_risk_regime"]
    cutoff = df["date"].max() - pd.Timedelta(days=365)
    df_full = df[df["date"] >= cutoff]

    results = []
    for n in [2, 3, 5, 10]:
        if n > len(all_assets):
            continue
        assets = all_assets[:n]
        df_sub = df_full[df_full["symbol"].isin(assets)]
        prices = load_prices_pivot(assets)
        if prices is None:
            continue
        for name in ["Softmax", "ERC"]:
            if name == "Softmax":
                records = build_allocation_softmax(df_sub, assets, regime_series)
            else:
                records = build_allocation_risk_based(df_sub, assets, regime_series, method="erc")
            if not records:
                continue
            port_ret, w_matrix = run_backtest_with_allocation(
                records, assets, prices, transaction_cost=TRANSACTION_COST
            )
            m = compute_metrics(port_ret)
            m["n_assets"] = n
            m["Strategy"] = name
            results.append(m)

    res = pd.DataFrame(results)
    if not res.empty:
        print(res.pivot_table(index="n_assets", columns="Strategy", values="Sharpe").to_string())
        res.to_csv(OUT_DIR / "robustness_test.csv", index=False)
    return res


def run_vic_vcb_validation():
    """Phase 4: VIC-VCB case - Softmax ~50-50, ERC ~36/64."""
    print("\n" + "=" * 60)
    print(" VIC-VCB VALIDATION")
    print("=" * 60)

    from src.portfolio_engine.risk_based_optimizer import optimize_allocation, get_covariance_matrix

    symbols = ["VIC", "VCB"]
    df = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        print("  [!] No VIC/VCB in signal")
        return

    # Get latest date
    last_date = df["date"].max()
    g = df[df["date"] == last_date]
    signals = g["signal"].values
    regime_series = pd.read_csv(RISK_REGIME_FILE, parse_dates=["date"]).set_index("date")["portfolio_risk_regime"]
    regime = regime_series.loc[last_date] if last_date in regime_series.index else "NORMAL"

    # Softmax (signal used for weight - tends to 50-50 when dispersion low)
    if regime == "HIGH":
        w_softmax = allocate_defensive(symbols, list(signals))
    else:
        w_softmax = allocate_balanced(symbols, signals)
    w_softmax = {k: w_softmax.get(k, 0) for k in symbols}

    # ERC
    ret, _ = load_returns(symbols, last_date)
    cov = get_covariance_matrix(ret) if ret is not None else None
    if cov is not None:
        from src.portfolio_engine.risk_based_optimizer import erc_weights, min_variance_weights
        # 2 assets: dung bounds rong de hien thi dung risk structure
        w_erc = erc_weights(cov, min_w=0.01, max_w=0.99)
        w_minvar = min_variance_weights(cov, min_w=0.01, max_w=0.99)
        w_erc = dict(zip(symbols, w_erc))
        w_minvar = dict(zip(symbols, w_minvar))
    else:
        w_erc = {s: 0.5 for s in symbols}
        w_minvar = {s: 0.5 for s in symbols}

    print(f"\n  Date: {last_date.date()}, Regime: {regime}")
    print("\n  Method      | VIC      | VCB")
    print("  " + "-" * 30)
    for name, w in [("Softmax", w_softmax), ("ERC", w_erc), ("MinVar", w_minvar)]:
        v1 = w.get("VIC", 0) * 100
        v2 = w.get("VCB", 0) * 100
        print(f"  {name:<11} | {v1:6.1f}% | {v2:6.1f}%")

    out = pd.DataFrame([
        {"method": "Softmax", "VIC": w_softmax.get("VIC"), "VCB": w_softmax.get("VCB")},
        {"method": "ERC", "VIC": w_erc.get("VIC"), "VCB": w_erc.get("VCB")},
        {"method": "MinVar", "VIC": w_minvar.get("VIC"), "VCB": w_minvar.get("VCB")},
    ])
    out.to_csv(OUT_DIR / "vic_vcb_weights.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'vic_vcb_weights.csv'}")


if __name__ == "__main__":
    import sys
    run_validation()
    if "--full" in sys.argv:
        run_robustness_test()
        run_vic_vcb_validation()
