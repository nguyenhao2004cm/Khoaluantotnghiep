"""
Shared data loader for presentation slides.
Loads portfolio returns, benchmark returns, regime from existing pipeline.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_DIR / "data_processed" / "prices" / "prices.csv"
ALLOCATION_FILE = PROJECT_DIR / "data_processed" / "portfolio" / "portfolio_allocation_final.csv"
POWERBI_DIR = PROJECT_DIR / "data_processed" / "powerbi"
REGIME_FILE = PROJECT_DIR / "data_processed" / "reporting" / "risk_regime_timeseries.csv"


def get_portfolio_returns():
    """Portfolio returns from powerbi timeseries (regime-aware allocation)."""
    path = POWERBI_DIR / "portfolio_timeseries.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    ret = df.set_index("date")["portfolio_return"].dropna()
    return ret


def get_benchmark_returns():
    """Equal-weight benchmark returns (same universe as portfolio)."""
    if not ALLOCATION_FILE.exists() or not PRICE_FILE.exists():
        return None
    alloc = pd.read_csv(ALLOCATION_FILE)
    alloc["symbol"] = alloc["symbol"].str.strip().str.upper()
    symbols = alloc["symbol"].unique().tolist()

    df = pd.read_csv(PRICE_FILE)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["symbol"].isin(symbols)].dropna(subset=["date", "symbol", "close"])
    if df.empty:
        return None
    prices = df.sort_values("date").groupby(["date", "symbol"], as_index=False).last()
    prices = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    ret = prices.pct_change(fill_method=None).dropna(how="all")
    n = len(symbols)
    eq_weights = 1.0 / n
    benchmark = ret.mul(eq_weights).sum(axis=1)
    return benchmark


def get_regime_series():
    """Portfolio risk regime (LOW/NORMAL/HIGH) by date."""
    if not REGIME_FILE.exists():
        return None
    df = pd.read_csv(REGIME_FILE, parse_dates=["date"])
    return df.set_index("date")["portfolio_risk_regime"].str.upper()


def get_market_returns():
    """Returns for stylized facts (Slide 1). Use portfolio returns as main series."""
    return get_portfolio_returns()


def get_portfolio_benchmark_regime():
    """Aligned portfolio, benchmark, regime on common dates."""
    port = get_portfolio_returns()
    bench = get_benchmark_returns()
    regime = get_regime_series()
    if port is None or bench is None:
        return None, None, None
    common = port.index.intersection(bench.index)
    if regime is not None:
        common = common.intersection(regime.index)
    if len(common) < 30:
        return None, None, None
    port_a = port.loc[common].dropna()
    bench_a = bench.loc[common].dropna()
    common = port_a.index.intersection(bench_a.index)
    port_a = port_a.loc[common]
    bench_a = bench_a.loc[common]
    regime_a = regime.loc[common] if regime is not None else None
    return port_a, bench_a, regime_a
