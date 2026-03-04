"""
Slide 3 — Performance & Drawdown: Sharpe, CAGR, Max Drawdown.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "report_images"
OUT_DIR.mkdir(exist_ok=True)
TRADING_DAYS = 252


def sharpe_ratio(r):
    """Annualized Sharpe ratio."""
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def cagr(r):
    """Compound annual growth rate."""
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    cumulative = (1 + r).prod()
    years = len(r) / TRADING_DAYS
    return float(cumulative ** (1 / years) - 1) if years > 0 else np.nan


def max_drawdown(r):
    """Maximum drawdown (negative = loss)."""
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    cumulative = (1 + r).cumprod()
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    return float(dd.min())


def plot_drawdown(r_port, r_bench, out_dir=None):
    """Drawdown comparison: portfolio vs benchmark."""
    out_dir = out_dir or OUT_DIR
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Serif"

    common = r_port.index.intersection(r_bench.index)
    r_port = r_port.loc[common].dropna()
    r_bench = r_bench.loc[common].dropna()
    common = r_port.index.intersection(r_bench.index)
    r_port = r_port.loc[common]
    r_bench = r_bench.loc[common]

    if len(r_port) < 2:
        plt.figure(figsize=(10, 5))
        plt.title("Drawdown Comparison")
        plt.savefig(out_dir / "slide3_drawdown.png", dpi=150, bbox_inches="tight")
        plt.close()
        return

    cum_port = (1 + r_port).cumprod()
    cum_bench = (1 + r_bench).cumprod()
    peak_port = cum_port.cummax()
    peak_bench = cum_bench.cummax()
    dd_port = (cum_port - peak_port) / peak_port
    dd_bench = (cum_bench - peak_bench) / peak_bench

    plt.figure(figsize=(12, 5))
    plt.fill_between(dd_port.index, dd_port.values * 100, 0, color="#3b6edc", alpha=0.3, label="Portfolio")
    plt.plot(dd_port.index, dd_port.values * 100, color="#3b6edc", linewidth=1.5)
    plt.plot(dd_bench.index, dd_bench.values * 100, color="#888", linestyle="--", label="Benchmark")
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    plt.tight_layout()
    plt.savefig(out_dir / "slide3_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_slide3(portfolio=None, benchmark=None, out_dir=None):
    """Generate Slide 3 figures and return metrics."""
    from ._data_loader import get_portfolio_benchmark_regime

    out_dir = out_dir or OUT_DIR
    if portfolio is None or benchmark is None:
        portfolio, benchmark, _ = get_portfolio_benchmark_regime()

    if portfolio is None or len(portfolio) < 30:
        return {
            "sharpe": np.nan,
            "cagr": np.nan,
            "max_drawdown": np.nan,
            "error": "Insufficient data",
        }

    sharpe = sharpe_ratio(portfolio)
    cagr_val = cagr(portfolio)
    mdd = max_drawdown(portfolio)

    plot_drawdown(portfolio, benchmark, out_dir=out_dir)

    return {
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "cagr": float(cagr_val) if not np.isnan(cagr_val) else None,
        "max_drawdown": float(mdd) if not np.isnan(mdd) else None,
    }
