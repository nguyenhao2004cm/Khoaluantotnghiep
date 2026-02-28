# =====================================================
# FILE: src/portfolio_engine/efficient_frontier.py
# =====================================================
"""
Efficient Frontier (REFERENCE ONLY)

IMPORTANT:
----------
This module is used STRICTLY for ex-post validation and visualization.

• Portfolio allocation is NOT optimized using Markowitz.
• Allocation weights are produced by an AI-driven, regime-aware system.
• The efficient frontier is plotted solely as a BENCHMARK.

Academic positioning:
---------------------
Markowitz (1952) – Modern Portfolio Theory
Used here only as a geometric reference, NOT as an optimization engine.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
from typing import Optional

# =====================================================
# CONFIG
# =====================================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_DIR / "data_processed" / "prices" / "prices.csv"
OUT_DIR = PROJECT_DIR / "Reports"
OUT_DIR.mkdir(exist_ok=True)


# =====================================================
# CORE: COMPUTE EFFICIENT FRONTIER (REFERENCE)
# =====================================================
def compute_efficient_frontier(
    symbols: list[str],
    lookback_days: int = 252,
    n_points: int = 40,
):
    """
    Compute Markowitz efficient frontier for a fixed asset universe.

    Parameters
    ----------
    symbols : list[str]
        Asset universe (same universe used by AI portfolio)
    lookback_days : int
        Historical window for estimating mean & covariance
    n_points : int
        Number of frontier points

    Returns
    -------
    frontier_vol : list[float]
    frontier_ret : list[float]
    """

    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])

    prices = (
        df.pivot(index="date", columns="symbol", values="close")
        .sort_index()
    )

    prices = prices[symbols].dropna().tail(lookback_days)

    if prices.shape[0] < 60:
        raise ValueError("Not enough data to compute efficient frontier")

    returns = prices.pct_change().dropna()

    mu = returns.mean().values * 252
    cov = returns.cov().values * 252

    n = len(symbols)

    # Portfolio functions
    def port_vol(w):
        return np.sqrt(w.T @ cov @ w)

    def port_ret(w):
        return w @ mu

    # Target return grid
    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    frontier_vol = []
    frontier_ret = []

    for r in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, r=r: w @ mu - r},
        )

        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        res = minimize(
            port_vol,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "disp": False},
        )

        if res.success:
            frontier_vol.append(res.fun)
            frontier_ret.append(r)

    return frontier_vol, frontier_ret


# =====================================================
# VISUALIZATION: FRONTIER + AI PORTFOLIO POSITION
# =====================================================
def plot_efficient_frontier_reference(
    symbols: list[str],
    portfolio_weights: dict[str, float],
    lookback_days: int = 252,
    risk_free_rate: float = 0.03,
    out_path: Optional[Path] = None,
):
    """
    Plot Efficient Frontier with AI-driven portfolio position.

    NOTE:
    -----
    This plot is for interpretation and reporting only.

    out_path: Nếu có, lưu vào đây (để PDF tìm được). Mặc định: Reports/efficient_frontier_reference.png
    """

    # ---------- Frontier ----------
    frontier_vol, frontier_ret = compute_efficient_frontier(
        symbols=symbols,
        lookback_days=lookback_days,
    )

    # ---------- Portfolio stats ----------
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])

    prices = (
        df.pivot(index="date", columns="symbol", values="close")
        .sort_index()
    )

    prices = prices[symbols].dropna().tail(lookback_days)
    returns = prices.pct_change().dropna()

    mu = returns.mean().values * 252
    cov = returns.cov().values * 252

    w = np.array([portfolio_weights.get(s, 0.0) for s in symbols])

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))

    sharpe = (
        (port_ret - risk_free_rate) / port_vol
        if port_vol > 1e-8 else np.nan
    )

    # ---------- Plot ----------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    plt.plot(
        frontier_vol,
        frontier_ret,
        linewidth=2.5,
        color="#3b6edc",
        label="Efficient Frontier (Reference)",
        zorder=2,
    )

    plt.scatter(
        port_vol,
        port_ret,
        marker="*",
        s=320,
        color="#FF6B6B",
        edgecolors="black",
        linewidths=1.4,
        label="AI-driven Portfolio",
        zorder=5,
    )

    plt.xlabel("Risk (Annualized Volatility)", fontsize=12)
    plt.ylabel("Expected Return (Annualized)", fontsize=12)

    plt.title(
        "Efficient Frontier (Reference Only)\n"
        "AI-driven Portfolio Position",
        fontsize=14,
    )

    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Percentage format
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%")
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%")
    )

    plt.tight_layout()

    save_path = out_path if out_path is not None else OUT_DIR / "efficient_frontier_reference.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "portfolio_return": port_ret,
        "portfolio_volatility": port_vol,
        "portfolio_sharpe": sharpe,
    }
