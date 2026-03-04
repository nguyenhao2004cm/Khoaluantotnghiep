"""
Slide 1 — Stylized Facts: Distribution, QQ plot, Volatility Clustering.
"""

from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "report_images"
OUT_DIR.mkdir(exist_ok=True)


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["figure.dpi"] = 120
    return plt


def plot_distribution(returns, out_dir=None):
    """Histogram + QQ plot, return Jarque-Bera stats."""
    out_dir = out_dir or OUT_DIR
    plt = _ensure_matplotlib()
    import seaborn as sns
    import scipy.stats as stats
    from statsmodels.stats.stattools import jarque_bera

    returns = returns.dropna()
    if len(returns) < 10:
        return {"jb_stat": np.nan, "jb_p": np.nan}

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(returns, kde=True, ax=ax[0])
    ax[0].set_title("Return Distribution")

    stats.probplot(returns, dist="norm", plot=ax[1])
    ax[1].set_title("QQ Plot")

    plt.tight_layout()
    plt.savefig(out_dir / "slide1_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    jb_stat, jb_p, _, _ = jarque_bera(returns)
    return {"jb_stat": float(jb_stat), "jb_p": float(jb_p)}


def plot_volatility_clustering(returns, window=30, out_dir=None):
    """Rolling volatility plot."""
    out_dir = out_dir or OUT_DIR
    plt = _ensure_matplotlib()

    rolling_vol = returns.dropna().rolling(window).std()
    rolling_vol = rolling_vol.dropna()
    if len(rolling_vol) < 5:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_vol.index, rolling_vol.values)
    plt.title(f"Rolling Volatility ({window}d)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(out_dir / "slide1_volatility.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_slide1(returns=None, out_dir=None):
    """Generate Slide 1 figures and return stats."""
    from ._data_loader import get_market_returns

    out_dir = out_dir or OUT_DIR
    returns = returns if returns is not None else get_market_returns()
    if returns is None or len(returns) < 10:
        return {"jb_stat": np.nan, "jb_p": np.nan, "error": "Insufficient return data"}

    stats = plot_distribution(returns, out_dir)
    plot_volatility_clustering(returns, out_dir=out_dir)
    return stats
