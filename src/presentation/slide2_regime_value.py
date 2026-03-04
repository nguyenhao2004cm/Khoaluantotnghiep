"""
Slide 2 — Regime Value: CVaR, Event Study around regime shifts.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "report_images"
OUT_DIR.mkdir(exist_ok=True)


def compute_cvar(r, alpha=0.05):
    """CVaR (Expected Shortfall) at tail alpha."""
    r = np.asarray(r)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return np.nan
    var = np.percentile(r, alpha * 100)
    tail = r[r <= var]
    return float(np.mean(tail)) if len(tail) > 0 else float(var)


def event_study(portfolio, benchmark, regime, pre=10, post=20):
    """
    Event study around Low → High regime shifts.
    Returns list of (portfolio_cvar - benchmark_cvar) per event.
    """
    regime = regime.reindex(portfolio.index).ffill().bfill()
    shifts = (regime == "HIGH") & (regime.shift(1) == "LOW")
    shift_dates = portfolio.index[shifts]

    event_results = []
    for idx in shift_dates:
        try:
            start = portfolio.index.get_loc(idx) - pre
            end = portfolio.index.get_loc(idx) + post + 1
            if start < 0 or end > len(portfolio):
                continue
            window = slice(start, end)
            p_slice = portfolio.iloc[window]
            b_slice = benchmark.iloc[window]
            p_cvar = compute_cvar(p_slice)
            b_cvar = compute_cvar(b_slice)
            event_results.append(p_cvar - b_cvar)
        except (KeyError, IndexError):
            continue
    return np.array(event_results) if event_results else np.array([])


def plot_event_curve(portfolio, benchmark, regime, pre=10, post=20, out_dir=None):
    """Average portfolio vs benchmark around regime shifts (Low→High)."""
    out_dir = out_dir or OUT_DIR
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Serif"

    regime = regime.reindex(portfolio.index).ffill().bfill()
    shifts = (regime == "HIGH") & (regime.shift(1) == "LOW")
    shift_dates = portfolio.index[shifts]

    if len(shift_dates) == 0:
        plt.figure(figsize=(10, 5))
        plt.title("CVaR around Regime Shift (no Low→High events)")
        plt.savefig(out_dir / "slide2_event.png", dpi=150, bbox_inches="tight")
        plt.close()
        return

    n_points = pre + post + 1
    port_curves = []
    bench_curves = []

    for idx in shift_dates:
        try:
            pos = portfolio.index.get_loc(idx)
            start = max(0, pos - pre)
            end = min(len(portfolio), pos + post + 1)
            p_slice = portfolio.iloc[start:end].values
            b_slice = benchmark.iloc[start:end].values
            if len(p_slice) >= 5 and len(b_slice) >= 5:
                port_curves.append(p_slice)
                bench_curves.append(b_slice)
        except (KeyError, IndexError):
            continue

    if not port_curves:
        plt.figure(figsize=(10, 5))
        plt.title("CVaR around Regime Shift")
        plt.savefig(out_dir / "slide2_event.png", dpi=150, bbox_inches="tight")
        plt.close()
        return

    # Pad to same length
    max_len = max(len(c) for c in port_curves)
    port_padded = [np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in port_curves]
    bench_padded = [np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in bench_curves]

    avg_port = np.nanmean(port_padded, axis=0)
    avg_bench = np.nanmean(bench_padded, axis=0)
    x = np.arange(len(avg_port)) - pre

    plt.figure(figsize=(10, 5))
    plt.plot(x, avg_port, label="Portfolio", color="#3b6edc")
    plt.plot(x, avg_bench, label="Benchmark", color="#888", linestyle="--")
    plt.axvline(0, color="gray", linestyle=":", alpha=0.7)
    plt.title("Returns around Regime Shift (Low → High)")
    plt.xlabel("Days relative to shift")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "slide2_event.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_slide2(portfolio=None, benchmark=None, regime=None, out_dir=None):
    """Generate Slide 2 figures and return stats."""
    from ._data_loader import get_portfolio_benchmark_regime

    out_dir = out_dir or OUT_DIR
    if portfolio is None or benchmark is None or regime is None:
        portfolio, benchmark, regime = get_portfolio_benchmark_regime()

    if portfolio is None or len(portfolio) < 30:
        return {
            "cvar_full_sample": np.nan,
            "cvar_high_regime": np.nan,
            "event_improvement_mean": np.nan,
            "error": "Insufficient data",
        }

    cvar_full = compute_cvar(portfolio)
    high_mask = regime == "HIGH"
    cvar_high = compute_cvar(portfolio[high_mask]) if high_mask.any() else np.nan

    events = event_study(portfolio, benchmark, regime)
    event_improvement = float(np.mean(events)) if len(events) > 0 else np.nan

    plot_event_curve(portfolio, benchmark, regime, out_dir=out_dir)

    return {
        "cvar_full_sample": float(cvar_full),
        "cvar_high_regime": float(cvar_high) if not np.isnan(cvar_high) else None,
        "event_improvement_mean": float(event_improvement) if not np.isnan(event_improvement) else None,
    }
