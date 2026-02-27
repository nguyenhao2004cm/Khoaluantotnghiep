# risk_based_optimizer.py - ERC, MinVar, Ledoit-Wolf cov 60d

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from sklearn.covariance import LedoitWolf
except ImportError:
    LedoitWolf = None

PRICE_FILE = PROJECT_DIR / "data_processed/prices/prices.csv"
ROLLING_WINDOW = 60
TRADING_DAYS = 252


def load_returns(symbols, as_of_date=None, lookback=ROLLING_WINDOW):
    """Load returns matrix: index=date, columns=symbols."""
    df = pd.read_csv(PRICE_FILE)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return None, None
    df = df.sort_values("date").groupby(["date", "symbol"], as_index=False).last()
    prices = df.pivot(index="date", columns="symbol", values="close").sort_index()

    ret = prices.pct_change(fill_method=None).dropna(how="all")
    ret = ret[list(symbols)]

    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date)
        ret = ret[ret.index <= as_of_date].tail(lookback)

    if len(ret) < 30:
        return None, None
    return ret, ret.index


def get_covariance_matrix(returns, use_ledoit_wolf=True):
    """
    Covariance matrix annualized.
    Ledoit-Wolf shrinkage để tránh singular matrix.
    """
    R = returns.values
    n, p = R.shape
    cov_sample = np.cov(R.T) * TRADING_DAYS

    if use_ledoit_wolf and LedoitWolf is not None and n >= p:
        try:
            lw = LedoitWolf().fit(returns)
            cov = lw.covariance_ * TRADING_DAYS
            return cov
        except Exception:
            pass

    # Fallback: add small regularization
    cov = cov_sample + np.eye(p) * 1e-6
    return cov


# =====================================
# MINIMUM VARIANCE PORTFOLIO
# =====================================
def min_variance_weights(cov, min_w=0.0, max_w=1.0):
    """
    min w' Σ w  s.t. sum(w)=1, min_w <= w <= max_w
    """
    n = cov.shape[0]
    bounds = [(min_w, max_w)] * n

    def obj(w):
        return w @ cov @ w

    w0 = np.ones(n) / n
    result = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"maxiter": 500},
    )
    if result.success:
        return result.x
    return w0


# =====================================
# EQUAL RISK CONTRIBUTION (ERC)
# =====================================
def _risk_contributions(w, cov):
    """Marginal risk contribution: (Σw)_i * w_i"""
    sigma_p = np.sqrt(w @ cov @ w) + 1e-12
    mc = cov @ w
    rc = w * mc / sigma_p
    return rc


def erc_weights(cov, min_w=0.0, max_w=1.0, tol=1e-6, max_iter=200):
    """
    Equal Risk Contribution: mỗi asset đóng góp risk bằng nhau.
    n=2: closed form w1 = sigma2/(sigma1+sigma2)
    n>2: iterative optimization
    """
    n = cov.shape[0]
    # Closed form cho 2 assets (khong clip de tranh ep 50-50)
    if n == 2:
        sig1 = np.sqrt(cov[0, 0])
        sig2 = np.sqrt(cov[1, 1])
        if sig1 + sig2 > 1e-12:
            w1 = sig2 / (sig1 + sig2)
            w2 = 1 - w1
            w = np.array([w1, w2])
            w = np.clip(w, 0.01, 0.99)  # Tranh 0, chi gioi han nhe
            w /= w.sum()
            return w
    bounds = [(min_w, max_w)] * n

    def obj(w):
        rc = _risk_contributions(w, cov)
        target = 1.0 / n
        return np.sum((rc - target) ** 2)

    w0 = np.ones(n) / n
    result = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"maxiter": max_iter},
    )
    if result.success:
        return result.x
    return w0


# =====================================
# TOP-K SELECTION + OPTIMIZATION
# =====================================
def optimize_allocation(
    symbols,
    signals,
    method="erc",
    as_of_date=None,
    top_k=8,
    min_weight=0.05,
    max_weight=0.25,
):
    """
    Chọn top-K theo signal, sau đó tối ưu weight theo risk structure.

    method: "erc" | "minvar"
    """
    # Chọn top-K theo signal
    idx = np.argsort(signals)[::-1][:top_k]
    selected = [symbols[i] for i in idx]

    ret, _ = load_returns(selected, as_of_date)
    if ret is None or len(ret) < 30:
        # Fallback: equal weight
        n = len(selected)
        return dict(zip(selected, np.ones(n) / n))

    cov = get_covariance_matrix(ret)

    if method == "minvar":
        w = min_variance_weights(cov, min_w=min_weight, max_w=max_weight)
    else:
        w = erc_weights(cov, min_w=min_weight, max_w=max_weight)

    return dict(zip(selected, w))


# =====================================
# REGIME-AWARE WRAPPER
# =====================================
def allocate_risk_based(
    symbols,
    signals,
    regime,
    method="erc",
    as_of_date=None,
    risk_profile="balanced",
):
    """
    Regime-aware allocation dùng ERC/MinVar thay softmax.

    HIGH regime: ERC + scale exposure (max 60% equity)
    LOW/NORMAL: ERC hoặc MinVar với full exposure
    """
    if regime == "HIGH":
        top_k = min(4, len(symbols))
        max_equity = 0.6
        min_w, max_w = 0.05, 0.25
    elif regime == "LOW":
        top_k = min(10, len(symbols))
        max_equity = 1.0
        min_w, max_w = 0.02, 0.30
    else:
        top_k = min(8, len(symbols))
        max_equity = 1.0
        min_w, max_w = 0.05, 0.25

    weights = optimize_allocation(
        symbols, signals, method=method, as_of_date=as_of_date,
        top_k=top_k, min_weight=min_w, max_weight=max_w,
    )

    w_arr = np.array(list(weights.values()))
    w_arr *= max_equity
    w_arr /= w_arr.sum()

    # Risk profile scaling
    if risk_profile == "conservative":
        w_arr *= 0.8
    elif risk_profile == "aggressive":
        w_arr *= 1.2
    w_arr /= w_arr.sum()

    return dict(zip(weights.keys(), w_arr))


# =====================================
# TEST
# =====================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" RISK-BASED OPTIMIZER TEST")
    print("=" * 60)

    test_symbols = ["VIC", "VCB", "FPT", "VNM", "HPG"]
    test_signals = np.array([0.5, 0.6, 0.7, 0.4, 0.8])  # FPT, VNM cao; VIC, VCB thấp

    for method in ["erc", "minvar"]:
        w = optimize_allocation(test_symbols, test_signals, method=method)
        print(f"\n{method.upper()} weights:")
        for s, v in sorted(w.items(), key=lambda x: -x[1]):
            print(f"  {s}: {v*100:.2f}%")
