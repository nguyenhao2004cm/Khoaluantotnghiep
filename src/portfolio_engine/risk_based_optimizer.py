# risk_based_optimizer.py - ERC, MinVar, Ledoit-Wolf
# Regime-specific covariance: Σ_LOW ≠ Σ_HIGH

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
REGIME_FILE = PROJECT_DIR / "data_processed/reporting/risk_regime_timeseries.csv"
ROLLING_WINDOW = 120  # 60 quá ngắn → covariance unstable → shrinkage mạnh → gần equal
ROLLING_WINDOW_EXTENDED = 180
MIN_OBS_REGIME = 30  # Tối thiểu quan sát cho covariance theo regime
TRADING_DAYS = 252

_REGIME_CACHE = None


def _load_regime_series():
    """Load portfolio_risk_regime (LOW/NORMAL/HIGH) theo ngày."""
    global _REGIME_CACHE
    if _REGIME_CACHE is not None:
        return _REGIME_CACHE
    if not REGIME_FILE.exists():
        _REGIME_CACHE = pd.Series(dtype=str)
        return _REGIME_CACHE
    df = pd.read_csv(REGIME_FILE, parse_dates=["date"])
    _REGIME_CACHE = df.set_index("date")["portfolio_risk_regime"].str.upper()
    return _REGIME_CACHE


def load_returns(symbols, as_of_date=None, lookback=None):
    """Load returns matrix: index=date, columns=symbols."""
    if lookback is None:
        lookback = ROLLING_WINDOW_EXTENDED if len(symbols) <= 8 else ROLLING_WINDOW
    df = pd.read_csv(PRICE_FILE)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return None, None
    df = df.sort_values("date").groupby(["date", "symbol"], as_index=False).last()
    prices = df.pivot(index="date", columns="symbol", values="close").sort_index()

    ret = prices.pct_change(fill_method=None).dropna(how="all")
    # Chỉ giữ symbols có dữ liệu (tránh KeyError khi 1 mã thiếu giá)
    avail = [c for c in symbols if c in ret.columns]
    if len(avail) < 2:
        return None, None
    ret = ret[avail]

    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date)
        ret = ret[ret.index <= as_of_date].tail(lookback)

    if len(ret) < 30:
        return None, None
    return ret, ret.index


def load_returns_by_regime(symbols, as_of_date, regime, lookback=None):
    """
    Load returns CHỈ từ các ngày thuộc regime (LOW/HIGH/NORMAL).
    Σ_regime = Cov(R_regime) — cấu trúc rủi ro theo regime.
    """
    if lookback is None:
        lookback = ROLLING_WINDOW_EXTENDED if len(symbols) <= 8 else ROLLING_WINDOW

    ret, _ = load_returns(symbols, as_of_date=None, lookback=252 * 3)  # Lấy 3 năm
    if ret is None or len(ret) < MIN_OBS_REGIME:
        return load_returns(symbols, as_of_date, lookback)[0]

    regime_series = _load_regime_series()
    if regime_series.empty:
        ret = ret[ret.index <= pd.Timestamp(as_of_date)].tail(lookback)
        return ret if len(ret) >= MIN_OBS_REGIME else None

    as_of_date = pd.Timestamp(as_of_date)
    ret = ret[ret.index <= as_of_date]
    common = ret.index.intersection(regime_series.index)
    if len(common) == 0:
        return ret.tail(lookback) if len(ret) >= MIN_OBS_REGIME else None

    mask = regime_series.reindex(common).fillna("NORMAL") == regime
    regime_dates = common[mask]
    ret_regime = ret.loc[ret.index.isin(regime_dates)].sort_index().tail(lookback)

    if len(ret_regime) < MIN_OBS_REGIME:
        # HIGH/LOW ít quan sát: dùng expanding (tất cả ngày regime có)
        ret_regime = ret.loc[ret.index.isin(regime_dates)].sort_index()
        if len(ret_regime) < 20:
            return load_returns(symbols, as_of_date, lookback)[0]  # Fallback full sample
    return ret_regime if len(ret_regime) >= MIN_OBS_REGIME else None


def _log_regime_diagnostics(cov, returns, regime, symbols):
    """Log phân hóa Σ theo regime (REGIME_COV_DEBUG=1)."""
    try:
        n = cov.shape[0]
        corr = np.corrcoef(returns.values.T) if len(returns) > 1 else np.eye(n)
        avg_corr = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0
        eigvals = np.linalg.eigvalsh(cov)
        eig_disp = np.std(eigvals) / (np.mean(eigvals) + 1e-12)
        vol = np.sqrt(np.diag(cov))
        print(f"  [REGIME={regime}] avg_corr={avg_corr:.4f} eig_disp={eig_disp:.4f} vol_mean={vol.mean():.4f}")
    except Exception:
        pass


def get_covariance_matrix(returns, use_ledoit_wolf=None):
    """
    Covariance matrix annualized.
    use_ledoit_wolf: True=shrink (mượt, dễ equal), False=sample cov (dispersion cao hơn).
    Mặc định False (USE_LEDOIT_WOLF=1 để bật LW).
    """
    if use_ledoit_wolf is None:
        use_ledoit_wolf = __import__("os").environ.get("USE_LEDOIT_WOLF", "0") == "1"
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
# BOUNDS HELPER (danh mục 2–3 cổ phiếu)
# =====================================
def _feasible_bounds(n, min_w, max_w):
    """
    Với n nhỏ (2–3), min_w/max_w mặc định có thể không khả thi (sum=1).
    VD: n=2, min_w=0.05, max_w=0.25 → không thể có w1+w2=1.
    """
    if n <= 1:
        return min_w, max_w
    # Cần 1/n nằm trong [min_w, max_w] để equal-weight khả thi
    lo = 1.0 / n
    min_ok = min(min_w, lo * 0.5)   # cho phép ít nhất 50% equal-weight
    max_ok = max(max_w, lo * 1.5)   # cho phép nhiều nhất 1.5x equal-weight
    return min_ok, max_ok


# =====================================
# MINIMUM VARIANCE PORTFOLIO
# =====================================
def min_variance_weights(cov, min_w=0.0, max_w=1.0):
    """
    min w' Σ w  s.t. sum(w)=1, min_w <= w <= max_w
    """
    n = cov.shape[0]
    min_eff, max_eff = _feasible_bounds(n, min_w, max_w)
    bounds = [(min_eff, max_eff)] * n

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
    # Closed form cho 2 assets: w1 = sigma2/(sigma1+sigma2) — risk parity chính xác
    if n == 2:
        sig1 = np.sqrt(cov[0, 0])
        sig2 = np.sqrt(cov[1, 1])
        if sig1 + sig2 > 1e-12:
            w1 = sig2 / (sig1 + sig2)
            w2 = 1 - w1
            w = np.array([w1, w2])
            w = np.clip(w, 0.01, 0.99)  # tránh 0, giới hạn nhẹ
            w /= w.sum()
            return w
    # n>=3: cần bounds khả thi (n=3 với min_w=0.05, max_w=0.25 không feasible)
    min_eff, max_eff = _feasible_bounds(n, min_w, max_w)
    bounds = [(min_eff, max_eff)] * n

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
SIGNAL_BLEND = 0.2  # Giữ ERC làm core, signal chỉ điều chỉnh nhẹ (0.5 → gần equal)


def optimize_allocation(
    symbols,
    signals,
    method="erc",
    as_of_date=None,
    top_k=8,
    min_weight=0.05,
    max_weight=0.25,
    signal_blend=SIGNAL_BLEND,
    regime=None,
):
    """
    Chọn top-K theo signal, tối ưu weight theo risk structure.
    regime: LOW/HIGH/NORMAL — dùng Σ_regime (covariance theo regime).
    """
    # Chọn top-K theo signal
    idx = np.argsort(signals)[::-1][:top_k]
    selected = [symbols[i] for i in idx]
    signals_selected = signals[idx]

    # Regime-specific covariance: Σ_LOW ≠ Σ_HIGH
    if regime and as_of_date and regime in ("LOW", "HIGH", "NORMAL"):
        ret = load_returns_by_regime(selected, as_of_date, regime)
    else:
        ret, _ = load_returns(selected, as_of_date)
    if ret is None or len(ret) < 30:
        # Fallback: signal-proportional (không equal weight)
        exp_s = np.exp(np.clip(signals_selected, -5, 5))
        w = exp_s / exp_s.sum()
        return dict(zip(selected, w))
    # Thứ tự phải khớp ret.columns (cov matrix)
    avail = [s for s in selected if s in ret.columns]
    if len(avail) < 2:
        n = len(avail)
        return dict(zip(avail, np.ones(n) / n))
    avail = list(ret.columns)  # Đảm bảo khớp cov
    signals_avail = np.array([signals[symbols.index(s)] for s in avail])

    cov = get_covariance_matrix(ret)

    # Regime diagnostics (bật với REGIME_COV_DEBUG=1)
    if regime and __import__("os").environ.get("REGIME_COV_DEBUG") == "1":
        _log_regime_diagnostics(cov, ret, regime, avail)

    if method == "minvar":
        w_risk = min_variance_weights(cov, min_w=min_weight, max_w=max_weight)
    else:
        w_risk = erc_weights(cov, min_w=min_weight, max_w=max_weight)

    # Blend với signal: tránh phân bổ đều, signal cao → weight cao
    exp_s = np.exp(np.clip(signals_avail, -5, 5))
    w_signal = exp_s / exp_s.sum()
    alpha = max(0.0, min(1.0, signal_blend))
    w = (1 - alpha) * w_risk + alpha * w_signal
    w = np.clip(w, 0.01, 0.99)
    w /= w.sum()

    return dict(zip(avail, w))


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
        regime=regime,
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
