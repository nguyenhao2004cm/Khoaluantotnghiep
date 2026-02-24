# =====================================================
# FILE: src/portfolio_engine/simple_optimizer.py
# =====================================================
"""
Simplified Portfolio Optimization - Always Works
Dựa trên Ma et al. (2020) nhưng dùng heuristics thay vì SLSQP
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_DIR / "data_processed/prices/prices.csv"


def optimize_inverse_volatility(symbols, lookback_days=252):
    """
    Inverse Volatility Weighting
    Risk thấp → Weight cao
    """
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = df.pivot(index="date", columns="symbol", values="close")
    prices = prices[symbols].dropna().tail(lookback_days)
    
    if len(prices) < 60:
        n = len(symbols)
        return dict(zip(symbols, [1/n]*n))
    
    returns = prices.pct_change().dropna()
    vols = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Inverse volatility
    inv_vol = 1 / (vols + 1e-8)
    weights = inv_vol / inv_vol.sum()
    
    # Apply bounds [5%, 25%]
    weights = np.clip(weights.values, 0.05, 0.25)
    weights = weights / weights.sum()
    
    return dict(zip(symbols, weights))


def optimize_risk_parity(symbols, lookback_days=252, max_iter=100):
    """
    Risk Parity - Mỗi asset đóng góp risk bằng nhau
    """
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = df.pivot(index="date", columns="symbol", values="close")
    prices = prices[symbols].dropna().tail(lookback_days)
    
    if len(prices) < 60:
        n = len(symbols)
        return dict(zip(symbols, [1/n]*n))
    
    returns = prices.pct_change().dropna()
    cov = returns.cov().values * 252  # Annualized
    
    n = len(symbols)
    
    # Start with inverse volatility
    vols = np.sqrt(np.diag(cov))
    weights = (1 / vols) / (1 / vols).sum()
    
    # Iterative reweighting
    for _ in range(max_iter):
        # Portfolio volatility
        port_vol = np.sqrt(weights @ cov @ weights)
        
        # Marginal risk contribution
        mrc = (cov @ weights) / port_vol
        
        # Risk contribution
        rc = weights * mrc
        
        # Target: equal risk contribution
        target_rc = port_vol / n
        
        # Adjust weights
        weights = weights * (target_rc / (rc + 1e-8))
        
        # Normalize
        weights = weights / weights.sum()
        
        # Check convergence
        if np.max(np.abs(rc - target_rc)) < 1e-4:
            break
    
    # Apply bounds
    weights = np.clip(weights, 0.05, 0.25)
    weights = weights / weights.sum()
    
    return dict(zip(symbols, weights))


def optimize_sharpe_heuristic(symbols, lookback_days=252):
    """
    Sharpe-based Heuristic (không dùng scipy.optimize)
    """
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = df.pivot(index="date", columns="symbol", values="close")
    prices = prices[symbols].dropna().tail(lookback_days)
    
    if len(prices) < 60:
        n = len(symbols)
        return dict(zip(symbols, [1/n]*n))
    
    returns = prices.pct_change().dropna()
    
    # Exponential weighted mean (gần đây quan trọng hơn)
    mean_ret = returns.ewm(span=60).mean().iloc[-1] * 252
    std_ret = returns.ewm(span=60).std().iloc[-1] * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (mean_ret - 0.03) / (std_ret + 1e-8)
    
    # Negative Sharpe → zero weight
    sharpe = sharpe.clip(lower=0)
    
    if sharpe.sum() == 0:
        # All negative Sharpe → equal weight
        weights = np.ones(len(symbols)) / len(symbols)
    else:
        # Softmax với temperature
        exp_sharpe = np.exp(sharpe.values / 0.5)
        weights = exp_sharpe / exp_sharpe.sum()
    
    # Apply bounds
    weights = np.clip(weights, 0.05, 0.25)
    weights = weights / weights.sum()
    
    return dict(zip(symbols, weights))


def optimize_minimum_variance_heuristic(symbols, lookback_days=252):
    """
    Minimum Variance Portfolio (heuristic approach)
    """
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = df.pivot(index="date", columns="symbol", values="close")
    prices = prices[symbols].dropna().tail(lookback_days)
    
    if len(prices) < 60:
        n = len(symbols)
        return dict(zip(symbols, [1/n]*n))
    
    returns = prices.pct_change().dropna()
    cov = returns.cov().values * 252
    
    # Inverse covariance approach
    try:
        inv_cov = np.linalg.inv(cov + np.eye(len(cov)) * 1e-5)
        ones = np.ones(len(symbols))
        
        # w = inv(Σ) @ 1 / (1' @ inv(Σ) @ 1)
        weights = inv_cov @ ones
        weights = weights / weights.sum()
        
        # Apply bounds
        weights = np.clip(weights, 0.05, 0.25)
        weights = weights / weights.sum()
        
        return dict(zip(symbols, weights))
    
    except np.linalg.LinAlgError:
        # Fallback to inverse volatility
        return optimize_inverse_volatility(symbols, lookback_days)


def optimize_adaptive(symbols, lookback_days=252):
    """
    Adaptive Method - Tự động chọn method tốt nhất
    """
    methods = {
        "sharpe": optimize_sharpe_heuristic,
        "min_var": optimize_minimum_variance_heuristic,
        "risk_parity": optimize_risk_parity,
        "inv_vol": optimize_inverse_volatility,
    }
    
    results = {}
    
    for name, func in methods.items():
        try:
            weights = func(symbols, lookback_days)
            
            # Evaluate performance
            df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
            prices = df.pivot(index="date", columns="symbol", values="close")
            prices = prices[symbols].dropna().tail(lookback_days)
            returns = prices.pct_change().dropna()
            
            w = np.array([weights[s] for s in symbols])
            port_returns = returns.values @ w
            
            # Sortino ratio (focus on downside)
            excess = port_returns - 0.03/252
            downside = excess[excess < 0]
            
            if len(downside) > 0:
                sortino = excess.mean() / downside.std() * np.sqrt(252)
            else:
                sortino = 10.0  # No downside → very good
            
            results[name] = (sortino, weights)
        
        except Exception as e:
            print(f"⚠️ Method {name} failed: {e}")
            continue
    
    if not results:
        # All failed → equal weight
        n = len(symbols)
        return dict(zip(symbols, [1/n]*n))
    
    # Select best method
    best_method = max(results.items(), key=lambda x: x[1][0])
    best_name, (best_sortino, best_weights) = best_method
    
    print(f"✅ Selected method: {best_name} (Sortino: {best_sortino:.2f})")
    
    return best_weights


# =====================================================
# UPDATED: src/decision/build_portfolio_allocation.py
# =====================================================
"""
REPLACE the Markowitz import with simple_optimizer
"""
import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ✅ CHANGE THIS LINE
from src.portfolio_engine.simple_optimizer import optimize_adaptive

TOP_K = 8
MAX_WEIGHT = 0.25
MIN_WEIGHT = 0.05

BASE_DIR = os.getcwd()
INPUT_FILE = os.path.join(BASE_DIR, "data_processed", "decision", "signal.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_processed", "portfolio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

# Filter low signals
signal_threshold = df.groupby("date")["signal"].transform(lambda x: x.quantile(0.3))
df = df[df["signal"] > signal_threshold].copy()

# Top K
df["rank"] = df.groupby("date")["signal"].rank(ascending=False)
topk = df[df["rank"] <= TOP_K].copy()


def hybrid_allocation(signal_weights, mvo_weights, alpha=0.6):
    """
    Hybrid: AI + Optimization
    """
    symbols = set(signal_weights.keys()) & set(mvo_weights.keys())
    
    hybrid = {}
    for s in symbols:
        hybrid[s] = (
            alpha * signal_weights.get(s, 0) +
            (1 - alpha) * mvo_weights.get(s, 0)
        )
    
    # Normalize
    total = sum(hybrid.values())
    if total > 0:
        hybrid = {k: v/total for k, v in hybrid.items()}
    else:
        # Fallback
        n = len(symbols)
        hybrid = {s: 1/n for s in symbols}
    
    return hybrid


# Process each date
allocation_records = []

for date, group in topk.groupby("date"):
    symbols = group["symbol"].tolist()
    signals = group["signal"].values
    
    # AI weights
    exp_s = np.exp(signals / 2.0)
    ai_weights_raw = exp_s / exp_s.sum()
    ai_weights_raw = np.clip(ai_weights_raw, MIN_WEIGHT, MAX_WEIGHT)
    ai_weights_raw = ai_weights_raw / ai_weights_raw.sum()
    
    ai_weights = dict(zip(symbols, ai_weights_raw))
    
    # Optimization weights
    try:
        mvo_weights = optimize_adaptive(symbols, lookback_days=252)
    except Exception as e:
        print(f"⚠️ Optimization failed on {date}: {e}")
        mvo_weights = {s: 1/len(symbols) for s in symbols}
    
    # Hybrid
    final_weights = hybrid_allocation(ai_weights, mvo_weights, alpha=0.6)
    
    for symbol, weight in final_weights.items():
        allocation_records.append({
            "date": date,
            "symbol": symbol,
            "allocation_weight": weight
        })

out = pd.DataFrame(allocation_records)
out.to_csv(
    os.path.join(OUTPUT_DIR, "portfolio_allocation_final.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("✅ portfolio_allocation_final.csv (HYBRID: 60% AI + 40% ADAPTIVE)")