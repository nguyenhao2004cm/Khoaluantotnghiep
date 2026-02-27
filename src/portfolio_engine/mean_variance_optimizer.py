# mean_variance_optimizer.py - MVO benchmark (standalone test)

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_DIR / "data_processed/prices/prices.csv"

def get_available_symbols():
    """
    Get list of symbols that actually exist in cleaned price data
    """
    try:
        symbols_file = PROJECT_DIR / "data_processed/prices/symbols.txt"
        if symbols_file.exists():
            with open(symbols_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
    except:
        pass
    
    # Fallback: read from CSV
    df = pd.read_csv(PRICE_FILE)
    return sorted(df['symbol'].unique().tolist())


def optimize_portfolio_weights(symbols, lookback_days=252):
    """
    Mean-Variance Optimization (Markowitz) với constraint
    
    ✅ FIXED: Automatically filter out symbols not in cleaned data
    """
    # Load prices
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = df.pivot(index="date", columns="symbol", values="close")
    
    # ✅ FIX: Only use symbols that exist in price data
    available_symbols = [s for s in symbols if s in prices.columns]
    
    if len(available_symbols) == 0:
        print("⚠️ No valid symbols found in price data")
        return {}
    
    if len(available_symbols) < len(symbols):
        removed = set(symbols) - set(available_symbols)
        print(f"⚠️ Removed symbols (not in cleaned data): {removed}")
        symbols = available_symbols
    
    prices = prices[symbols].dropna()
    
    if len(prices) < 60:
        print(f"⚠️ Insufficient data ({len(prices)} days)")
        return {s: 1/len(symbols) for s in symbols}
    
    # Lookback window
    prices = prices.tail(lookback_days)
    
    # Returns
    returns = prices.pct_change().dropna()
    
    # Expected return & covariance
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    
    # Add regularization
    cov = cov + np.eye(len(cov)) * 1e-8
    
    n = len(symbols)
    
    # ✅ OBJECTIVE: Maximize Sharpe Ratio
    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w) + 1e-8
        sharpe = (port_return - 0.03) / port_vol
        return -sharpe
    
    # ✅ CONSTRAINTS
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]
    
    bounds = [(0.05, 0.25)] * n
    w0 = np.ones(n) / n
    
    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300}
    )
    
    if result.success:
        return dict(zip(symbols, result.x))
    else:
        print(f"⚠️ Optimization failed: {result.message}")
        return dict(zip(symbols, [1/n]*n))


def hybrid_allocation(signal_weights, mvo_weights, alpha=0.7):
    """
    Kết hợp AI signal + Mean-Variance Optimization
    
    alpha: trọng số AI signal (0.7 = 70% AI, 30% MVO)
    """
    symbols = set(signal_weights.keys()) & set(mvo_weights.keys())
    
    if not symbols:
        print("⚠️ No common symbols between signal and MVO weights")
        return signal_weights
    
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
        hybrid = {s: 1/len(symbols) for s in symbols}
    
    return hybrid


# =====================================================
# USAGE EXAMPLE (FIXED)
# =====================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MEAN-VARIANCE OPTIMIZER TEST")
    print("="*60 + "\n")
    
    # ✅ FIX: Get actual available symbols from cleaned data
    available = get_available_symbols()
    
    print(f"Total available symbols: {len(available)}")
    
    # ✅ SMART: Use symbols from latest allocation if available
    try:
        alloc_file = PROJECT_DIR / "data_processed/portfolio/portfolio_allocation_final.csv"
        if alloc_file.exists():
            alloc_df = pd.read_csv(alloc_file)
            # Get most recent allocation
            latest_date = alloc_df['date'].max()
            latest_symbols = alloc_df[alloc_df['date'] == latest_date]['symbol'].unique().tolist()
            
            # Filter to available only
            test_symbols = [s for s in latest_symbols if s in available]
            
            if len(test_symbols) >= 5:
                print(f"\nUsing symbols from latest allocation ({latest_date}):")
                print(f"   {test_symbols}\n")
            else:
                # Fallback to first 5 available
                test_symbols = available[:5] if len(available) >= 5 else available
                print(f"\nUsing first {len(test_symbols)} available symbols:")
                print(f"   {test_symbols}\n")
        else:
            # No allocation file - use first N available
            test_symbols = available[:5] if len(available) >= 5 else available
            print(f"\nUsing first {len(test_symbols)} available symbols:")
            print(f"   {test_symbols}\n")
    
    except Exception as e:
        print(f"⚠️ Could not load allocation: {e}")
        test_symbols = available[:5] if len(available) >= 5 else available
        print(f"\nUsing first {len(test_symbols)} available symbols:")
        print(f"   {test_symbols}\n")
    
    if len(test_symbols) < 2:
        print("❌ Need at least 2 symbols for MVO")
        exit(1)
    
    # Test MVO optimizer
    optimal_weights = optimize_portfolio_weights(test_symbols)
    
    if optimal_weights:
        print("✅ OPTIMIZED WEIGHTS (Mean-Variance):")
        for symbol, weight in sorted(optimal_weights.items(), key=lambda x: -x[1]):
            print(f"   {symbol}: {weight*100:.2f}%")
        
        # Test hybrid approach
        signal_weights = {s: 1/len(test_symbols) for s in test_symbols}
        
        hybrid = hybrid_allocation(signal_weights, optimal_weights, alpha=0.6)
        
        print("\n✅ HYBRID WEIGHTS (60% AI + 40% MVO):")
        for symbol, weight in sorted(hybrid.items(), key=lambda x: -x[1]):
            print(f"   {symbol}: {weight*100:.2f}%")
        
        # Validation
        total = sum(hybrid.values())
        print(f"\n✅ Weight sum: {total:.6f}")
        
        if abs(total - 1.0) < 0.01:
            print("✅ Weights validated successfully")
        else:
            print("⚠️ Warning: Weights do not sum to 1")
    
    else:
        print("❌ Optimization failed")
    
    print("\n" + "="*60)
    print("NOTE: This is a reference implementation.")
    print("Main portfolio uses Pure AI allocation (no MVO).")
    print("="*60 + "\n")