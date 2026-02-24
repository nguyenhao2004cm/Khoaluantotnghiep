# =====================================================
# FILE: src/portfolio_engine/markowitz_optimizer.py (FIXED)
# =====================================================
"""
Markowitz Mean-Variance Optimization
Fixed: Multiple solver support + fallback
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import cvxpy as cp

PROJECT_DIR = Path(__file__).resolve().parents[2]
PRICE_FILE = PROJECT_DIR / "data_processed/prices/prices.csv"


def compute_expected_returns(prices_df, method="mean"):
    """
    Tính expected return - Ma et al. Equation (3)
    """
    returns = prices_df.pct_change().dropna()
    
    if method == "mean":
        return returns.mean() * 252
    elif method == "exponential":
        return returns.ewm(span=60).mean().iloc[-1] * 252
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_covariance_matrix(prices_df, method="sample"):
    """
    Tính covariance matrix - Ma et al. Section 2.4
    """
    returns = prices_df.pct_change().dropna()
    
    if method == "sample":
        return returns.cov() * 252
    elif method == "exponential":
        return returns.ewm(span=60).cov() * 252
    else:
        raise ValueError(f"Unknown method: {method}")



def max_sharpe_portfolio(
    symbols,
    lookback_days=252,
    risk_free_rate=0.03
):
    """
    Fixed version with better error handling
    """
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = (
        df.pivot(index="date", columns="symbol", values="close")
          .loc[:, symbols]
          .dropna()
          .tail(lookback_days)
    )

    if len(prices) < 60:
        raise ValueError("❌ Not enough price data")

    returns = prices.pct_change().dropna()
    
    # Remove inf/nan
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    # Expected return & Covariance
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = len(symbols)
    
    # ✅ CHECK FOR ZERO VOLATILITY
    vols = np.sqrt(np.diag(cov))
    
    if np.any(vols < 1e-8):
        print("⚠️ Warning: Removing zero-volatility assets")
        valid_idx = vols >= 1e-8
        
        if np.sum(valid_idx) == 0:
            raise ValueError("❌ All assets have zero volatility")
        
        symbols = [s for i, s in enumerate(symbols) if valid_idx[i]]
        mu = mu[valid_idx]
        cov = cov[np.ix_(valid_idx, valid_idx)]
        vols = np.sqrt(np.diag(cov))
        n = len(symbols)

    # Try CVXPY first (silent)
    import cvxpy as cp
    
    try:
        w = cp.Variable(n)
        
        port_return = mu @ w
        port_variance = cp.quad_form(w, cov)
        port_vol = cp.sqrt(port_variance)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0.05,
            w <= 0.25
        ]
        
        problem = cp.Problem(
            cp.Maximize((port_return - risk_free_rate) / port_vol),
            constraints
        )
        
        # Try available solvers
        for solver in [cp.SCS, cp.OSQP, cp.CVXOPT]:
            try:
                problem.solve(solver=solver, verbose=False)
                if w.value is not None:
                    w_opt = w.value
                    mu_p = mu @ w_opt
                    sigma_p = np.sqrt(w_opt @ cov @ w_opt)
                    sharpe_p = (mu_p - risk_free_rate) / sigma_p
                    
                    return {
                        "weights": dict(zip(symbols, w_opt)),
                        "expected_return": mu_p,
                        "volatility": sigma_p,
                        "sharpe": sharpe_p
                    }
            except:
                continue
    
    except:
        pass
    
    # Scipy fallback
    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-8:
            return 1e10
        sharpe = (port_return - risk_free_rate) / port_vol
        return -sharpe
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.05, 0.25)] * n
    
    # ✅ SAFE INITIAL GUESS
    w0 = np.ones(n) / n  # Start with equal weights
    
    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-6}
    )
    
    if result.success:
        w_opt = result.x
        mu_p = mu @ w_opt
        sigma_p = np.sqrt(w_opt @ cov @ w_opt)
        sharpe_p = (mu_p - risk_free_rate) / sigma_p
        
        return {
            "weights": dict(zip(symbols, w_opt)),
            "expected_return": mu_p,
            "volatility": sigma_p,
            "sharpe": sharpe_p
        }
    
    # ✅ FINAL FALLBACK: Minimum variance
    print("⚠️ Scipy failed, using minimum variance...")
    
    try:
        inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-5)
        ones = np.ones(n)
        w_final = inv_cov @ ones
        w_final = w_final / w_final.sum()
        w_final = np.clip(w_final, 0.05, 0.25)
        w_final = w_final / w_final.sum()
    except:
        # Ultimate fallback: equal weights
        w_final = np.ones(n) / n
    
    mu_p = mu @ w_final
    sigma_p = np.sqrt(w_final @ cov @ w_final)
    sharpe_p = (mu_p - risk_free_rate) / sigma_p
    
    return {
        "weights": dict(zip(symbols, w_final)),
        "expected_return": mu_p,
        "volatility": sigma_p,
        "sharpe": sharpe_p
    }

# =====================================================
# LEGACY FUNCTIONS (for compatibility)
# =====================================================

def semi_absolute_deviation(weights, returns, expected_returns):
    """
    Ma et al. (2020) - Equation (4-5)
    Semi-Absolute Deviation (SAD)
    """
    portfolio_returns = returns @ weights
    expected_portfolio_return = expected_returns @ weights
    
    downside_deviations = np.maximum(
        expected_portfolio_return - portfolio_returns, 
        0
    )
    
    sad = downside_deviations.mean()
    l2_penalty = 0.01 * np.sum(weights ** 2)
    
    return sad + l2_penalty


def optimize_markowitz_sad(symbols, lookback_days=252, target_return=None):
    """
    SAD Optimization (legacy, may fail)
    """
    df = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    prices = df.pivot(index="date", columns="symbol", values="close")
    prices = prices[symbols].dropna().tail(lookback_days)
    
    if len(prices) < 60:
        n = len(symbols)
        return dict(zip(symbols, [1/n]*n))
    
    returns = prices.pct_change().dropna()
    expected_returns = compute_expected_returns(prices, method="exponential")
    
    n = len(symbols)
    
    def objective(w):
        return semi_absolute_deviation(w, returns.values, expected_returns.values)
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    if target_return is not None:
        constraints.append({
            "type": "ineq", 
            "fun": lambda w: (expected_returns.values @ w) - target_return
        })
    
    bounds = [(0.05, 0.25)] * n
    
    vols = returns.std().values
    w0 = (1 / vols) / (1 / vols).sum()
    w0 = np.clip(w0, 0.05, 0.25)
    w0 = w0 / w0.sum()
    
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-6}
    )
    
    if result.success:
        return dict(zip(symbols, result.x))
    else:
        return dict(zip(symbols, [1/n]*n))


def optimize_max_sharpe(symbols, lookback_days=252, risk_free_rate=0.03):
    """
    Wrapper for max_sharpe_portfolio (returns only weights)
    """
    try:
        result = max_sharpe_portfolio(symbols, lookback_days, risk_free_rate)
        return result["weights"]
    except Exception:
        n = len(symbols)
        return dict(zip(symbols, [1/n]*n))

