# src/portfolio_engine/performance_metrics_genai.py

import numpy as np
import pandas as pd


def compute_performance_metrics(
    portfolio_df: pd.DataFrame,
    initial_capital: float = 10000.0,
    trading_days: int = 252
) -> dict:
    """
    Compute performance metrics CONSISTENT with portfolio builder & report

    Assumptions:
    - portfolio_value already reflects compounding from initial_capital
    - portfolio_df is AFTER weight lag (no look-ahead)
    """

    df = portfolio_df.copy()

    # =========================
    # BASIC SERIES
    # =========================
    returns = df["portfolio_return"].dropna()
    values = df["portfolio_value"].dropna()

    if len(values) < 2:
        raise ValueError(" Not enough data to compute performance metrics")

    # =========================
    # CAGR (ĐÚNG CHUẨN)
    # =========================
    start_value = initial_capital
    end_value = values.iloc[-1]

    n_days = len(values)
    years = n_days / trading_days

    cagr = (end_value / start_value) ** (1 / years) - 1

    # =========================
    # SHARPE
    # =========================
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(trading_days)
    else:
        sharpe = np.nan

    # =========================
    # MAX DRAWDOWN
    # =========================
    if "drawdown" in df.columns:
        max_dd = df["drawdown"].min()
    else:
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        max_dd = drawdown.min()

    # =========================
    # VOLATILITY (ANNUAL)
    # =========================
    vol_annual = returns.std() * np.sqrt(trading_days) if len(returns) > 0 else np.nan

    # =========================
    # TOTAL RETURN
    # =========================
    total_return = (end_value / start_value) - 1.0

    # =========================
    # SORTINO (downside deviation)
    # =========================
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 1 else np.nan
    sortino = (returns.mean() * trading_days) / downside_vol if downside_vol and downside_vol > 0 else np.nan

    # =========================
    # CVaR 5%
    # =========================
    var_5 = np.quantile(returns, 0.05)
    cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else var_5

    # =========================
    # BETA vs VNIndex (placeholder if no benchmark)
    # =========================
    beta = np.nan  # Backend can compute when benchmark data available

    return {
        "start_balance": start_value,
        "end_balance": end_value,
        "CAGR": cagr,
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Volatility (annual)": vol_annual,
        "Max Drawdown": max_dd,
        "CVaR_5": cvar_5,
        "Sortino Ratio": sortino,
        "Beta": beta,
        "Observations": n_days,
    }
