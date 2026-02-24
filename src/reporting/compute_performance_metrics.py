# C:\Users\ASUS\fintech-project\src\portfolio_engine\performance_metrics_genai.py

import numpy as np
import pandas as pd


def compute_performance_metrics(
    portfolio_df: pd.DataFrame,
    start_date: str | None = None,
    trading_days: int = 252
) -> dict:
    """
    Compute portfolio performance metrics on a DEFINED investment window.

    Parameters
    ----------
    portfolio_df : DataFrame
        Must contain: date, portfolio_value, portfolio_return, drawdown
    start_date : str, optional
        Investment start date (YYYY-MM-DD)
    trading_days : int
        Trading days per year (default: 252)

    Returns
    -------
    dict
    """

    df = portfolio_df.copy()

    # =========================
    # 1. APPLY INVESTMENT WINDOW
    # =========================
    if start_date is not None:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= pd.to_datetime(start_date)]

    # Remove invalid early rows
    df = df.dropna(subset=["portfolio_value", "portfolio_return"])

    if len(df) < 30:
        raise ValueError(
            " Not enough data after start_date to compute performance metrics"
        )

    # =========================
    # 2. BASIC SERIES
    # =========================
    returns = df["portfolio_return"]
    values = df["portfolio_value"]

    # =========================
    # 3. CAGR (Correct Window)
    # =========================
    n_days = len(df)
    cagr = (values.iloc[-1] / values.iloc[0]) ** (
        trading_days / n_days
    ) - 1

    # =========================
    # 4. SHARPE (Robust)
    # =========================
    ret_std = returns.std()

    if ret_std > 1e-8:
        sharpe = returns.mean() / ret_std * np.sqrt(trading_days)
    else:
        sharpe = np.nan

    # =========================
    # 5. MAX DRAWDOWN (TRUE DD)
    # =========================
    if "drawdown" in df.columns:
        max_dd = df["drawdown"].min()
    else:
        cummax = values.cummax()
        drawdown = values / cummax - 1
        max_dd = drawdown.min()

    # =========================
    # 6. FINAL VALUE
    # =========================
    final_value = values.iloc[-1]

    return {
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Final Value": final_value,
        "Start Date": df["date"].iloc[0],
        "End Date": df["date"].iloc[-1],
        "Observations": n_days
    }
