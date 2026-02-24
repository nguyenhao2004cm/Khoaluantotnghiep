#C:\Users\ASUS\fintech-project\src\portfolio_engine\performance_metrics.py
import pandas as pd
import numpy as np


# =====================================
# BASIC PERFORMANCE METRICS
# =====================================
def compute_cagr(portfolio_value, trading_days=252):
    """
    CAGR = (V_end / V_start)^(1 / years) - 1
    """
    total_periods = len(portfolio_value)
    years = total_periods / trading_days

    return (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1


def compute_annualized_volatility(portfolio_returns, trading_days=252):
    """
    Annualized volatility
    """
    return portfolio_returns.std() * np.sqrt(trading_days)


def compute_sharpe_ratio(portfolio_returns, risk_free_rate=0.0, trading_days=252):
    """
    Sharpe ratio (standard, no innovation)
    """
    excess_return = portfolio_returns - risk_free_rate / trading_days
    return excess_return.mean() / portfolio_returns.std() * np.sqrt(trading_days)


# =====================================
# DRAWDOWN
# =====================================
def compute_drawdown(portfolio_value):
    """
    Drawdown series:
    DD_t = V_t / max(V_0..V_t) - 1
    """
    rolling_max = portfolio_value.cummax()
    drawdown = portfolio_value / rolling_max - 1
    return drawdown


def compute_max_drawdown(drawdown):
    """
    Maximum drawdown
    """
    return drawdown.min()


# =====================================
# SUMMARY TABLE
# =====================================
def performance_summary(df_portfolio):
    """
    Input:
        DataFrame with columns:
        - portfolio_return
        - portfolio_value
    """
    returns = df_portfolio["portfolio_return"]
    value = df_portfolio["portfolio_value"]

    summary = {
        "CAGR": compute_cagr(value),
        "Annualized Volatility": compute_annualized_volatility(returns),
        "Sharpe Ratio": compute_sharpe_ratio(returns),
        "Max Drawdown": compute_max_drawdown(compute_drawdown(value))
    }

    return pd.Series(summary)

# =====================================
# ANNUAL RETURNS
# =====================================
def compute_annual_returns(df_portfolio):
    """
    Compute annual geometric returns.
    """
    df = df_portfolio.copy()
    df["year"] = df["date"].dt.year

    annual_returns = (
        df.groupby("year")["portfolio_return"]
          .apply(lambda x: (1 + x).prod() - 1)
    )

    return annual_returns

# =====================================
# RUN TEST
# =====================================
if __name__ == "__main__":
    from portfolio_builder import build_portfolio

    df_port = build_portfolio(initial_capital=10000)

    dd = compute_drawdown(df_port["portfolio_value"])
    summary = performance_summary(df_port)
    annual_returns = compute_annual_returns(df_port)
    print("\nPERFORMANCE SUMMARY:")
    print(summary)

    print("\nMAX DRAWDOWN:", summary["Max Drawdown"])
    

    print("\nANNUAL RETURNS:")
    print(annual_returns)