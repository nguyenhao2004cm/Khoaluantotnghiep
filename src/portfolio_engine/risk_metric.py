# src/portfolio_engine/risk_metrics.py

import numpy as np
import pandas as pd

def compute_risk_metrics(portfolio_df: pd.DataFrame) -> dict:
    returns = portfolio_df["portfolio_return"]

    vol = returns.std() * np.sqrt(252)
    downside = returns[returns < 0].std() * np.sqrt(252)
    max_dd = portfolio_df["drawdown"].min()

    return {
        "Volatility": vol,
        "Downside Risk": downside,
        "Max Drawdown": max_dd
    }
