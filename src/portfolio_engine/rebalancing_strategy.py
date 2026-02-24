
# =====================================================
# FILE 3: src/portfolio_engine/rebalancing_strategy.py (NEW)
# =====================================================
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
ALLOCATION_FILE = PROJECT_DIR / "data_processed/portfolio/portfolio_allocation_final.csv"

def apply_rebalancing_rules(df_alloc):
    """
    Thêm các quy tắc rebalancing động:
    1. Monthly rebalancing (không daily → giảm transaction cost)
    2. Threshold-based rebalancing (chỉ rebalance khi drift > 5%)
    3. Stop-loss khi drawdown > 20%
    """
    df = df_alloc.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    #  RULE 1: Monthly rebalancing only
    df["month"] = df["date"].dt.to_period("M")
    df = df.groupby(["month", "symbol"], as_index=False).last()
    df = df.drop(columns=["month"])
    
    #  RULE 2: Weight drift detection
    # (cần portfolio_value để tính → implement trong portfolio_builder)
    
    return df


def add_stop_loss_logic(df_portfolio):
    """
    Stop-loss: khi drawdown > threshold, chuyển sang defensive portfolio
    """
    df = df_portfolio.copy()
    
    # Tính drawdown
    rolling_max = df["portfolio_value"].cummax()
    drawdown = df["portfolio_value"] / rolling_max - 1
    
    # Trigger stop-loss
    STOP_LOSS_THRESHOLD = -0.20  # -20%
    df["stop_loss_active"] = drawdown < STOP_LOSS_THRESHOLD
    
    return df