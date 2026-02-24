"""
Regime-Aware, User-Constrained Portfolio Allocation
AI governs risk – User provides constraints
Khi chạy từ web: đọc assets, start_date, end_date, risk_appetite từ web_run_config.json.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.web_run_config import load_web_run_config

# =====================================================
# CONFIG (USER-CONSTRAINED) – mặc định khi không có config web
# =====================================================

USER_ASSETS = ["FPT", "VNM", "HPG", "TCB", "VIC"]
RISK_PROFILE = "balanced"  # {"conservative", "balanced", "aggressive"}

TOP_K = 8
MAX_WEIGHT = 0.25
MIN_WEIGHT = 0.05

BASE_DIR = os.getcwd()

SIGNAL_FILE = BASE_DIR + "/data_processed/decision/signal.csv"
RISK_REGIME_FILE = PROJECT_DIR / "data_processed/reporting/risk_regime_timeseries.csv"
OUTPUT_DIR = BASE_DIR + "/data_processed/portfolio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# RISK PROFILE SCALING
# =====================================================

def apply_risk_profile(weights, profile):
    if profile == "conservative":
        weights *= 0.8
    elif profile == "aggressive":
        weights *= 1.2
    return weights / weights.sum()

# =====================================================
# ALLOCATION STRATEGIES (FIXED)
# =====================================================

def allocate_defensive(symbols, signals):
    """
    HIGH RISK → Capital preservation
    - Fewer assets
    - Flat / equal-risk allocation
    """
    n = min(4, len(symbols))
    idx = np.argsort(signals)[::-1][:n]

    weights = np.ones(n) / n
    weights *= 0.8  # 20% implicit cash
    weights = weights / weights.sum()

    return dict(zip([symbols[i] for i in idx], weights))


def allocate_balanced(symbols, signals):
    n = min(TOP_K, len(symbols))
    idx = np.argsort(signals)[::-1][:n]

    temp = 2.0
    exp_s = np.exp(signals[idx] / temp)
    weights = exp_s / exp_s.sum()

    weights = np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)
    weights = weights / weights.sum()

    return dict(zip([symbols[i] for i in idx], weights))


def allocate_aggressive(symbols, signals):
    n = min(10, len(symbols))
    idx = np.argsort(signals)[::-1][:n]

    temp = 1.0
    exp_s = np.exp(signals[idx] / temp)
    weights = exp_s / exp_s.sum()

    weights = np.clip(weights, MIN_WEIGHT * 0.5, MAX_WEIGHT * 1.2)
    weights = weights / weights.sum()

    return dict(zip([symbols[i] for i in idx], weights))

# =====================================================
# LOAD PORTFOLIO REGIME
# =====================================================

def load_portfolio_regime():
    df = pd.read_csv(RISK_REGIME_FILE, parse_dates=["date"])
    return df.set_index("date")["portfolio_risk_regime"]

PORTFOLIO_REGIME = load_portfolio_regime()

# =====================================================
# MAIN
# =====================================================

def main():
    print("\n REGIME-AWARE PORTFOLIO ALLOCATION (FIXED)\n")

    config = load_web_run_config()
    if config:
        assets = [s.strip().upper() for s in config.get("assets", [])]
        risk_profile = config.get("risk_appetite", RISK_PROFILE)
        start_date = pd.to_datetime(config["start_date"]) if config.get("start_date") else None
        end_date = pd.to_datetime(config["end_date"]) if config.get("end_date") else None
    else:
        assets = USER_ASSETS
        risk_profile = RISK_PROFILE
        start_date = None
        end_date = None

    df = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])

    #  USER CONSTRAINT
    df = df[df["symbol"].isin(assets)]

    records = []

    for date, g in df.groupby("date"):
        if date not in PORTFOLIO_REGIME.index:
            continue
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue

        symbols = g["symbol"].tolist()
        signals = g["signal"].values

        regime = PORTFOLIO_REGIME.loc[date]

        if regime == "HIGH":
            weights = allocate_defensive(symbols, signals)
        elif regime == "LOW":
            weights = allocate_aggressive(symbols, signals)
        else:
            weights = allocate_balanced(symbols, signals)

        w = np.array(list(weights.values()))
        w = apply_risk_profile(w, risk_profile)

        for (sym, _), weight in zip(weights.items(), w):
            records.append({
                "date": date,
                "symbol": sym,
                "allocation_weight": weight,
                "risk_regime": regime,
                "risk_profile": risk_profile
            })

    out = pd.DataFrame(records)
    out.to_csv(
        OUTPUT_DIR + "/portfolio_allocation_final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(" Allocation completed & stabilized")
    if not out.empty:
        print(out.groupby("date")["allocation_weight"].sum().describe())


if __name__ == "__main__":
    main()
