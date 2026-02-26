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

# Walk-forward: chỉ xử lý dates <= cutoff
DATA_CUTOFF = os.environ.get("DATA_CUTOFF_DATE")

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
PRICE_FILE = PROJECT_DIR / "data_processed/prices/prices.csv"
OUTPUT_DIR = BASE_DIR + "/data_processed/portfolio"

# Volatility targeting: scale exposure khi realized_vol > target
TARGET_VOL = 0.15
VOL_LOOKBACK = 20

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
    HIGH RISK → Capital preservation + CASH buffer
    - Fewer assets (4)
    - Max equity 60%, Cash 40% (như quỹ thực tế)
    """
    n = min(4, len(symbols))
    idx = np.argsort(signals)[::-1][:n]

    weights = np.ones(n) / n
    weights = weights / weights.sum()  # normalize
    max_equity_weight = 0.6  # 40% cash khi regime HIGH
    weights *= max_equity_weight

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
# VOLATILITY TARGETING (khi regime HIGH)
# =====================================================
def _load_vol_series(symbols):
    """Load rolling vol series 1 lần, reuse."""
    if not PRICE_FILE.exists():
        return None
    try:
        df = pd.read_csv(PRICE_FILE)
        df["symbol"] = df["symbol"].str.strip().str.upper()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["symbol"].isin(symbols)]
        if df.empty:
            return None
        pivot = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret = pivot.pct_change(fill_method=None).dropna(how="all")
        cols = [c for c in ret.columns if c in symbols]
        if not cols:
            return None
        eq_weight = 1.0 / len(cols)
        port_ret = (ret[cols] * eq_weight).sum(axis=1)
        vol_series = port_ret.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
        return vol_series
    except Exception:
        return None

def get_volatility_scale(date, symbols, vol_series=None):
    """Scale = min(1, target_vol / realized_vol)."""
    if vol_series is None:
        vol_series = _load_vol_series(symbols)
    if vol_series is None:
        return 1.0
    try:
        v = vol_series[vol_series.index <= date].tail(1)
        if v.empty or pd.isna(v.iloc[0]) or v.iloc[0] <= 0:
            return 1.0
        scale = min(1.0, TARGET_VOL / v.iloc[0])
        return max(0.3, scale)
    except Exception:
        return 1.0


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

    # Preload vol series cho volatility targeting (1 lần)
    vol_series = _load_vol_series(assets)

    records = []

    for date, g in df.groupby("date"):
        if date not in PORTFOLIO_REGIME.index:
            continue
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue
        if DATA_CUTOFF is not None and date > pd.to_datetime(DATA_CUTOFF):
            continue

        symbols = g["symbol"].tolist()
        signals = g["signal"].values

        regime = PORTFOLIO_REGIME.loc[date]

        if regime == "HIGH":
            weights = allocate_defensive(symbols, signals)
            # Volatility targeting: scale exposure khi realized_vol cao
            vol_scale = get_volatility_scale(date, symbols, vol_series)
            weights = {k: v * vol_scale for k, v in weights.items()}
            w = np.array(list(weights.values()))
            # Không normalize khi HIGH (giữ tổng < 1 = cash buffer)
        elif regime == "LOW":
            weights = allocate_aggressive(symbols, signals)
            w = np.array(list(weights.values()))
            w = apply_risk_profile(w, risk_profile)
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
