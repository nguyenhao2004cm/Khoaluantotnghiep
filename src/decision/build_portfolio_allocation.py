# Regime-Aware ERC allocation. Web: reads web_run_config.json

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.web_run_config import load_web_run_config

DATA_CUTOFF = os.environ.get("DATA_CUTOFF_DATE")
ALLOCATION_METHOD = os.environ.get("ALLOCATION_METHOD", "erc").lower().strip()
REBALANCE_FREQ = os.environ.get("REBALANCE_FREQ", "monthly").lower().strip()

# Khi khong co web config: dung toan thi truong (tat ca symbol trong signal)
# Web config: chi gioi han assets khi user chon
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


def load_portfolio_regime():
    df = pd.read_csv(RISK_REGIME_FILE, parse_dates=["date"])
    return df.set_index("date")["portfolio_risk_regime"]

PORTFOLIO_REGIME = load_portfolio_regime()

def _get_weights_for_date(symbols, signals, regime, risk_profile, date, vol_series):
    """Return dict {symbol: weight}. method: erc | minvar | softmax."""
    method = ALLOCATION_METHOD

    if method == "erc":
        from src.portfolio_engine.risk_based_optimizer import allocate_risk_based
        weights = allocate_risk_based(
            symbols, signals, regime, method="erc",
            as_of_date=date, risk_profile=risk_profile
        )
        if regime == "HIGH":
            vol_scale = get_volatility_scale(date, symbols, vol_series)
            weights = {k: v * vol_scale for k, v in weights.items()}
        return weights

    elif method == "minvar":
        from src.portfolio_engine.risk_based_optimizer import allocate_risk_based
        weights = allocate_risk_based(
            symbols, signals, regime, method="minvar",
            as_of_date=date, risk_profile=risk_profile
        )
        if regime == "HIGH":
            vol_scale = get_volatility_scale(date, symbols, vol_series)
            weights = {k: v * vol_scale for k, v in weights.items()}
        return weights

    else:
        # softmax: baseline only, NOT production
        if regime == "HIGH":
            weights = allocate_defensive(symbols, signals)
            vol_scale = get_volatility_scale(date, symbols, vol_series)
            weights = {k: v * vol_scale for k, v in weights.items()}
            w = np.array(list(weights.values()))
        elif regime == "LOW":
            weights = allocate_aggressive(symbols, signals)
            w = np.array(list(weights.values()))
            w = apply_risk_profile(w, risk_profile)
        else:
            weights = allocate_balanced(symbols, signals)
            w = np.array(list(weights.values()))
            w = apply_risk_profile(w, risk_profile)
        return dict(zip(weights.keys(), w))


def main():
    print(f"\n REGIME-AWARE PORTFOLIO ALLOCATION (method={ALLOCATION_METHOD})")
    print(f" Production: ERC. Softmax = baseline only.\n")

    df = pd.read_csv(SIGNAL_FILE, parse_dates=["date"])
    config = load_web_run_config()
    if config and config.get("assets"):
        assets = [s.strip().upper() for s in config["assets"]]
        risk_profile = config.get("risk_appetite", RISK_PROFILE)
        start_date = pd.to_datetime(config["start_date"]) if config.get("start_date") else None
        end_date = pd.to_datetime(config["end_date"]) if config.get("end_date") else None
    else:
        assets = df["symbol"].unique().tolist()
        risk_profile = RISK_PROFILE
        start_date = None
        end_date = None
    if not assets:
        assets = df["symbol"].unique().tolist()

    df = df[df["symbol"].isin(assets)]

    # Preload vol series cho volatility targeting (1 lần)
    vol_series = _load_vol_series(assets)

    # Web config: build signal lookup với forward-fill (mã thiếu signal dùng ngày gần nhất)
    signal_lookup = None
    if config and config.get("assets"):
        df_sig = df.pivot(index="date", columns="symbol", values="signal").reindex(columns=assets)
        df_sig = df_sig.sort_index().ffill().bfill().fillna(0)  # forward-fill, backfill, còn NaN→0
        signal_lookup = df_sig

    records = []
    last_weights_by_month = {}  # For monthly rebalance: (year, month) -> weights dict

    for date, g in df.groupby("date"):
        if date not in PORTFOLIO_REGIME.index:
            continue
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue
        if DATA_CUTOFF is not None and date > pd.to_datetime(DATA_CUTOFF):
            continue

        # Web config: đảm bảo TẤT CẢ assets user chọn có trong mỗi ngày
        # Mã thiếu signal dùng signal ngày gần nhất (forward-fill)
        if config and config.get("assets") and signal_lookup is not None:
            symbols = list(assets)
            row = signal_lookup.loc[date] if date in signal_lookup.index else signal_lookup.iloc[-1]
            signals = np.array([float(row[s]) if s in row.index and pd.notna(row[s]) else 0.0 for s in symbols])
        elif config and config.get("assets"):
            symbols_in_date = g["symbol"].tolist()
            symbols = list(assets)
            signals_list = []
            for sym in symbols:
                if sym in symbols_in_date:
                    sig = g.loc[g["symbol"] == sym, "signal"].iloc[0]
                    signals_list.append(float(sig))
                else:
                    signals_list.append(0.0)
            signals = np.array(signals_list)
        else:
            symbols = g["symbol"].tolist()
            signals = g["signal"].values
        regime = PORTFOLIO_REGIME.loc[date]

        # Monthly rebalance: only optimize on 1st trading day of month
        if REBALANCE_FREQ == "monthly" and ALLOCATION_METHOD in ("erc", "minvar"):
            month_key = (date.year, date.month)
            prev_key = (date.year, date.month - 1) if date.month > 1 else (date.year - 1, 12)
            if date.day > 1:
                wdict = last_weights_by_month.get(month_key) or last_weights_by_month.get(prev_key)
                if wdict:
                    for sym, weight in wdict.items():
                        records.append({
                            "date": date,
                            "symbol": sym,
                            "allocation_weight": weight,
                            "risk_regime": regime,
                            "risk_profile": risk_profile
                        })
                    continue

        weights = _get_weights_for_date(symbols, signals, regime, risk_profile, date, vol_series)
        if REBALANCE_FREQ == "monthly" and ALLOCATION_METHOD in ("erc", "minvar"):
            last_weights_by_month[(date.year, date.month)] = weights
        w = np.array(list(weights.values()))

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
