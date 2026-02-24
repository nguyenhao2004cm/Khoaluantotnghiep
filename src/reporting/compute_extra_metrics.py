# src/reporting/compute_extended_metrics.py

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis

# =========================
# CONFIG
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[2]

PORTFOLIO_FILE = (
    PROJECT_DIR
    / "data_processed"
    / "powerbi"
    / "portfolio_timeseries.csv"
)

OUT_DIR = PROJECT_DIR / "data_processed" / "reporting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "performance_extended.csv"

TRADING_DAYS = 252
VAR_LEVEL = 0.05



# =========================
# LOAD DATA
# =========================
df = pd.read_csv(PORTFOLIO_FILE, parse_dates=["date"])
df = df.sort_values("date")

returns = df["portfolio_return"].dropna()
values = df["portfolio_value"].dropna()

# =========================
# BASIC METRICS
# =========================
start_balance = values.iloc[0]
end_balance = values.iloc[-1]

num_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
cagr = (end_balance / start_balance) ** (1 / num_years) - 1

mean_daily = returns.mean()
mean_annual = mean_daily * TRADING_DAYS

vol_daily = returns.std()
vol_annual = vol_daily * np.sqrt(TRADING_DAYS)

# =========================
# DOWNSIDE / SORTINO
# =========================
downside_returns = returns[returns < 0]
downside_dev = downside_returns.std() * np.sqrt(TRADING_DAYS)
sortino = mean_annual / downside_dev if downside_dev > 0 else np.nan

# =========================
# SHARPE
# =========================
sharpe = mean_annual / vol_annual if vol_annual > 0 else np.nan

# =========================
# DRAW DOWN
# =========================
max_drawdown = df["drawdown"].min()

# =========================
# EXTREME DAYS
# =========================
best_day = returns.max()
worst_day = returns.min()
positive_days = (returns > 0).mean()

# =========================
# RISK METRICS
# =========================
var_5 = np.quantile(returns, VAR_LEVEL)
cvar_5 = returns[returns <= var_5].mean()

skewness = skew(returns)
kurt = kurtosis(returns)

metrics = [
    ("start_balance", start_balance),
    ("end_balance", end_balance),

    ("cagr", cagr),
    ("mean_return_annual", mean_annual),
    ("volatility_annual", vol_annual),

    ("sharpe_ratio", sharpe),
    ("sortino_ratio", sortino),

    ("max_drawdown", max_drawdown),
    ("best_day_return", best_day),
    ("worst_day_return", worst_day),
    ("positive_days_ratio", positive_days),

    ("var_5", var_5),
    ("cvar_5", cvar_5),

    ("skewness", skewness),
    ("excess_kurtosis", kurt),
]

# =========================
# BUILD TABLE
# =========================
metrics = [
    ("start_balance", start_balance),
    ("end_balance", end_balance),

    ("cagr", cagr),
    ("mean_return_annual", mean_annual),
    ("volatility_annual", vol_annual),

    ("sharpe_ratio", sharpe),
    ("sortino_ratio", sortino),

    ("max_drawdown", max_drawdown),
    ("best_day_return", best_day),
    ("worst_day_return", worst_day),
    ("positive_days_ratio", positive_days),

    ("var_5", var_5),
    ("cvar_5", cvar_5),

    ("skewness", skewness),
    ("excess_kurtosis", kurt),
]


df_metrics = pd.DataFrame(metrics, columns=["metric", "value"])
df_metrics.to_csv(OUT_FILE, index=False)


print("Saved performance metrics to:", OUT_FILE)
