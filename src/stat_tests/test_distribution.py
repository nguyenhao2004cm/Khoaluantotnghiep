import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# =====================================
# CONFIG
# =====================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data_processed" / "powerbi"

# =====================================
# LOAD DATA
# =====================================
def load_portfolio_returns():
    df = pd.read_csv(DATA_DIR / "portfolio_timeseries.csv")
    returns = df["portfolio_return"].dropna()
    return returns


# =====================================
# NORMALITY TESTS
# =====================================
def jarque_bera_test(returns):
    jb_stat, jb_p = stats.jarque_bera(returns)
    return {
        "JB Statistic": jb_stat,
        "p-value": jb_p,
        "Normality Rejected": jb_p < 0.05
    }


def shapiro_test(returns, max_n=5000):
    # Shapiro không dùng cho mẫu quá lớn
    sample = returns.sample(min(len(returns), max_n), random_state=42)
    stat, p = stats.shapiro(sample)
    return {
        "Shapiro Statistic": stat,
        "p-value": p,
        "Normality Rejected": p < 0.05
    }


def tail_properties(returns):
    return {
        "Skewness": stats.skew(returns),
        "Excess Kurtosis": stats.kurtosis(returns, fisher=True)
    }


# =====================================
# RUN ALL
# =====================================
def run_distribution_tests():
    returns = load_portfolio_returns()

    results = {
        "Jarque-Bera": jarque_bera_test(returns),
        "Shapiro-Wilk": shapiro_test(returns),
        "Tail Properties": tail_properties(returns)
    }

    return results


if __name__ == "__main__":
    res = run_distribution_tests()
    for k, v in res.items():
        print(f"\n{k}")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")
