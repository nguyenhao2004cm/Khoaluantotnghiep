import numpy as np
import pandas as pd
from scipy import stats

# =========================
# LOAD DATA
# =========================
DATA_FILE = "data_processed/powerbi/portfolio_timeseries.csv"
df = pd.read_csv(DATA_FILE)

returns = df["portfolio_return"].dropna().values
T = len(returns)

alpha = 0.05

# =========================
# 1. VaR ESTIMATION
# =========================

# Gaussian VaR
mu = np.mean(returns)
sigma = np.std(returns, ddof=1)
var_gaussian = mu + sigma * stats.norm.ppf(alpha)

# Student-t VaR
df_t, loc_t, scale_t = stats.t.fit(returns)
var_student = loc_t + scale_t * stats.t.ppf(alpha, df_t)

# Historical VaR
var_hist = np.quantile(returns, alpha)

# =========================
# 2. BACKTESTING FUNCTION
# =========================
def kupiec_test(returns, var, alpha):
    violations = returns < var
    x = np.sum(violations)
    n = len(returns)

    pi_hat = x / n
    LR = -2 * (
        (n - x) * np.log((1 - alpha) / (1 - pi_hat)) +
        x * np.log(alpha / pi_hat)
    )
    p_value = 1 - stats.chi2.cdf(LR, df=1)

    return {
        "violations": int(x),
        "expected": n * alpha,
        "LR_stat": LR,
        "p_value": p_value,
        "reject": p_value < 0.05
    }

# =========================
# 3. RUN TESTS
# =========================
results = {
    "Gaussian": kupiec_test(returns, var_gaussian, alpha),
    "Student-t": kupiec_test(returns, var_student, alpha),
    "Historical": kupiec_test(returns, var_hist, alpha)
}

# =========================
# 4. PRINT RESULTS
# =========================
print("\nVaR Backtesting (Kupiec POF Test)")
print("=" * 45)

for model, res in results.items():
    print(f"\n{model} VaR")
    print(f"  Violations : {res['violations']}")
    print(f"  Expected   : {res['expected']:.2f}")
    print(f"  LR stat    : {res['LR_stat']:.4f}")
    print(f"  p-value    : {res['p_value']:.6f}")
    print(f"  Reject H0  : {res['reject']}")
