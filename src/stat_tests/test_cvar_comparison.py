import numpy as np
import pandas as pd
from scipy import stats

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data_processed/powerbi/portfolio_timeseries.csv")
returns = df["portfolio_return"].dropna().values

alpha = 0.05

# =========================
# GAUSSIAN CVaR
# =========================
mu = np.mean(returns)
sigma = np.std(returns, ddof=1)
z = stats.norm.ppf(alpha)
cvar_gaussian = mu - sigma * stats.norm.pdf(z) / alpha

# =========================
# STUDENT-T CVaR
# =========================
df_t, loc_t, scale_t = stats.t.fit(returns)
t_alpha = stats.t.ppf(alpha, df_t)

cvar_student = (
    loc_t
    - scale_t
    * (stats.t.pdf(t_alpha, df_t) / alpha)
    * ((df_t + t_alpha**2) / (df_t - 1))
)

# =========================
# HISTORICAL CVaR
# =========================
var_hist = np.quantile(returns, alpha)
cvar_hist = returns[returns <= var_hist].mean()

# =========================
# PRINT RESULTS
# =========================
print("\nCVaR Comparison (5%)")
print("=" * 40)
print(f"Gaussian CVaR  : {cvar_gaussian:.4f}")
print(f"Student-t CVaR : {cvar_student:.4f}")
print(f"Historical CVaR: {cvar_hist:.4f}")
