import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg # type: ignore
from statsmodels.stats.diagnostic import het_arch # type: ignore

# =========================
# LOAD DATA
# =========================
DATA_FILE = "data_processed/powerbi/portfolio_timeseries.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
returns = df["portfolio_return"].dropna()

# =========================
# REMOVE MEAN DYNAMICS (AR)
# =========================
model = AutoReg(returns, lags=1, old_names=False)
res = model.fit()
residuals = res.resid

# =========================
# ARCH LM TEST
# =========================
lags = 12
lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=lags)

# =========================
# OUTPUT
# =========================
print("\nARCH LM Test (on AR(1) residuals)")
print(f"  LM Statistic : {lm_stat:.4f}")
print(f"  p-value      : {lm_pvalue:.4e}")
print(f"  F Statistic  : {f_stat:.4f}")
print(f"  F p-value    : {f_pvalue:.4e}")

print("\nConclusion:")
if lm_pvalue < 0.05:
    print("  Volatility clustering detected (ARCH effect present)")
else:
    print("  No significant ARCH effect detected")
