import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
BOOTSTRAP_N = 10000
ALPHA = 0.05

RESULT_FILE = "data_processed/stat_tests/cvar_after_switch_results.csv"

# =========================
# LOAD DATA
# =========================
res = pd.read_csv(RESULT_FILE)

delta = res["delta"].dropna().values
n = len(delta)

print("\nBootstrap Test for ΔCVaR (Portfolio - Benchmark)")
print("=" * 60)
print(f"Observations: {n}")
print(f"Mean ΔCVaR  : {delta.mean():.5f}")

# =========================
# BOOTSTRAP
# =========================
boot_means = np.empty(BOOTSTRAP_N)

for i in range(BOOTSTRAP_N):
    sample = np.random.choice(delta, size=n, replace=True)
    boot_means[i] = sample.mean()

# =========================
# ONE-SIDED P-VALUE
# =========================
p_value = np.mean(boot_means >= 0)

# =========================
# CONFIDENCE INTERVAL
# =========================
ci_low = np.percentile(boot_means, 100 * ALPHA)
ci_high = np.percentile(boot_means, 100 * (1 - ALPHA))

# =========================
# OUTPUT
# =========================
print("\nBootstrap Results")
print("-" * 60)
print(f"Mean ΔCVaR (bootstrap) : {boot_means.mean():.5f}")
print(f"{int((1-ALPHA)*100)}% CI              : [{ci_low:.5f}, {ci_high:.5f}]")
print(f"One-sided p-value      : {p_value:.5f}")

print("\nConclusion:")
if p_value < ALPHA:
    print("  Reject H0")
    print("  Portfolio REDUCES tail risk vs benchmark after regime switches.")
else:
    print("  Fail to reject H0")
    print("  No statistical evidence of tail risk reduction.")
