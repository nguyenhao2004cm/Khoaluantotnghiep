# =====================================================
# RUN EXTENDED VALIDATION
# Trình tự: Diagnostics → Regime → ADF → Stability
# =====================================================

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

scripts = [
    "src/diagnostics/descriptive_risk_features.py",
    "src/diagnostics/correlation_analysis.py",
    "src/diagnostics/latent_factor_analysis.py",
    "src/diagnostics/regime_discrimination_test.py",
    "src/diagnostics/adf_stationarity_test.py",
    "src/diagnostics/latent_stability_test.py",
]

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" EXTENDED VALIDATION (Q1 Empirical Standards)")
    print("=" * 60)

    for script in scripts:
        print(f"\n▶ Running {script} ...")
        result = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_DIR
        )
        if result.returncode != 0:
            print(f"\n⚠ Pipeline stopped at: {script}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(" All extended validation completed.")
    print(" Output: data_processed/diagnostics/")
    print("=" * 60 + "\n")
