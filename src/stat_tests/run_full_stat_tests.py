# =====================================================
# RUN FULL STATISTICAL TESTS
# =====================================================

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

scripts = [
    "src/stat_tests/test_volatility.py",
    "src/stat_tests/test_distribution.py",
    "src/stat_tests/test_var_backtest.py",
    "src/stat_tests/test_cvar_comparison.py",
    "src/stat_tests/test_cvar_regime.py",
    "src/stat_tests/test_cvar_vs_benchmark.py",
    "src/stat_tests/test_cer_comparison.py",
    "src/stat_tests/test_sharpe_difference.py",
    "src/stat_tests/test_dm_tail_loss.py",
    "src/stat_tests/test_es_dominance.py",
    "src/stat_tests/test_cvar_window_sensitivity.py",
    "src/stat_tests/test_cvar_after_regime_switch.py",  
    "src/stat_tests/test_cvar_bootstrap.py",           
]

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" FULL STATISTICAL TESTS (stat_tests)")
    print("=" * 60)

    for script in scripts:
        print(f"\n>>> Running {script} ...")
        result = subprocess.run(
            [sys.executable, script],
            cwd=PROJECT_DIR
        )
        if result.returncode != 0:
            print(f"\nPipeline stopped at: {script}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(" All statistical tests completed.")
    print("=" * 60 + "\n")
