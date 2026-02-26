# =====================================================
# WALK-FORWARD BACKTEST (tránh look-ahead bias)

# =====================================================

import os
import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

def run_with_cutoff(cutoff_date: str):
    """Chạy regime + allocation với data cutoff."""
    env = os.environ.copy()
    env["DATA_CUTOFF_DATE"] = cutoff_date
    steps = [
        ("src.reporting.build_portfolio_risk_regime", "Build regime (cutoff)"),
        ("src.decision.build_portfolio_allocation", "Build allocation (cutoff)"),
    ]
    for module, name in steps:
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_DIR,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed: {name}")


def main():
    """
    Walk-forward: với mỗi năm Y, dùng data đến 31/12/(Y-1).
    Cần chạy run_all.py trước để có đủ data.
    """
    print("\n" + "=" * 60)
    print(" WALK-FORWARD BACKTEST")
    print("=" * 60)
    print(" Lưu ý: Cần DATA_CUTOFF_DATE cho toàn pipeline.")
    print(" Hiện tại build_portfolio_risk_regime hỗ trợ DATA_CUTOFF_DATE.")
    print(" Chạy: set DATA_CUTOFF_DATE=2019-12-31 && python run_all.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
