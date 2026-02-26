# =====================================================
# RUN ALL BACKTEST PHASES
# =====================================================

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.backtest.multi_year_backtest import run_multi_year_backtest
from src.backtest.benchmark_comparison import run_benchmark_comparison
from src.backtest.allocation_charts import run_allocation_charts
from src.backtest.crisis_stress_test import run_crisis_stress_test


def main():
    print("\n" + "=" * 70)
    print(" BACKTEST — FULL PIPELINE")
    print("=" * 70)
    run_multi_year_backtest()
    run_benchmark_comparison()
    run_allocation_charts()
    run_crisis_stress_test()
    print("\n" + "=" * 70)
    print(" BACKTEST COMPLETED")
    print(" Outputs in: data_processed/backtest/")
    print("=" * 70)


if __name__ == "__main__":
    main()
