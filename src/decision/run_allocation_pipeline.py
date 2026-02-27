# run_allocation_pipeline.py - Diagnostics + Validation

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

scripts = [
    ("src.decision.allocation_diagnostics", "1. Problem Diagnosis (signal dispersion, VIC-VCB risk structure)"),
    ("src.decision.allocation_validation", "2. Empirical Validation (Softmax vs MV vs ERC)"),
]

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" ALLOCATION PIPELINE - Quy trinh khac phuc phan bo danh muc")
    print("=" * 70)

    for module, desc in scripts:
        print(f"\n>>> {desc}")
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=PROJECT_DIR
        )
        if result.returncode != 0:
            print(f"\nPipeline stopped at: {module}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED")
    print("=" * 70)
    print("\nOutput: data_processed/diagnostics/, data_processed/backtest/")
    print("=" * 70 + "\n")
