# =====================================================
# VIC-VCB VALIDATION - Phase 4
# Chay rieng: Softmax vs ERC vs MinVar cho VIC+VCB
# Expected: Softmax ~50-50, ERC ~36% VIC / 64% VCB (vol khac nhau)
# =====================================================

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.decision.allocation_validation import run_vic_vcb_validation

if __name__ == "__main__":
    run_vic_vcb_validation()
