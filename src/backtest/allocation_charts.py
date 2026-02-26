# =====================================================
# PHASE 3 — DANH MỤC ĐỀ XUẤT & BIỂU ĐỒ
# Mục tiêu: Hiển thị "Danh mục đề xuất qua từng năm là gì?"
# =====================================================

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =====================================
# CONFIG
# =====================================
ALLOC_FILE = PROJECT_DIR / "data_processed" / "backtest" / "allocation_by_year.csv"
OUT_DIR = PROJECT_DIR / "data_processed" / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_allocation_charts():
    """Tạo bảng allocation wide + stack bar chart + regime overlay."""
    print("\n" + "=" * 60)
    print(" PHASE 3 — ALLOCATION CHARTS")
    print("=" * 60)

    if not ALLOC_FILE.exists():
        print(f" Run multi_year_backtest first. Missing: {ALLOC_FILE}")
        return

    df = pd.read_csv(ALLOC_FILE)
    df["year"] = df["year"].astype(int)

    # Bảng wide: Year | Regime | FPT | VNM | HPG | TCB | ... | Cash
    pivot = df.pivot_table(
        index=["year", "regime"],
        columns="symbol",
        values="allocation_weight",
        aggfunc="first"
    ).reset_index()

    # Đảm bảo Cash có cột
    if "Cash" not in pivot.columns:
        pivot["Cash"] = 0
    pivot["Cash"] = pivot["Cash"].fillna(0)
    total = pivot.drop(columns=["year", "regime"]).sum(axis=1)
    pivot["Cash"] = pivot["Cash"] + (1 - total).clip(lower=0)
    pivot = pivot.fillna(0)

    out_table = OUT_DIR / "allocation_by_year_wide.csv"
    pivot.to_csv(out_table, index=False, encoding="utf-8-sig")
    print(f"\n Saved: {out_table}")
    print("\n Allocation by year (wide):")
    print(pivot.to_string(index=False))

    # Stack bar chart
    symbols = [c for c in pivot.columns if c not in ("year", "regime")]
    pivot_plot = pivot.set_index("year")[symbols]
    pivot_plot = pivot_plot.fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_plot.plot(kind="bar", stacked=True, ax=ax, width=0.7, colormap="tab20")
    ax.set_xlabel("Year")
    ax.set_ylabel("Allocation weight")
    ax.set_title("Portfolio allocation by year (stacked)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    chart_path = OUT_DIR / "allocation_stack_bar.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {chart_path}")

    # Regime overlay (colored bars by year)
    regime_df = pivot[["year", "regime"]].drop_duplicates().sort_values("year")
    regime_map = {"HIGH": "#e74c3c", "NORMAL": "#3498db", "LOW": "#2ecc71"}
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    years = regime_df["year"].tolist()
    colors = [regime_map.get(str(r).upper(), "#95a5a6") for r in regime_df["regime"]]
    ax2.bar(years, [1] * len(years), color=colors, width=0.7)
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("")
    ax2.set_xlabel("Year")
    ax2.set_title("Regime overlay by year")
    ax2.set_yticks([])
    handles = [Patch(facecolor=regime_map.get(r, "#95a5a6"), label=r) for r in ["HIGH", "NORMAL", "LOW"]]
    ax2.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    regime_path = OUT_DIR / "regime_overlay.png"
    plt.savefig(regime_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {regime_path}")

    return pivot


if __name__ == "__main__":
    run_allocation_charts()
