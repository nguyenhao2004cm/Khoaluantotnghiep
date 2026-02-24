# =====================================================
# REGIME DISCRIMINATION TEST
# Latent factors có phân biệt HIGH vs LOW regime không?
# =====================================================

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

LATENT_DIR = PROJECT_DIR / "data_processed" / "latent"
REGIME_FILE = PROJECT_DIR / "data_processed" / "reporting" / "risk_regime_timeseries.csv"
OUT_DIR = PROJECT_DIR / "data_processed" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LATENT_DIM = 3


def cohens_d(x, y):
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x) + (ny - 1) * np.var(y)) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def load_latent_with_regime():
    """Load latent + merge với portfolio regime theo date."""
    regime = pd.read_csv(REGIME_FILE, parse_dates=["date"])
    regime = regime[["date", "portfolio_risk_regime"]]

    dfs = []
    for f in LATENT_DIR.glob("*_latent.csv"):
        df = pd.read_csv(f, parse_dates=["date"])
        df["symbol"] = f.stem.replace("_latent", "")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No latent files found. Run encode_latent first.")

    latent = pd.concat(dfs, ignore_index=True)
    merged = latent.merge(regime, on="date", how="inner")
    return merged


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" REGIME DISCRIMINATION TEST")
    print("=" * 60)

    df = load_latent_with_regime()

    high_mask = df["portfolio_risk_regime"] == "HIGH"
    low_mask = df["portfolio_risk_regime"] == "LOW"

    n_high = high_mask.sum()
    n_low = low_mask.sum()
    print(f"\nHIGH regime: {n_high} obs | LOW regime: {n_low} obs")

    if n_high < 10 or n_low < 10:
        print("\n⚠ Not enough HIGH/LOW obs. Regime discrimination skipped.")
        print("  (Run build_portfolio_risk_regime with relaxed thresholds)")
    else:
        results = []

        for i in range(LATENT_DIM):
            col = f"latent_{i}"
            z_high = df.loc[high_mask, col].values
            z_low = df.loc[low_mask, col].values

            t_stat, t_pval = stats.ttest_ind(z_high, z_low)
            u_stat, mw_pval = stats.mannwhitneyu(z_high, z_low, alternative="two-sided")
            d = cohens_d(z_high, z_low)

            results.append({
                "Latent": f"Z{i+1}",
                "t_stat": t_stat,
                "t_pvalue": t_pval,
                "MW_pvalue": mw_pval,
                "Cohens_d": d,
                "Pass_p005": t_pval < 0.05,
                "Pass_effect_05": abs(d) > 0.5
            })

        res_df = pd.DataFrame(results)
        out_file = OUT_DIR / "regime_discrimination_results.csv"
        res_df.to_csv(out_file, index=False)

        print("\n" + res_df.round(4).to_string())
        print(f"\nSaved to: {out_file}")

        # Boxplot
        plot_df = df[df["portfolio_risk_regime"].isin(["HIGH", "LOW"])].copy()
        fig, axes = plt.subplots(1, LATENT_DIM, figsize=(12, 4))
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            high_vals = plot_df.loc[plot_df["portfolio_risk_regime"] == "HIGH", f"latent_{i}"]
            low_vals = plot_df.loc[plot_df["portfolio_risk_regime"] == "LOW", f"latent_{i}"]
            ax.boxplot([low_vals.dropna(), high_vals.dropna()], labels=["LOW", "HIGH"])
            ax.set_title(f"Z{i+1} by Regime")
            ax.set_ylabel(f"Z{i+1}")
        plt.suptitle("Latent Factors by Regime (HIGH vs LOW)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "regime_discrimination_boxplot.png", dpi=150)
        plt.close()
        print(f"\nBoxplot saved: {OUT_DIR / 'regime_discrimination_boxplot.png'}")

    print("\n" + "=" * 60 + "\n")
