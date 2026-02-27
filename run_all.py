# run_all.py - Full pipeline

import sys
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
sys.path.append(str(PROJECT_DIR))  # import src

def run_step(script_path, step_name):
    print("\n" + "=" * 70)
    print(f" {step_name}")
    print("=" * 70)

    # Module call: python -m ...
    if script_path.startswith("-m"):
        module = script_path.replace("-m ", "")
        result = subprocess.run(
            [PYTHON, "-m", module],
            cwd=PROJECT_DIR
        )
    else:
        result = subprocess.run(
            [PYTHON, script_path],
            cwd=PROJECT_DIR
        )

    if result.returncode != 0:
        print(f"\n PIPELINE STOPPED AT STEP: {step_name}")
        sys.exit(1)

    print(f" COMPLETED: {step_name}")

def main():
    print("\n" + "=" * 70)
    print(" PORTFOLIO OPTIMIZATION – FULL PIPELINE")
    print("=" * 70)

    steps = [

        #=====================================================
        # STAGE 0 – PRICE DATA
        # =====================================================
        ("src/data/prepare_prices.py",
         "Prepare price panel from raw data"),


        # =====================================================
        # STAGE 1 – FEATURE ENGINEERING
        # =====================================================
        ("src/features/build_features.py",
         "Build technical & market features"),
        ("src/features/build_risk_features.py",
         "Build technical & market risk features"),
        #=====================================================
        # STAGE 2 – LATENT REPRESENTATION
        # =====================================================
        ("src/models/train_autoencoder.py",
         "Train market autoencoder"),

        ("-m src.models.encode_latent",
         "Encode latent features"),

        ("src/models/train_lstm_latent.py",
         "Train LSTM on latent (for risk prediction)"),

        # =====================================================
        # STAGE 3 – RISK ESTIMATION
        # =====================================================
        ("-m src.models.compute_risk_score",
         "Compute risk score from latent space"),

        ("src/models/normalize_risk_signal.py",
         "Normalize risk signal & regimes"),

        # =====================================================
        # STAGE 4 – DECISION SIGNAL
        # =====================================================
        ("src/decision/build_signal.py",
         "Build trading / allocation signals"),

        # =====================================================
        # STAGE 5 – PORTFOLIO ALLOCATION
        # =====================================================
        ("src/reporting/build_portfolio_risk_regime.py",
         "Build risk regime"),

        ("src/decision/build_portfolio_allocation.py",
         "Build portfolio allocation (ERC - Regime-Aware Equal Risk Contribution)"),

        # =====================================================
        # STAGE 6 – PORTFOLIO SIMULATION
        # =====================================================
        ("src/portfolio_engine/portfolio_builder.py",
         "Simulate portfolio performance"),

        ("src/portfolio_engine/performance_metrics_genai.py",
         "Compute extended performance & risk metrics"),

        # =====================================================
        # STAGE 7 – REPORTING DATASETS
        # =====================================================
        ("src/reporting/compute_annual_returns_table.py",
         "Build annual returns table"),

        ("src/reporting/compute_extra_metrics.py",
         "Build extra performance metrics"),

        ("src/reporting/compute_drawdown_periods.py",
         "Build drawdown periods table"),

        ("src/portfolio_engine/asset_risk_return.py",
         "Build asset risk-return dataset"),

        ("src/portfolio_engine/efficient_frontier.py",
         "Build efficient frontier"),

        ("src/portfolio_engine/correlation_matrix.py",
         "Build correlation matrix & AI summary"),

        # =====================================================
        # STAGE 8 – EXPORT FOR DASHBOARD
        # =====================================================
        ("data_processed/powerbi/export_powerbi_data.py",
         "Export datasets for Power BI"),

        # =====================================================
        # STAGE 9 – BACKTEST (multi-year, benchmark, crisis)
        # =====================================================
        ("-m src.backtest.run_all_backtest",
         "Run backtest: multi-year, benchmark comparison, allocation charts, crisis stress test"),

        # =====================================================
        # STAGE 10 – AUTO PDF REPORT
        # =====================================================
        ("-m src.reporting.report_pdf",
         "Generate PDF portfolio report with AI commentary"),
    ]

    for script, name in steps:
        run_step(script, name)

    print("\n" + "=" * 70)
    print(" FULL PIPELINE EXECUTED SUCCESSFULLY")
    print(" PDF REPORT READY IN: Reports/")
    print("=" * 70)


if __name__ == "__main__":
    main()
