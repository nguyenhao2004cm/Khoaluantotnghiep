# Chạy pipeline tối ưu từ web: config -> allocation -> export powerbi -> reporting -> PDF
# Trả về dict cho PortfolioResult + đường dẫn PDF

import sys
import subprocess
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]


def run_optimization_for_web(
    assets: list,
    start_date: str,
    end_date: str,
    risk_appetite: str,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Ghi web_run_config, chạy pipeline từ build_portfolio_allocation -> export_powerbi -> reporting -> report_pdf.
    Trả về dict dùng cho PortfolioResult (metrics, allocation, growth_series) + pdf_path.
    """
    from src.utils.web_run_config import save_web_run_config

    save_web_run_config(assets, start_date, end_date, risk_appetite, initial_capital)

    steps = [
        ("-m", "src.decision.build_portfolio_allocation"),
        (str(PROJECT_DIR / "data_processed" / "powerbi" / "export_powerbi_data.py"),),
        (str(PROJECT_DIR / "src" / "reporting" / "compute_annual_returns_table.py"),),
        (str(PROJECT_DIR / "src" / "reporting" / "compute_extra_metrics.py"),),
        (str(PROJECT_DIR / "src" / "reporting" / "compute_drawdown_periods.py"),),
        (str(PROJECT_DIR / "src" / "portfolio_engine" / "asset_risk_return.py"),),
        (str(PROJECT_DIR / "src" / "portfolio_engine" / "efficient_frontier.py"),),
        (str(PROJECT_DIR / "src" / "portfolio_engine" / "correlation_matrix.py"),),
        ("-m", "src.reporting.report_pdf"),
    ]

    import os
    env = dict(os.environ)
    env.setdefault("ALLOCATION_METHOD", "erc")
    for args in steps:
        cmd = [sys.executable] + list(args)
        r = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"Pipeline step failed: {' '.join(cmd)}\n{r.stderr or r.stdout}")

    # Đọc kết quả để trả API
    return _build_portfolio_result_from_outputs(start_date, end_date, risk_appetite, initial_capital)


def _build_portfolio_result_from_outputs(
    start_date: str, end_date: str, risk_appetite: str, initial_capital: float = 10000.0
) -> dict:
    """Đọc powerbi/ và portfolio/ -> dict PortfolioResult + pdf_path."""
    import pandas as pd

    powerbi_dir = PROJECT_DIR / "data_processed" / "powerbi"
    portfolio_dir = PROJECT_DIR / "data_processed" / "portfolio"
    reports_dir = PROJECT_DIR / "Reports"

    # growth_series từ portfolio_timeseries
    ts_path = powerbi_dir / "portfolio_timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError("portfolio_timeseries.csv not found after pipeline")
    df_ts = pd.read_csv(ts_path, parse_dates=["date"])
    df_ts = df_ts.sort_values("date")
    growth_series = [
        {"date": row["date"].strftime("%Y-%m-%d"), "value": float(row["portfolio_value"])}
        for _, row in df_ts.iterrows()
    ]

    # allocation từ asset_allocation_current hoặc portfolio_allocation_final (ngày mới nhất)
    allocation = []
    alloc_path = powerbi_dir / "asset_allocation_current.csv"
    if alloc_path.exists():
        alloc_df = pd.read_csv(alloc_path)
        if "symbol" in alloc_df.columns and "allocation_weight" in alloc_df.columns:
            allocation = [{"asset": str(r["symbol"]).strip(), "weight": float(r["allocation_weight"])} for _, r in alloc_df.iterrows()]
    if not allocation and (portfolio_dir / "portfolio_allocation_final.csv").exists():
        alloc_df = pd.read_csv(portfolio_dir / "portfolio_allocation_final.csv", parse_dates=["date"])
        latest = alloc_df["date"].max()
        alloc_df = alloc_df[alloc_df["date"] == latest]
        allocation = [{"asset": str(r["symbol"]).strip(), "weight": float(r["allocation_weight"])} for _, r in alloc_df.iterrows()]

    # metrics từ performance_metrics_genai hoặc portfolio_summary
    initial_cap = initial_capital
    try:
        from src.portfolio_engine.performance_metrics_genai import compute_performance_metrics
        from src.portfolio_engine.portfolio_builder import build_portfolio
        df_port = build_portfolio(initial_capital=initial_cap)
        perf = compute_performance_metrics(df_port, initial_capital=initial_cap)
        cagr = float(perf.get("CAGR", 0))
        max_dd = float(perf.get("Max Drawdown", 0))
        cvar_5 = float(perf.get("CVaR_5", -0.02))
        sharpe = perf.get("Sharpe Ratio")
        vol = perf.get("Volatility (annual)")
        total_return = perf.get("Total Return")
        sortino = perf.get("Sortino Ratio")
        beta = perf.get("Beta")
    except Exception:
        summary_path = powerbi_dir / "portfolio_summary.csv"
        perf_ext = PROJECT_DIR / "data_processed" / "reporting" / "performance_extended.csv"
        if summary_path.exists():
            sum_df = pd.read_csv(summary_path)
            m = dict(zip(sum_df.iloc[:, 0].astype(str).str.strip(), sum_df.iloc[:, 1]))
            cagr = float(m.get("cagr", m.get("CAGR", 0)))
            max_dd = float(m.get("max_drawdown", m.get("Max Drawdown", 0)))
            cvar_5 = float(m.get("cvar_5", -0.02))
            sharpe = m.get("sharpe_ratio") or m.get("Sharpe Ratio")
            vol = m.get("volatility_annual") or m.get("Annualized Volatility")
            sb, eb = m.get("start_balance"), m.get("end_balance")
            total_return = (float(eb) / float(sb) - 1) if sb and eb else None
            sortino = m.get("sortino_ratio")
            beta = None
        else:
            cagr = 0.0
            max_dd = 0.0
            cvar_5 = -0.02
            sharpe = None
            vol = None
            total_return = None
            sortino = None
            beta = None
        if perf_ext.exists():
            try:
                pe = pd.read_csv(perf_ext)
                pm = dict(zip(pe.iloc[:, 0], pe.iloc[:, 1]))
                if sharpe is None:
                    sharpe = pm.get("sharpe_ratio")
                if vol is None:
                    vol = pm.get("volatility_annual")
                if total_return is None and "end_balance" in pm and "start_balance" in pm:
                    total_return = pm["end_balance"] / pm["start_balance"] - 1
                if sortino is None:
                    sortino = pm.get("sortino_ratio")
            except Exception:
                pass

    # regime: từ risk_regime_timeseries ngày cuối
    regime = "NORMAL"
    try:
        reg_df = pd.read_csv(PROJECT_DIR / "data_processed" / "reporting" / "risk_regime_timeseries.csv", parse_dates=["date"])
        regime = str(reg_df.iloc[-1].get("portfolio_risk_regime", "NORMAL")).upper()
        if regime not in ("LOW", "NORMAL", "HIGH"):
            regime = "NORMAL"
    except Exception:
        pass

    # strategy: map risk_appetite -> defensive/balanced/aggressive
    strategy_map = {"conservative": "defensive", "balanced": "balanced", "aggressive": "aggressive"}
    strategy = strategy_map.get(risk_appetite, "balanced")

    pdf_path = None
    if reports_dir.exists():
        pdfs = list(reports_dir.glob("*.pdf"))
        if pdfs:
            pdf_path = str(max(pdfs, key=lambda p: p.stat().st_mtime))

    result = {
        "start_date": start_date,
        "end_date": end_date,
        "regime": regime,
        "strategy": strategy,
        "metrics": {
            "cagr": cagr,
            "max_drawdown": max_dd,
            "cvar_5": cvar_5,
            "volatility_annual": vol,
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "sortino_ratio": sortino,
            "beta": beta,
        },
        "allocation": allocation,
        "growth_series": growth_series,
    }
    if pdf_path:
        result["pdf_path"] = pdf_path
    return result
