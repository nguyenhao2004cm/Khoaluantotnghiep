#C:\Users\ASUS\fintech-project\data_processed\powerbi\export_powerbi_data.py
import pandas as pd
import sys
import time
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.portfolio_engine.portfolio_builder import build_portfolio
from src.portfolio_engine.performance_metrics import compute_drawdown, performance_summary, compute_annual_returns
from src.portfolio_engine.holdings import current_holdings, average_holdings
from src.portfolio_engine.asset_risk_return import (
    load_asset_returns,
    compute_asset_stats,
    compute_efficient_frontier
)

# =====================================
# CONFIG
# =====================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "data_processed" / "powerbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_to_csv(df: pd.DataFrame, path: Path, max_retries: int = 3):
    """Write CSV via temp file to avoid PermissionError when target is locked (e.g. by Power BI)."""
    path = Path(path)
    temp_path = path.with_suffix(".tmp")
    for attempt in range(max_retries):
        try:
            df.to_csv(temp_path, index=False, encoding="utf-8-sig")
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                fallback = path.with_name(path.stem + "_new.csv")
                df.to_csv(fallback, index=False, encoding="utf-8-sig")
                raise PermissionError(
                    f"Cannot write to {path}. Close Power BI/Excel if open. "
                    f"Data saved to {fallback}"
                )


# =====================================
# EXPORT PORTFOLIO TIME SERIES
# =====================================
def _get_initial_capital():
    try:
        from src.utils.web_run_config import load_web_run_config
        cfg = load_web_run_config()
        if cfg and cfg.get("initial_capital") is not None:
            return float(cfg["initial_capital"])
    except Exception:
        pass
    return 10000.0


def export_portfolio_timeseries():
    cap = _get_initial_capital()
    df_port = build_portfolio(initial_capital=cap)

    df_port["drawdown"] = compute_drawdown(df_port["portfolio_value"])

    out = df_port[[
        "date",
        "portfolio_value",
        "portfolio_return",
        "drawdown"
    ]]

    _safe_to_csv(out, OUT_DIR / "portfolio_timeseries.csv")

    print(" Exported portfolio_timeseries.csv")


# =====================================
# EXPORT PORTFOLIO SUMMARY (KPI)
# =====================================
def export_portfolio_summary():
    cap = _get_initial_capital()
    df_port = build_portfolio(initial_capital=cap)
    summary = performance_summary(df_port)

    df_summary = summary.reset_index()
    df_summary.columns = ["metric", "value"]

    _safe_to_csv(df_summary, OUT_DIR / "portfolio_summary.csv")

    print(" Exported portfolio_summary.csv")




# =====================================
# EXPORT ASSET ALLOCATION (CURRENT)
# =====================================
def export_asset_allocation_current():
    _, holdings = current_holdings()
    out = holdings[["symbol", "allocation_weight"]].copy()
    total = out["allocation_weight"].sum()
    if total > 0:
        out["allocation_weight"] = out["allocation_weight"] / total

    out.to_csv(
        OUT_DIR / "asset_allocation_current.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(" Exported asset_allocation_current.csv")


# =====================================
# EXPORT ASSET ALLOCATION (AVERAGE)
# =====================================
def export_asset_allocation_average():
    avg = average_holdings()

    avg = avg.rename(columns={
        "allocation_weight": "avg_weight"
    })

    _safe_to_csv(avg, OUT_DIR / "asset_allocation_average.csv")

    print(" Exported asset_allocation_average.csv")



# =====================================
# EXPORT ANNUAL RETURNS
# =====================================
def export_annual_returns():
    cap = _get_initial_capital()
    df_port = build_portfolio(initial_capital=cap)
    annual = compute_annual_returns(df_port)

    df_annual = (
        annual.reset_index()
              .rename(columns={"year": "year", "portfolio_return": "annual_return"})
    )

    # compute_annual_returns trả về Series, nên sửa đúng tên cột
    df_annual.columns = ["year", "annual_return"]

    _safe_to_csv(df_annual, OUT_DIR / "annual_returns.csv")

    print(" Exported annual_returns.csv")




# =====================================
# EXPORT ASSET RISK–RETURN
# =====================================
def export_asset_risk_return():
    # Ưu tiên danh mục user chọn (web config) thay vì holdings
    try:
        from src.utils.web_run_config import load_web_run_config
        cfg = load_web_run_config()
        if cfg and cfg.get("assets"):
            symbols = [s.strip().upper() for s in cfg["assets"]]
        else:
            _, holdings = current_holdings()
            symbols = holdings["symbol"].tolist()
    except Exception:
        _, holdings = current_holdings()
        symbols = holdings["symbol"].tolist()

    asset_returns = load_asset_returns(symbols)
    stats = compute_asset_stats(asset_returns)

    out = stats.reset_index().rename(columns={
        "index": "symbol",
        "Volatility": "volatility",
        "Expected Return": "expected_return"
    })

    _safe_to_csv(out, OUT_DIR / "asset_risk_return.csv")

    print(" Exported asset_risk_return.csv")


# =====================================
# EXPORT EFFICIENT FRONTIER
# =====================================
def export_efficient_frontier():
    # Ưu tiên danh mục user chọn (web config) thay vì holdings
    try:
        from src.utils.web_run_config import load_web_run_config
        cfg = load_web_run_config()
        if cfg and cfg.get("assets"):
            symbols = [s.strip().upper() for s in cfg["assets"]]
        else:
            _, holdings = current_holdings()
            symbols = holdings["symbol"].tolist()
    except Exception:
        _, holdings = current_holdings()
        symbols = holdings["symbol"].tolist()

    asset_returns = load_asset_returns(symbols)

    mean_returns = asset_returns.mean() * 252
    cov_matrix = asset_returns.cov() * 252

    vols, rets = compute_efficient_frontier(
        mean_returns.values,
        cov_matrix.values
    )

    df_frontier = pd.DataFrame({
        "volatility": vols,
        "expected_return": rets
    })

    _safe_to_csv(df_frontier, OUT_DIR / "efficient_frontier.csv")

    print(" Exported efficient_frontier.csv")

if __name__ == "__main__":
    export_portfolio_timeseries()
    export_portfolio_summary()
    export_asset_allocation_current()
    export_asset_allocation_average()
    export_annual_returns()
    export_asset_risk_return()
    export_efficient_frontier()


