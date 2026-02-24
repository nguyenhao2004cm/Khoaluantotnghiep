# POST /api/portfolio/optimize
# Nhận PortfolioRequest -> gọi pipeline (run_optimization_web) -> trả PortfolioResult + sinh PDF

from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from .schemas import (
    PortfolioRequest,
    PortfolioResult,
    PortfolioMetrics,
    AllocationItem,
    GrowthPoint,
)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

PROJECT_DIR = Path(__file__).resolve().parents[2]
POWERBI_DIR = PROJECT_DIR / "data_processed" / "powerbi"

_last_result = None


def get_last_optimization_result():
    return _last_result


def set_last_optimization_result(result):
    global _last_result
    _last_result = result


@router.post("/optimize", response_model=PortfolioResult)
def post_portfolio_optimize(body: PortfolioRequest):
    """
    Tối ưu danh mục theo start_date, end_date (mặc định hôm nay), assets, risk_appetite.
    Chỉ trả dữ liệu thật từ pipeline (build_portfolio_allocation -> export_powerbi -> report_pdf).
    Không trả dữ liệu mẫu; nếu pipeline lỗi thì trả 503 và hướng dẫn chạy run_all.py.
    """
    end_date = body.end_date or datetime.now().strftime("%Y-%m-%d")
    initial_capital = body.initial_capital if body.initial_capital is not None else 10000.0
    try:
        from .run_optimization_web import run_optimization_for_web
        raw = run_optimization_for_web(
            assets=body.assets,
            start_date=body.start_date,
            end_date=end_date,
            risk_appetite=body.risk_appetite,
            initial_capital=initial_capital,
        )
        result = _raw_to_portfolio_result(raw)
        set_last_optimization_result(result.model_dump())
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("Pipeline failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline tối ưu chưa chạy hoặc thiếu dữ liệu. "
                "Vui lòng từ thư mục gốc dự án chạy: python run_all.py "
                "Sau khi chạy xong, thử lại tối ưu từ web. Chi tiết: " + str(e)
            ),
        )


def _sanitize_float(v):
    """Chuyển NaN/Inf thành None để JSON serialize được."""
    if v is None:
        return None
    try:
        f = float(v)
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _raw_to_portfolio_result(raw: dict) -> PortfolioResult:
    m = raw["metrics"]
    metrics = PortfolioMetrics(
        cagr=_sanitize_float(m.get("cagr")) or 0,
        max_drawdown=_sanitize_float(m.get("max_drawdown")) or 0,
        cvar_5=_sanitize_float(m.get("cvar_5")) or 0,
        volatility_annual=_sanitize_float(m.get("volatility_annual")),
        sharpe_ratio=_sanitize_float(m.get("sharpe_ratio")),
        total_return=_sanitize_float(m.get("total_return")),
        sortino_ratio=_sanitize_float(m.get("sortino_ratio")),
        beta=_sanitize_float(m.get("beta")),
    )
    allocation = [
        AllocationItem(asset=str(x["asset"]), weight=_sanitize_float(x.get("weight")) or 0)
        for x in raw["allocation"]
    ]
    growth_series = [
        GrowthPoint(date=str(x["date"]), value=_sanitize_float(x.get("value")) or 0)
        for x in raw["growth_series"]
    ]
    return PortfolioResult(
        start_date=raw["start_date"],
        end_date=raw["end_date"],
        regime=raw["regime"],
        strategy=raw["strategy"],
        metrics=metrics,
        allocation=allocation,
        growth_series=growth_series,
    )


def _stub_portfolio_result(req: PortfolioRequest) -> PortfolioResult:
    n = max(1, len(req.assets))
    w = 1.0 / n
    allocation = [AllocationItem(asset=a, weight=round(w, 4)) for a in req.assets[:10]]
    growth_series = [
        GrowthPoint(date=req.start_date, value=1.0),
        GrowthPoint(date=req.end_date, value=1.05),
    ]
    metrics = PortfolioMetrics(
        cagr=0.05,
        max_drawdown=-0.10,
        cvar_5=-0.02,
        volatility_annual=0.15,
        sharpe_ratio=0.8,
    )
    return PortfolioResult(
        start_date=req.start_date,
        end_date=req.end_date,
        regime="NORMAL",
        strategy="balanced",
        metrics=metrics,
        allocation=allocation,
        growth_series=growth_series,
    )


@router.get("/timeseries")
def get_portfolio_timeseries():
    """
    Dữ liệu giống figure 'portfolio growth' và 'drawdown' trong report_pdf.py.
    Nguồn: data_processed/powerbi/portfolio_timeseries.csv
    """
    path = POWERBI_DIR / "portfolio_timeseries.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa có portfolio_timeseries.csv. Hãy chạy tối ưu trước.")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                "portfolio_value": float(r["portfolio_value"]),
                "drawdown": float(r.get("drawdown", 0.0)),
            }
        )
    return out


@router.get("/asset-risk-return")
def get_asset_risk_return():
    """
    Dữ liệu giống figure 'risk-return scatter' trong report_pdf.py.
    Nguồn: data_processed/powerbi/asset_risk_return.csv
    """
    path = POWERBI_DIR / "asset_risk_return.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa có asset_risk_return.csv. Hãy chạy tối ưu trước.")
    df = pd.read_csv(path)
    # kỳ vọng các cột: symbol, volatility, expected_return
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "symbol": str(r.get("symbol", "")).strip(),
                "volatility": float(r.get("volatility", 0.0)),
                "expected_return": float(r.get("expected_return", 0.0)),
            }
        )
    return out


@router.get("/annual-returns")
def get_annual_returns():
    """
    Dữ liệu giống bảng/biểu đồ annual returns trong report_pdf.py.
    Nguồn: data_processed/powerbi/annual_returns.csv
    """
    path = POWERBI_DIR / "annual_returns.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa có annual_returns.csv. Hãy chạy tối ưu trước.")
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "year": int(r.get("year")),
                "annual_return": float(r.get("annual_return", 0.0)),
            }
        )
    return out
