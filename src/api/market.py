# GET /api/market/overview
# Load risk_regime_timeseries + aggregation → JSON (mirror MarketOverviewData)

from pathlib import Path
from fastapi import APIRouter
import pandas as pd

from .schemas import (
    MarketOverviewData,
    MarketOverviewSummary,
    SectorRisk,
    SectorGroup,
    HeatmapAsset,
)

router = APIRouter(prefix="/api/market", tags=["market"])

PROJECT_DIR = Path(__file__).resolve().parents[2]
RISK_REGIME_FILE = PROJECT_DIR / "data_processed" / "reporting" / "risk_regime_timeseries.csv"


@router.get("/overview", response_model=MarketOverviewData)
def get_market_overview():
    """
    Trả dữ liệu toàn cảnh thị trường (regime, timeline, sector).
    Nguồn: risk_regime_timeseries.csv + (tuỳ chọn) sector aggregation.
    """
    if not RISK_REGIME_FILE.exists():
        return _fallback_market_overview()

    df = pd.read_csv(RISK_REGIME_FILE, parse_dates=["date"])
    df = df.sort_values("date")

    # Current regime = last row
    last = df.iloc[-1]
    current_regime = str(last.get("portfolio_risk_regime", "NORMAL")).upper()
    if current_regime not in ("LOW", "NORMAL", "HIGH"):
        current_regime = "NORMAL"

    # Regime stability: số ngày liên tiếp cùng regime (đơn giản: 7)
    regime_stability_days = 7

    # Risk score series: last 90 days
    tail = df.tail(90)
    risk_score_series = [
        {"date": row["date"].strftime("%Y-%m-%d"), "score": float(row.get("mean_risk_z_smooth", 0) * 20 + 50)}
        for _, row in tail.iterrows()
    ]

    # Regime timeline
    regime_timeline = [
        {"date": row["date"].strftime("%Y-%m-%d"), "regime": str(row.get("portfolio_risk_regime", "NORMAL")).upper()}
        for _, row in tail.iterrows()
    ]

    # Placeholder heatmap / sector_risks (có thể thay bằng aggregation thật)
    heatmap = [
        SectorGroup(name="Tài chính", assets=[
            HeatmapAsset(ticker="TCB", change=-2.4, marketCap=150, riskScore=78),
            HeatmapAsset(ticker="MBB", change=-1.8, marketCap=120, riskScore=72),
        ]),
        SectorGroup(name="Công nghệ", assets=[
            HeatmapAsset(ticker="FPT", change=1.2, marketCap=130, riskScore=45),
        ]),
    ]
    sector_risks = [
        SectorRisk(name="Ngân hàng", riskLevel="HIGH", score=82, trend="up"),
        SectorRisk(name="Công nghệ", riskLevel="NORMAL", score=54, trend="stable"),
    ]
    summary = MarketOverviewSummary(high_ratio=0.35, normal_ratio=0.50, low_ratio=0.15)

    return MarketOverviewData(
        current_regime=current_regime,
        regime_stability_days=regime_stability_days,
        risk_score_series=risk_score_series,
        regime_timeline=regime_timeline,
        heatmap=heatmap,
        sector_risks=sector_risks,
        summary=summary,
    )


def _fallback_market_overview() -> MarketOverviewData:
    """Khi chưa có file CSV."""
    return MarketOverviewData(
        current_regime="NORMAL",
        regime_stability_days=0,
        risk_score_series=[],
        regime_timeline=[],
        heatmap=[],
        sector_risks=[],
        summary=MarketOverviewSummary(high_ratio=0.33, normal_ratio=0.34, low_ratio=0.33),
    )
