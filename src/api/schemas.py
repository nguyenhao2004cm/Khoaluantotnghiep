# src/api/schemas.py
# Data contract: mirror 100% với Trang web/types.ts

from pydantic import BaseModel
from typing import List, Literal, Optional

# --- Request (Frontend → Backend) ---

RiskAppetite = Literal["conservative", "balanced", "aggressive"]


class PortfolioRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: Optional[str] = None  # Mặc định = hôm nay (backend auto)
    assets: List[str]
    risk_appetite: RiskAppetite
    initial_capital: Optional[float] = 10000.0  # Giá trị danh mục ban đầu (VNĐ)


# --- Market Overview ---

Regime = Literal["LOW", "NORMAL", "HIGH"]


class HeatmapAsset(BaseModel):
    ticker: str
    change: float
    marketCap: float
    riskScore: float


class SectorGroup(BaseModel):
    name: str
    assets: List[HeatmapAsset]


class SectorRisk(BaseModel):
    name: str
    riskLevel: Regime
    score: float
    trend: Literal["up", "down", "stable"]


class MarketOverviewSummary(BaseModel):
    high_ratio: float
    normal_ratio: float
    low_ratio: float


class MarketOverviewData(BaseModel):
    current_regime: Regime
    regime_stability_days: int
    risk_score_series: List[dict]  # [{ date, score }]
    regime_timeline: List[dict]     # [{ date, regime }]
    heatmap: List[SectorGroup]
    sector_risks: List[SectorRisk]
    summary: MarketOverviewSummary


# --- Portfolio Result (Backend → Frontend) ---

AllocationStrategy = Literal["defensive", "balanced", "aggressive"]


class PortfolioMetrics(BaseModel):
    cagr: float
    max_drawdown: float
    cvar_5: float
    volatility_annual: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    total_return: Optional[float] = None
    sortino_ratio: Optional[float] = None
    beta: Optional[float] = None


class AllocationItem(BaseModel):
    asset: str
    weight: float


class GrowthPoint(BaseModel):
    date: str
    value: float


class PortfolioResult(BaseModel):
    start_date: str
    end_date: str
    regime: Regime
    strategy: AllocationStrategy
    metrics: PortfolioMetrics
    allocation: List[AllocationItem]
    growth_series: List[GrowthPoint]


# --- AI Chat ---

class ChatContext(BaseModel):
    market_regime: Regime
    user_assets: List[str]
    portfolio_metrics: PortfolioMetrics
    current_strategy: AllocationStrategy


class ChatMessagePart(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: Literal["user", "model"]
    parts: List[ChatMessagePart]


class ChatRequest(BaseModel):
    message: str
    context: ChatContext
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
