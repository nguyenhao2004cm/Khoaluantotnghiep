/* =========================
   ENUMS
========================= */

export enum RiskRegime {
  LOW = "LOW",
  NORMAL = "NORMAL",
  HIGH = "HIGH",
}

export enum RiskAppetite {
  CONSERVATIVE = "conservative",
  BALANCED = "balanced",
  AGGRESSIVE = "aggressive",
}

export const RiskAppetiteLabel: Record<RiskAppetite, string> = {
  [RiskAppetite.CONSERVATIVE]: "Thận trọng",
  [RiskAppetite.BALANCED]: "Cân bằng",
  [RiskAppetite.AGGRESSIVE]: "Chấp nhận rủi ro",
};

export type AllocationStrategy =
  | "defensive"
  | "balanced"
  | "aggressive";

/* =========================
   REQUEST → BACKEND
========================= */

export interface PortfolioRequest {
  start_date: string;
  end_date?: string; // Mặc định = hôm nay (backend auto)
  assets: string[];
  risk_appetite: RiskAppetite;
  initial_capital?: number; // Giá trị danh mục ban đầu (VNĐ), mặc định 10000
}

/* =========================
   MARKET OVERVIEW
========================= */

export interface HeatmapAsset {
  ticker: string;
  change: number;
  marketCap: number;
  riskScore: number;
}

export interface SectorGroup {
  name: string;
  assets: HeatmapAsset[];
}

export interface SectorRisk {
  name: string;
  riskLevel: RiskRegime;
  score: number;
  trend: "up" | "down" | "stable";
}

export interface MarketOverviewData {
  current_regime: RiskRegime;
  regime_stability_days: number;

  risk_score_series: {
    date: string;
    score: number;
  }[];

  regime_timeline: {
    date: string;
    regime: RiskRegime;
  }[];

  heatmap: SectorGroup[];
  sector_risks: SectorRisk[];

  summary: {
    high_ratio: number;
    normal_ratio: number;
    low_ratio: number;
  };
}

/* =========================
   PORTFOLIO RESULT
========================= */

export interface PortfolioMetrics {
  cagr: number;
  max_drawdown: number;
  cvar_5: number;

  volatility_annual?: number;
  sharpe_ratio?: number;
  total_return?: number;
  sortino_ratio?: number;
  beta?: number;
}

export interface AllocationItem {
  asset: string;
  weight: number;
}

export interface GrowthPoint {
  date: string;
  value: number;
}

export interface PortfolioResult {
  start_date: string;
  end_date: string;

  regime: RiskRegime;
  strategy: AllocationStrategy;

  metrics: PortfolioMetrics;
  allocation: AllocationItem[];
  growth_series: GrowthPoint[];
}

/* =========================
   AI CHAT
========================= */

export interface ChatContext {
  market_regime: RiskRegime;
  user_assets: string[];
  portfolio_metrics: PortfolioMetrics;
  current_strategy: AllocationStrategy;
}

export interface ChatMessage {
  role: "user" | "model";
  parts: { text: string }[];
}

/** Alias for AI Advisor component */
export type AdvisorContext = ChatContext;
