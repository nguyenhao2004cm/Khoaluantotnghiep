
import { RiskRegime, MarketOverviewData } from './types';

export const ASSET_LIST = [
  "FPT", "VNM", "HPG", "TCB", "VIC", "MSN", "MWG", "MBB", "VHM", "GAS", "REE", "PNJ"
];

/**
 * MOCK FUNCTION
 * Used for frontend development only.
 * Replaced by backend API: GET /api/market/overview
 */
const generateRiskData = () => {
  return Array.from({ length: 40 }, (_, i) => {
    const baseDate = new Date();
    baseDate.setDate(baseDate.getDate() - (40 - i));
    const dateStr = baseDate.toISOString().split('T')[0];
    const score = 40 + Math.random() * 40 + (i > 30 ? 15 : 0);
    return {
      date: dateStr,
      score: Math.min(100, score),
      regime: score > 75 ? RiskRegime.HIGH : (score > 45 ? RiskRegime.NORMAL : RiskRegime.LOW)
    };
  });
};

const riskSeries = generateRiskData();

/**
 * MOCK MARKET OVERVIEW DATA
 * ----------------------------------
 * This data simulates the output of:
 *   GET /api/market/overview
 * Backend source:
 *   Autoencoder latent risk + regime normalization
 * Used ONLY for UI development & demonstration
 */
export const MOCK_MARKET_DATA: MarketOverviewData = {
  current_regime: RiskRegime.HIGH,
  regime_stability_days: 12,
  risk_score_series: riskSeries.map(d => ({ date: d.date, score: d.score })),
  regime_timeline: riskSeries.map(d => ({ date: d.date, regime: d.regime })),
  sector_risks: [
    { name: 'Ngân hàng', riskLevel: RiskRegime.HIGH, score: 82, trend: 'up' },
    { name: 'Công nghệ', riskLevel: RiskRegime.NORMAL, score: 54, trend: 'stable' },
    { name: 'Bất động sản', riskLevel: RiskRegime.HIGH, score: 88, trend: 'up' },
    { name: 'Tiêu dùng', riskLevel: RiskRegime.LOW, score: 32, trend: 'down' },
    { name: 'Năng lượng', riskLevel: RiskRegime.NORMAL, score: 48, trend: 'stable' },
  ],
  heatmap: [
    {
      name: 'Tài chính',
      assets: [
        { ticker: 'TCB', change: -2.4, marketCap: 150, riskScore: 78 },
        { ticker: 'MBB', change: -1.8, marketCap: 120, riskScore: 72 },
        { ticker: 'SSI', change: -3.5, marketCap: 80, riskScore: 85 },
        { ticker: 'VCB', change: -0.5, marketCap: 450, riskScore: 40 },
      ]
    },
    {
      name: 'Công nghệ',
      assets: [
        { ticker: 'FPT', change: 1.2, marketCap: 130, riskScore: 45 },
        { ticker: 'CMG', change: 0.8, marketCap: 20, riskScore: 52 },
      ]
    },
    {
      name: 'Sản xuất',
      assets: [
        { ticker: 'HPG', change: -2.1, marketCap: 180, riskScore: 75 },
        { ticker: 'VNM', change: 0.4, marketCap: 160, riskScore: 35 },
        { ticker: 'MSN', change: -1.2, marketCap: 90, riskScore: 65 },
      ]
    },
    {
      name: 'Bất động sản',
      assets: [
        { ticker: 'VIC', change: -4.2, marketCap: 200, riskScore: 92 },
        { ticker: 'VHM', change: -3.8, marketCap: 210, riskScore: 90 },
        { ticker: 'NVL', change: -6.5, marketCap: 40, riskScore: 98 },
      ]
    }
  ],
  summary: {
    high_ratio: 0.35,
    normal_ratio: 0.50,
    low_ratio: 0.15,
  }
};

