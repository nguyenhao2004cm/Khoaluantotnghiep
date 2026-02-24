import { PortfolioRequest, PortfolioResult } from "../types";
import { API_BASE } from "../apiConfig";

export type PortfolioTimeseriesPoint = {
  date: string;
  portfolio_value: number;
  drawdown: number;
};

export type AssetRiskReturnPoint = {
  symbol: string;
  expected_return: number;
  volatility: number;
};

export type AnnualReturnPoint = {
  year: number;
  annual_return: number;
};

export const portfolioService = {
  async optimize(payload: PortfolioRequest): Promise<PortfolioResult> {
    const res = await fetch(`${API_BASE}/api/portfolio/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      let msg = "Không thể tối ưu danh mục.";
      try {
        const j = await res.json();
        if (typeof (j as { detail?: string }).detail === "string") msg = (j as { detail: string }).detail;
      } catch {
        if (res.status === 503) msg = "Pipeline chưa chạy hoặc thiếu dữ liệu. Vui lòng chạy python run_all.py từ thư mục gốc dự án, sau đó thử lại.";
        else if (res.status >= 500) msg = "Lỗi backend. Vui lòng kiểm tra backend đang chạy (port 8000).";
      }
      throw new Error(msg);
    }

    return res.json();
  },

  async getTimeseries(): Promise<PortfolioTimeseriesPoint[]> {
    const res = await fetch(`${API_BASE}/api/portfolio/timeseries`);
    if (!res.ok) throw new Error("Không lấy được portfolio_timeseries.");
    return res.json();
  },

  async getAssetRiskReturn(): Promise<AssetRiskReturnPoint[]> {
    const res = await fetch(`${API_BASE}/api/portfolio/asset-risk-return`);
    if (!res.ok) throw new Error("Không lấy được asset_risk_return.");
    return res.json();
  },

  async getAnnualReturns(): Promise<AnnualReturnPoint[]> {
    const res = await fetch(`${API_BASE}/api/portfolio/annual-returns`);
    if (!res.ok) throw new Error("Không lấy được annual_returns.");
    return res.json();
  },

  async downloadReport(): Promise<Blob> {
    const res = await fetch(`${API_BASE}/api/report/pdf`);
    if (!res.ok) {
      if (res.headers.get("content-type")?.includes("application/json")) {
        const j = await res.json().catch(() => ({}));
        throw new Error((j as { detail?: string }).detail || "Chưa có báo cáo PDF.");
      }
      throw new Error("Chưa có báo cáo PDF. Vui lòng chạy tối ưu danh mục thành công trước.");
    }
    return res.blob();
  },
};
