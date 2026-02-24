import React from "react";
import { Lightbulb } from "lucide-react";
import type { PortfolioResult } from "../types";
import { fmtPct } from "../utils/format";

interface InsightBoxProps {
  result: PortfolioResult;
  topAsset?: string;
}

/**
 * AI Insight Box - insight ngắn gọn dưới KPI
 * Placeholder: hiển thị metrics cơ bản. Có thể tích hợp LLM sau.
 */
const InsightBox: React.FC<InsightBoxProps> = ({ result, topAsset }) => {
  const m = result.metrics;
  const sharpe = m.sharpe_ratio;
  const topWeight = result.allocation[0];
  const insights: string[] = [];

  if (sharpe != null && !Number.isNaN(sharpe)) {
    if (sharpe > 1) insights.push(`Sharpe ${parseFloat(sharpe.toFixed(2))} — danh mục có mức rủi ro–lợi nhuận tốt.`);
    else if (sharpe > 0) insights.push(`Sharpe ${parseFloat(sharpe.toFixed(2))} — mức rủi ro trung bình.`);
  }
  if (topWeight) {
    insights.push(`${topWeight.asset} đóng góp ${fmtPct(topWeight.weight)} tổng danh mục.`);
  }
  if (m.max_drawdown < -0.1) {
    insights.push(`MDD ${fmtPct(m.max_drawdown)} — cần theo dõi sát khi thị trường biến động.`);
  }

  if (insights.length === 0) {
    insights.push("Chạy phân tích để xem insight chi tiết.");
  }

  return (
    <div className="bg-zinc-900/80 border border-sky-500/30 rounded-2xl p-5 flex gap-4">
      <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-sky-500/20 flex items-center justify-center">
        <Lightbulb size={20} className="text-sky-400" />
      </div>
      <div className="flex-1 space-y-1">
        <p className="text-sm font-medium text-zinc-300">Insight</p>
        <ul className="text-sm text-zinc-400 space-y-0.5">
          {insights.map((s, i) => (
            <li key={i}>• {s}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default InsightBox;
