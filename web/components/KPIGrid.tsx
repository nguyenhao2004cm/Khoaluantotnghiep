import React from "react";
import { fmtPct } from "../utils/format";
import type { PortfolioMetrics } from "../types";

interface KPIGridProps {
  metrics: PortfolioMetrics;
}

const KPI_ITEMS: {
  key: keyof PortfolioMetrics;
  label: string;
  color: string;
  format: (v: number) => string;
}[] = [
  { key: "cagr", label: "CAGR", color: "text-blue-400", format: (v) => fmtPct(v) },
  { key: "total_return", label: "Total Return", color: "text-emerald-400", format: (v) => fmtPct(v) },
  { key: "sharpe_ratio", label: "Sharpe", color: "text-sky-400", format: (v) => parseFloat(Number(v).toFixed(2)).toString() },
  { key: "volatility_annual", label: "Vol", color: "text-amber-400", format: (v) => fmtPct(v) },
  { key: "max_drawdown", label: "MDD", color: "text-red-400", format: (v) => fmtPct(v) },
  { key: "cvar_5", label: "CVaR 5%", color: "text-orange-400", format: (v) => fmtPct(v) },
];

const PortfolioOptimizerKPIGrid: React.FC<KPIGridProps> = ({ metrics }) => (
  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
    {KPI_ITEMS.map(({ key, label, color, format }) => {
      const raw = metrics[key];
      const val = typeof raw === "number" && !Number.isNaN(raw) ? raw : null;
      return (
        <div
          key={key}
          className="bg-zinc-900 p-5 rounded-2xl border border-zinc-800"
        >
          <p className="text-xs text-zinc-400 mb-1 truncate">{label}</p>
          <p className={`text-xl font-bold ${color}`}>
            {val !== null ? format(val) : "—"}
          </p>
        </div>
      );
    })}
  </div>
);

export default PortfolioOptimizerKPIGrid;
