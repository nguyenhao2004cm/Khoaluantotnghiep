import React from "react";
import { StockSelector, type CompanySuggestion } from "./StockSelector";
import { StartDatePicker } from "./StartDatePicker";
import { RiskAppetite, RiskAppetiteLabel } from "../types";
import { X } from "lucide-react";

export type TimeRange = "3M" | "6M" | "1Y" | "3Y" | "MAX";

interface ConfigBarProps {
  selectedAssets: string[];
  onAddStock: (item: CompanySuggestion) => void;
  onRemoveStock: (ma: string) => void;
  startDate: Date;
  onStartDateChange: (d: Date) => void;
  riskAppetite: RiskAppetite;
  onRiskChange: (r: RiskAppetite) => void;
  initialCapital: number;
  onInitialCapitalChange: (v: number) => void;
  timeRange?: TimeRange;
  onTimeRangeChange?: (r: TimeRange) => void;
  onRun: () => void;
  isProcessing: boolean;
}

const RISK_OPTIONS = [
  RiskAppetite.CONSERVATIVE,
  RiskAppetite.BALANCED,
  RiskAppetite.AGGRESSIVE,
] as const;

const ConfigBar: React.FC<ConfigBarProps> = ({
  selectedAssets,
  onAddStock,
  onRemoveStock,
  startDate,
  onStartDateChange,
  riskAppetite,
  onRiskChange,
  initialCapital,
  onInitialCapitalChange,
  timeRange = "1Y",
  onTimeRangeChange,
  onRun,
  isProcessing,
}) => {
  const setRange = (r: TimeRange) => {
    const d = new Date();
    if (r === "3M") d.setMonth(d.getMonth() - 3);
    else if (r === "6M") d.setMonth(d.getMonth() - 6);
    else if (r === "1Y") d.setFullYear(d.getFullYear() - 1);
    else if (r === "3Y") d.setFullYear(d.getFullYear() - 3);
    else d.setFullYear(2020);
    onStartDateChange(d);
    onTimeRangeChange?.(r);
  };
  return (
  <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 sm:p-6 shadow-md">
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6 items-end">
      {/* Danh mục cổ phiếu */}
      <div className="lg:col-span-3">
        <label className="text-sm text-zinc-400 block mb-2">
          Danh mục cổ phiếu
        </label>
        <StockSelector onSelect={onAddStock} placeholder="Nhập mã cổ phiếu..." />
        <div className="mt-3 flex flex-wrap gap-2">
          {selectedAssets.map((ma) => (
            <span
              key={ma}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-sky-500/20 text-sky-400 rounded-xl text-sm font-medium border border-sky-500/50"
            >
              {ma}
              <button
                onClick={() => onRemoveStock(ma)}
                className="hover:bg-sky-500/30 rounded-full p-0.5 transition-colors"
                aria-label={`Xóa ${ma}`}
              >
                <X size={12} />
              </button>
            </span>
          ))}
        </div>
      </div>

      {/* Giá trị danh mục ban đầu */}
      <div className="lg:col-span-2">
        <label className="text-sm text-zinc-400 block mb-2">
          Giá trị danh mục ban đầu (VNĐ)
        </label>
        <input
          type="number"
          min={1000}
          step={1000}
          value={initialCapital}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v) && v >= 0) onInitialCapitalChange(v);
          }}
          className="w-full px-4 py-3 rounded-xl border border-[#2a2a2a] bg-[#1a1a1a] text-white focus:ring-2 focus:ring-sky-500 focus:border-sky-500 outline-none text-sm"
          placeholder="10000"
        />
      </div>

      {/* Time Filter */}
      <div className="lg:col-span-2">
        <label className="text-sm text-zinc-400 block mb-2">Khoảng thời gian</label>
        <div className="flex flex-wrap gap-1.5">
          {(["3M", "6M", "1Y", "3Y", "MAX"] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`px-3 py-2 rounded-lg text-xs font-medium transition-all min-w-[2.5rem] ${
                timeRange === r
                  ? "bg-sky-500 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Ngày bắt đầu */}
      <div className="lg:col-span-2">
        <StartDatePicker
          value={startDate}
          onChange={onStartDateChange}
          label="Ngày bắt đầu tối ưu"
        />
      </div>

      {/* Khẩu vị rủi ro */}
      <div className="lg:col-span-2">
        <label className="text-sm text-zinc-400 block mb-2">
          Khẩu vị rủi ro
        </label>
        <div className="flex flex-wrap gap-2">
          {RISK_OPTIONS.map((key) => (
            <button
              key={key}
              onClick={() => onRiskChange(key)}
              className={`px-3 py-2 sm:px-4 rounded-xl border text-xs sm:text-sm font-medium transition-all ${
                riskAppetite === key
                  ? "bg-sky-500 border-sky-500 text-white"
                  : "border-zinc-700 text-zinc-400 hover:border-sky-500"
              }`}
            >
              {RiskAppetiteLabel[key]}
            </button>
          ))}
        </div>
      </div>

      {/* Nút chạy */}
      <div className="lg:col-span-2">
        <button
          onClick={onRun}
          disabled={isProcessing || selectedAssets.length === 0}
          className="w-full bg-sky-500 hover:bg-sky-400 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white py-3 rounded-xl font-medium transition-all min-h-[44px] touch-manipulation"
        >
          {isProcessing ? "Đang phân tích..." : "Phân tích"}
        </button>
      </div>
    </div>
  </div>
  );
};

export default ConfigBar;
