import React, { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  Customized,
} from "recharts";
import { portfolioService } from "../services/portfolioService";
import { RiskAppetite, PortfolioResult } from "../types";
import ConfigBar from "./ConfigBar";
import Footer from "./Footer";
import KPIGrid from "./KPIGrid";
import InsightBox from "./InsightBox";
import { fmtPct, fmtNum, fmtCompact } from "../utils/format";
import { Download, Target } from "lucide-react";
import type { TimeRange } from "./ConfigBar";

const COLORS = ["#38bdf8", "#22d3ee", "#60a5fa", "#93c5fd", "#e0f2fe"];

const PortfolioOptimizer: React.FC = () => {
  const [startDate, setStartDate] = useState<Date>(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return d;
  });
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
  const [riskAppetite, setRiskAppetite] = useState<RiskAppetite>(
    RiskAppetite.BALANCED
  );
  const [initialCapital, setInitialCapital] = useState(10000);
  const [timeRange, setTimeRange] = useState<TimeRange>("1Y");
  const [result, setResult] = useState<PortfolioResult | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [timeseries, setTimeseries] = useState<
    { date: string; portfolio_value: number; drawdown: number }[] | null
  >(null);
  const [assetRiskReturn, setAssetRiskReturn] = useState<
    { symbol: string; expected_return: number; volatility: number }[] | null
  >(null);
  const [annualReturns, setAnnualReturns] = useState<
    { year: number; annual_return: number }[] | null
  >(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const addStock = (item: { ma: string; ten: string }) => {
    if (!selectedAssets.includes(item.ma)) {
      setSelectedAssets([...selectedAssets, item.ma]);
    }
  };

  const removeStock = (ma: string) => {
    setSelectedAssets(selectedAssets.filter((a) => a !== ma));
  };

  const runOptimization = async () => {
    setIsProcessing(true);
    setResult(null);
    setActiveIndex(null);
    setTimeseries(null);
    setAssetRiskReturn(null);
    setAnnualReturns(null);

    try {
      const payload = {
        start_date: startDate.toISOString().split("T")[0],
        assets: selectedAssets,
        risk_appetite: riskAppetite,
        initial_capital: initialCapital,
      };

      const res = await portfolioService.optimize(payload);
      setResult(res);

      const [ts, rr, ar] = await Promise.all([
        portfolioService.getTimeseries(),
        portfolioService.getAssetRiskReturn(),
        portfolioService.getAnnualReturns(),
      ]);
      setTimeseries(ts);
      setAssetRiskReturn(rr);
      setAnnualReturns(ar);
    } catch (err) {
      console.error(err);
      alert(
        err instanceof Error
          ? err.message
          : "Không thể tối ưu danh mục. Vui lòng thử lại."
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* I. Config Bar - full width ngang */}
      <ConfigBar
        selectedAssets={selectedAssets}
        onAddStock={addStock}
        onRemoveStock={removeStock}
        startDate={startDate}
        onStartDateChange={setStartDate}
        riskAppetite={riskAppetite}
        onRiskChange={setRiskAppetite}
        initialCapital={initialCapital}
        onInitialCapitalChange={setInitialCapital}
        timeRange={timeRange}
        onTimeRangeChange={setTimeRange}
        onRun={runOptimization}
        isProcessing={isProcessing}
      />

      {/* Empty / Loading state */}
      {!result && !isProcessing && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-8 sm:p-16 flex flex-col items-center justify-center text-center">
          <Target size={48} className="text-zinc-600 mb-4 sm:size-14" />
          <h4 className="text-zinc-400 text-lg font-medium mb-2">
          </h4>
          <p className="text-zinc-500 text-sm max-w-md">
            Nhập mã cổ phiếu, chọn ngày bắt đầu và khẩu vị rủi ro, sau đó nhấn
            Phân tích.
          </p>
        </div>
      )}

      {isProcessing && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-8 sm:p-16 flex flex-col items-center justify-center text-center">
          <div className="w-12 h-12 sm:w-16 sm:h-16 border-4 border-zinc-700 border-t-sky-500 rounded-full animate-spin mb-6" />
          <h4 className="text-white text-lg font-medium">
            Đang tối ưu hóa danh mục
          </h4>
          <p className="text-zinc-500 text-sm mt-2">
          </p>
        </div>
      )}

      {/* II. Kết quả - layout mới */}
      {result && !isProcessing && (
        <div className="space-y-8 animate-slide-up">
          {/* KPI: CAGR | Total Return | Sharpe | Vol | MDD | CVaR */}
          <KPIGrid metrics={result.metrics} />

          {/* AI Insight Box */}
          <InsightBox result={result} />

          {/* Nút tải PDF – đặt giữa, tách khỏi biểu đồ */}
          <div className="flex justify-center">
            <button
              onClick={async () => {
                try {
                  const blob = await portfolioService.downloadReport();
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = "Portfolio_Optimization_Report.pdf";
                  a.click();
                  window.URL.revokeObjectURL(url);
                } catch (e) {
                  console.error(e);
                  alert(
                    e instanceof Error
                      ? e.message
                      : "Chưa có báo cáo PDF. Vui lòng chạy tối ưu danh mục thành công trước."
                  );
                }
              }}
              className="bg-sky-500 hover:bg-sky-400 text-white px-6 py-3 rounded-xl font-medium transition-all flex items-center gap-2"
            >
              <Download size={18} />
              Tải báo cáo PDF
            </button>
          </div>

          {/* Biểu đồ tăng trưởng – full width */}
          <div className="bg-zinc-900 p-4 sm:p-8 rounded-2xl border border-zinc-800">
            <h2 className="text-lg font-semibold text-white mb-6">
              Biểu đồ tăng trưởng tài sản
            </h2>
            <div className="h-[260px] sm:h-[320px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={result.growth_series} margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" opacity={0.4} />
                  <XAxis dataKey="date" hide />
                  <YAxis
                    stroke="#cbd5e1"
                    tick={{ fill: "#e2e8f0", fontSize: 12 }}
                    tickFormatter={(v) => fmtCompact(Number(v)) + " đ"}
                    width={80}
                    tickCount={5}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#18181b",
                      border: "1px solid #3f3f46",
                      borderRadius: "8px",
                      color: "#ffffff",
                    }}
                    formatter={(v: number) => [fmtNum(Number(v)), "Giá trị danh mục"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#38bdf8"
                    strokeWidth={2}
                    fill="#38bdf840"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Drawdown + Risk-Return – 2 cột */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">
              <h3 className="mb-4 font-medium text-white">Giá trị sụt giảm danh mục</h3>
              <div className="h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeseries || []}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" opacity={0.4} />
                    <XAxis dataKey="date" hide />
                    <YAxis
                      stroke="#cbd5e1"
                      tick={{ fill: "#e2e8f0", fontSize: 12 }}
                      tickFormatter={(v) => fmtPct(Number(v))}
                      tickCount={5}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#18181b",
                        border: "1px solid #3f3f46",
                        color: "#ffffff",
                      }}
                      formatter={(v: number) => [fmtPct(Number(v)), "Giá trị sụt giảm"]}
                    />
                    <Line
                      type="monotone"
                      dataKey="drawdown"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">
              <h3 className="mb-4 font-medium text-white">Biểu đồ rủi ro và lợi nhuận</h3>
              <div className="h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 35, right: 35, bottom: 45, left: 55 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" opacity={0.4} />
                    <XAxis
                      type="number"
                      dataKey="volatility"
                      stroke="#cbd5e1"
                      tick={{ fill: "#e2e8f0", fontSize: 12 }}
                      tickFormatter={(v) => fmtPct(Number(v))}
                      tickMargin={8}
                      label={{
                        value: "Độ biến động",
                        position: "insideBottom",
                        offset: -8,
                        fill: "#94a3b8",
                        fontSize: 12,
                      }}
                      tickCount={5}
                      domain={["auto", "auto"]}
                    />
                    <YAxis
                      type="number"
                      dataKey="expected_return"
                      stroke="#cbd5e1"
                      tick={{ fill: "#e2e8f0", fontSize: 12 }}
                      tickFormatter={(v) => fmtPct(Number(v))}
                      tickMargin={10}
                      width={70}
                      label={{
                        value: "Lợi nhuận kỳ vọng",
                        angle: -90,
                        position: "insideLeft",
                        offset: 15,
                        fill: "#94a3b8",
                        fontSize: 12,
                      }}
                      tickCount={5}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: "3 3" }}
                      contentStyle={{
                        backgroundColor: "#18181b",
                        border: "1px solid #ececf5",
                        borderRadius: "8px",
                        color: "#ffffff",
                      }}
                      itemStyle={{ color: "#ffffff" }}
                      labelStyle={{ color: "#ffffff", fontWeight: 600 }}
                      formatter={(v: number, name: string) => [
                        fmtPct(Number(v)),
                        name === "expected_return" ? "Lợi nhuận" : "Rủi ro",
                      ]}
                    />
                    <Scatter
                      data={assetRiskReturn || []}
                      fill="#38bdf8"
                      stroke="#e0f2fe"
                      strokeWidth={1.5}
                      shape="circle"
                      label={{
                        dataKey: "symbol",
                        position: "top",
                        fill: "#ffffff",
                        fontSize: 11,
                        fontWeight: 600,
                        offset: 8,
                      }}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Lợi nhuận năm (dài) | Phân bổ tài sản (ngắn) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-zinc-900 p-4 sm:p-6 rounded-2xl border border-zinc-800">
              <h3 className="mb-4 font-medium text-white">
                Lợi nhuận hàng năm
              </h3>
              <div className="h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={annualReturns || []}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" opacity={0.4} />
                    <XAxis
                      dataKey="year"
                      stroke="#cbd5e1"
                      tick={{ fill: "#e2e8f0", fontSize: 12 }}
                    />
                    <YAxis
                      stroke="#cbd5e1"
                      tick={{ fill: "#e2e8f0", fontSize: 12 }}
                      tickFormatter={(v) => fmtPct(Number(v))}
                      tickCount={5}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#18181b",
                        border: "1px solid #3f3f46",
                        color: "#ffffff",
                      }}
                      formatter={(v: number) => [fmtPct(Number(v)), "Lợi nhuận"]}
                    />
                    <Bar
                      dataKey="annual_return"
                      fill="#38bdf8"
                      name="Lợi nhuận"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="lg:col-span-1 bg-zinc-900 p-4 sm:p-6 rounded-2xl border border-zinc-800">
              <h3 className="mb-4 font-medium text-white">
                Phân bổ danh mục tài sản
              </h3>
              <div className="h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Customized
                      component={(props: { width?: number; height?: number }) => {
                        const w = props.width ?? 400;
                        const h = props.height ?? 240;
                        return (
                          <text
                            x={w / 2}
                            y={h / 2}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fill="#e2e8f0"
                            fontSize={14}
                            fontWeight={600}
                          >
                          </text>
                        );
                      }}
                    />
                    <Pie
                      data={result.allocation}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={95}
                      paddingAngle={3}
                      dataKey="weight"
                      nameKey="asset"
                      onMouseEnter={(_: unknown, index: number) => setActiveIndex(index)}
                      onMouseLeave={() => setActiveIndex(null)}
                    >
                      {result.allocation.map((_, i) => (
                        <Cell
                          key={i}
                          fill={COLORS[i % COLORS.length]}
                          opacity={activeIndex === null ? 1 : i === activeIndex ? 1 : 0.35}
                        />
                      ))}
                    </Pie>
                    <Legend
                      verticalAlign="bottom"
                      align="center"
                      iconType="circle"
                      iconSize={10}
                      formatter={(value) => value}
                      wrapperStyle={{
                        marginTop: 20,   
                        paddingTop: 10,
                        color: "#e2e8f0",
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "transparent",
                        border: "1px solid rgba(255,255,255,0.7)",
                        borderRadius: "10px",
                        padding: "6px 10px",
                      }}
                      itemStyle={{ color: "#ffffff", fontWeight: 500 }}
                      labelStyle={{ color: "#ffffff", fontWeight: 600 }}
                      cursor={{ fill: "rgba(255,255,255,0.05)" }}
                      formatter={(v: number, n: string) => [fmtPct(Number(v)), n]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer học thuật */}
      <Footer />
    </div>
  );
};

export default PortfolioOptimizer;
