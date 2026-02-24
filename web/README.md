# AI-Driven Regime-Aware Portfolio Optimization — Frontend

Web platform trình bày toàn cảnh thị trường theo Risk Regime, tối ưu danh mục và AI Chatbot giải thích (không khuyến nghị mua/bán).

**Stack:** React 18, Vite, TypeScript, Recharts, Lucide Icons.

## Chạy local

**Yêu cầu:** Node.js (khuyến nghị LTS).

1. Cài dependency:
   ```bash
   npm install
   ```
2. Chạy dev:
   ```bash
   npm run dev
   ```
   Ứng dụng chạy tại `http://localhost:3000`. Request `/api/*` được proxy tới `http://localhost:8000` (cần backend chạy sẵn).

3. Build:
   ```bash
   npm run build
   ```
4. Xem bản build:
   ```bash
   npm run preview
   ```

## Kiến trúc

- **Frontend** chỉ gửi input, nhận output và hiển thị. Không gọi Gemini trực tiếp, không tính toán tài chính.
- **Backend** (Python/Node) chịu trách nhiệm: Market Regime, Portfolio Optimizer, PDF Report, Gemini AI Gateway.
- Chi tiết: xem [ARCHITECTURE.md](./ARCHITECTURE.md).

## Cấu trúc chính

- `components/` — MarketOverview, PortfolioOptimizer, AIAdvisor (3 tab).
- `services/` — `portfolioService` (optimize + tải PDF), `geminiService` (chat qua `/api/advisor/chat`).
- `types.ts` — Data contract frontend ↔ backend (snake_case).
- `constants.ts` — ASSET_LIST, MOCK_MARKET_DATA (chỉ dùng khi chưa gắn API market).

## Backend cần cung cấp

- `GET /api/market/overview` — MarketOverviewData
- `POST /api/portfolio/optimize` — Body: PortfolioRequest → PortfolioResult
- `GET /api/report/download` hoặc `GET /api/report/pdf` — file PDF
- `POST /api/advisor/chat` — Body: { message, context, history } → { answer }

API key Gemini chỉ đặt ở backend; frontend không chứa key.
