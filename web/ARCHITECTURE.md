# Kiến trúc hệ thống — AI-Driven Regime-Aware Portfolio Optimization

## I. Mục tiêu tổng thể

Xây dựng **AI-Driven Regime-Aware Portfolio Optimization Web Platform** nhằm:

1. Trình bày toàn cảnh thị trường chứng khoán Việt Nam theo chế độ rủi ro (Risk Regime).
2. Cho phép người dùng nhập danh mục + thời gian đầu tư để tối ưu hóa danh mục.
3. Cung cấp AI Chatbot để:
   - Giải thích kết quả
   - Tư vấn chiến lược theo Regime
   - Không vi phạm quy tắc học thuật (không khuyến nghị mua/bán cụ thể)

---

## II. Kiến trúc tổng thể (bắt buộc)

```
┌─────────────────────────┐
│        FRONTEND         │  (React 18 + Vite)
│─────────────────────────│
│ Tab 1: Market Overview  │
│ Tab 2: Portfolio Input  │
│ Tab 3: AI Chatbot        │
└───────────┬─────────────┘
            │ HTTP / JSON (proxy /api → backend)
┌───────────▼─────────────┐
│        BACKEND          │  (Python / Node)
│─────────────────────────│
│ Market Regime Engine     │
│ Portfolio Optimizer      │
│ Performance Metrics      │
│ PDF Generator            │
│ Gemini AI Gateway        │
└─────────────────────────┘
```

**Nguyên tắc bắt buộc:**

- Frontend **không** gọi trực tiếp `@google/genai`.
- Frontend **không** tính toán tài chính.
- Frontend chỉ: gửi input, nhận output, hiển thị + giải thích.

---

## III. Phân rã chức năng theo 3 tab

### Tab 1 — Toàn cảnh thị trường (Market Overview)

**Mục tiêu:** Cung cấp bức tranh rủi ro vĩ mô (Regime hiện tại, thời gian ổn định, rủi ro theo ngành).

**Dữ liệu Backend → Frontend:** `MarketOverviewData` (trong `types.ts`):

- `current_regime`, `regime_stability_days`
- `risk_score_series`, `regime_timeline`
- `sector_risks`, `heatmap`, `summary`

**Nguồn dữ liệu backend:** Giá lịch sử HOSE, Risk Score từ Autoencoder + Regime Switching Model.

**API:** `GET /api/market/overview` (khi gắn backend). Hiện tại dùng `MOCK_MARKET_DATA` từ `constants.ts`.

**Frontend:** Chỉ hiển thị (line chart Risk Score, Regime Timeline, Sector Risk Cards, Heatmap). Không input từ user ở tab này.

---

### Tab 2 — Tối ưu danh mục (Portfolio Optimization)

**Mục tiêu:** Người dùng chọn tài sản, khoảng thời gian; hệ thống tối ưu danh mục theo Regime.

**Input Frontend → Backend:** `PortfolioRequest` (trong `types.ts`):

```json
{
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "assets": ["FPT", "VNM", "HPG"],
  "risk_appetite": "conservative" | "balanced" | "aggressive"
}
```

`start_date` là **bắt buộc**.

**API:** `POST /api/portfolio/optimize`

**Output Backend → Frontend:** `PortfolioResult`:

- `start_date`, `end_date`, `regime`, `strategy`
- `metrics`: `cagr`, `max_drawdown`, `cvar_5` (snake_case, khớp backend)
- `allocation`: `{ asset, weight }[]`
- `growth_series`: `{ date, value }[]`

**PDF:** Backend sinh báo cáo. Frontend: `GET /api/report/download` hoặc `GET /api/report/pdf` (tùy backend). Nút "Xuất Báo cáo PDF" gọi service tải blob.

---

### Tab 3 — AI Chatbot (Explainability Layer)

**Mục tiêu:** Giải thích kết quả, trả lời câu hỏi theo Regime, **không** khuyến nghị mua/bán.

**Luồng:** User hỏi → Frontend gửi `message` + `context` + `history` → Backend (Gemini) → Frontend hiển thị `answer`.

**API:** `POST /api/advisor/chat`

**Context gửi cho AI:** `ChatContext` (market_regime, user_assets, portfolio_metrics, current_strategy). Gemini **chỉ chạy ở backend**.

**Quy tắc AI (học thuật):** Không dự đoán giá; không bảo mua/bán; chỉ giải thích logic & rủi ro; ngôn ngữ học thuật. System prompt đặt ở backend.

---

## IV. Cấu trúc thư mục

**Frontend (hiện tại):**

```
Trang web/
├── components/
│   ├── MarketOverview.tsx
│   ├── PortfolioOptimizer.tsx
│   └── AIAdvisor.tsx
├── services/
│   ├── portfolioService.ts   # POST /api/portfolio/optimize, GET report
│   └── geminiService.ts      # POST /api/advisor/chat (fetch, không Gemini trực tiếp)
├── types.ts
├── constants.ts
├── App.tsx
├── index.tsx
└── vite.config.ts            # proxy /api → http://localhost:8000
```

**Backend (đề xuất):**

```
backend/
├── api/
│   ├── market.py
│   ├── portfolio.py
│   ├── report.py
│   └── ai_chat.py
├── engine/
│   ├── regime_detection.py
│   ├── optimizer.py
│   └── performance_metrics.py
```

---

## V. Tóm tắt API Frontend sử dụng

| Tab / Chức năng   | Method | Endpoint                  | Ghi chú                    |
|-------------------|--------|---------------------------|----------------------------|
| Market Overview   | GET    | /api/market/overview      | Khi có backend; hiện mock  |
| Portfolio Optimize| POST   | /api/portfolio/optimize   | Body: PortfolioRequest     |
| PDF Report        | GET    | /api/report/download hoặc /api/report/pdf | Trả về file PDF |
| AI Chat           | POST   | /api/advisor/chat         | Body: message, context, history → answer |

Frontend không biết chi tiết implementation backend; chỉ gọi API theo contract trong `types.ts`.
