# Quy trình & Cấu trúc tích hợp Web Frontend với Backend Python

**Vai trò tài liệu:** Thiết kế hệ thống / Tài liệu kỹ thuật tích hợp (có thể đưa vào luận văn).

---

## I. Nguyên tắc kiến trúc (bất biến)

### Nguyên tắc 1 – Python là “single source of truth”

- Frontend **không** tính toán tài chính.
- Frontend **không** suy luận rủi ro.
- Frontend **không** tối ưu hóa danh mục.

**→ Web chỉ là presentation layer + user input layer.**

---

### Nguyên tắc 2 – Không làm lại pipeline AI

- Autoencoder, Risk regime, Allocation, Backtest, Statistics, PDF report: **giữ nguyên** toàn bộ code hiện tại.
- **Chỉ** bọc (wrap) pipeline bằng REST API.

---

### Nguyên tắc 3 – Giao tiếp duy nhất qua REST API

- JSON vào / JSON ra.
- PDF download qua HTTP.

---

## II. Kiến trúc tổng thể sau khi ghép Web

```
┌─────────────────────────┐
│  React Frontend         │  (Trang web/)
│  - MarketOverview       │
│  - PortfolioOptimizer   │
│  - AIAdvisor            │
└───────────▲─────────────┘
            │ REST (JSON / PDF)
            ▼
┌────────────────────────────────┐
│ Python Backend (FastAPI)        │
│                                 │
│  API Layer (src/api/)           │
│  ├── market.py                  │
│  ├── portfolio.py               │
│  ├── report.py                  │
│  └── chat.py                    │
│                                 │
│  Core AI Engine (EXISTING)       │
│  ├── risk engine                │
│  ├── optimizer                  │
│  ├── backtest                   │
│  └── statistics                 │
└────────────────────────────────┘
```

**Lưu ý:** Web **không** “nhúng” Python; Web **gọi** Python qua API.

---

## III. Cấu trúc thư mục cần bổ sung (Backend)

### Thêm thư mục API (không động core)

```
src/
├── api/
│   ├── __init__.py
│   ├── app.py          # FastAPI app + CORS + mount routers
│   ├── schemas.py      # Pydantic models (mirror types.ts)
│   ├── market.py      # GET /api/market/overview
│   ├── portfolio.py    # POST /api/portfolio/optimize
│   ├── report.py      # GET /api/report/download
│   └── chat.py        # POST /api/advisor/chat
├── reporting/         # (existing)
├── decision/          # (existing)
├── portfolio_engine/  # (existing)
└── ...
```

**Vai trò:** `src/api/` là **adapter layer** giữa web và AI engine.

---

## IV. Data contract – schemas.py

File `src/api/schemas.py` **phải mirror 100%** với `Trang web/types.ts`.

- **PortfolioRequest:** `start_date`, `end_date`, `assets`, `risk_appetite` (conservative | balanced | aggressive).
- **PortfolioResult:** `start_date`, `end_date`, `regime`, `strategy` (defensive | balanced | aggressive), `metrics` (cagr, max_drawdown, cvar_5), `allocation`, `growth_series`.
- **ChatContext / ChatMessage** cho endpoint chat.

**Đây chính là data contract** giữa frontend và backend.

---

## V. API cụ thể tương ứng 3 tab Web

### Tab 1 – Market Overview

| Mục | Nội dung |
|-----|----------|
| **API** | `GET /api/market/overview` |
| **Backend** | Load `data_processed/reporting/risk_regime_timeseries.csv`, risk_normalized, sector aggregation → trả JSON. |
| **Response** | `current_regime`, `regime_stability_days`, `risk_score_series`, `regime_timeline`, `heatmap`, `sector_risks`, `summary` (high_ratio, normal_ratio, low_ratio). |
| **Frontend** | `MarketOverview.tsx` chỉ render; khi có backend thì thay `MOCK_MARKET_DATA` bằng `fetch('/api/market/overview')`. |

---

### Tab 2 – Portfolio Optimizer (quan trọng nhất)

| Mục | Nội dung |
|-----|----------|
| **API** | `POST /api/portfolio/optimize` |
| **Request** | `{ start_date, end_date, assets[], risk_appetite }` (JSON). |
| **Backend** | Ghi `web_run_config.json` (assets, start_date, end_date, risk_appetite) → chạy pipeline: `build_portfolio_allocation` (đọc config) → `export_powerbi_data` → reporting steps → `report_pdf` (đọc config cho INVESTMENT_START_DATE). Trả JSON + PDF trong `Reports/`. |
| **Response** | JSON: `start_date`, `end_date`, `regime`, `strategy`, `metrics` (cagr, max_drawdown, cvar_5), `allocation[]`, `growth_series[]`. |
| **Frontend** | `PortfolioOptimizer.tsx` gọi `portfolioService.optimize(payload)` → render kết quả; nút "Xuất PDF" gọi `GET /api/report/pdf` hoặc `/api/report/download`. |

---

### Tab 3 – AI Advisor

| Mục | Nội dung |
|-----|----------|
| **API** | `POST /api/advisor/chat` |
| **Request** | `{ message, context, history }` (context: market_regime, user_assets, portfolio_metrics, current_strategy). |
| **Backend** | Nhận request → gọi Gemini/LLM với system prompt (regime, metrics, strategy) → trả text. **API key Gemini chỉ ở backend.** |
| **Response** | `{ answer: string }`. |
| **Frontend** | `AIAdvisor.tsx` gọi `geminiService.getAdvice(message, context, history)` → hiển thị `answer`. |

---

## VI. PDF Report

| Mục | Nội dung |
|-----|----------|
| **API** | `GET /api/report/download` |
| **Backend** | Dùng lại `src/reporting/` (matplotlib, reportlab, report_pdf.py). Render report từ kết quả lần optimize gần nhất (hoặc từ session/cache) → trả file PDF. |
| **Response** | `FileResponse(..., media_type="application/pdf", filename="AI_Portfolio_Report.pdf")`. |
| **Frontend** | Link `<a href="/api/report/download" target="_blank">Download PDF</a>` hoặc nút gọi `portfolioService.downloadReport()` rồi trigger download. |

**Giữ nguyên triết lý pipeline PDF ban đầu;** API chỉ expose qua HTTP.

---

## VII. Các file / module không cần sửa

**Không đụng:**

- Autoencoder, Risk regime, Allocation logic (decision/, models/).
- Statistical tests (stat_tests/).
- Backtest code (backtest/, portfolio_engine/).

**Chỉ wrap chúng bằng API** (gọi từ `src/api/portfolio.py`, `market.py`, v.v.).

---

## VIII. Checklist triển khai

### Backend

- [ ] Thêm FastAPI app (`src/api/app.py`), CORS cho origin frontend.
- [ ] Viết 4 endpoint: market, portfolio, report, chat.
- [ ] `schemas.py` mirror `types.ts` (PortfolioRequest, PortfolioResult, ChatContext, ChatMessage).
- [ ] Map pipeline hiện có vào `portfolio.py` (start_date, end_date, assets, risk_appetite → gọi decision + portfolio_engine + reporting).
- [ ] Export PDF từ kết quả optimize (report_pdf.py hoặc tương đương) → `GET /api/report/download`.

### Frontend

- [ ] MarketOverview: thay `MOCK_MARKET_DATA` bằng `fetch('/api/market/overview')` khi backend sẵn sàng.
- [ ] PortfolioOptimizer: đã gọi `POST /api/portfolio/optimize`; kiểm tra loading / error state.
- [ ] AIAdvisor: đã gọi `POST /api/advisor/chat`; không đổi UI.
- [ ] Nút/link PDF: `GET /api/report/download` hoặc `downloadReport()` blob.

### Chạy tích hợp (để có dữ liệu thật và PDF)

1. **Chạy pipeline một lần** (từ thư mục gốc dự án) để có signal, regime, powerbi, PDF:
   ```bash
   python run_all.py
   ```
   Sau bước này trong `Reports/` sẽ có file PDF; `data_processed/powerbi/` có portfolio_timeseries, allocation, v.v.

2. **Khởi động backend:**
   ```bash
   uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Khởi động frontend:** trong `Trang web/` chạy `npm run dev` (Vite proxy `/api` → `http://localhost:8000`).

4. **Mở trình duyệt:** `http://localhost:3000` → tab Tối ưu danh mục → nhập danh sách cổ phiếu, thời gian → Khởi chạy. Kết quả và biểu đồ trên web = dữ liệu thật từ pipeline; nút "Xuất Báo cáo PDF" tải file từ `Reports/`.

---

## IX. Tóm tắt luồng nghiệp vụ

1. **User** mở web → chọn tab Market / Portfolio / AI.
2. **Market:** Frontend gọi `GET /api/market/overview` → hiển thị regime, timeline, heatmap.
3. **Portfolio:** User nhập start_date, end_date, assets, risk_appetite → Frontend gửi `POST /api/portfolio/optimize` → Backend chạy pipeline AI → trả JSON → Frontend hiển thị metrics, allocation, growth_series; user bấm “Xuất PDF” → `GET /api/report/download` → tải file.
4. **AI:** User gửi câu hỏi + context → Frontend gửi `POST /api/advisor/chat` → Backend gọi Gemini → trả `answer` → Frontend hiển thị.

Toàn bộ logic tài chính và AI nằm ở backend; frontend chỉ nhập liệu, gọi API và hiển thị.
