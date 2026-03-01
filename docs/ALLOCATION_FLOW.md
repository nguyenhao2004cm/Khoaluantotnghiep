# Luồng phân bổ danh mục đầu tư

## File chứa trọng lượng danh mục (nguồn chính)

| File | Mô tả | Được tạo bởi |
|------|-------|--------------|
| **`data_processed/portfolio/portfolio_allocation_final.csv`** | Trọng lượng phân bổ (date, symbol, allocation_weight) | `src/decision/build_portfolio_allocation.py` |

→ Đây là file chính chứa tỷ trọng các cổ phiếu.

---

## Thư mục `data_processed/decision` — KHÔNG phải nguồn trọng lượng

| File | Mô tả | Được tạo bởi |
|------|-------|--------------|
| `signal.csv` | Tín hiệu trading (đầu vào cho allocation) | `src/decision/build_signal.py` |
| `*_weight.csv` (VNM_weight.csv, ...) | Legacy / backtest đơn lẻ | Không nằm trong pipeline `run_all` |

→ Các file `*_weight.csv` **không** được tạo lại bởi `run_all.py`.

---

## Luồng chạy trong `run_all.py`

```
Stage 4: build_signal.py          → signal.csv
Stage 5: build_portfolio_risk_regime.py → risk_regime_timeseries.csv
Stage 5: build_portfolio_allocation.py  → portfolio_allocation_final.csv  ← TRỌNG LƯỢNG
Stage 8: export_powerbi_data.py   → asset_allocation_current.csv (copy cho web)
```

---

## Code tính trọng lượng

| File code | Vai trò |
|-----------|---------|
| `src/decision/build_portfolio_allocation.py` | Điều phối, ghi `portfolio_allocation_final.csv` |
| `src/portfolio_engine/risk_based_optimizer.py` | Tính ERC/MinVar, covariance theo regime |

---

## Cách chạy lại để cập nhật trọng lượng

```bash
python run_all.py
```

Hoặc chỉ chạy allocation:

```bash
python -m src.decision.build_portfolio_allocation
```

→ File `data_processed/portfolio/portfolio_allocation_final.csv` sẽ được ghi lại.
