# Đồng bộ dữ liệu Power BI / portfolio_timeseries

## Vấn đề
- **Dữ liệu gốc** (data_raw, prices.csv) đã cập nhật tới **13/2/2026**
- **portfolio_timeseries.csv** chỉ tới **16/1/2026**

## Nguyên nhân
`portfolio_timeseries.csv` được tạo bởi `export_powerbi_data.py` → `build_portfolio()`.
Chuỗi phụ thuộc:
```
prices.csv → risk_features → latent → risk → risk_normalized → risk_regime
                                                              → signal
                                                              → portfolio_allocation
                                                              → portfolio_timeseries
```

Các file trung gian (latent, risk, risk_regime, signal, allocation) chỉ được cập nhật khi **chạy lại pipeline**. Nếu pipeline chưa chạy sau khi cập nhật raw data, portfolio_timeseries sẽ dừng ở ngày cũ.

## Giải pháp
**Chạy lại toàn bộ pipeline** để đồng bộ dữ liệu:

```powershell
python run_all.py
```

Hoặc chạy từ Stage 1 (sau khi đã cập nhật data_raw):

```powershell
python src/data/prepare_prices.py
python src/features/build_features.py
python src/features/build_risk_features.py
# ... tiếp tục theo run_all.py
```

## Lưu ý
- `encode_latent.py` đọc từ `data_processed/risk_features` (đã sửa từ `data_processed/features`)
- `export_powerbi_data.py` không có logic lọc ngày – nó chỉ xuất theo allocation + prices hiện có
