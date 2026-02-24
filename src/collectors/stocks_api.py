import os
import pandas as pd
from datetime import timedelta
from vnstock import Vnstock

# ================================
# CONFIG
# ================================
DATA_DIR = "data_raw/stocks"
os.makedirs(DATA_DIR, exist_ok=True)

PRICE_COLS = ["open", "high", "low", "close"]


# ================================
# UTILS
# ================================
def normalize_price_unit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa đơn vị giá:
    - Nếu giá < 100 → đang ở đơn vị 'nghìn đồng' → nhân 1000
    """
    median_price = df[PRICE_COLS].median().median()

    if median_price < 100:
        df[PRICE_COLS] = df[PRICE_COLS] * 1000

    return df


# ================================
# CORE API
# ================================
def get_stock_history(symbol: str, start="2018-01-01", end=None):

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")

        df = stock.quote.history(
            start=start,
            end=end,
            interval="1D"
        )

        if df is None or df.empty:
            return None

        if "time" in df.columns:
            df.rename(columns={"time": "date"}, inplace=True)
        elif "date" not in df.columns:
            raise ValueError("Không tìm thấy cột date/time")

        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = normalize_price_unit(df)
        df["symbol"] = symbol

        return df

    except Exception as e:
        print(f" Lỗi khi lấy {symbol}: {e}")
        return None


# ================================
# UPDATE LOGIC
# ================================
def update_stock_daily(symbol: str):
    """
    - Chưa có file → tải full
    - Có file → append ngày mới
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    # ===== LẦN ĐẦU =====
    if not os.path.exists(file_path):
        print(f"⬇️ Tải lần đầu: {symbol}")
        df = get_stock_history(symbol)
        if df is not None:
            df.to_csv(file_path, index=False)
        return

    # ===== UPDATE =====
    df_old = pd.read_csv(file_path, parse_dates=["date"])
    last_date = df_old["date"].max()

    start_new = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    if pd.to_datetime(start_new) > pd.to_datetime(end):
        print(f" {symbol} đã cập nhật đến {last_date.date()}")
        return

    print(f"🔄 Cập nhật {symbol}: {start_new} → {end}")
    df_new = get_stock_history(symbol, start=start_new, end=end)


    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.drop_duplicates(subset=["date"], inplace=True)
    df_all.sort_values("date", inplace=True)

    df_all.to_csv(file_path, index=False)
