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
def _get_price_cols(df: pd.DataFrame) -> list:
    """Lấy các cột giá có trong df."""
    return [c for c in PRICE_COLS if c in df.columns]


def normalize_price_unit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa đơn vị giá an toàn:
    - Nếu giá trung vị < 1000 → khả năng ở đơn vị nghìn → nhân 1000
    - Nếu giá > 10 triệu → có thể lỗi → chia 1000
    """
    cols = _get_price_cols(df)
    if not cols:
        return df

    median_price = df[cols].median().median()

    if median_price < 1000:
        df[cols] = df[cols] * 1000
    elif median_price > 10_000_000:
        df[cols] = df[cols] / 1000

    return df


def validate_price_scale(df: pd.DataFrame, symbol: str) -> None:
    """
    Validation phát hiện lỗi scale.
    Cổ phiếu VN: ngân hàng 10k–60k, bluechip 40k–150k, penny < 10k.
    """
    cols = _get_price_cols(df)
    if not cols or "close" not in df.columns:
        return

    med = df["close"].median()
    if med < 1000:
        print(f"  {symbol}: gia qua nho (median={med:.0f}) -> co the sai don vi")
    elif med > 5_000_000:
        print(f"  {symbol}: gia qua lon (median={med:.0f}) -> can kiem tra")


# ================================
# CORE API
# ================================
def get_stock_history(symbol: str, start="2015-01-01", end=None):

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
    - Chưa có file → tải full từ 2015
    - Có file → chuẩn hóa data cũ, append ngày mới
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    # ========= FIRST BUILD =========
    if not os.path.exists(file_path):
        print(f"  Build full history: {symbol}")
        df = get_stock_history(symbol, start="2015-01-01")
        if df is not None:
            validate_price_scale(df, symbol)
            df.to_csv(file_path, index=False)
        return

    # ========= LOAD OLD =========
    df_old = pd.read_csv(file_path, parse_dates=["date"])
    df_old = normalize_price_unit(df_old)

    last_date = df_old["date"].max()
    start_new = last_date + timedelta(days=1)
    end = pd.Timestamp.today()

    if start_new > end:
        print(f"  {symbol} up-to-date")
        return

    print(f"  Updating {symbol}: {start_new.date()} -> {end.date()}")
    df_new = get_stock_history(symbol, start=start_new.strftime("%Y-%m-%d"))

    if df_new is None:
        return

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.drop_duplicates(subset=["date"], inplace=True)
    df_all.sort_values("date", inplace=True)

    validate_price_scale(df_all, symbol)
    df_all.to_csv(file_path, index=False)
