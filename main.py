import time
from datetime import datetime

from src.collectors.stocks_api import update_stock_daily
from src.collectors.symbols_api import get_hose_symbols
from src.features.build_features import run as build_features


TICKERS = get_hose_symbols()


def run_daily_pipeline():
    today = datetime.now().strftime("%Y-%m-%d")

    print("\n===============================")
    print("UPDATE STOCK DATA")
    print("===============================")

    for i, t in enumerate(TICKERS, 1):
        try:
            print(f"[{i}/{len(TICKERS)}] {t}")
            update_stock_daily(t)

            #  CHỐNG RATE LIMIT
            time.sleep(7)

        except Exception as e:
            print(f" Lỗi {t}: {e}")
            print(" Ngủ 10s do lỗi rate limit...")
            time.sleep(10)

    print("\n===============================")
    print("BUILD FEATURES")
    print("===============================")

    build_features()


    print("\n===============================")
    print(" PIPELINE HOÀN TẤT")
    print("===============================\n")


if __name__ == "__main__":
    run_daily_pipeline()
                        