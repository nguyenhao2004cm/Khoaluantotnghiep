import requests
import pandas as pd
import random
import time
from datetime import datetime, timedelta
import re

url = "https://sjc.com.vn/GoldPrice/Services/PriceService.ashx"

session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://sjc.com.vn",
    "Referer": "https://sjc.com.vn/bieu-do-gia-vang",
}

# chuyển /Date(...)/ → datetime
def convert_sjc_date(date_str):
    try:
        ts = int(re.search(r"\d+", date_str).group())
        return datetime.fromtimestamp(ts / 1000)
    except:
        return None

# ===========================
# HÀM FETCH CHUẨN CHÍNH XÁC
# ===========================
def fetch_chunk(start_date, end_date, retries=5):

    payload = {
        "method": "GetGoldPriceHistory",
        "goldPriceId": "1",
        "fromDate": start_date.strftime("%d/%m/%Y"),   # ĐÚNG FORMAT
        "toDate": end_date.strftime("%d/%m/%Y"),
        "pageSize": "5000",    # BẮT BUỘC
        "pageIndex": "1"       # BẮT BUỘC
    }

    for attempt in range(retries):
        try:
            res = session.post(url, data=payload, headers=headers, timeout=10)
            data = res.json()

            # Khi sai format → message Invalid Value
            if not data.get("success"):
                print("⚠ API trả về lỗi:", data)
                return []

            print(f"✔ Lấy thành công {payload['fromDate']} -> {payload['toDate']}")
            return data["data"]

        except Exception as e:
            print(f"❌ Lỗi lần {attempt+1}: {e}")
            time.sleep(random.uniform(1, 3))

    return []

# ===========================
# CHẠY TỪ 2015 → 2025
# ===========================
start = datetime(2015, 1, 1)
end = datetime(2025, 12, 12)

all_rows = []
cur_start = start

while cur_start <= end:

    cur_end = min(cur_start + timedelta(days=89), end)
    print(f"\nĐang lấy block: {cur_start.strftime('%d/%m/%Y')} -> {cur_end.strftime('%d/%m/%Y')}")

    rows = fetch_chunk(cur_start, cur_end)

    for r in rows:
        r["GroupDate"] = convert_sjc_date(r["GroupDate"])

    all_rows.extend(rows)

    time.sleep(random.uniform(1, 2))

    cur_start = cur_end + timedelta(days=1)


df = pd.DataFrame(all_rows)
df.to_csv("sjc_history_full.csv", index=False, encoding="utf-8-sig")

print("\nDONE — Tổng số dòng:", len(df))
