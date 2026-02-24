import pandas as pd
from pathlib import Path


def get_hose_symbols():
    file_path = Path("data_raw/company.xlsx")

    if not file_path.exists():
        print(f" Không tìm thấy file: {file_path}")
        return []

    try:
        # Đọc Excel chuẩn
        df = pd.read_excel(file_path, engine="openpyxl")

        symbol_col = "Mã chứng khoán"

        if symbol_col not in df.columns:
            print(" Không tìm thấy cột 'Mã chứng khoán'")
            print(" Các cột hiện có:", df.columns.tolist())
            return []

        symbols = (
            df[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
            .tolist()
        )

        print(f" Đã lấy {len(symbols)} mã cổ phiếu từ company.xlsx")

        return symbols

    except Exception as e:
        print(" Lỗi đọc file company.xlsx:", e)
        print(" Kiểm tra xem file có đúng là Excel hay không")
        return []
