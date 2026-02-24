# API tìm kiếm công ty từ company.xlsx
# Cột B (index 1) = tên công ty, Cột E (index 4) = mã chứng khoán

import pandas as pd
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["company"])

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EXCEL_PATH = PROJECT_DIR / "data_raw" / "company.xlsx"

_df_cache = None


def _load_company_df():
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    if not EXCEL_PATH.exists():
        return None
    try:
        df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
        cols = df.columns.tolist()
        # Ưu tiên tên cột, fallback theo index (B=1, E=4)
        if "Mã chứng khoán" in df.columns:
            df["ma_ck"] = df["Mã chứng khoán"]
        elif "ma_ck" in [c.lower() for c in cols]:
            idx = [c.lower() for c in cols].index("ma_ck")
            df["ma_ck"] = df.iloc[:, idx]
        elif len(cols) >= 5:
            df["ma_ck"] = df.iloc[:, 4]
        else:
            return None
        if "Tên công ty" in df.columns:
            df["ten_cong_ty"] = df["Tên công ty"]
        elif len(cols) >= 2:
            df["ten_cong_ty"] = df.iloc[:, 1]
        else:
            df["ten_cong_ty"] = ""
        df["ma_ck"] = df["ma_ck"].astype(str).str.strip().str.upper()
        df["ten_cong_ty"] = df["ten_cong_ty"].astype(str).str.strip()
        _df_cache = df
        return df
    except Exception:
        return None


@router.get("/search-company")
def search_company(query: str):
    """Tìm kiếm công ty theo mã CK. Trả về [{ma, ten}, ...]"""
    df = _load_company_df()
    if df is None or "ma_ck" not in df.columns:
        return []
    q = str(query).strip().upper()
    if not q:
        return []
    # Tìm theo mã CK hoặc tên công ty
    mask_ma = df["ma_ck"].str.contains(q, na=False, case=False)
    mask_ten = df["ten_cong_ty"].str.upper().str.contains(q, na=False)
    filtered = df[mask_ma | mask_ten].head(10)
    return [
        {"ma": row["ma_ck"], "ten": row["ten_cong_ty"]}
        for _, row in filtered.iterrows()
    ]
